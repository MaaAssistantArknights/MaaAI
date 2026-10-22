"""Unified, bounded PaddleOCR fine-tuning trial with immutable data and guard sets.

No auto-extension, hidden AMP, label truncation, or inference-runtime change.
"""

import argparse, copy, json, os, random, sys, time
from collections import Counter
from pathlib import Path

os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.70")
# This entry always consumes a frozen plan; keep stochastic kernel policy fixed.
os.environ.setdefault("FLAGS_cudnn_deterministic", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import cv2, numpy as np, paddle, yaml
from workspace import ROOT, RT

sys.path.insert(0, str(RT / "vendor/PaddleOCR"))
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.postprocess import build_post_process
from label_contract import load_encoder
from training_batches import prepare_training_batches
from dataset_contract import load_contract, training_partition
from released_input import width_groups
from machine_resources import (
    gpu_status,
    load_policy,
    require_safe_temperature,
    require_training_capacity,
)
from file_utils import sha, acquire_training_lock
from recipe_settings import load_recipe


def write(path, obj):
    """原子写入训练过程的 JSON 状态。"""
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def read(path):
    """读取训练和保留能力样本的逐行 JSON 清单。"""
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    """校验冻结计划并运行有限步数训练、选版和失败停止。"""
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["char", "word"], required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--language", choices=["zh-CN", "zh-TW", "en", "ja", "ko"])
    p.add_argument("--training-profile", type=Path, required=True)
    p.add_argument("--sampling", choices=["historical", "balanced"])
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument(
        "--input-policy",
        choices=["fixed", "released"],
        default="fixed",
    )
    p.add_argument("--run-plan", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pause-at", type=int)
    p.add_argument("--seed", type=int, default=20260920)
    a = p.parse_args()
    from training_state import (
        validate_plan,
        save_state,
        inspect_state,
        restore_state,
    )

    training_lock = acquire_training_lock(RT)
    plan = json.loads(a.run_plan.read_text())
    validate_plan(a, plan)
    if a.pause_at is not None and not 0 < a.pause_at < a.steps:
        raise ValueError("pause must precede completion")
    profiles = json.loads(a.training_profile.read_text())
    if a.language and a.kind != "word":
        raise ValueError("language profile applies to WordOCR")
    contract = ROOT / profiles["contract"]
    contract_data = load_contract(contract)
    input_shape = contract_data["input_shape"]
    ctc_steps = contract_data["ctc_time_steps"]
    max_length = contract_data["encoder_max_length"]
    if a.kind == "char" and profiles.get("char"):
        input_shape = contract_data["char_input_shape"]
        ctc_steps = contract_data["char_ctc_time_steps"]
        max_length = contract_data["char_encoder_max_length"]
    audit = json.loads((a.dataset / "audit.json").read_text())
    # A successful generator run alone is insufficient: bind training to the
    # exact admitted manifest and rendering recipe that were reviewed.
    if (
        not audit["passed"]
        or audit["admission"] != "bounded_trial_only"
        or audit["manifest_sha256"] != sha(a.dataset / "manifest.jsonl")
        or audit["recipe_sha256"] != sha(a.dataset / "recipe.json")
    ):
        raise ValueError("unvalidated/changed dataset")
    resource_policy_path = Path(plan["resource_policy"])
    resource_policy = load_policy(resource_policy_path)
    state0 = require_training_capacity(RT, resource_policy)
    if a.resume:
        if not a.output.is_dir() or (a.output / "result.json").exists():
            raise ValueError("resume needs an unfinished run")
    else:
        a.output.mkdir(parents=True, exist_ok=False)
    model_kind = (
        profiles["languages"][a.language]["model"]
        if a.language
        else profiles["char"]["model"]
    )
    dictionary = (
        profiles["languages"][a.language].get("dictionary") if a.language else None
    )
    encoder, encoder_meta = load_encoder(
        model_kind, max_length, RT / dictionary if dictionary else None
    )
    recipe = json.loads((a.dataset / "recipe.json").read_text())
    if (
        a.language
        and recipe.get("profile_sha256")
        and recipe["profile_sha256"] != sha(a.training_profile)
    ):
        raise ValueError("dataset training profile differs")
    if (
        a.language
        and recipe.get("encoder_metadata", {})
        .get(model_kind, {})
        .get("dictionary_sha256", encoder_meta["dictionary_sha256"])
        != encoder_meta["dictionary_sha256"]
    ):
        raise ValueError("dataset dictionary differs")
    config = Path(plan["model_config"])
    checkpoint = Path(plan["checkpoint"])
    cfg = yaml.safe_load(config.read_text())
    if a.language:
        cfg["Global"].update(
            character_dict_path=encoder_meta["dictionary_path"],
            max_text_length=max_length,
            d2s_train_image_shape=input_shape,
            distributed=False,
        )
        cfg["Architecture"]["Head"]["out_channels_list"] = {
            "CTCLabelDecode": len(encoder.ctc_encode.character),
            "NRTRLabelDecode": len(encoder.gtc_encode.character),
        }
        for head in cfg["Architecture"]["Head"]["head_list"]:
            if "NRTRHead" in head:
                head["NRTRHead"]["max_text_length"] = max_length
    if a.checkpoint:
        checkpoint = a.checkpoint
    if sha(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("initial weights differ from plan")
    config = a.output / "model.yml"
    if a.resume and config.read_text() != yaml.safe_dump(cfg, sort_keys=False):
        raise ValueError("model config changed")
    if not a.resume:
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    paddle.set_device(f"gpu:{resource_policy['gpu_index']}")
    paddle.seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    cv2.setNumThreads(1)
    model = build_model(copy.deepcopy(cfg["Architecture"]))
    state = paddle.load(str(checkpoint))
    expected = model.state_dict()
    if set(state) != set(expected) or any(
        tuple(state[k].shape) != tuple(expected[k].shape) for k in expected
    ):
        raise ValueError("checkpoint architecture mismatch")
    model.set_state_dict(state)
    model.eval()
    post = build_post_process(cfg["PostProcess"], cfg["Global"])
    with paddle.no_grad():
        shape = list(model(paddle.zeros([1, *input_shape])).shape)
    if shape[1] != ctc_steps:
        raise ValueError("actual CTC time steps differ")
    all_rows = read(a.dataset / "manifest.jsonl")
    route = "CharOCR" if a.kind == "char" else "WordOCR"
    selected_rows = [
        r
        for r in all_rows
        if r["route"] == route and (not a.language or r["language"] == a.language)
    ]
    # Recheck the partition at consumption: an older or edited audit must not
    # let the trainer use a different split from the one the auditor inspected.
    for row in selected_rows:
        training_partition(row)
    base = {
        part: [
            {**r, "image": str((a.dataset / r["image"]).resolve())}
            for r in selected_rows
            if r["text_partition"] == part
        ]
        for part in ["train", "dev"]
    }
    for r in base["train"] + base["dev"]:
        verified_real = (
            r.get("appearance_status") == "verified_real_crop"
            and r.get("training_eligible") is True
            and r.get("formal_evaluation_eligible") is False
            and r.get("split") == "train"
        )
        if (
            r["trial_contract_sha256"] != sha(contract)
            or not r["bounded_trial_eligible"]
            or (r.get("source_session") is not None and not verified_real)
        ):
            raise ValueError("trial admission changed")
    # Guard images never enter optimizer batches; they are used only to catch
    # capability loss before a checkpoint can be selected.
    guard_path = Path(plan["guard_manifest"])
    guards = read(guard_path)
    guard_excluded = Counter()
    for row in guards:
        row["image"] = str((guard_path.parent / row["image"]).resolve())
    base["guard"] = guards
    adapter, native_widths, capacity, batch, evaluate = prepare_training_batches(
        base,
        model,
        encoder,
        post,
        input_shape,
        ctc_steps,
        a.kind,
        a.batch,
        a.input_policy,
    )
    del state, expected
    loss_fn = build_loss(copy.deepcopy(cfg["Loss"]))
    optimizer = paddle.optimizer.Adam(
        learning_rate=a.learning_rate,
        parameters=model.parameters(),
        grad_clip=paddle.nn.ClipGradByGlobalNorm(5),
    )
    initial = model.state_dict()["head.ctc_head.fc.weight"].numpy().copy()
    started = time.monotonic()
    history = []
    baseline = (
        json.loads((a.output / "baseline.json").read_text())
        if a.resume
        else {name: evaluate(name) for name in ["dev", "guard"]}
    )
    if not a.resume:
        write(a.output / "baseline.json", baseline)
        # The fixed initial checkpoint stays in the immutable external model
        # directory; only its identity is repeated in this run.
        write(
            a.output / "initial-reference.json",
            {
                "checkpoint": str(checkpoint),
                "sha256": sha(checkpoint),
                "reason": "bounded comparisons restart from fixed initialization; no redundant initial checkpoint copy",
            },
        )
    record = {
        "kind": a.kind,
        "language": a.language,
        "model_kind": model_kind,
        "selection_policy": "released_maa_checkpoint_selection",
        "steps": a.steps,
        "batch": a.batch,
        "learning_rate": a.learning_rate,
        "training_profile_sha256": sha(a.training_profile),
        "config_sha256": sha(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "dataset": str(a.dataset.resolve()),
        "dataset_audit_sha256": sha(a.dataset / "audit.json"),
        "contract_sha256": sha(contract),
        "resource_policy_sha256": sha(resource_policy_path),
        "gpu_before": state0,
        "shape_measured": shape,
        "counts": {k: len(v) for k, v in base.items()},
        "guard_excluded": dict(guard_excluded),
        "sampling": a.sampling,
        "eval_every": a.eval_every,
        "freeze_bn_statistics": True,
        "amp": False,
        "input_shape": input_shape,
        "input_policy": a.input_policy,
        "released_preprocessor": adapter.identity() if adapter else None,
        "measured_native_capacity": capacity,
        "native_width_counts": {k: dict(Counter(v)) for k, v in native_widths.items()}
        if adapter
        else None,
        "logical_batch_policy": "same scheduled examples per optimizer step; native-width microgroups and sample-weighted loss shared by both arms"
        if adapter
        else "single fixed-width batch",
        "guard_manifests": {
            k: [
                {
                    "id": r["id"],
                    "sha256": r["sha256"],
                    "label": r["label"],
                    "scene": r["scene"],
                }
                for r in v
            ]
            for k, v in base.items()
            if k == "guard"
        },
    }
    record.update(
        run_plan_sha256=sha(a.run_plan),
        purpose=plan["purpose"],
        seed=a.seed,
        checkpoint_policy="atomic full-state recovery; latest two plus selected weights",
    )
    if not a.resume:
        write(a.output / "run-config.json", record)
    rng = np.random.default_rng(a.seed)
    from recipe_sampling import schedule

    executable_recipe = load_recipe(plan["recipe_config"])
    order, exposures = schedule(
        base["train"],
        a.kind,
        a.sampling,
        a.steps * a.batch,
        seed=a.seed + 2,
        policy_path=ROOT / executable_recipe["sampling_policy"],
    )
    position = 0
    if not a.resume:
        write(a.output / "sampling.json", exposures)
    print(
        json.dumps(
            {
                "event": "baseline",
                "dev_correct": baseline["dev"]["correct"],
                "dev_count": baseline["dev"]["count"],
                "guard_correct": baseline["guard"]["correct"],
                "guard_count": baseline["guard"]["count"],
                "gpu": gpu_status(resource_policy),
            }
        ),
        flush=True,
    )
    start_step = 0
    elapsed_before = 0.0
    identity = {
        "plan_sha256": sha(a.run_plan),
        "model_config_sha256": sha(config),
        "dataset_manifest_sha256": sha(a.dataset / "manifest.jsonl"),
        "initial_sha256": sha(checkpoint),
        "code": {
            n: sha(ROOT / n)
            for n in [
                "scripts/train_model.py",
                "scripts/training_batches.py",
                "scripts/training_state.py",
                "scripts/checkpoint_select.py",
                "scripts/recipe_sampling.py",
                "scripts/released_input.py",
                "scripts/label_contract.py",
                "environment/uv.lock",
            ]
        },
        "input_adapter": adapter.identity() if adapter else None,
    }
    if a.resume:
        folder, meta = inspect_state(a.output, identity)
        saved = restore_state(folder, model, optimizer, rng)
        if not np.array_equal(saved["order"], order):
            raise ValueError("sampling schedule changed")
        history = saved["history"]
        position = meta["position"]
        start_step = meta["step"]
        elapsed_before = meta["elapsed"]
        write(a.output / "resume-validation.json", saved["restoration_validation"])
        if plan.get("actual_maa_evaluation", True) and start_step % a.eval_every == 0:
            from checkpoint_select import evaluate_checkpoint

            gate = evaluate_checkpoint(a.output, folder, plan, config)
            if plan.get("stop_on_gate_failure", False) and (
                not gate["export_passed"]
                or gate["new_real_errors_vs_current"]
                or gate.get("new_real_errors_vs_official", [])
            ):
                raise ValueError("resume cannot bypass a failed checkpoint gate")
        if position != start_step * a.batch:
            raise ValueError("sampling cursor does not match step")
        print(
            json.dumps({"event": "resumed", "step": start_step, "position": position}),
            flush=True,
        )
    if start_step >= a.steps:
        raise ValueError("the frozen plan has no remaining optimizer steps")
    for step in range(start_step + 1, a.steps + 1):
        # Bind stochastic layers to the optimizer step, independent of resume.
        step_seed = (a.seed + step * 104729) % (2**31 - 1)
        paddle.seed(step_seed)
        random.seed(step_seed)
        np.random.seed(step_seed)
        model.train()
        # Keep the known BN-stat drift risk bounded in this first small-data trial.
        for layer in model.sublayers():
            if isinstance(layer, paddle.nn.BatchNorm2D):
                layer.eval()
        if position + a.batch > len(order):
            raise ValueError("fixed sampling budget exhausted")
        idx = order[position : position + a.batch]
        position += a.batch
        groups = width_groups(idx, native_widths["train"]) if adapter else [idx]
        value = 0.0
        for group in groups:
            # Native widths differ. Accumulate a sample-weighted logical batch
            # before one optimizer step; extra width groups must not mean extra updates.
            b = batch("train", group)
            losses = loss_fn(model(b[0], data=b[1:]), b)
            loss = losses["loss"] * (len(group) / len(idx))
            part = float(loss.item())
            if not np.isfinite(part):
                raise FloatingPointError("nonfinite loss")
            value += part
            loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        if step == 1 or step % a.eval_every == 0 or step == a.steps:
            scores = {name: evaluate(name) for name in ["dev", "guard"]}
            # This guard compares against initialization on GPU and is diagnostic
            # for planned runs; checkpoint selection uses exported MAA predictions.
            old_wrong = {e["id"] for e in baseline["guard"]["errors"]}
            new_wrong = {e["id"] for e in scores["guard"]["errors"]}
            regressions = sorted(new_wrong - old_wrong)
            eligible = not regressions
            core_non_decrease = all(
                g["correct"] >= baseline["dev"]["groups"][k]["correct"]
                for k, g in scores["dev"]["groups"].items()
            )
            status = gpu_status(resource_policy)
            row = {
                "step": step,
                "loss": value,
                "elapsed_seconds": elapsed_before + time.monotonic() - started,
                "scores": scores,
                "new_guard_regressions": regressions,
                "eligible": eligible and core_non_decrease,
                "gpu": status,
            }
            history.append(row)
            write(a.output / "history.json", history)
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "step": step,
                        "loss": value,
                        "dev_correct": scores["dev"]["correct"],
                        "guard_correct": scores["guard"]["correct"],
                        "new_guard_regressions": len(regressions),
                        "gpu": status,
                        "elapsed_seconds": row["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
            require_safe_temperature(status, resource_policy)
        if step % a.eval_every == 0 or step == a.steps or step == a.pause_at:
            if plan["purpose"] == "formal_training":
                validate_plan(a, plan)
            saved = save_state(
                a.output,
                model,
                optimizer,
                step,
                position,
                order,
                rng,
                history,
                identity,
                elapsed_before + time.monotonic() - started,
            )
            if step % a.eval_every == 0 or step == a.steps:
                if plan.get("actual_maa_evaluation", True):
                    from checkpoint_select import evaluate_checkpoint

                    gate = evaluate_checkpoint(a.output, saved, plan, config)
                    if plan.get("stop_on_gate_failure", False) and (
                        not gate["export_passed"]
                        or gate["new_real_errors_vs_current"]
                        or gate.get("new_real_errors_vs_official", [])
                    ):
                        write(
                            a.output / "stopped.json",
                            {
                                "step": step,
                                "reason": "export or real-scene regression",
                                "gate": gate,
                            },
                        )
                        return
            if step == a.pause_at:
                write(
                    a.output / "paused.json",
                    {
                        "step": step,
                        "checkpoint": str(saved),
                        "purpose": "readiness recovery check",
                    },
                )
                return

    os.link(saved / "model.pdparams", a.output / "last.pdparams")
    changed = not np.array_equal(
        initial, model.state_dict()["head.ctc_head.fc.weight"].numpy()
    )
    if not changed:
        raise ValueError("weights did not update")
    result = {
        "kind": a.kind,
        "language": a.language,
        "model_kind": model_kind,
        "selection_policy": "released_maa_checkpoint_selection",
        "steps": a.steps,
        "sample_presentations": a.steps * a.batch,
        "head_updated": changed,
        "baseline": baseline,
        "final": history[-1]["scores"],
        "elapsed_seconds": elapsed_before + time.monotonic() - started,
        "gpu_peak_allocated_bytes": paddle.device.cuda.max_memory_allocated(),
        "last_sha256": sha(a.output / "last.pdparams"),
        "input_policy": a.input_policy,
        "status": "planned_run_completed",
        "purpose": plan["purpose"],
        "recovery_available": True,
        "actual_selection": "selection.json"
        if (a.output / "selection.json").exists()
        else None,
        "formal_acceptance": False,
        "next": "review actual MAA checkpoint results; no automatic extension or release",
    }
    write(a.output / "result.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ["baseline", "final"]}),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        print(json.dumps({"event": "failed", "error": repr(e)}), flush=True)
        raise

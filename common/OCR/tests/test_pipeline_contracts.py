"""Contracts for data admission, paired evaluation and export isolation."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from acceptance_cases import compare_cases
from dataset_contract import training_partition
from evaluation_policy import evaluation_failed
from file_utils import sha
from machine_resources import gpu_status, load_policy
from prepare_training_plan import prepare
from PIL import Image
from recipe_settings import check_plan_recipe, check_profile, load_recipe
from training_preflight import audit_rows


class _AEncoder:
    ctc_encode = SimpleNamespace(dict={"A": 0}, character=["A"])
    gtc_encode = SimpleNamespace(character=["", "A"])

    def __call__(self, _row):
        return {"length": 1, "label_ctc": [0], "label_gtc": [0, 1]}


def test_training_plan_rejects_dataset_from_another_profile(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    manifest = dataset / "manifest.jsonl"
    manifest.write_text("")
    recipe = dataset / "recipe.json"
    recipe.write_text(json.dumps({"profile_sha256": "0" * 64}))
    (dataset / "audit.json").write_text(json.dumps({
        "passed": True,
        "manifest_sha256": sha(manifest),
        "recipe_sha256": sha(recipe),
    }))
    profile = Path(__file__).resolve().parents[1] / "pipeline/configs/training.json"
    spec = tmp_path / "train.json"
    spec.write_text(json.dumps({
        "route": "zh-CN", "dataset": str(dataset), "profile": str(profile),
        "checkpoint": "initial.pdparams", "model_config": "model.yml",
        "guard_manifest": "guard.jsonl",
    }))
    with pytest.raises(ValueError, match="different training profiles"):
        prepare(spec, tmp_path / "plan.json")


def test_train_dev_disagreement_is_rejected_by_audit_and_consumer(
    tmp_path, monkeypatch
):
    import training_preflight

    monkeypatch.setattr(training_preflight, "RT", tmp_path)
    Image.new("RGB", (8, 4), "white").save(tmp_path / "sample.png")
    (tmp_path / "source.json").write_text('{"text":"A"}')
    base = {
        "label": "A",
        "language": "en",
        "route": "WordOCR",
        "scene": "item_name",
        "image": "sample.png",
        "sha256": sha(tmp_path / "sample.png"),
        "source": {
            "path": "source.json",
            "sha256": sha(tmp_path / "source.json"),
            "json_pointer": "/text",
        },
        "text_partition": "train",
    }
    rows = [dict(base, id="train", split="train"), dict(base, id="dev", split="dev")]
    report, _checked = audit_rows(rows, tmp_path, {"v6": _AEncoder()})
    assert not report["passed"]
    assert report["issue_counts"]["split_partition_mismatch"] == 1
    with pytest.raises(ValueError, match="must match"):
        training_partition(rows[1])


def test_same_pixels_cannot_cross_train_and_dev(tmp_path, monkeypatch):
    import training_preflight

    monkeypatch.setattr(training_preflight, "RT", tmp_path)
    Image.new("RGB", (8, 4), "white").save(tmp_path / "same.png")
    (tmp_path / "source.json").write_text('{"text":"A"}')
    row = {
        "id": "train", "label": "A", "language": "en", "route": "WordOCR",
        "scene": "item_name", "image": "same.png", "sha256": sha(tmp_path / "same.png"),
        "source": {"path": "source.json", "sha256": sha(tmp_path / "source.json"),
                   "json_pointer": "/text"},
        "split": "train", "text_partition": "train",
    }
    report, _ = audit_rows([row, dict(row, id="dev", split="dev", text_partition="dev")],
                           tmp_path, {"v6": _AEncoder()})
    assert not report["passed"]
    assert report["issue_counts"]["identical_pixels_cross_partition"] == 1


def test_full_analyzer_failure_is_a_quality_regression_even_with_correct_text():
    cases = [
        {
            "id": "case",
            "route": "zh-CN",
            "mode": "recruit",
            "source_scope": "fixture",
            "truth": {"texts": ["A"]},
        }
    ]
    official = [
        {
            "id": "case",
            "mode": "recruit",
            "analyze_ok": True,
            "results": [{"text": "A"}],
        }
    ]
    candidate = [dict(official[0], analyze_ok=False)]
    frames = compare_cases(cases, official, candidate)
    assert frames["new_truth_regressions"] == []
    assert frames["new_analyzer_failures"] == ["case"]
    summary = {"failed_timing_routes": [], "frames": frames, "recognition": {}}
    assert evaluation_failed({"purpose": "evaluation"}, summary)
    assert not evaluation_failed({"purpose": "integration"}, summary)



def test_unasserted_depot_change_needs_exact_reviewed_exception():
    from acceptance_cases import business_digest

    case = {"id": "x", "route": "zh-CN", "mode": "depot", "source_scope": "fixture",
            "truth": {"texts": ["A"]}}
    official = {"id": "x", "mode": "depot", "analyze_ok": True,
                "results": [{"text": "A", "quantity": 1}]}
    candidate = {"id": "x", "mode": "depot", "analyze_ok": True,
                 "results": [{"text": "A", "quantity": 2}]}
    summary = {"failed_timing_routes": [], "recognition": {}}
    summary["frames"] = compare_cases([case], [official], [candidate])
    assert summary["frames"]["new_truth_regressions"] == []
    assert summary["frames"]["unreviewed_business_changes"] == ["x"]
    assert evaluation_failed({"purpose": "evaluation"}, summary)
    case["approved_business_change"] = {
        "reason": "Manually checked quantity in source screenshot",
        "candidate_canonical_sha256": business_digest(candidate),
    }
    summary["frames"] = compare_cases([case], [official], [candidate])
    assert summary["frames"]["unreviewed_business_changes"] == []
    assert not evaluation_failed({"purpose": "evaluation"}, summary)
    summary["frames"] = compare_cases([case], [official], [official])
    assert summary["frames"]["unreviewed_business_changes"] == ["x"]
    assert evaluation_failed({"purpose": "evaluation"}, summary)
    candidate["results"][0]["quantity"] = 3
    summary["frames"] = compare_cases([case], [official], [candidate])
    assert summary["frames"]["unreviewed_business_changes"] == ["x"]


def test_export_artifacts_belong_to_each_attempt(tmp_path, monkeypatch):
    import multilingual_verify as exporter

    release = tmp_path / "release"
    monkeypatch.setattr(exporter, "RELEASE", release)
    keys = tmp_path / "keys.txt"
    keys.write_text("A\n")
    config = tmp_path / "model.yml"
    config.write_text(yaml.safe_dump({"Global": {"character_dict_path": str(keys)}}))
    checkpoint = tmp_path / "model.pdparams"
    checkpoint.write_bytes(b"fixed-weights")

    def fake_run(command, _log, timeout=600):
        command = list(map(str, command))
        if "--output" in command:
            output = Path(command[command.index("--output") + 1])
            if command[1].endswith("export_model.py"):
                output.mkdir()
            elif command[1].endswith("repair_onnx_attributes.py"):
                output.write_bytes(checkpoint.read_bytes())
        if "--save_file" in command:
            Path(command[command.index("--save_file") + 1]).write_bytes(b"converter")

    monkeypatch.setattr(exporter, "run", fake_run)
    packs = []
    for branch in ("a", "b"):
        attempt = tmp_path / branch / "run-01" / "attempt-01"
        attempt.mkdir(parents=True)
        pack, report = exporter.export(
            checkpoint,
            config,
            "run-01-500-attempt-01",
            attempt,
            [{"id": "case", "scene": "item_name", "label": "A"}],
        )
        assert report["passed"]
        packs.append(pack)
    assert packs[0].resolve() != packs[1].resolve()
    assert (
        packs[0].joinpath("rec/inference.onnx").resolve()
        != packs[1].joinpath("rec/inference.onnx").resolve()
    )
    assert packs[0].joinpath("rec/inference.onnx").read_bytes() == b"fixed-weights"
    with pytest.raises(FileExistsError, match="reuse"):
        exporter.export(
            checkpoint,
            config,
            "run-01-500-attempt-01",
            packs[0].parent,
            [{"id": "case", "scene": "item_name", "label": "A"}],
        )


def test_versioned_recipe_matches_frozen_route_and_profile():
    recipe = load_recipe()
    assert recipe["base_seed"] == 20260922
    assert recipe["expansion_seed"] == 20260925
    assert sum(recipe["pool_per_scene"]["word"].values()) == 10000
    assert sum(recipe["pool_per_scene"]["char"].values()) == 10000
    profile = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "pipeline/configs/training.json"
        ).read_text()
    )
    check_profile(recipe, profile)
    rows = [
        {"route": "WordOCR", "language": "en", "split": "train"} for _ in range(10000)
    ]
    check_plan_recipe(recipe, "en", "historical", rows, formal=True)
    with pytest.raises(ValueError, match="sampling"):
        check_plan_recipe(recipe, "en", "balanced", rows, formal=True)
    with pytest.raises(ValueError, match="count"):
        check_plan_recipe(recipe, "en", "historical", rows[:-1], formal=True)
    assert recipe["routes"]["char"]["base_train_images"] == 10000
    assert recipe["routes"]["char"]["formal_train_images"] == 20000
    base_char = [{"route": "CharOCR", "split": "train"}] * 10000
    with pytest.raises(ValueError, match="count"):
        check_plan_recipe(recipe, "char", "historical", base_char, formal=True)
    check_plan_recipe(recipe, "char", "historical", base_char * 2, formal=True)


def test_gpu_monitoring_uses_cuda_uuid_despite_physical_index_or_visibility(monkeypatch):
    import machine_resources

    seen = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-selected,GPU-other")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "GPU-selected,GPU-other")
    monkeypatch.setattr(machine_resources, "cuda_device_uuid", lambda device: {0: "GPU-selected", 1: "GPU-other"}[device])

    def query(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(stdout="7, GPU-other, 2, 15947, 36, 0\n")

    monkeypatch.setattr(machine_resources.subprocess, "run", query)
    status = gpu_status({**load_policy(), "gpu_index": 1})
    assert status["index"] == 7 and status["cuda_index"] == 1
    assert len(seen) == 1 and "--id=GPU-other" in seen[0]
    monkeypatch.setattr(machine_resources.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="7, GPU-wrong, 2, 15947, 36, 0\n"))
    with pytest.raises(RuntimeError, match="different device UUID"):
        gpu_status({**load_policy(), "gpu_index": 1})


def test_cuda_device_uuid_uses_runtime_properties_and_rejects_failure(monkeypatch):
    import machine_resources
    from cuda.bindings import runtime

    properties = SimpleNamespace(uuid=SimpleNamespace(bytes=bytes(range(16))))
    seen = []

    def query(device):
        seen.append(device)
        return 0, properties

    monkeypatch.setattr(runtime, "cudaGetDeviceProperties", query)
    assert machine_resources.cuda_device_uuid(2) == "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"
    assert seen == [2]
    monkeypatch.setattr(runtime, "cudaGetDeviceProperties", lambda device: (101, None))
    with pytest.raises(RuntimeError, match="unavailable"):
        machine_resources.cuda_device_uuid(2)


@pytest.mark.parametrize("field,value", [("gpu_index", -1), ("gpu_index", True),
                                        ("minimum_free_gpu_memory_mib", float("nan")),
                                        ("maximum_gpu_temperature_c", 0)])
def test_invalid_resource_policy_stops_before_training(tmp_path, field, value):
    policy = {**load_policy(), field: value}
    path = tmp_path / "resources.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        load_policy(path)


def test_native_build_preserves_target_specific_abi_and_output_types():
    from build_native import commands

    builds = commands()
    assert len(builds) == 4
    outputs = {Path(command[-1]).name: command for command in builds}
    assert set(outputs) == {"libmaa_rec_input.so", "maa_ocr_probe", "fd_tensor_probe", "maa_acceptance_probe"}
    for name, command in outputs.items():
        assert "-std=c++20" in command and "-stdlib=libc++" in command
        assert ("-shared" in command) == name.endswith(".so")
        assert ("-lMaaCore" in command) == (not name.endswith(".so"))
        assert ("-l:libonnxruntime.so.1" in command) == (name in {"fd_tensor_probe", "maa_acceptance_probe"})
    with pytest.raises(ValueError, match="unknown native target"):
        commands(["unknown"])


def test_active_asset_locks_have_one_owner_per_download():
    configs = Path(__file__).resolve().parents[1] / "pipeline/configs"
    seen = {}
    for name in ["dataset/game-assets.lock.json", "render-assets.lock.json", "weights.lock.json"]:
        rows = json.loads((configs / name).read_text())["files"]
        for row in rows:
            assert row["path"] not in seen
            seen[row["path"]] = row["sha256"]
    styles = json.loads((configs / "styles.json").read_text())
    for template in styles["templates"]:
        assert seen[template["path"]] == template["sha256"]
    for style in styles["styles"].values():
        assert seen[style["font"]["path"]] == style["font"]["sha256"]


def test_profile_rejects_weight_settings_that_do_not_drive_the_train_spec():
    configs = Path(__file__).resolve().parents[1] / "pipeline/configs"
    profile = json.loads((configs / "training.json").read_text())
    profile["languages"]["en"]["checkpoint"] = "unused.pdparams"
    with pytest.raises(ValueError, match="unknown model route fields"):
        check_profile(load_recipe(), profile)


def test_dataset_contract_rejects_conflicting_scene_and_route_capacities(tmp_path):
    from dataset_contract import load_contract

    path = Path(__file__).resolve().parents[1] / "pipeline/configs/dataset/contract.json"
    contract = load_contract(path)
    contract["char_ctc_time_steps"] += 1
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="family capacity differs"):
        load_contract(changed)


def test_training_plan_freezes_external_resource_policy(tmp_path, monkeypatch):
    import prepare_training_plan as planner

    root = Path(__file__).resolve().parents[1] / "pipeline"
    release = tmp_path / "release"
    monkeypatch.setattr(planner, "RT", tmp_path)
    monkeypatch.setattr(planner, "RELEASE", release)
    for name in ["build/maa_ocr_probe", "build/fd_tensor_probe", "build/libmaa_rec_input.so",
                 "release/libMaaCore.so", "release/libfastdeploy_ppocr.so", "release/libonnxruntime.so.1"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    data = tmp_path / "data"
    data.mkdir()
    manifest = data / "manifest.jsonl"
    manifest.write_text(json.dumps({"route": "WordOCR", "language": "en", "split": "train", "text_partition": "train"}) + "\n")
    profile = root / "configs/training.json"
    recipe = data / "recipe.json"
    recipe.write_text(json.dumps({"profile_sha256": sha(profile)}))
    (data / "audit.json").write_text(json.dumps({"passed": True, "manifest_sha256": sha(manifest), "recipe_sha256": sha(recipe)}))
    dictionary = tmp_path / "keys.txt"
    dictionary.write_text("A\n")
    (tmp_path / "model.yml").write_text(yaml.safe_dump({"Global": {"character_dict_path": str(dictionary)}}))
    (tmp_path / "weights.pdparams").write_bytes(b"fixture")
    for name in ["guard", "evaluation"]:
        image = tmp_path / (name + ".png")
        Image.new("RGB", (8, 4), name == "guard" and "white" or "black").save(image)
        (tmp_path / (name + ".jsonl")).write_text(json.dumps({"id": name, "image": str(image), "sha256": sha(image)}) + "\n")
    for name in ["official", "current"]:
        (tmp_path / (name + ".json")).write_text(json.dumps({"predictions": [{"id": "evaluation"}]}))
    policy_path = tmp_path / "custom-resources.json"
    policy_path.write_text(json.dumps({**load_policy(), "gpu_index": 2}))
    spec = tmp_path / "train.json"
    spec.write_text(json.dumps({"route": "en", "purpose": "readiness_probe", "sampling": "historical",
        "dataset": "data", "profile": str(profile), "checkpoint": "weights.pdparams", "model_config": "model.yml",
        "guard_manifest": "guard.jsonl", "resource_policy": "custom-resources.json",
        "evaluation": {"manifest": "evaluation.jsonl", "official": "official.json", "current": "current.json"}}))
    plan = planner.prepare(spec, tmp_path / "plan.json")
    assert plan["resource_policy"] == str(policy_path)
    assert plan["files"][str(policy_path)] == sha(policy_path)
    assert str(root / "configs/resource-policy.json") not in plan["files"]


@pytest.mark.parametrize(
    "command",
    [
        "setup",
        "fetch",
        "corpus",
        "generate",
        "char-rework",
        "prepare-model",
        "plan",
        "train",
        "export-check",
        "doctor",
        "evaluate",
        "package",
        "freeze-evaluation",
    ],
)
def test_public_entry_loads_and_displays_help(command, tmp_path):
    entry = Path(__file__).resolve().parents[1] / "pipeline/pipeline.py"
    result = subprocess.run(
        [sys.executable, str(entry), "--runtime", str(tmp_path), command, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (command, result.stderr)
    assert "usage:" in result.stdout

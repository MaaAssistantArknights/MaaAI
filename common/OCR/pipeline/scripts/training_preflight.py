"""Audit image/label records and export only complete, losslessly encodable lists.

No rejected-row filtering or implicit text normalization is performed. Explicit
source-to-visible-text transformations are verified against their source fields. A single rejected row
blocks all list output. Diagnostic lists do not certify scene fidelity or model quality.
"""

import argparse
import collections
import json
from pathlib import Path
from PIL import Image
from game_text import resolve_pointer
from label_contract import inspect_label, load_encoder
from file_utils import sha
from workspace import ROOT, RT
from recruitment_tags import recruitment_tags
from dataset_contract import training_partition

HELDOUT_THEMES = set(
    json.loads((ROOT / "configs/dataset/split-policy.json").read_text())[
        "theme_holdout"
    ]
)
def inside(base, name):
    """限制清单所指文件必须位于数据目录内。"""
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("path leaves data directory")
    return path


def audit_rows(rows, dataset, encoders, purpose="diagnostic", trial_contract=None):
    """逐图核验文字、来源、分区、图像和训练接纳契约。"""
    # Reject the whole list on any bad row. Filtering it here would silently
    # change the intended scene proportions and the frozen train/dev split.
    issues = []
    checked = []
    ids = set()
    pixels = {}
    cache = {}
    labels = {}
    groups = {}
    if purpose not in {"diagnostic", "production", "training_trial"}:
        raise ValueError("unknown admission purpose")
    if purpose == "training_trial" and trial_contract is None:
        raise ValueError("training trial requires an explicit versioned contract")
    contract_path = trial_contract
    trial = (
        json.loads(contract_path.read_text()) if purpose == "training_trial" else None
    )
    for row in rows:
        problems = []
        rid = row.get("id")
        label = row.get("label")
        if not rid or rid in ids:
            problems.append("missing_or_duplicate_id")
        ids.add(rid)
        language = row.get("language")
        route = row.get("route")
        model = (
            "v6-ascii"
            if route == "CharOCR"
            else ("v5-korean" if language == "ko" else "v6")
        )
        if language not in {"en", "ja", "ko", "zh-CN", "zh-TW"} or route not in {
            "WordOCR",
            "CharOCR",
        }:
            problems.append("unsupported_route_or_language")
        if model not in encoders:
            raise ValueError("required route encoder missing: " + model)
        result = inspect_label(
            label,
            encoders[model],
            row.get("ctc_time_steps"),
            literal_ascii=bool(
                row.get("literal_ascii")
                and route == "CharOCR"
                and row.get("scene") in {"ascii_replay", "spaced_ascii"}
            ),
        )
        problems += result["problems"]
        if (
            route == "CharOCR"
            and isinstance(label, str)
            and any(not 32 <= ord(c) <= 126 for c in label)
        ):
            problems.append("non_ascii_char_label")
        if (
            row.get("scene") == "recruitment"
            and language in {"en", "ja", "ko", "zh-CN", "zh-TW"}
            and label not in recruitment_tags(language)
        ):
            problems.append("not_in_released_recruitment_tags")
        if row.get("missing_glyphs"):
            problems.append("font_missing_glyphs")
        split = row.get("text_partition")
        try:
            training_partition(row)
        except ValueError:
            problems.append("split_partition_mismatch")
        if split not in {"train", "dev"}:
            problems.append("reserved_or_unknown_partition")
        if isinstance(label, str):
            if labels.setdefault(label, split) != split:
                problems.append("same_text_cross_partition")
        group = row.get("leakage_group")
        if group and groups.setdefault(group, split) != split:
            problems.append("source_group_cross_partition")
        if row.get("heldout_theme") or row.get("theme") in HELDOUT_THEMES:
            problems.append("heldout_theme_reserved")
        # Evaluation images and held-out themes are never admitted as training
        # examples, even if their label and encoder capacity happen to be valid.
        if row.get("formal_evaluation_eligible") is True:
            problems.append("evaluation_sample_reserved")
        if purpose == "training_trial":
            family = trial["families"].get(row.get("scene"))
            allowed_statuses = (family or {}).get(
                "appearance_statuses", [(family or {}).get("appearance_status")]
            )
            if (
                not family
                or family["route"] != route
                or row.get("appearance_status") not in allowed_statuses
            ):
                problems.append("trial_family_contract_mismatch")
            if row.get("trial_contract_sha256") != sha(contract_path):
                problems.append("trial_contract_changed")
            if row.get("bounded_trial_eligible") is not True:
                problems.append("not_admitted_for_bounded_trial")
            if row.get("source_session") is not None and not (
                row.get("appearance_status") == "verified_real_crop"
                and row.get("training_eligible") is True
                and row.get("formal_evaluation_eligible") is False
                and split == "train"
            ):
                problems.append("trial_disallows_unapproved_capture_pixels")
            if trial.get("languages") and language not in trial["languages"]:
                problems.append("trial_language_contract_mismatch")
            if row.get("ctc_time_steps") != (family or {}).get(
                "ctc_time_steps", trial.get("ctc_time_steps", 40)
            ):
                problems.append("trial_input_capacity_mismatch")
        if purpose == "production" and split == "train":
            if row.get("training_eligible") is not True:
                problems.append("not_approved_for_training")
            if row.get("calibration_status") != "verified":
                problems.append("appearance_unverified")
            if row.get("task_mapping_status") != "verified":
                problems.append("task_mapping_unverified")
            if row.get("ctc_time_steps") is None:
                problems.append("ctc_time_capacity_unverified")
        image = None
        try:
            image = inside(dataset, row["image"])
            if any(c in str(image) for c in "\t\r\n"):
                raise ValueError("unsafe training-list path")
            if sha(image) != row["sha256"]:
                raise ValueError("image hash changed")
            with Image.open(image) as im:
                im.load()
                rgb = im.convert("RGB")
                import hashlib

                digest = hashlib.sha256(
                    str(rgb.size).encode() + rgb.tobytes()
                ).hexdigest()
            previous = pixels.setdefault(digest, (label, split))
            if previous[0] != label:
                problems.append("identical_pixels_conflicting_labels")
            if previous[1] != split:
                problems.append("identical_pixels_cross_partition")
            source = row["source"]
            p = inside(RT, source["path"])
            key = (str(p), source["sha256"])
            if key not in cache:
                if sha(p) != source["sha256"]:
                    raise ValueError("source hash changed")
                cache[key] = json.loads(p.read_text())
            raw = resolve_pointer(cache[key], source["json_pointer"])
            if "text_transform" in row:
                from text_targets import validate_transform

                validate_transform(row, raw)
            elif raw != label:
                raise ValueError("label differs from exact source field")
        except (OSError, KeyError, ValueError, IndexError, TypeError) as e:
            problems.append("image_or_source: " + str(e))
        if problems:
            issues.append(
                {
                    "id": rid,
                    "language": language,
                    "label": label,
                    **result,
                    "problems": sorted(set(problems)),
                }
            )
        checked.append(
            {
                "id": rid,
                "model": model,
                "split": split,
                "image": str(image),
                "label": label,
            }
        )
    if (
        purpose in {"production", "training_trial"}
        and rows
        and not any(r.get("text_partition") == "train" for r in rows)
    ):
        issues.append({"id": None, "problems": ["no_training_rows"]})
    if not rows:
        issues.append({"id": None, "problems": ["empty_dataset"]})
    return {
        "samples": len(rows),
        "rejected_rows": len(issues),
        "issue_counts": dict(
            collections.Counter(p for r in issues for p in r["problems"])
        ),
        "issues": issues,
        "passed": not issues,
        "purpose": purpose,
        "formal_data_acceptance": False,
    }, checked


def export_lists(checked, output, result):
    """只在全量审计通过后导出训练与开发清单。"""
    if not result["passed"]:
        raise ValueError("preflight failed; no training list emitted")
    groups = collections.defaultdict(list)
    for row in checked:
        groups[(row["model"], row["split"])].append(
            row["image"] + "\t" + row["label"] + "\n"
        )
    for (model, split), lines in groups.items():
        path = output / f"{model}-{split}.txt"
        with path.open("x") as f:
            f.writelines(lines)


def bind_capacity(rows, pairs, evidence, height, width):
    """按实测模型结构、字典和输入尺寸绑定 CTC 容量。"""
    tables = {}
    for model, (_, meta) in pairs.items():
        measured = evidence["models"][model]
        if measured["encoder"]["dictionary_sha256"] != meta["dictionary_sha256"]:
            raise ValueError("capacity dictionary differs")
        if measured["encoder"]["config_sha256"] != meta["config_sha256"]:
            raise ValueError("capacity architecture config differs")
        if (
            measured["encoder"]["configured_max_length"]
            != meta["configured_max_length"]
        ):
            raise ValueError("capacity encoder length differs")
        shape = measured["shapes"][f"{height}x{width}"]
        if not shape["supported"]:
            raise ValueError("unsupported measured input size")
        tables[model] = shape["ctc_time_steps"]
    result = []
    for row in rows:
        model = (
            "v6-ascii"
            if row.get("route") == "CharOCR"
            else ("v5-korean" if row.get("language") == "ko" else "v6")
        )
        result.append(
            {
                **row,
                "ctc_time_steps": tables[model],
                "diagnostic_input_shape": [3, height, width],
            }
        )
    return result


def main():
    """按指定用途审计数据集并写出结果与可选清单。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest-name", default="manifest.jsonl")
    p.add_argument("--route", choices=["WordOCR", "CharOCR"])
    p.add_argument("--encoder-max-length", type=int)
    p.add_argument("--capacity", type=Path)
    p.add_argument("--input-height", type=int, default=48)
    p.add_argument("--input-width", type=int, default=480)
    p.add_argument(
        "--purpose",
        choices=["diagnostic", "production", "training_trial"],
        default="diagnostic",
    )
    p.add_argument("--trial-contract", type=Path)
    p.add_argument("--export", action="store_true")
    a = p.parse_args()
    if (a.purpose == "training_trial") != (a.trial_contract is not None):
        p.error("--trial-contract is required only for training_trial")
    if a.output.exists():
        raise FileExistsError("choose a new output directory")
    path = inside(a.dataset, a.manifest_name)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    if a.encoder_max_length is not None and (
        a.capacity is None or a.purpose != "diagnostic"
    ):
        raise ValueError(
            "custom encoder length is diagnostic only and requires measured capacity"
        )
    excluded = []
    if a.route:
        excluded = [
            {
                "id": r["id"],
                "route": r.get("route"),
                "reason": "explicit route selection",
            }
            for r in rows
            if r.get("route") != a.route
        ]
        rows = [r for r in rows if r.get("route") == a.route]
    pairs = {
        model: load_encoder(model, a.encoder_max_length)
        for model in ["v6", "v5-korean", "v6-ascii"]
    }
    if a.capacity:
        rows = bind_capacity(
            rows,
            pairs,
            json.loads(a.capacity.read_text()),
            a.input_height,
            a.input_width,
        )
    result, checked = audit_rows(
        rows, a.dataset, {m: v[0] for m, v in pairs.items()}, a.purpose,
        a.trial_contract,
    )
    result.update(
        manifest_sha256=sha(path),
        models={m: v[1] for m, v in pairs.items()},
        implementation={
            name: sha(Path(__file__).parent / name)
            for name in ["training_preflight.py", "label_contract.py"]
        },
        exported=False,
        source_text_changed=False,
        selection_exclusions=excluded,
        capacity_sha256=sha(a.capacity) if a.capacity else None,
        diagnostic_input_shape=[3, a.input_height, a.input_width]
        if a.capacity
        else None,
    )
    a.output.mkdir(parents=True)
    if a.export and result["passed"]:
        export_lists(checked, a.output, result)
        result["exported"] = True
    (a.output / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["samples", "rejected_rows", "passed", "purpose", "exported"]
            },
            ensure_ascii=False,
        )
    )
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

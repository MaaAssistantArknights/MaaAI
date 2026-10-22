"""Shared export and released-runtime equivalence gate."""

import argparse, json, subprocess, sys
from pathlib import Path
import yaml
from file_utils import sha
from workspace import ROOT, RT
from workspace import RELEASE


def run(args, log, timeout=600):
    """执行导出或发布探针并保存独立日志。"""
    with log.open("x") as f:
        subprocess.run(
            [str(a) for a in args],
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )


def pack(path, model, keys, kind="word"):
    """组装识别模型与字典，并引用未修改的官方检测模型。"""
    (path / "rec").mkdir(parents=True)
    (path / "det").symlink_to(
        RELEASE
        / ("resource/PaddleCharOCR/det" if kind == "char" else "resource/PaddleOCR/det")
    )
    (path / "rec/inference.onnx").symlink_to(model)
    (path / "rec/keys.txt").symlink_to(keys)
    return path


def export(checkpoint, config, variant, d, rows, kind="word"):
    """完成模型导出、ONNX 修复及多后端同输入一致性检查。"""
    # Keep the model under this evaluation attempt. A name based only on the
    # run basename and step can collide across otherwise independent runs.
    model = d / "model"
    if model.exists():
        raise FileExistsError("refuse to reuse an export directory: " + str(model))
    run(
        [
            sys.executable,
            ROOT / "scripts/export_model.py",
            "--config",
            config,
            "--checkpoint",
            checkpoint.with_suffix(""),
            "--output",
            model,
        ],
        d / "export.log",
    )
    run(
        [
            Path(sys.executable).with_name("paddle2onnx"),
            "--model_dir",
            model,
            "--model_filename",
            "inference.json",
            "--params_filename",
            "inference.pdiparams",
            "--save_file",
            model / "inference.onnx",
            "--opset_version",
            17,
            "--enable_auto_update_opset",
            "False",
            "--optimize_tool",
            "None",
        ],
        d / "onnx.log",
    )
    # Match scalar semantics to PIR before any graph stabilization/optimization.
    (model / "inference.onnx").rename(model / "converter-raw.onnx")
    run(
        [
            sys.executable,
            ROOT / "scripts/repair_onnx_attributes.py",
            "--pir",
            model / "inference.json",
            "--onnx",
            model / "converter-raw.onnx",
            "--output",
            model / "inference.onnx",
            "--report",
            model / "attribute-repair.json",
        ],
        d / "attributes.log",
    )
    # These graph transforms were validated for the ASCII route only. They do
    # not replace the eager/PIR/ORT/released-runtime comparison that follows.
    if kind == "char":
        (model / "inference.onnx").rename(model / "raw.onnx")
        run(
            [
                sys.executable,
                ROOT / "scripts/stabilize_pointwise.py",
                model / "raw.onnx",
                model / "pointwise.onnx",
            ],
            d / "pointwise.log",
        )
        run(
            [
                sys.executable,
                ROOT / "scripts/optimize_onnx.py",
                model / "pointwise.onnx",
                model / "inference.onnx",
            ],
            d / "optimization.log",
        )
    keys = Path(yaml.safe_load(config.read_text())["Global"]["character_dict_path"])
    selected = []
    for family in dict.fromkeys(r["scene"] for r in rows):
        group = [r for r in rows if r["scene"] == family]
        # First and longest cases cover short and dynamic-width paths.
        chosen = group[:3] + sorted(group, key=lambda r: len(r["label"]))[-4:]
        selected.extend({r["id"]: r for r in chosen}.values())
    mp = d / "equivalence-manifest.jsonl"
    mp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    tensors = d / "tensors"
    tensors.mkdir()
    run(
        [RT / "build/fd_tensor_probe", model / "inference.onnx", keys, mp, tensors],
        d / "capture.log",
    )
    (tensors / "provenance.json").write_text(
        json.dumps({"model": sha(model / "inference.onnx")})
    )
    try:
        run(
            [
                sys.executable,
                ROOT / "scripts/check_equivalence.py",
                "--model-dir",
                model,
                "--variant",
                variant,
                "--checkpoint",
                checkpoint,
                "--config",
                config,
                "--tensor-manifest",
                tensors / "manifest.json",
                "--report-dir",
                d,
                "--report-tag",
                variant,
            ],
            d / "equivalence.log",
        )
        passed = True
    except subprocess.CalledProcessError:
        passed = False
    return pack(d / "pack", model / "inference.onnx", keys, kind), {
        "passed": passed,
        "samples": len(selected),
        "report": str(d / f"export-equivalence-{variant}.json"),
        "model_sha256": sha(model / "inference.onnx"),
        "checkpoint_sha256": sha(checkpoint),
        "model_dir": str(model.resolve()),
    }


def main():
    """核对导出输入并显式执行转换及数值检查。"""
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--kind',choices=['word','char'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(l) for l in a.manifest.read_text().splitlines() if l]
    if not rows:raise ValueError('equivalence needs real inputs')
    for r in rows:
        r['image']=str((a.manifest.parent/r['image']).resolve())
        if sha(Path(r['image']))!=r['sha256']:raise ValueError('input changed')
    a.output.mkdir(parents=True,exist_ok=False)
    # Randomness only separates artifact names, not model/data contents.
    import uuid
    variant='export-'+uuid.uuid4().hex[:12];pack,gate=export(a.checkpoint.resolve(),a.config.resolve(),variant,a.output.resolve(),rows,a.kind)
    (a.output/'result.json').write_text(json.dumps({'pack':str(pack),'gate':gate,'release_approved':False},indent=2)+'\n')
    raise SystemExit(0 if gate['passed'] else 1)


if __name__ == "__main__":
    main()

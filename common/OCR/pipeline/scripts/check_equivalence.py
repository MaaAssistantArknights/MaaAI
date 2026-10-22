"""Compare identical released-FastDeploy inputs across eager/PIR/ONNX/released ORT."""

import argparse, copy, hashlib, json, sys
from pathlib import Path
import numpy as np, paddle, onnxruntime as ort, yaml
from workspace import RT

sys.path.insert(0, str(RT / "vendor/PaddleOCR"))
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--variant", default="compatible")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--report-dir", type=Path, required=True)
parser.add_argument("--tensor-manifest", type=Path, required=True)
parser.add_argument("--report-tag")
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()
variant = args.variant
model_dir = args.model_dir.resolve()
if not model_dir.is_dir():
    raise ValueError("export model directory is missing")
paddle.set_device("cpu")
paddle.set_flags({"FLAGS_paddle_num_threads": 3})
cfg = yaml.safe_load(args.config.read_text())
model = build_model(copy.deepcopy(cfg["Architecture"]))
weights = paddle.load(str(args.checkpoint))
expected = model.state_dict()
if set(weights) != set(expected) or any(
    tuple(weights[k].shape) != tuple(expected[k].shape) for k in expected
):
    raise ValueError("checkpoint/config mismatch")
model.set_state_dict(weights)
model.eval()
post = build_post_process(cfg["PostProcess"], cfg["Global"])
static = paddle.jit.load(str(model_dir / "inference"))
static.eval()
opt = ort.SessionOptions()
opt.intra_op_num_threads = 3
session = ort.InferenceSession(
    str(model_dir / "inference.onnx"),
    sess_options=opt,
    providers=["CPUExecutionProvider"],
)
tensor_manifest = args.tensor_manifest
rows = json.loads(tensor_manifest.read_text())
checks = []
failures = []
if not rows:
    raise ValueError("released tensor evidence is empty")
provenance = tensor_manifest.with_name("provenance.json")
if not provenance.exists():
    raise ValueError("missing tensor provenance")
if (
    json.loads(provenance.read_text())["model"]
    != hashlib.sha256((model_dir / "inference.onnx").read_bytes()).hexdigest()
):
    raise ValueError("captured tensors belong to a different ONNX model")


def compare(name, reference, actual):
    """比较两份推理张量的形状及数值容差。"""
    same = reference.shape == actual.shape
    ok = same and np.allclose(actual, reference, atol=1e-4, rtol=1e-3)
    return {
        "pair": name,
        "shape_match": same,
        "max_abs": float(abs(reference - actual).max()) if same else None,
        "allclose": bool(ok),
    }


for row in rows:
    x = np.fromfile(row["input"]["path"], dtype="<f4").reshape(row["input"]["shape"])
    released = np.fromfile(row["output"]["path"], dtype="<f4").reshape(
        row["output"]["shape"]
    )
    with paddle.no_grad():
        eager = model(paddle.to_tensor(x)).numpy()
        pir = static(paddle.to_tensor(x)).numpy()
    exported = session.run(None, {session.get_inputs()[0].name: x})[0]
    pairs = [
        compare("PIR/python-ORT", pir, exported),
        compare("python-ORT/released-ORT", exported, released),
        compare("eager/PIR", eager, pir),
        compare("eager/ORT-1.30-diagnostic", eager, exported),
        compare("eager/released-ORT", eager, released),
    ]
    texts = {
        "eager": post(eager)[0][0],
        "pir": post(pir)[0][0],
        "python_ort": post(exported)[0][0],
        "released_fd": row["text"],
    }
    ok = all(c["allclose"] for c in pairs) and len(set(texts.values())) == 1
    result = {
        "id": row["id"],
        "shape": list(x.shape),
        "pairs": pairs,
        "decoded": texts,
        "pass": ok,
    }
    checks.append(result)
    if not ok:
        failures.append(result)
# Dynamic batch/width checks beyond the actual D0 images.
shape_checks = []
rng = np.random.default_rng(20260917)
for shape in [(1, 3, 48, 32), (1, 3, 48, 160), (2, 3, 48, 320), (1, 3, 48, 640)]:
    x = rng.uniform(-1, 1, shape).astype("float32")
    with paddle.no_grad():
        eager = model(paddle.to_tensor(x)).numpy()
        pir = static(paddle.to_tensor(x)).numpy()
    exported = session.run(None, {session.get_inputs()[0].name: x})[0]
    pairs = [
        compare("PIR/python-ORT", pir, exported),
        compare("eager/PIR", eager, pir),
        compare("eager/ONNX", eager, exported),
    ]
    shape_checks.append({"shape": shape, "pairs": pairs})
report = {
    "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    "onnx_sha256": hashlib.sha256(
        (model_dir / "inference.onnx").read_bytes()
    ).hexdigest(),
    "model_dir": str(model_dir),
    "tensor_manifest_sha256": hashlib.sha256(tensor_manifest.read_bytes()).hexdigest(),
    "checkpoint": str(args.checkpoint),
    "variant": variant,
    "samples": len(rows),
    "failures": failures,
    "shape_checks": shape_checks,
    "checks": checks,
    "atol": 1e-4,
    "rtol": 1e-3,
    "passed": not failures
    and all(c["allclose"] for r in shape_checks for c in r["pairs"]),
}
args.report_dir.mkdir(parents=True, exist_ok=True)
(args.report_dir / f"export-equivalence-{args.report_tag or variant}.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print("samples", len(rows), "failures", len(failures), "passed", report["passed"])
if not report["passed"]:
    raise SystemExit(1)

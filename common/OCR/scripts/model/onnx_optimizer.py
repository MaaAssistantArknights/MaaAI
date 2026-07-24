import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", nargs="?", default="models/output/inference.onnx")
parser.add_argument("output", nargs="?", default="models/output/inference_optimized.onnx")
args = parser.parse_args()

import onnx
import onnxoptimizer

model = onnx.load(args.input)
new_model = onnxoptimizer.optimize(model)
onnx.save(new_model, args.output)

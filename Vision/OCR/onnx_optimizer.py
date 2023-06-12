import onnx
import onnxoptimizer
model = onnx.load("./inference.onnx")
new_model = onnxoptimizer.optimize(model)
onnx.save(new_model,"inference_optimized.onnx")
exit()

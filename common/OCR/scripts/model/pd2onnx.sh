paddle2onnx --model_dir "${1:-./models/output}" \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file "${2:-./models/output/inference.onnx}" \
--enable_dev_version True

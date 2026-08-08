paddle2onnx --model_dir "${1:-./models/output/PP-OCRv6_medium_rec/inference}" \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file "${2:-./models/output/PP-OCRv6_medium_rec/inference.onnx}" \
--enable_dev_version True

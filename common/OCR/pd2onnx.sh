paddle2onnx --model_dir ./ \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file inference.onnx \
--enable_dev_version True

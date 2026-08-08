#!/usr/bin/env bash
set -euo pipefail

# paddle2onnx 无 cp313 wheel，需在 PaddleOCR 环境中安装后运行：
#   pip install paddle2onnx
if ! command -v paddle2onnx > /dev/null 2>&1; then
    echo "未找到 paddle2onnx，请先在 PaddleOCR 环境中安装：pip install paddle2onnx"
    exit 1
fi

paddle2onnx \
    --model_dir "${1:-./models/output/PP-OCRv6_medium_rec/inference}" \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file "${2:-./models/output/PP-OCRv6_medium_rec/inference.onnx}" \
    --enable_dev_version True

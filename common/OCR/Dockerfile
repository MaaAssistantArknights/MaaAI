ARG PRETRAINED_MODEL=PP-OCRv6_medium_rec_pretrained.pdparams
ARG VERSION=3.0.0
FROM paddlepaddle/paddle:${VERSION}-gpu-cuda11.8-cudnn8.9-trt8.6

ARG PRETRAINED_MODEL

RUN git clone https://gitee.com/paddlepaddle/PaddleOCR.git  /PaddleOCR  --depth=1 && \
    cd /PaddleOCR && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple 

COPY datasets/generated /workspace/datasets/generated
COPY models/pretrained/$PRETRAINED_MODEL /workspace/models/pretrained/$PRETRAINED_MODEL
COPY models/configs /workspace/models/configs
WORKDIR /workspace

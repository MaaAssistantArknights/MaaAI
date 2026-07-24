ARG OCR_LANG=zh_CN
ARG PRETRAINED_MODEL=ch_PP-OCRv3_rec_train
ARG VERSION=22.05
FROM nvcr.io/nvidia/paddlepaddle:${VERSION}-py3

ARG OCR_LANG
ARG PRETRAINED_MODEL

RUN git clone https://gitee.com/paddlepaddle/PaddleOCR.git  /PaddleOCR  --depth=1 && \
    cd /PaddleOCR && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple 

COPY datasets/generated/$OCR_LANG /workspace/datasets/generated/$OCR_LANG
COPY datasets/generated/render/$OCR_LANG /workspace/datasets/generated/render/$OCR_LANG
COPY models/pretrained/$PRETRAINED_MODEL /workspace/models/pretrained/$PRETRAINED_MODEL
COPY models/configs /workspace/models/configs
WORKDIR /workspace

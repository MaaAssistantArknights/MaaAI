ARG OCR_LANG=zh_CN
ARG PRETRAINED_MODEL=ch_PP-OCRv3_rec_distillation
ARG VERSION=22.05
FROM nvcr.io/nvidia/paddlepaddle:{22.05}-py3

RUN git clone https://gitee.com/paddlepaddle/PaddleOCR.git  /PaddleOCR  --depth=1 && \
    cd /PaddleOCR && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple 

COPY output/$OCR_LANG /workspace/output/$OCR_LANG
COPY output/render/$OCR_LANG /workspace/output/render/$OCR_LANG
COPY pretrained_model/$PRETRAINED_MODEL /workspace/pretrained_model/$PRETRAINED_MODEL
COPY *.yml  /workspace/
WORKDIR /workspace

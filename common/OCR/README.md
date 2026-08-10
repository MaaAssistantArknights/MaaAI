# OCR

基于 PaddleOCR（PP-OCRv6），整理《明日方舟》所有文本生成数据集进行训练

本项目主要提供生成数据集的脚本，以及训练完的模型产物

PP-OCRv6 为多语言统一模型（中/英/日/韩/繁中一网打尽），因此所有游戏客户端（cn/en/kr/jp/tw）的数据合并为一份多语言数据集，只微调一个 rec 模型即可

## 目录结构

```text
OCR/
├── datasets/
│   ├── custom/       # 手工裁剪并标注的补充数据
│   ├── generated/    # 生成的语料、图片、多语言标签（不提交）
│   ├── keys/         # 各语言基础字符字典
│   └── render.yaml   # 合成图片配置
├── game_data/        # 外部游戏数据、字体和 text_renderer（不提交）
├── models/
│   ├── configs/      # PaddleOCR 训练配置
│   ├── output/       # checkpoint 和导出模型（不提交）
│   └── pretrained/   # 下载的预训练模型（不提交）
└── scripts/
    ├── data/         # 语料、标签、数据集合并等处理脚本
    ├── model/        # 模型转换与优化脚本
    └── generate_dataset.sh

pyproject.toml / uv.lock  # uv 依赖管理（数据生成 + 模型转换）
```

## 使用方法

目前仅有 rec（识别）模型，检测模型可以直接用飞桨官方的。需要将对应的 rec 模型文件和 keys.txt 替换成 [release](https://github.com/MaaAssistantArknights/ArknightsTrainingData/releases/latest) 包里的

可选使用方法：

- 最简单：使用 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 推理，参考 [通用 OCR 产线文档](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/OCR.html)
- 最推荐：使用 [FastDeploy](https://github.com/PaddlePaddle/FastDeploy) 部署，可自由选择 ONNX Runtime, Paddle Inference, TensorRT, OpenVINO 等后端进行推理
- 最折腾：使用 [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) 转换为 ONNX 模型，使用 [RapidOCR](https://github.com/RapidAI/RapidOCR) + ONNX Runtime 进行推理

## 依赖管理

数据集生成与模型转换脚本的 Python 依赖由 [uv](https://docs.astral.sh/uv/) 管理：

```bash
# 首次安装依赖（生成 .venv 与 uv.lock）
uv sync

# 模型优化（onnx / onnxoptimizer）需要
uv sync --extra model
```

> 注意：`paddle2onnx`（Paddle 静态图转 ONNX）无 Python 3.13 的 wheel，请放到 PaddleOCR 环境中安装（`pip install paddle2onnx`）后使用 `scripts/model/pd2onnx.sh`。

## 训练方法

**如果你不需要重新训练，请忽略该内容**  
**推荐在 Linux 或 WSL2 中进行，Windows 也能跑，但很慢，所以不推荐**

1. 安装 CUDA, CUDNN

    没啥好说的，自己 Google（

2. 安装 PaddlePaddle

    <https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/windows-conda.html>

3. 搭建 PaddleOCR 环境

    ```bash
    git clone https://github.com/PaddlePaddle/PaddleOCR.git
    pip install -r PaddleOCR/requirements.txt
    ```

4. 整理你自己的数据集

    如果想增加某些场景的识别率，参考 [datasets/custom](./datasets/custom/README.md)，把额外的数据集放进来
    （没有自己数据集的可以忽略这一步）

5. 生成数据集

    ```bash
    # 依次生成 zh_CN / zh_TW / ja_JP / ko_KR / en_US 五种客户端数据，并合并为多语言数据集
    # 脚本会自动 uv sync 依赖；小规模测试可用 NUM_IMG=200 覆盖图片数量
    bash ./scripts/generate_dataset.sh
    ```

6. 开始训练

    ```bash
    python PaddleOCR/tools/train.py -c models/configs/PP-OCRv6_medium_rec.yml
    ```

    一些配置文件中可能要修改的项：
    - `num_workers`: 读取数据集的进程数。不能大于你的 CPU 线程数，但是太大了也没意义，不造成性能瓶颈就行，一般 4 或者 8 就差不多了。Windows 不支持这项，调也没用，所以很慢
    - `batch_size_per_card`: batch size, 一般来说越大越快，但会吃更多显存，自己看着调
    - `lr.learning_rate`: 学习率，原则上要和 batch size 等比例调整

7. 断点训练

    把配置文件中 `checkpoints` 那项指向上次保存的权重即可

8. 评估

    ```bash
    python PaddleOCR/tools/eval.py -c models/configs/PP-OCRv6_medium_rec.yml -o Global.pretrained_model=./models/output/PP-OCRv6_medium_rec/best_accuracy.pdparams
    ```

9. 导出模型

    ```bash
    python PaddleOCR/tools/export_model.py -c models/configs/PP-OCRv6_medium_rec.yml -o Global.pretrained_model=./models/output/PP-OCRv6_medium_rec/best_accuracy.pdparams Global.save_inference_dir=./models/output/PP-OCRv6_medium_rec/inference
    ```

只是个大致的流程，都还是 PaddleOCR 的那套，更多详细的参数等请参考 PaddleOCR 的文档

## 模型转换

```bash
# Paddle 静态图导出为 ONNX（需在 PaddleOCR 环境中安装 paddle2onnx）
bash ./scripts/model/pd2onnx.sh

# ONNX 优化（需要 uv sync --extra model）
uv run --extra model python ./scripts/model/onnx_optimizer.py
```

## 训练方法 (Docker)

如果你是用恰好有 nvidia-docker 并且不想折腾环境可以试试 Docker, 本教程假设你知道一些常用的 Docker 命令

0. 依赖

- `docker` 以及 `nvidia-docker` 具体安装流程参考 [Nvidia文档](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) 
- 本仓库提供的 Dockerfile 基于 Paddle 3.0.0 GPU 镜像（CUDA 11.8 / cuDNN 8.9 / TensorRT 8.6）

1. 获取镜像

    ```bash
    docker build -t maa_train . \   # 以下为可选参数
        --build-arg VERSION=3.0.0 \ # Paddle 镜像的版本，默认为 3.0.0
        --build-arg PRETRAINED_MODEL=PP-OCRv6_medium_rec_pretrained.pdparams # 预训练权重文件名，默认为 v6 medium
    ``` 

2. 运行镜像

    ```bash
    # 如果启动失败，可尝试删除 --ulimit memlock=-1 或者添加 sudo 运行
    docker --gpus all --shm-size=1g --ulimit memlock=-1 run -it maa_train /bin/bash
    ```

    进入容器后，将第六步中PaddleOCR的位置替换为`../PaddleOCR`，即

    ```bash
    python ../PaddleOCR/tools/train.py -c models/configs/PP-OCRv6_medium_rec.yml
    ```

## 开源库

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): Awesome multilingual OCR toolkits based on PaddlePaddle
- [ArknightsGamedata](https://github.com/ArknightsAssets/ArknightsGamedata): 《明日方舟》游戏数据
- [text_renderer](https://github.com/Sanster/text_renderer): Generate text images for training deep learning ocr model
- [source-han-sans](https://github.com/adobe-fonts/source-han-sans): Source Han Sans | 思源黑体 | 思源黑體 | 思源黑體 香港 | 源ノ角ゴシック | 본고딕

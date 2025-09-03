# OCR

基于 PaddleOCR，整理《明日方舟》所有文本生成数据集进行训练  

本项目主要提供生成数据集的脚本，以及训练完的模型产物

## 使用方法

目前仅有 rec（识别）模型，检测模型可以直接用飞桨官方的。需要将对应的 rec 模型文件和 keys.txt 替换成 [release](https://github.com/MaaAssistantArknights/ArknightsTrainingData/releases/latest) 包里的

可选使用方法：

- 最简单：使用 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 推理，参考 [使用方法](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/quickstart.md)
- 最推荐：使用 [FastDeploy](https://github.com/PaddlePaddle/FastDeploy) 部署，可自由选择 ONNX Runtime, Paddle Inference, TensorRT, OpenVINO 等后端进行推理
- 最折腾：使用 [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) 转换为 ONNX 模型，使用 [RapidOCR](https://github.com/RapidAI/RapidOCR) + ONNX Runtime 进行推理

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

    如果想增加某些场景的识别率，参考 [my_data](./my_data/README.md)，把额外的数据集放进来  
    （没有自己数据集的可以忽略这一步）

5. 生成数据集

    ```bash
    # 默认只生成简中数据，其他语言改下开头的变量即可
    sh ./steps.sh
    ```

6. 开始训练

    ```bash
    # 简中的，其他语言请替换成对应的配置
    python PaddleOCR/tools/train.py -c ch_PP-OCRv3_rec_distillation.yml
    ```

    一些配置文件中可能要修改的项：
    - `num_workers`: 读取数据集的进程数。不能大于你的 CPU 线程数，但是太大了也没意义，不造成性能瓶颈就行，一般 4 或者 8 就差不多了。Windows 不支持这项，调也没用，所以很慢
    - `batch_size_per_card`: batch size, 一般来说越大越快，但会吃更多显存，自己看着调
    - `lr.values` / `learning_rate`: 学习率，原则上要和 batch size 等比例调整

7. 断点训练

    把配置文件中 `checkpoints` 那项去掉注释

8. 导出模型

    ```bash
    python PaddleOCR/tools/export_model.py -c ch_PP-OCRv3_rec_distillation.yml -o Global.checkpoints=./output/ja_JP/model/epoch/latest/best_accuracy
    ```

只是个大致的流程，都还是 PaddleOCR 的那套，更多详细的参数等请参考 PaddleOCR 的文档

## 训练方法 (Docker)

如果你是用恰好有 nvidia-docker 并且不想折腾环境可以试试 Docker, 本教程假设你知道一些常用的 Docker 命令

0. 依赖

- `docker` 以及 `nvidia-docker` 具体安装流程参考 [Nvidia文档](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) 
- 对应版本的 CUDA, 本仓库提供的 Dockerfile 默认版本支持 nvidia 驱动版本 >= 515 (CUDA >= 11.7)

1. 获取镜像

    ```bash
    docker build -t maa_train . \   # 以下为可选参数
        --build-arg VERSION=22.05 \ # Nvdia 镜像的版本，默认为22.05， 可选的版本参考之前的链接
        --build-arg OCR_LANG=zh_CN \ # 训练数据集的语言，docker将拷贝对应语言数据集到镜像，默认zh_CN, 可选`zh_CN | ja_JP | zh_TW | en_US`
        --build_arg PRETRAINED_MODEL=ch_PP-OCRv3_rec_distillation # 预训练模型权重名称, 默认为简中知识蒸馏模型
    ``` 

2. 运行镜像

    ```bash
    # 如果启动失败，可尝试删除 --ulimit memlock=-1 或者添加 sudo 运行
    docker --gpus all --shm-size=1g --ulimit memlock=-1 run -it maa_train /bin/bash
    ```

    进入容器后，将第六步中PaddleOCR的位置替换为`../PaddleOCR`，即

    ```bash
    python ../PaddleOCR/tools/train.py -c ch_PP-OCRv3_rec_distillation.yml 
    ```

## 开源库

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): Awesome multilingual OCR toolkits based on PaddlePaddle
- [ArknightsGameData](https://github.com/Kengxxiao/ArknightsGameData): 《明日方舟》游戏数据
- [text_renderer](https://github.com/Sanster/text_renderer): Generate text images for training deep learning ocr model
- [source-han-sans](https://github.com/adobe-fonts/source-han-sans): Source Han Sans | 思源黑体 | 思源黑體 | 思源黑體 香港 | 源ノ角ゴシック | 본고딕

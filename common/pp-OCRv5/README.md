# pp-OCRv5 训练 Pipeline

本项目基于PaddleOCR最新模型（pp-OCRv5）为MAA（MaaAssistantArknights）训练专用的OCR识别模型。该README将介绍完整的训练流程。

由于 [ArknightsGameData](https://github.com/Kengxxiao/ArknightsGameData) 改版后只有zh_CN的数据，此pipeline将专注于重构中文识别模型的训练。其它语言的训练亦可参考本流程。

## 项目结构

```
pp-OCRv5/
├── configs/                    # 配置文件
│   ├── PP-OCRv5_server_rec.yml # 训练配置
│   └── maa_config.py          # 数据生成配置
├── output/                     # 输出目录
│   ├── render/                # 渲染的图片数据
│   ├── ppocr_format/          # PaddleOCR格式数据
│   └── zh_CN/                 # 中文数据
├── utils/                      # 工具脚本
├── PaddleOCR/                  # PaddleOCR源码
└── text_renderer/              # 文本渲染器
```

## Pipeline 总览

- **目标**: 为MAA训练新的OCR识别模型
- **步骤**:
  1. 下载游戏数据和对应字体
  2. 生成数据集
  3. 搭建 PaddleOCR 环境并且进行推理
  4. 导出onnx

## 详细步骤

### 1. 下载数据和字体

运行`download.py`文件下载数据以及字体：

```bash
python download.py
```

### 2. 生成训练数据

#### 2.1 生成文字数据

使用仓库提供的`wording.py`和`number.py`先生成文字数据：

```bash
python utils/wording.py
python utils/number.py
```

#### 2.2 拉取text_renderer

```bash
git clone https://github.com/oh-my-ocr/text_renderer
```

#### 2.3 安装text_renderer依赖

```bash
cd text_renderer
pip install -r requirements.txt
```

#### 2.4 生成训练数据

使用仓库提供的`maa_config.py`和text_renderer生成对应的训练数据：

```bash
python text_renderer/main.py --config configs/maa_config.py
```

#### 2.5 转换数据格式

使用仓库提供的`convert_ppocr.py`将训练数据改成PaddleOCR可以接收的形式：

```bash
python utils/convert_ppocr.py
```

该脚本会：
- 读取`output/render/`目录下的所有配置文件
- 将每个配置的数据转换为PaddleOCR格式：`图像路径\t文本内容`
- 生成单独的训练文件
- 输出到`output/ppocr_format/`目录

#### 2.6 训练测试集分割

使用`train_test_split.py`将数据分割为训练集和测试集：

```bash
python utils/train_test_split.py
```

该脚本会：
- 读取`output/ppocr_format/`目录下的所有配置文件
- 将所有数据合并并随机打乱
- 按8:2的比例分割为训练集和测试集
- 生成`train_list.txt`和`test_list.txt`两个文件

### 3. 搭建 PaddleOCR 环境并训练 （具体参考paddleOCR官网）

#### 3.1 搭建PaddleOCR环境

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
pip install -r PaddleOCR/requirements.txt
```

#### 3.2 下载预训练模型

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams 
```

#### 3.3 训练

**重要**: 确保配置文件中的路径设置正确。训练前请检查以下路径：

- 训练数据目录: `./output/render/`
- 训练标签文件: `./output/ppocr_format/train_list.txt`
- 测试标签文件: `./output/ppocr_format/test_list.txt`
- 字符字典: `./output/zh_CN/keys.txt`

开始训练：

```bash
python PaddleOCR/tools/train.py -c configs/PP-OCRv5_server_rec.yml -o Global.pretrained_model=./PP-OCRv5_server_rec_pretrained.pdparams
```

训练参数说明：
- `-c`: 指定配置文件
- `-o`: 覆盖配置参数
- `Global.pretrained_model`: 预训练模型路径

训练过程中会输出到 `./output/PP-OCRv5_server_rec/` 目录。

#### 3.4 导出模型

使用训练好的模型进行推理导出：

```bash
python PaddleOCR/tools/export_model.py -c configs/PP-OCRv5_server_rec.yml -o Global.pretrained_model=./output/PP-OCRv5_server_rec/best_accuracy.pdparams Global.save_inference_dir="./PP-OCRv5_server_rec_infer/"
```

参数说明：
- `Global.pretrained_model`: 使用训练过程中保存的最佳模型
- `Global.save_inference_dir`: 推理模型输出目录

#### 3.5 获取 ONNX 模型

参考官方文档：[获取ONNX模型](https://www.paddleocr.ai/main/version3.x/deployment/obtaining_onnx_models.html?h=onnx)

使用paddle2onnx转换模型：

```bash
paddle2onnx \
    --model_dir ./PP-OCRv5_server_rec_infer/ \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file ./PP-OCRv5_server_rec.onnx \
    --opset_version 11
```

参数说明：
- `--model_dir`: Paddle推理模型目录
- `--model_filename`: 模型文件名
- `--params_filename`: 参数文件名
- `--save_file`: 输出ONNX模型路径
- `--opset_version`: ONNX opset版本（推荐11）


## 开源库

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): Awesome multilingual OCR toolkits based on PaddlePaddle
- [ArknightsGameData](https://github.com/Kengxxiao/ArknightsGameData): 《明日方舟》游戏数据
- [text_renderer](https://github.com/oh-my-ocr/text_renderer): Generate text images for training deep learning ocr model
- [source-han-sans](https://github.com/adobe-fonts/source-han-sans): Source Han Sans | 思源黑体 | 思源黑體 | 思源黑體 香港 | 源ノ角ゴシック | 본고딕

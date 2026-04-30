# Skill Ready 模型训练与推理

本项目提供了一个完整的机器学习流水线，用于训练、验证和导出技能就绪分类模型。该项目支持多种深度学习模型，包括TIMM库中的预训练模型和自定义模型。

## 项目结构

```
skill_ready/
├── configs/                 # 配置文件目录
│   └── mobilenetv4_conv_small.yaml    # MobileNetV4小模型配置
├── core/                    # 核心功能模块
│   ├── base_model.py       # 基础模型类（包含数据变换）
│   ├── data_loader.py      # 数据集和数据加载工具
│   └── model_builder.py    # 模型创建工厂
├── models/                  # 模型实现
│   ├── TimmModel.py        # TIMM库模型包装器
│   └── CustomModel.py      # 自定义模型包装器
├── utils/                   # 工具函数
│   ├── img_process.py      # 图像处理工具
│   ├── logger.py           # 训练和验证日志记录
│   ├── loss.py             # 损失函数实现
│   └── path_manager.py     # 路径管理工具
├── train.py                # 训练脚本
├── split_dataset.py        # 从训练集划分验证集
├── deduplicate_dataset.py  # 对相似截图做哈希去重
├── validate.py             # PyTorch模型验证
├── export.py               # ONNX模型导出
├── onnx_inference.py       # ONNX模型验证
├── export.md               # 导出文档
└── requirements.txt        # Python依赖包
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖包
pip install -r requirements.txt
```

### 2. 数据准备

将数据集按以下结构组织：
```
datasets/
├── train/   # 训练集
│   ├── c
│   ├── n
│   ├── y
│   ├── c_err   # 可选: 历史误判但真实标签为 c 的难例
│   ├── n_err   # 可选: 历史误判但真实标签为 n 的难例
│   └── y_err   # 可选: 历史误判但真实标签为 y 的难例
└── val/     # 验证集
    ├── c
    ├── n
    └── y
```

支持的图像格式：`.png`, `.jpg`, `.jpeg`

训练时会将 `*_err` 目录映射回对应原类别，并按配置中的 `training.sampler.err_folder_weight` 提高采样权重，用于难例增强。

### 3. 模型训练

```bash
# 先对训练集去重，重复图会被移到单独目录
python deduplicate_dataset.py --input_dir datasets/train --duplicates_dir datasets/duplicates/train --distance_threshold 4 --clear_duplicates

# 从训练集抽取 10% 到验证集
python split_dataset.py --train_dir datasets/train --val_dir datasets/val --val_ratio 0.1 --clear_val

# 使用MobileNetV4小模型训练
python train.py --config configs/mobilenetv4_conv_small.yaml

# 在上一次训练结果基础上继续微调
python train.py --config configs/mobilenetv4_conv_small.yaml --weights checkpoints/mobilenetv4_conv_small/skill_ready_cls.pth
```

`split_dataset.py` 支持直接读取 `train` 下的 `c/n/y/c_err/n_err/y_err` 六个目录，并在划分时按真实标签归并到 `val/c`、`val/n`、`val/y`。
默认会复制文件，适合你按版本保留训练集与验证集各一份；如果你确实要把样本从 train 抽走到 val，可以加 `--move`。

`deduplicate_dataset.py` 会对 `train` 下的 `c/n/y/c_err/n_err/y_err` 六个目录按真实标签分组做 dHash 去重，只在同一真实类别内比较相似度，避免跨类误删。重复图片会被移动到单独的 `duplicates` 目录，并保留原始来源子目录；去重明细会写入 `dedup_report.tsv`。
默认阈值是 4，适合处理短时间连续截图导致的高度相似样本；如果发现去重过严或过松，可以调 `--distance_threshold`。

推荐迭代方式：当前版本收集两份新数据，一份直接补到 `train`，一份单独补到 `val`；等下一批新数据到来后，再把上一轮 `val` 中已经看过的样本并回 `train`，重新划一轮新的验证集。

如果是增量数据微调，可以直接传 `--weights` 加载上一版模型权重作为起点；当前实现不会恢复历史 epoch 和优化器状态，只做轻量热启动。

### 4. 模型验证

验证阶段的 `Accuracy` 会按验证集自身的类别数量做直接反比加权，不会因为样本来自 `*_err` 目录而额外提高权重。

```bash
# PyTorch模型验证
python validate.py --config configs/mobilenetv4_conv_small.yaml --val_path datasets/val

# 将误判样本直接回灌到训练集的 *_err 目录
python validate.py --config configs/mobilenetv4_conv_small.yaml --val_path datasets/val --hard_example_dir datasets/train
```

### 5. 模型导出

```bash
# 默认加载 checkpoints/{model_name}/skill_ready_cls.pth 后导出为 ONNX
python export.py --config configs/mobilenetv4_conv_small.yaml

# 显式指定要导出的 pth 权重
python export.py --config configs/mobilenetv4_conv_small.yaml --weights checkpoints/mobilenetv4_conv_small/skill_ready_cls.pth
```

`export.py` 在导出前会先加载 `.pth` 权重。默认加载 `checkpoints/{model_name}/skill_ready_cls.pth`，也可以通过 `--weights` 指定其他权重文件。

### 6. ONNX模型验证

```bash
# ONNX模型验证和性能测试
python onnx_inference.py --config configs/mobilenetv4_conv_small.yaml --val_path datasets/val

# 导出误判样本到训练集的 *_err 目录
python onnx_inference.py --config configs/mobilenetv4_conv_small.yaml --val_path datasets/val --hard_example_dir datasets/train
```

## 配置说明

配置文件采用YAML格式，支持以下参数：

### 模型配置
- `type`: 模型类型（'timm' 或 'custom'）
- `name`: 模型名称（如 'mobilenetv4_conv_small'）
- `pretrained`: 是否使用预训练权重
- `num_classes`: 输出类别数量
- `drop_path_rate`: Dropout正则化率

### 训练配置
- `batch_size`: 训练批次大小
- `epochs`: 训练轮数
- `optimizer`: 优化器配置（类型、学习率、权重衰减）
- `use_cuda_amp`: 启用混合精度训练
- `lr_scheduler`: 当前支持带 warmup 的 cosine 调度，训练中会实际生效
- `early_stopping`: 按验证集加权准确率提前停止训练
- `loss`: 损失函数类型（'focal' 或 'CE'）
- `loss.class_weight.enabled`: 是否按训练集文件数量自动启用类别权重；当前采用更温和的 `1/sqrt(样本数)` 方式

### 数据配置
- `dataset_path`: 数据集目录路径
- `input_size`: 输入图像尺寸
- `split_ratio`: 训练/验证集分割比例

## 模型架构

### 数据加载和预处理
参考文件: `core/data_loader.py` 和 `core/base_model.py`

#### 数据集
- **SkillIconDataset**: 用于加载图像数据，支持的图像文件格式包括 `.png`, `.jpg`, `.jpeg`。

### 支持的模型
- **TIMM模型**: TIMM库中的任何模型
- **自定义模型**: 自定义模型实现

## 输出文件

### 训练输出
- **检查点**: 保存在 `checkpoints/{model_name}/`
- **日志**: TensorBoard日志保存在 `logs/train/{model_name}/`
- **图形化报告**: 训练曲线、混淆矩阵、每类 Precision/Recall/F1 柱状图保存在对应日志目录
- **最佳模型**: `skill_ready_cls.pth`

训练、验证、ONNX 验证结束后，控制台会直接打印报告目录以及 `classification_report.txt`、`val_confusion_matrix.png`、`val_classification_metrics.png` 等文件路径。

### ONNX输出
- **ONNX模型**: `onnx/{model_name}/skill_ready_cls.onnx`
- **调试图像**: 错误分类的图像保存用于分析

### 导出流程

#### 导出流程
参考文件: `export.py` 与 `export.md`

- **加载配置文件**: 获取模型配置
- **创建模型实例**
- **加载 `.pth` 权重**: 默认读取 `checkpoints/{model_name}/skill_ready_cls.pth`，也可用 `--weights` 覆盖
- **导出 ONNX 模型**: 使用 `torch.onnx.export()` 导出 ONNX 模型
  - **注意**: 如果不加载 `.pth` 权重，导出的 ONNX 模型只会是初始化状态


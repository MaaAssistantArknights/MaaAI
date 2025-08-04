# Skill Ready 模型训练与推理

本项目提供了一个完整的机器学习流水线，用于训练、验证和导出技能就绪分类模型。该项目支持多种深度学习模型，包括TIMM库中的预训练模型和自定义模型。

## 项目结构

```
skill_ready/
├── configs/                 # 配置文件目录
│   ├── mobilenetv4_conv_small.yaml    # MobileNetV4小模型配置
│   └── convnextv2_large.yaml          # ConvNeXtV2大模型配置
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
├── validate.py             # PyTorch模型验证
├── export.py               # ONNX模型导出
├── onnx_inference.py       # ONNX模型验证
├── export.md               # 导出文档
└── requirements.txt        # Python依赖包
```

## 主要特性

- **多模型支持**: 支持TIMM库模型和自定义模型
- **数据增强**: 完整的训练数据增强流水线
- **混合精度训练**: 支持CUDA AMP加速训练
- **ONNX导出**: 将训练好的模型导出为ONNX格式用于部署
- **全面日志记录**: TensorBoard集成，包含混淆矩阵和分类报告
- **性能测试**: PyTorch和ONNX模型的推理速度测试
- **自动类别权重**: 针对不平衡数据集的自动类别权重计算

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
│   └── y
└── val/     # 验证集
    ├── c
    ├── n
    └── y
```

支持的图像格式：`.png`, `.jpg`, `.jpeg`

### 3. 模型训练

```bash
# 使用MobileNetV4小模型训练
python train.py --config configs/mobilenetv4_conv_small.yaml

# 使用ConvNeXtV2大模型训练
python train.py --config configs/convnextv2_large.yaml
```

### 4. 模型验证

```bash
# PyTorch模型验证
python validate.py --config configs/mobilenetv4_conv_small.yaml --val_path datasets/val
```

### 5. 模型导出

```bash
# 导出为ONNX格式
python export.py --config configs/mobilenetv4_conv_small.yaml
```

### 6. ONNX模型验证

```bash
# ONNX模型验证和性能测试
python onnx_inference.py --config configs/mobilenetv4_conv_small.yaml --val_path datasets/val
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
- `loss`: 损失函数类型（'focal' 或 'CE'）

### 数据配置
- `dataset_path`: 数据集目录路径
- `input_size`: 输入图像尺寸
- `split_ratio`: 训练/验证集分割比例

## 模型架构

### 基础模型特性
- **数据增强**: 随机裁剪、翻转、颜色抖动、随机擦除
- **标准化**: ImageNet均值/标准差标准化
- **类别权重**: 自动计算不平衡数据集的类别权重

### 数据加载和预处理
参考文件: `core/data_loader.py` 和 `core/base_model.py`

#### 数据集
- **SkillIconDataset**: 用于加载图像数据，支持的图像文件格式包括 `.png`, `.jpg`, `.jpeg`。

#### 数据转换

**训练集转换**:
- **RandomResizedCrop**: 随机裁剪并调整大小到 `input_size`（例如 64x64），`scale` 参数控制裁剪的比例
- **RandomHorizontalFlip**: 以 0.5 的概率水平翻转图像
- **ColorJitter**: 随机调整图像的亮度、对比度和饱和度
- **RandomErasing**: 随机擦除图像的一部分
- **ToTensor**: 将图像转换为 Tensor
- **Normalize**: 使用均值 `[0.485, 0.456, 0.406]` 和标准差 `[0.229, 0.224, 0.225]` 对图像进行归一化

**验证集转换**:
- **Resize**: 调整大小到 72x72
- **CenterCrop**: 中心裁剪到 `input_size`（例如 64x64）
- **ToTensor**: 将图像转换为 Tensor
- **Normalize**: 使用均值 `[0.485, 0.456, 0.406]` 和标准差 `[0.229, 0.224, 0.225]` 对图像进行归一化

### 支持的模型
- **TIMM模型**: TIMM库中的任何模型
- **自定义模型**: 自定义模型实现

### 模型结构
参考文件: `models/TimmModel.py`

- 使用 `timm` 库创建 `mobilenetv4_conv_small` 模型或其他 `timm` 库支持的模型
- 修改分类头以适应小数据集
- 模型期望的输入形状为 `(batch_size, 3, input_size, input_size)`，其中 `input_size` 在配置文件中定义

## 输出文件

### 训练输出
- **检查点**: 保存在 `checkpoints/{model_name}/`
- **日志**: TensorBoard日志保存在 `logs/train/{model_name}/`
- **最佳模型**: `skill_ready_cls.pth`

### 验证输出
- **结果**: 详细的验证结果文本格式
- **混淆矩阵**: 可视化保存为PNG
- **分类报告**: 每个类别的详细指标

### ONNX输出
- **ONNX模型**: `onnx/{model_name}/skill_ready_cls.onnx`
- **调试图像**: 错误分类的图像保存用于分析

### 验证和导出流程

#### 验证流程
参考文件: `validate.py`

- **加载配置文件**: 获取模型和数据配置
- **创建数据加载器**: 使用 `val_transform` 对验证集数据进行转换
- **加载预训练模型权重**
- **推理**: 对验证集数据进行推理，并记录指标

#### 导出流程
参考文件: `export.py`

- **加载配置文件**: 获取模型配置
- **创建模型实例**
- **导出 ONNX 模型**: 使用 `torch.onnx.export()` 导出 ONNX 模型
  - **注意**: 导出的函数中务必加载 `.pth` 权重，否则导出的 ONNX 模型将只是初始化状态

#### ONNX 模型推理前后处理流程

为了确保 ONNX 模型推理结果与 `validate.py` 中的验证流程一致，需要进行以下前后处理：

**推理前处理**:
- **图像加载**: 使用 PIL 或其他图像处理库加载图像
- **图像大小调整**: 如果图像大小与模型期望的输入大小不一致，需要进行调整（验证集使用 `Resize (72x72)` 和 `CenterCrop (64x64)`）
- **图像格式转换**: 确保图像为 RGB 格式
- **转换为 Tensor**: 将图像转换为 Tensor
- **归一化**: 使用与训练时相同的均值和标准差进行归一化  
  `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- **添加 Batch 维度**: 将图像 Tensor 转换为形状 `(1, 3, 64, 64)`

**推理后处理**:
- **输出解析**: ONNX 模型的输出是 logits，需要使用 softmax 函数将其转换为概率
- **类别映射**: 将概率映射到类别名称（例如 `c`, `n`, `y`）

**重要提示**: 为了确保 ONNX 模型推理结果与 `validate.py` 一致，请注意以下几点：
- 使用与 `validate.py` 相同的预处理流程，包括图像大小调整、格式转换、Tensor 转换和归一化
- 使用与训练时相同的均值和标准差进行归一化
- 在推理后，使用 softmax 函数将 logits 转换为概率，并将概率映射到类别名称

## 性能指标

流水线跟踪和报告以下指标：
- **准确率**: 整体分类准确率
- **每类指标**: 每个类别的精确率、召回率、F1分数
- **混淆矩阵**: 预测结果的可视化表示
- **推理速度**: 延迟和FPS测量

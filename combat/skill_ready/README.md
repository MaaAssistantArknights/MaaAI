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
- **最佳模型**: `skill_ready_cls.pth`

### ONNX输出
- **ONNX模型**: `onnx/{model_name}/skill_ready_cls.onnx`
- **调试图像**: 错误分类的图像保存用于分析

### 导出流程

#### 导出流程
参考文件: `export.py`

- **加载配置文件**: 获取模型配置
- **创建模型实例**
- **导出 ONNX 模型**: 使用 `torch.onnx.export()` 导出 ONNX 模型
  - **注意**: 导出的函数中务必加载 `.pth` 权重，否则导出的 ONNX 模型将只是初始化状态


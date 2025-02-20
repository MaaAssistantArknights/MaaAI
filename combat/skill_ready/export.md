## 1. 数据加载和预处理
参考文件: `core/data_loader.py` 和 `core/base_model.py`

### 数据集
- **SkillIconDataset**: 用于加载图像数据，支持的图像文件格式包括 `.png`, `.jpg`, `.jpeg`。

### 数据转换

#### 训练集
- **RandomResizedCrop**: 随机裁剪并调整大小到 `input_size`（例如 64x64），`scale` 参数控制裁剪的比例。
- **RandomHorizontalFlip**: 以 0.5 的概率水平翻转图像。
- **ColorJitter**: 随机调整图像的亮度、对比度和饱和度。
- **RandomErasing**: 随机擦除图像的一部分。
- **ToTensor**: 将图像转换为 Tensor。
- **Normalize**: 使用均值 `[0.485, 0.456, 0.406]` 和标准差 `[0.229, 0.224, 0.225]` 对图像进行归一化。

#### 验证集
- **Resize**: 调整大小到 72x72。
- **CenterCrop**: 中心裁剪到 `input_size`（例如 64x64）。
- **ToTensor**: 将图像转换为 Tensor。
- **Normalize**: 使用均值 `[0.485, 0.456, 0.406]` 和标准差 `[0.229, 0.224, 0.225]` 对图像进行归一化。

---

## 2. 模型结构
参考文件: `models/mobilenetv4.py`

- 使用 `timm` 库创建 `mobilenetv4_conv_small` 模型。
- 修改分类头以适应小数据集。
- 模型期望的输入形状为 `(batch_size, 3, input_size, input_size)`，其中 `input_size` 在 `configs/mobilenetv4.yaml` 中定义。

---

## 3. 验证流程
参考文件: `validate.py`

- **加载配置文件**: 获取模型和数据配置。
- **创建数据加载器**: 使用 `val_transform` 对验证集数据进行转换。
- **加载预训练模型权重**。
- **推理**: 对验证集数据进行推理，并记录指标。

---

## 4. 导出流程
参考文件: `export.py`

- **加载配置文件**: 获取模型配置。
- **创建模型实例**。
- **导出 ONNX 模型**: 使用 `model.export_onnx(output_path)` 导出 ONNX 模型。
  - **注意**: 导出的函数中务必加载 `.pth` 权重，否则导出的 ONNX 模型将只是初始化状态。

---

## 5. ONNX 模型推理前后处理流程

为了确保 ONNX 模型推理结果与 `validate.py` 中的验证流程一致，需要进行以下前后处理：

### 推理前处理
- **图像加载**: 使用 PIL 或其他图像处理库加载图像。
- **图像大小调整**: 如果图像大小与模型期望的输入大小不一致，需要进行调整（验证集使用 `Resize (72x72)` 和 `CenterCrop (64x64)`）。
- **图像格式转换**: 确保图像为 RGB 格式。
- **转换为 Tensor**: 将图像转换为 Tensor。
- **归一化**: 使用与训练时相同的均值和标准差进行归一化  
  `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- **添加 Batch 维度**: 将图像 Tensor 转换为形状 `(1, 3, 64, 64)`。

### 推理后处理
- **输出解析**: ONNX 模型的输出是 logits，需要使用 softmax 函数将其转换为概率。
- **类别映射**: 将概率映射到类别名称（例如 `c`, `n`, `y`）。

---

## 总结
为了确保 ONNX 模型推理结果与 `validate.py` 一致，请注意以下几点：
- 使用与 `validate.py` 相同的预处理流程，包括图像大小调整、格式转换、Tensor 转换和归一化。
- 使用与训练时相同的均值和标准差进行归一化。
- 在推理后，使用 softmax 函数将 logits 转换为概率，并将概率映射到类别名称。

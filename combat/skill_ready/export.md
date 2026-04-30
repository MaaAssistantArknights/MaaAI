# ONNX 导出说明

## 默认行为

运行下面的命令时：

```bash
python export.py --config configs/mobilenetv4_conv_small.yaml
```

导出脚本会自动执行以下步骤：

1. 读取配置文件。
2. 创建模型结构。
3. 默认加载 `checkpoints/{model_name}/skill_ready_cls.pth`。
4. 将当前权重导出为 `onnx/{model_name}/skill_ready_cls.onnx`。

这意味着如果你已经训练并生成了最佳检查点，直接运行上面的命令就会导出带权重的 ONNX 模型，不需要手动再改代码。

## 显式指定权重

如果你想导出其他 `.pth` 文件，可以显式传入：

```bash
python export.py --config configs/mobilenetv4_conv_small.yaml --weights checkpoints/mobilenetv4_conv_small/skill_ready_cls.pth
```

## 注意事项

- 导出前必须加载 `.pth` 权重，否则导出的 ONNX 只有初始化参数。
- 如果默认检查点不存在，脚本会直接报错，不会导出空模型。
- 导出使用的是 CPU 设备，不影响最终 ONNX 的推理使用方式。
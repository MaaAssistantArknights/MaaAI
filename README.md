# MAA AI

明日方舟相关深度学习模型，最佳实践：[MAA](https://github.com/MaaAssistantArknights/MaaAssistantArknights)

## [光学文字识别](common/OCR)

基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 的明日方舟 OCR 模型 finetuning 全流程，包括数据生成、训练、模型转换及优化。详情请参考 [README](common/OCR/README.md)

## [战斗技能 Ready 识别](combat/skill_ready)

![skill_ready_1](https://user-images.githubusercontent.com/18511905/223741336-47fce2de-1d09-4b9c-a09e-16c805686d63.png)
![skill_ready_2](https://user-images.githubusercontent.com/18511905/223743166-cc6143c4-3c02-4495-b0da-6f1dcd724393.png)
![skill_ready_3](https://user-images.githubusercontent.com/18511905/223743312-2cb43115-d3a3-4e69-97c9-74e75ef0baab.png)

三分类 [MobileNetv4 Small](https://arxiv.org/abs/2404.10518) 网络，模型大小 9M，CPU 推理耗时在 1ms 以内  
输入需要是 720p 原图下截取的 64x64 的图标（上面三张这种）

区分三种技能状态，有(y) / 无(n) / 可关闭(c)

- [C++ 推理参考](https://github.com/MaaAssistantArknights/MaaAssistantArknights/blob/dev/src/MaaCore/Vision/Battle/BattlefieldClassifier.cpp)
- [Python 推理参考](https://github.com/MaaAssistantArknights/MaaAI/blob/main/combat/skill_ready/onnx_inference.py)

## [战斗干员（血条）检测](combat/operators)

![image](https://user-images.githubusercontent.com/18511905/229085758-a32f6379-0eb5-421d-baee-700d327d2f17.png)

[YOLOv8](https://github.com/ultralytics/ultralytics) N 检测模型，模型大小 12M，CPU 推理耗时约 50ms  
输入是 16:9 的图片缩放到 640x640

- [C++ 推理参考](https://github.com/MaaAssistantArknights/MaaAssistantArknights/blob/dev/src/MaaCore/Vision/Battle/BattlefieldDetector.cpp)

## [战斗干员方向识别](combat/deploy_direction)

![2023-03-20_16-30-54-725_帕拉斯_1](https://user-images.githubusercontent.com/18511905/229086254-2869d975-a89a-47a8-991c-d69d07fa416f.png)
![2023-03-20_16-59-48-497_深海色_2](https://user-images.githubusercontent.com/18511905/229086346-3eb296b8-4a52-42e0-ace2-bbfc94d812c8.png)

四分类 [MnistSimpleCNN](https://arxiv.org/abs/2008.10400) 网络，模型大小 18M，CPU 推理耗时约 20ms  
输入是 720p 下以格子正中心 96x96 的图片  

- [C++ 推理参考](https://github.com/MaaAssistantArknights/MaaAssistantArknights/blob/dev/src/MaaCore/Vision/Battle/BattlefieldClassifier.cpp)

## 其他

敬请期待

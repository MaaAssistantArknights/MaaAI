# MAA AI

明日方舟相关深度学习模型，最佳实践：[MAA](https://github.com/MaaAssistantArknights/MaaAssistantArknights)

## [战斗技能 Ready 识别](combat/skill_ready)

![skill_ready_1](https://user-images.githubusercontent.com/18511905/223741336-47fce2de-1d09-4b9c-a09e-16c805686d63.png)
![skill_ready_2](https://user-images.githubusercontent.com/18511905/223743166-cc6143c4-3c02-4495-b0da-6f1dcd724393.png)
![skill_ready_3](https://user-images.githubusercontent.com/18511905/223743312-2cb43115-d3a3-4e69-97c9-74e75ef0baab.png)

二分类 [Inception v1](https://arxiv.org/abs/1409.4842v1) 网络，模型大小 450k，CPU 推理速度在 1ms 以内  
数据集包含了一些传统算法较难处理的情况，准确率在 99% 以上  
输入需要是 720p 原图下截取的 64x64 的图标（上面三张这种）

- [C++ 推理参考](https://github.com/MaaAssistantArknights/MaaAssistantArknights/blob/dev/src/MaaCore/Vision/Miscellaneous/BattleSkillReadyImageAnalyzer.cpp)
- [Python 推理参考](https://github.com/MaaAssistantArknights/MaaAI/blob/main/combat/skill_ready/inference.py)

## [战斗干员（血条）检测](combat/operators)

![image](https://user-images.githubusercontent.com/18511905/229085758-a32f6379-0eb5-421d-baee-700d327d2f17.png)

[YOLOv8](https://github.com/ultralytics/ultralytics) 检测模型，输入是 16:9 的图片缩放到 640x640。集成示例 TODO

## [战斗干员方向识别](combat/deploy_direction)

![2023-03-20_16-30-54-725_帕拉斯_1](https://user-images.githubusercontent.com/18511905/229086254-2869d975-a89a-47a8-991c-d69d07fa416f.png)
![2023-03-20_16-59-48-497_深海色_2](https://user-images.githubusercontent.com/18511905/229086346-3eb296b8-4a52-42e0-ace2-bbfc94d812c8.png)

四分类 [MnistSimpleCNN](https://arxiv.org/abs/2008.10400) 网络，输入是 720p 下以格子正中心 96x96 的图片  
对于部分被严重遮挡的干员，识别效果并不理想，建议多识别几帧对输出 loss 值做个累加，后续可能会做个动态 batch 啥的。集成示例 TODO

## 其他

敬请期待

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

## 其他

敬请期待

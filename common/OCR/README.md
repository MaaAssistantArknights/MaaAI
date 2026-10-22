# MAA OCR 识别模型训练

本目录用于生成明日方舟 OCR 训练数据，按需微调五条 WordOCR 路线与 CharOCR，导出可交给 MAA 的识别模型和指标。简中、繁中、英文、日文及 Char 使用 PP-OCRv6 small；韩文使用 Korean PP-OCRv5 mobile。

当前建议通用 WordOCR 优先使用官方预训练权重，Char 使用 ASCII 专用微调。已测模型、数据规模、结果和适用范围见[模型评估](docs/evaluation.md)。

从 [训练流程](pipeline/README.md) 开始：在 Linux x64 上创建独立工作目录，安装 [同一套 v5/v6 环境](pipeline/environment/pyproject.toml)，按固定来源获取游戏文字、字体和预训练权重；生成数据后冻结训练计划，训练、导出并检查与 MAA 当前发布接口的一致性。所有下载素材、图片、检查点、ONNX、日志和完整评估结果都写到仓库外，不提交 Git。

| 目录 | 用途 |
|---|---|
| [pipeline](pipeline/README.md) | 当前可执行的数据准备、训练、导出与交付流程；配置和环境锁都在此处 |
| [docs/evaluation.md](docs/evaluation.md) | 最终模型选择、结果及评估方法；附 JSON 与公开样本 CSV |
| `tests/` | 数据、导出、评估和训练恢复的自动化测试 |

新的游戏数据、模型或 MAA 版本应建立新的素材锁、数据与训练计划，并按同一接口重新评估，不能直接沿用旧结果。当前验证平台为 Linux x64；推荐组合在新版 Windows 和外服客户端的质量仍待验证。

本目录只维护 v6/v5 流程，旧 v3 的生成器、训练配置及转换脚本已退役，可从 Git 历史查阅。数据标签应保留完整文字，避免按所有空格拆分或重复追加；感谢 SherkeyXD 在 [MaaAI #14](https://github.com/MaaAssistantArknights/MaaAI/pull/14) 中提出的标签处理改进思路。

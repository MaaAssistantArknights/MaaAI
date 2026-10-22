# 可复现的数据、训练与导出入口

这是仓库唯一的 v6/v5 正式训练入口，覆盖五语言 WordOCR 和独立 Char 子集。按下文依次取得固定素材、生成数据、冻结计划、训练、导出并以 MAA 接口评估；每步写出来源和哈希。合成图或开发集成绩不能代替独立实图验收。

当前通用 WordOCR 优先采用官方预训练模型，是否微调由实图对照决定；也可跳过数据生成和训练，直接按第 3、5 节准备及导出官方模型。各路线的结果和验证范围见[模型评估](../docs/evaluation.md)。

代码位于本目录，所有字体、游戏资源、模型、数据和运行输出放到调用者指定的外部目录。路径不依赖原试验场，也不修改已安装的 PaddleOCR。当前实际适配并测试的是 Linux x64、NVIDIA GPU、由锁文件固定的 MAA v6.20.1 接口；Windows 用于候选部署验收，不在这里宣称训练平台兼容。

依赖环境、执行配置与代码都在本目录；游戏资源、字体、图片、检查点及输出写入外部 `OCR_WORK`。数据集契约位于 `configs/dataset/`，不会读取旧试验场的阶段目录。

## 1. 环境与固定依赖

在 `common/OCR` 目录执行。下文 `$OCR_WORK` 是你选择的外部工作目录；示例 `/path/to/ocr-runtime` 必须替换。不要将它放入 Git。

```bash
export OCR_WORK=/path/to/ocr-runtime
mkdir -p "$OCR_WORK"
export UV_PROJECT_ENVIRONMENT="$OCR_WORK/envs/train"
export UV_CACHE_DIR="$OCR_WORK/cache/uv"
uv sync --project pipeline/environment --locked
uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" setup --native
uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" doctor
```

`setup` 只下载固定版本到指定目录：PaddleOCR、MAA 发布源码、MaaUtils、发布包、预训练权重；`--native` 另装固定的本地编译器，构建发布预处理、OCR 和张量探针。现有版本或哈希不匹配时停止，不覆盖现有依赖。版本与文件来源见 `configs/sources.lock.json`、`runtime.lock.json`、`weights.lock.json` 和 `native-toolchain.lock.txt`。环境采用 Paddle 3.4.0 / Paddle2ONNX 2.1.0；升级后需重验转换适配，不能只改版本号。

首次安装需要网络、兼容的 NVIDIA 驱动及发布包所需系统 ABI。升级 MAA 时需同时核对源码、发布包和动态库，并重新构建探针；旧版本结果不能替代新版本验证。

## 2. 数据源、字体与语料

```bash
uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" fetch --lock pipeline/configs/dataset/game-assets.lock.json
uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" fetch --lock pipeline/configs/render-assets.lock.json
uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" corpus \
  --lock pipeline/configs/dataset/game-assets.lock.json \
  --fonts pipeline/configs/fonts.json --output "$OCR_WORK/corpus.jsonl"
```

`fetch --offline` 只核对已有文件。语料来自固定提交的 ArknightsAssets/ArknightsGamedata，按明确字段提取各服名称、物品、公招、基建/技能文字；每项保留文件哈希和 JSON 指针。字体覆盖不足、富文本和占位符不会偷偷截断或猜测展开。字体和 21 张物品背景均由素材锁单独下载到 `data/`，不依赖某个旧版本的发布目录；代码仓库不包含这些二进制。

语料合并与模型字典合并是两件事：语料可统一管理并按语言/任务选择；v6 保留官方字符顺序，Char 按 ASCII 子集映射，韩文按实际缺失字符追加。不能将多语言 keys 排序合并后直接配给已有预训练输出层。

## 3. 准备输出层和生成数据

为通用 v6、ASCII 和韩文生成带绝对字典路径的模型配置。输出目录必须是新目录；模型准备只调整字典对应关系，不代表已经微调或证明精度。

```bash
uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" prepare-model \
  --config "$OCR_WORK/vendor/PaddleOCR/configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml" \
  --checkpoint "$OCR_WORK/weights/PP-OCRv6_small_rec_pretrained.pdparams" \
  --dictionary "$OCR_WORK/vendor/PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt" --output "$OCR_WORK/models/word"
uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" prepare-model \
  --config "$OCR_WORK/vendor/PaddleOCR/configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml" \
  --checkpoint "$OCR_WORK/weights/PP-OCRv6_small_rec_pretrained.pdparams" \
  --dictionary pipeline/configs/ascii_keys.txt --output "$OCR_WORK/models/char"
uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" prepare-model \
  --config "$OCR_WORK/vendor/PaddleOCR/configs/rec/PP-OCRv5/multi_language/korean_PP-OCRv5_mobile_rec.yml" \
  --checkpoint "$OCR_WORK/weights/korean_PP-OCRv5_mobile_rec_pretrained.pdparams" \
  --dictionary "$OCR_WORK/vendor/PaddleOCR/ppocr/utils/dict/ppocrv5_korean_dict.txt" \
  --append-characters '★' '《' '》' --output "$OCR_WORK/models/ko"
uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" generate \
  --recipe-config pipeline/configs/dataset/recipe.json \
  --style-recipe pipeline/configs/styles.json --corpus "$OCR_WORK/corpus.jsonl" \
  --output "$OCR_WORK/data/pilot" --pilot
```

先生成 pilot：六路线、各六类、训练/开发各 10 张，共 720 张；逐图重放、来源/标签/字典/分组检查全部通过才写 `audit.json`。正式数据分两次生成：

```bash
uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" generate \
  --recipe-config pipeline/configs/dataset/recipe.json \
  --style-recipe pipeline/configs/styles.json --corpus "$OCR_WORK/corpus.jsonl" \
  --output "$OCR_WORK/data/base"
uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" generate \
  --recipe-config pipeline/configs/dataset/recipe.json \
  --style-recipe pipeline/configs/styles.json --corpus "$OCR_WORK/corpus.jsonl" \
  --expansion-base "$OCR_WORK/data/base" --expansion-route char \
  --output "$OCR_WORK/data/formal"
```

基础集每条 Word 路线与 Char 各有 10,000 训练图 + 2,000 开发图，共 72,000 张；只扩 Char 后共 82,000 张，Char 为 20,000 + 2,000。正式 Char 计划要求扩量后的数量；扩量只是本次冻结配方的选型，不代表任意扩量都有收益。训练曝光比例由配方指定的 `sampling.json` 控制。
数据数量与采样入口为 `configs/dataset/recipe.json`，模型与字典配置为 `configs/training.json`。新训练必须基于实际源码和输入重新冻结计划；改动数据或代码后不能继续使用旧计划。

`styles.json` 保存当前已选的字体、字号、外观参数和公开背景引用，不包含私人截图。更换游戏数据来源时还要检查字形覆盖、字段提取和画面外观；仅更新素材 URL 不等于新数据通过训练验收。外服字体和主题仍需客户端实图对照。

## 4. 冻结计划后再训练

复制 `configs/train-spec.example.json` 和 `configs/resource-policy.json` 到自己的工作目录，在 spec 的 `resource_policy` 中填写后者路径，再填写数据与模型路径和路线。必须提供：已校验的数据集、训练 profile、初始模型与模型配置、保留能力样本清单，以及同一评判清单下旧发布模型/当前候选的原生预测。不可拿待训练模型的答案当真值，也不能为空缺材料自动生成“通过”。

- `guard_manifest` 每行包含 `id,image,sha256,label,scene`；若为 ASCII 字面样本，保留 `literal_ascii`。
- `evaluation.manifest` 每行另含 `mode,expected_output,subset`，任务模式需 `task_name`；图像路径为绝对路径，由原生探针读取。
- 两份参考预测是发布接口探针的 `{"predictions":[...]}` 输出，案例 ID 必须与评判清单完全一致。
- 新评判清单的原文案例必须携带 `evidence_kind`：合成图为 `synthetic`，已复核真实图为 `reviewed_real`。旧 `raw/final/`、`raw/v3/`、`raw/real/` 清单按原前缀兼容；其它未标来源的案例会被拒绝。
- 路线选 `zh-CN/zh-TW/en/ja/ko/char`；初始配置分别使用 word/ko/char 的 `model.yml`。自己的材料位置和预算全部写进新 spec，不修改旧冻结证据。

```bash
uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" plan \
  --spec "$OCR_WORK/train-spec.json" --output "$OCR_WORK/plan.json"
uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" train \
  --plan "$OCR_WORK/plan.json" --output "$OCR_WORK/runs/train-01" --print-command
# 审阅冻结输入后，去掉 --print-command 才真正开始；中断后加 --resume。
```

`resource-policy.json` 的 `gpu_index` 是 CUDA 可见序号，与 `paddle.set_device("gpu:N")` 一致；支持在多卡机器上选一张训练，也遵循 `CUDA_VISIBLE_DEVICES` 与容器的可见设备设置。监测通过 CUDA UUID 关联 `nvidia-smi`，不会把逻辑序号当作物理序号。当前支持完整 GPU，MIG 分区未适配。默认显存、温度和磁盘门槛来自已测的 16 GB 配置；不同设备应在外部策略文件中明确调整，并重新冻结计划。

计划会锁定数据、代码、资源策略、权重、字典、发布规则、运行库与参考结果。正式计划要求原生 MAA 评估及失败停止，不允许通过关闭检查来宣称正式训练。Char 训练使用固定尺寸预处理，通用文字训练使用 MAA 发布版预处理；两者都会经过同一发布接口的导出检查。训练设置有限步数、工作目录互斥、空闲显存/磁盘和温度检查；保存两份恢复点、随机状态、采样位置与优化器。GPU 后续计算不保证逐位一致，恢复完整性与质量门槛分开判断。

若真实失败来自 CharOCR，可用 `char-rework` 在冻结 Char 数据上加入已复核的真实关卡裁剪。spec 必须记录原图哈希、可见真值、MAA ROI/阈值、复核人和训练变体数；生成图不得进入独立准确率评判。加入裁剪后需新建冻结计划，命令见[评估与交付](#6-评估与交付)。

训练时应保留计划、指标、最终权重及其哈希，输出均写入外部工作目录。后续训练建立新记录，不覆盖已有结果。

## 5. 导出和候选检查

```bash
uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" export-check \
  --checkpoint "$OCR_WORK/runs/train-01/recovery/step-0001000/model.pdparams" \
  --config "$OCR_WORK/runs/train-01/model.yml" --manifest "$OCR_WORK/evaluation.jsonl" \
  --kind word --output "$OCR_WORK/runs/export-01"
```

若不微调通用模型，可在 `prepare-model` 后直接把 `--checkpoint` 指向输出的 `initial.pdparams`、`--config` 指向 `model.yml`，以真实评判输入执行同一个 `export-check`；不需要先生成训练图或执行 `train`。韩文第 3 节示例在官方字典末尾追加微调所需的三个符号；若直接导出官方韩文基底，去掉 `--append-characters`，使用官方原字典新建输出目录。扩展字典新增 `★`、`《`、`》` 的输出层参数需要训练，不能将其未经训练的结果称为原始官方模型。

Char 使用 `--kind char`。入口依次导出、按源 PIR 修复属性、执行必要的标准 ONNX 变换、比较 eager/PIR/Python ORT/发布库输出；失败返回非零。内部导出的 Paddle 中间产物不能直接当作已验收 ONNX。训练检查点也调用同一套实际接口检查。

部署前在目标平台上用同一发布库、检测器、任务规则和输入，对照原资源与候选。分别检查资源加载、原始文字和完整业务输出；对应语言需要客户端实图，加载成功不代表识别质量通过。当前入口不更换检测器、任务替换规则或推理后端；性能默认只记录，不设硬门槛。

正式整帧验收除已标注真值和分析成功状态外，还逐例比较任务输出；未标注的数量、顺序等业务变化默认阻止通过。若人工确认某一处变化正确，可在冻结案例写入 `approved_business_change: {"reason": "复核依据", "candidate_canonical_sha256": "该候选规范化业务输出的 SHA-256"}`。哈希由 `acceptance_cases.business_digest()` 计算；候选输出再变或例外不再适用时必须重新复核。只变置信分数不算业务变化。

## 6. 评估与交付

```bash
ocr() { uv run --project pipeline/environment python pipeline/pipeline.py --runtime "$OCR_WORK" "$@"; }
```

若独立真实截图揭示 CharOCR 的明确错误，复核可见文字、原图哈希及 MAA 的 ROI／阈值后，才用 `char-rework` 把裁剪及有限变体加入**训练**数据。命令要求已审计的冻结基础集和写明 `reviewer`、`review_status`、`source_revision`、`variants` 的来源 spec；输出的图都禁止作为独立验收真值。

```bash
ocr char-rework --base "$OCR_WORK/data/formal" \
  --spec "$OCR_WORK/char-rework/source-spec.json" \
  --output "$OCR_WORK/data/char-rework"
```

需要开发集选版时，用 `freeze-evaluation` 固定图像、真值、来源类别和哈希，再用 `evaluate` 将候选与官方发布资源作同接口配对比较。新来源必须明确 `evidence_kind`：合成图为 `synthetic`，已复核实图为 `reviewed_real`；不能让训练图进入独立验收。

```bash
ocr freeze-evaluation --spec "$OCR_WORK/eval-inputs.json" --output "$OCR_WORK/frozen-eval"
ocr evaluate --spec "$OCR_WORK/evaluation.json" --output "$OCR_WORK/evaluation-result"
```

`evaluation.json` 可分别指定 `frames`、`recognition`、`timing`；整帧案例包含图片、SHA-256、路线、任务模式和人工真值。原版与候选必须使用同一批图片、MAA 发布库、任务规则和线程设置。正式计时采用固定预算并报告 P95 **响应时间比率**；当前 `timing_policy: record_only` 只记录速度，不将其设为模型质量否决项。业务输出若改变而没有明确真值或经复核的例外，评估应失败。

六条路线的 ONNX 与字典选定后，用 `package` 按逐文件 SHA-256 生成交接包；它不改变本地 MAA 安装，也不代表发布审批。

```bash
ocr package --spec "$OCR_WORK/package.json" --output "$OCR_WORK/handoff"
```

来源更新时依次新建素材锁、核对抽取字段／字典／字体、生成新数据、审计场景覆盖、建立新训练计划，再按上述方式比较官方发布版。每次结果记录对应的源码、输入和模型哈希；不要覆盖旧记录。

## 7. 代码与配置

| 环节 | 实现 | 生效配置 |
|---|---|---|
| 环境与素材 | `setup_runtime`、`build_native`、`source_assets`、`source_corpus` | `sources.lock.json`、`runtime.lock.json`、`weights.lock.json`、工具链锁；`dataset/game-assets.lock.json` 固定游戏表，`render-assets.lock.json` 固定字体和背景 |
| 数据与模型准备 | `prepare_model`、`recipe_dataset`、`core_training_data`、`training_preflight` | `training.json` 选模型与字典；`dataset/recipe.json` 选数量与采样；`styles.json`、`fonts.json` 定外观；`dataset/contract.json` 定输入尺寸与编码容量，`dataset/split-policy.json` 定分区 |
| 训练与恢复 | `prepare_training_plan`、`run_training_plan`、`train_model`、`training_batches`、`training_state` | 外部 train spec 决定初始权重、学习率和步数，外部 resource policy 决定设备和资源限制，`sampling.json` 决定曝光比例 |
| 导出与评估 | `multilingual_verify`、`checkpoint_select`、`evaluate`、`acceptance_cases` | 外部图像清单与评估 spec 决定真值、任务和计时，外部 package spec 决定交付资源 |

配置中的 `id`、`version`、`scope` 和来源描述用于辨认输入；它们不是训练开关。模型配置中没有隐式的 checkpoint 默认值，训练只读取冻结 spec 明确指定的权重。第三方提交、文件哈希与字符顺序均属于实际输入，不能随意改写。`docs/` 仅提供结果，不被流程作为默认输入。

保持 `run_training_plan` 为轻量启动层，可以在没有导入 Paddle 或游戏素材的情况下打印命令及帮助；张量转换与发布接口检查保留独立子进程，避免训练期修改已安装依赖。

在 `common/OCR` 执行 `uv run --project pipeline/environment python -m pytest tests -q` 运行自动化测试。测试使用小型夹具；完整训练、原生编译和目标平台集成需各自环境。新的游戏数据、模型、代码或配置都需要新锁与训练计划，不直接续用旧计划。

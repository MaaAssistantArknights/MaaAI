#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "推荐在 Python 虚拟环境中运行"
echo "国内用户请挂代理，或者自己想办法将以下 repo 及字体资源放到对应目录下"

# 支持的游戏客户端，对应游戏数据仓库中的语言
clients=(zh_CN zh_TW ja_JP ko_KR en_US)
# 每种语言对应的思源黑体字体子集；en_US 复用 CN（含拉丁字符）
declare -A font_langs=(
    [zh_CN]=CN [zh_TW]=TW [ja_JP]=JP [ko_KR]=KR [en_US]=CN
)

num_img=200000  # 每种语言生成的图片数量

game_data_dir="game_data/ArknightsGamedata"
renderer_dir="game_data/text_renderer"
fonts_dir="game_data/fonts"
pretrained_model="models/pretrained"

mkdir -p "$game_data_dir" "$renderer_dir" "$fonts_dir" "$pretrained_model" "datasets/generated"

if [ ! -d "$game_data_dir/.git" ]; then
    git clone https://github.com/ArknightsAssets/ArknightsGamedata.git --depth=1 "$game_data_dir"
else
    git -C "$game_data_dir" pull --ff-only
fi

if [ ! -d "$renderer_dir/.git" ]; then
    git clone https://github.com/Sanster/text_renderer --depth=1 "$renderer_dir"
else
    git -C "$renderer_dir" pull --ff-only
fi
python3 -m pip install -r "$renderer_dir/requirements.txt"

for fl in CN TW JP KR; do
    wget -nc "https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSans${fl}.zip" -P "$fonts_dir"
    yes | unzip "$fonts_dir/SourceHanSans${fl}.zip" -d "$fonts_dir" > /dev/null
done

# PP-OCRv6 为多语言统一模型，只需一份预训练权重
# 其他档位：PP-OCRv6_small_rec_pretrained.pdparams / PP-OCRv6_tiny_rec_pretrained.pdparams
wget -nc "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams" -P "$pretrained_model"

###### 以下是离线操作了 ######

for client in "${clients[@]}"; do
    echo "=== 生成 $client 数据 ==="
    font_lang="${font_langs[$client]}"
    ls "$fonts_dir/SubsetOTF/$font_lang"/* > "$fonts_dir/fonts_${client}.txt"

    python3 ./scripts/data/wording.py "$client"
    python3 ./scripts/data/number.py -l "$client"

    num_img_fraction=$((num_img / 100))
    num_short_img=$((num_img_fraction * 30))
    num_long_img=$((num_img_fraction * 60))
    num_number_img=$((num_img_fraction * 10))
    output="datasets/generated/render"
    fonts_list="$fonts_dir/fonts_${client}.txt"

    python3 "$renderer_dir/main.py" --fonts_list "$fonts_list" --config_file datasets/render.yaml --img_width=0 --corpus_dir "datasets/generated/$client/short/" --corpus_mode=list --num_img "$num_short_img" --chars_file="datasets/generated/$client/keys.txt" --strict --output_dir="$output/$client/short"
    python3 "$renderer_dir/main.py" --fonts_list "$fonts_list" --config_file datasets/render.yaml --img_width=0 --corpus_dir "datasets/generated/$client/long/" --corpus_mode=chn --length=7 --num_img "$num_long_img" --chars_file="datasets/generated/$client/keys.txt" --strict --output_dir="$output/$client/long"
    python3 "$renderer_dir/main.py" --fonts_list "$fonts_list" --config_file datasets/render.yaml --img_width=0 --corpus_dir "datasets/generated/$client/number/" --corpus_mode=list --num_img "$num_number_img" --chars_file="datasets/generated/$client/keys.txt" --strict --output_dir="$output/$client/number"

    python3 ./scripts/data/train_test_split.py "$output/$client/short/default/tmp_labels.txt" -o "$output/$client/short/default"
    python3 ./scripts/data/train_test_split.py "$output/$client/long/default/tmp_labels.txt" -o "$output/$client/long/default"
    python3 ./scripts/data/train_test_split.py "$output/$client/number/default/tmp_labels.txt" -o "$output/$client/number/default"

    python3 ./scripts/data/build_ppocr_labels.py "$output/$client" "datasets/generated/$client" "$client"
done

# 合并各语言为一份多语言字典和标签，供 PP-OCRv6 单模型训练
python3 ./scripts/data/merge_dataset.py --langs "${clients[@]}"

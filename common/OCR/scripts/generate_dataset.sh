#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "推荐在 Python 虚拟环境中运行"
echo "国内用户请挂代理，或者自己想办法将以下 repo 及字体资源放到对应目录下"

num_img=200000  # 总的生成图片数量
client="zh_CN"  # 生成哪种语言的图片，"zh_CN" | "zh_TW" | "ja_JP" | "ko_KR"
fontLang="CN"   # 下载哪种语言的字体 # "CN" | "TW" | "JP" | "KR"，和上面的要改一起改

echo "num_img: $num_img, client: $client, fontLang: $fontLang"

game_data_dir="game_data/ArknightsGameData"
renderer_dir="game_data/text_renderer"

mkdir -p game_data models/pretrained datasets/generated

if [ ! -d "$game_data_dir" ]; then
    git clone https://github.com/Kengxxiao/ArknightsGameData --depth=1 "$game_data_dir"
else
    git -C "$game_data_dir" pull --ff-only
fi

if [ ! -d "$renderer_dir" ]; then
    git clone https://github.com/Sanster/text_renderer --depth=1 "$renderer_dir"
else
    git -C "$renderer_dir" pull --ff-only
fi
python3 -m pip install -r "$renderer_dir/requirements.txt"

fonts_dir='game_data/fonts'
wget -nc https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSans$fontLang.zip -P $fonts_dir

# 下载你需要的哪个语言的即可，这几个应该不用挂代理也行
pretrained_model="models/pretrained"

wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar -P $pretrained_model
tar -xvf "$pretrained_model/ch_PP-OCRv3_rec_train.tar" -C "$pretrained_model"

wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_train.tar -P $pretrained_model
tar -xvf "$pretrained_model/chinese_cht_PP-OCRv3_rec_train.tar" -C "$pretrained_model"

wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_train.tar -P $pretrained_model
tar -xvf "$pretrained_model/japan_PP-OCRv3_rec_train.tar" -C "$pretrained_model"

wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_train.tar -P $pretrained_model
tar -xvf "$pretrained_model/korean_PP-OCRv3_rec_train.tar" -C "$pretrained_model"


###### 以下是离线操作了 ######

yes | unzip "$fonts_dir/SourceHanSans$fontLang.zip" -d "$fonts_dir"
ls "$PWD/$fonts_dir/SubsetOTF/$fontLang"/* > "$fonts_dir/fonts.txt"

python3 ./scripts/data/wording.py "$client"
python3 ./scripts/data/number.py -l "$client"

num_img_fraction=`expr $num_img / 100`
num_short_img=`expr $num_img_fraction \* 30`
num_long_img=`expr $num_img_fraction \* 60`
num_number_img=`expr $num_img_fraction \* 10`
output='datasets/generated/render'

python3 "$renderer_dir/main.py" --fonts_list "$fonts_dir/fonts.txt" --config_file datasets/render.yaml --img_width=0 --corpus_dir "datasets/generated/$client/short/" --corpus_mode=list --num_img "$num_short_img" --chars_file="datasets/generated/$client/keys.txt" --strict --output_dir="$output/$client/short"
python3 "$renderer_dir/main.py" --fonts_list "$fonts_dir/fonts.txt" --config_file datasets/render.yaml --img_width=0 --corpus_dir "datasets/generated/$client/long/" --corpus_mode=chn --length=7 --num_img "$num_long_img" --chars_file="datasets/generated/$client/keys.txt" --strict --output_dir="$output/$client/long"
python3 "$renderer_dir/main.py" --fonts_list "$fonts_dir/fonts.txt" --config_file datasets/render.yaml --img_width=0 --corpus_dir "datasets/generated/$client/number/" --corpus_mode=list --num_img "$num_number_img" --chars_file="datasets/generated/$client/keys.txt" --strict --output_dir="$output/$client/number"

python3 ./scripts/data/train_test_split.py "$output/$client/short/default/tmp_labels.txt" -o "$output/$client/short/default"
python3 ./scripts/data/train_test_split.py "$output/$client/long/default/tmp_labels.txt" -o "$output/$client/long/default"
python3 ./scripts/data/train_test_split.py "$output/$client/number/default/tmp_labels.txt" -o "$output/$client/number/default"

python3 ./scripts/data/build_ppocr_labels.py "$output/$client" "datasets/generated/$client" "$client"

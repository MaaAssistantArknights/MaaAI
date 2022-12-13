echo "注意请 cd 到当前目录下再运行，并推荐在 python 虚拟环境中运行"
echo "国内用户请挂代理，或者自己想办法将以下 repo 及字体资源放到对应目录下"

num_img=100000  # 总的生成图片数量
client="zh_CN"  # 生成哪种语言的图片，"zh_CN" | "zh_TW" | "ja_JP" | "ko_KR"
fontLang="CN"   # 下载哪种语言的字体 # "ZH" | "TW" | "JP" | "KR"，和上面的要改一起改

echo "num_img: $num_img, client: $client, fontLang: $fontLang"

if [ ! -d 'ArknightsGameData' ]; then
    git clone https://github.com/Kengxxiao/ArknightsGameData --depth=1
else
    git -C ArknightsGameData pull
fi

if [ ! -d 'text_renderer' ]; then
    git clone https://github.com/Sanster/text_renderer --depth=1
else
    git -C text_renderer pull
fi
python3 -m pip install -r text_renderer/requirements.txt

fonts_dir='fonts'
wget https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSans$fontLang.zip -P $fonts_dir

# 下载你需要的哪个语言的即可，这几个应该不用挂代理也行
pretrained_model="pretrained_model"

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar -P $pretrained_model
tar -xvf $pretrained_model/ch_PP-OCRv3_rec_train.tar -C $pretrained_model

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_train.tar -P $pretrained_model
tar -xvf $pretrained_model/chinese_cht_PP-OCRv3_rec_train.tar -C $pretrained_model

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_train.tar -P $pretrained_model
tar -xvf $pretrained_model/japan_PP-OCRv3_rec_train.tar -C $pretrained_model

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_train.tar -P $pretrained_model
tar -xvf $pretrained_model/korean_PP-OCRv3_rec_train.tar -C $pretrained_model


###### 以下是离线操作了 ######

yes | unzip fonts/SourceHanSans$fontLang.zip -d $fonts_dir
ls $PWD/$fonts_dir/SubsetOTF/$fontLang/* > $fonts_dir/fonts.txt

python3 ./utils/wording.py $client
python3 ./utils/number.py $client

num_img_fraction=`expr $num_img / 100`
num_short_img=`expr $num_img_fraction \* 30`
num_long_img=`expr $num_img_fraction \* 60`
num_number_img=`expr $num_img_fraction \* 10`
output='output/render' # 下面有些 python 脚本是 hardcode的，这里的输出目录不建议修改

python3 ./text_renderer/main.py --fonts_list $fonts_dir/fonts.txt --config_file render.yaml --img_width=0 --corpus_dir output/$client/short/ --corpus_mode=list --num_img $num_short_img --chars_file=output/$client/keys.txt --strict --output_dir=$output/$client/short
python3 ./text_renderer/main.py --fonts_list $fonts_dir/fonts.txt --config_file render.yaml --img_width=0 --corpus_dir output/$client/long/ --corpus_mode=chn --length=7 --num_img $num_long_img --chars_file=output/$client/keys.txt --strict --output_dir=$output/$client/long
python3 ./text_renderer/main.py --fonts_list $fonts_dir/fonts.txt --config_file render.yaml --img_width=0 --corpus_dir output/$client/number/ --corpus_mode=list --num_img $num_number_img --chars_file=output/$client/keys.txt --strict --output_dir=$output/$client/number

python3 ./utils/half_and_half.py $output/$client/short/default/tmp_labels.txt $output/$client/short/default
python3 ./utils/half_and_half.py $output/$client/long/default/tmp_labels.txt $output/$client/long/default
python3 ./utils/half_and_half.py $output/$client/number/default/tmp_labels.txt $output/$client/number/default

python3 ./utils/rename_for_ppocr.py ./output/render/$client ./output/$client $client

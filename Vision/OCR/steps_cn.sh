echo "注意请 cd 到当前目录下再运行！"
echo "国内用户请挂代理，或者自己想办法将以下 repo 及字体资源放到对应目录下"

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

fonts_dir='fonts'
wget https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansCN.zip -P $fonts_dir
yes | unzip fonts/SourceHanSansCN.zip -d $fonts_dir

ls $PWD/$fonts_dir/SubsetOTF/CN/* > $fonts_dir/fonts.txt

###### 以下是离线操作了 ######

num_img=100000  # 总的生成图片数量

python3 ./utils/wording.py
python3 ./utils/number.py

num_img_fraction=`expr $num_img / 100`
num_short_img=`expr $num_img_fraction \* 28`
num_long_img=`expr $num_img_fraction \* 70`
num_number_img=`expr $num_img_fraction \* 2`
render='output/render'
python3 ./text_renderer/main.py --fonts_list $fonts_dir/fonts.txt --config_file render.yaml --img_width=0 --corpus_dir output/zh_CN/short/ --corpus_mode=list --num_img $num_short_img --chars_file=output/zh_CN/keys.txt --strict --output_dir=$render/zh_CN/short
python3 ./text_renderer/main.py --fonts_list $fonts_dir/fonts.txt --config_file render.yaml --img_width=0 --corpus_dir output/zh_CN/long/ --corpus_mode=chn --length=7 --num_img $num_long_img --chars_file=output/zh_CN/keys.txt --strict --output_dir=$render/zh_CN/long
python3 ./text_renderer/main.py --fonts_list $fonts_dir/fonts.txt --config_file render.yaml --img_width=0 --corpus_dir output/zh_CN/number/ --corpus_mode=list --num_img $num_number_img --chars_file=output/zh_CN/keys.txt --strict --output_dir=$render/zh_CN/number

python ./utils/half_and_half.py $render/zh_CN/short/default/tmp_labels.txt $render/zh_CN/short/default
python ./utils/half_and_half.py $render/zh_CN/long/default/tmp_labels.txt $render/zh_CN/long/default
python ./utils/half_and_half.py $render/zh_CN/number/default/tmp_labels.txt $render/zh_CN/number/default

python ./utils/rename_for_ppocr.py ./output/render/zh_CN ./output/zh_CN zh_CN

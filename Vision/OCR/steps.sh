echo "不建议直接执行，最好照着敲，有哪步报错了及时解决下"

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
render='output/render'

wget https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansCN.zip -p $fonts_dir
wget https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansTW.zip -p $fonts_dir
wget https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansJP.zip -p $fonts_dir
wget https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansKR.zip -p $fonts_dir
yes | unzip fonts/SourceHanSansCN.zip -d $fonts_dir
yes | unzip fonts/SourceHanSansTW.zip -d $fonts_dir
yes | unzip fonts/SourceHanSansJP.zip -d $fonts_dir
yes | unzip fonts/SourceHanSansKR.zip -d $fonts_dir

ls $PWD/SubsetOTF/CN/* > $fonts_dir/fonts.txt

python3 ./wording.py
python3 ./text_renderer/main.py --fonts_list $fonts_dir/fonts.txt --config_file render.yaml --img_width=0 --corpus_dir output/zh_CN/short/ --corpus_mode=list --num_img 20000 --chars_file=output/zh_CN/keys.txt --strict --output_dir=$render/zh_CN/short
python3 ./text_renderer/main.py --fonts_list $fonts_dir/fonts.txt --config_file render.yaml --img_width=0 --corpus_dir output/zh_CN/long/ --corpus_mode=chn --length=7 --num_img 60000 --chars_file=output/zh_CN/keys.txt --strict --output_dir=$render/zh_CN/long

head -n 16000 $render/zh_CN/short/default/tmp_labels.txt > $render/zh_CN/short/default/train.txt
tail -n 4000 $render/zh_CN/short/default/tmp_labels.txt > $render/zh_CN/short/default/test.txt

head -n 48000 $render/zh_CN/long/default/tmp_labels.txt > $render/zh_CN/long/default/train.txt
tail -n 12000 $render/zh_CN/long/default/tmp_labels.txt > $render/zh_CN/long/default/test.txt

ls $PWD/$fonts_dir/SubsetOTF/CN/* > $fonts_dir/fonts.txt
ls $PWD/$fonts_dir/SubsetOTF/TW/* >> $fonts_dir/fonts.txt
ls $PWD/$fonts_dir/SubsetOTF/JP/* >> $fonts_dir/fonts.txt
ls $PWD/$fonts_dir/SubsetOTF/KR/* >> $fonts_dir/fonts.txt

python3 ./number.py
python3 ./text_renderer/main.py --fonts_list $fonts_dir/fonts.txt --config_file render.yaml --img_width=0 --corpus_dir output/numbers/corpus/ --corpus_mode=list --num_img 50000 --chars_file=output/numbers/keys.txt --strict --output_dir=$render/numbers/

head -n 40000 $render/numbers/default/tmp_labels.txt > $render/numbers/default/train.txt
tail -n 10000 $render/numbers/default/tmp_labels.txt > $render/numbers/default/test.txt

python ./rename_for_ppocr.py

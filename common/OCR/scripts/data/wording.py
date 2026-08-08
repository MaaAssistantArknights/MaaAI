import argparse
import os
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("client", choices=("zh_CN", "zh_TW", "ja_JP", "ko_KR", "en_US"))
parser.add_argument("--game-data", type=Path, default=Path("game_data/ArknightsGameData"))
parser.add_argument("--fonts-dir", type=Path, default=Path("game_data/fonts/SubsetOTF"))
parser.add_argument("--keys-dir", type=Path, default=Path("datasets/keys"))
parser.add_argument("--output-dir", type=Path, default=Path("datasets/generated"))
args = parser.parse_args()
client = args.client

from fontTools.ttLib import TTFont

# 每种语言对应思源黑体的字体子集；en_US 复用 CN（含拉丁字符）
FONT_LANG_MAP = {
    "zh_CN": "CN",
    "zh_TW": "TW",
    "ja_JP": "JP",
    "ko_KR": "KR",
    "en_US": "CN",
}

unicode_map = {}

font_dir = args.fonts_dir / FONT_LANG_MAP[client]
for f in os.listdir(font_dir):
    if not f.endswith("otf"):
        continue
    fontType = os.path.join(font_dir, f)
    font = TTFont(fontType)
    unicode_map = font['cmap'].tables[0].ttFont.getBestCmap()
    break


def parse_line(line):
    result = set()
    in_string = False
    string_start = 0
    string_end = 0
    pre_char = ""
    has_non_ascii = False
    for index in range(len(line)):
        char = line[index]
        if char == "\"" and pre_char != "\\":
            if not in_string:
                string_start = index + 1
            else:
                string_end = index
                if has_non_ascii:
                    wording = line[string_start:string_end]
                    # 小火龙档案里有段乱码，屏蔽掉
                    if r'■■■■■■■■■■■■■■■■■■\n■■■■■■■■■■\n■■■■■\n\n' in wording:
                        break
                    wording = re.sub(r"<.*?>", "", wording)
                    wording = re.sub(r"{.*?}", "", wording)
                    wording = wording.replace("\\\\", "\\")
                    wording = wording.replace("\\\"", "\"")
                    wording = wording.replace("\\n", "\n")
                    wording = wording.replace("\\t", "\n")
                    wording = wording.replace("\t", "\n")
                    wording = wording.replace("......", "\n")
                    wording = wording.replace("\r", "")
                    wording = wording.replace(" ", "")
                    lines = [line for line in wording.split(
                        "\n") if line and line != ' ']
                    loc_lines = set()
                    for l in lines:
                        not_support = False
                        for w in l:
                            if ord(w) not in unicode_map.keys():
                                not_support = True
                                break
                        if not not_support:
                            loc_lines.add(l)
                    result.update(loc_lines)
            in_string = not in_string
            has_non_ascii = False
        elif in_string and ord(char) > 127:
            has_non_ascii = True
        pre_char = char
    return result


def find_all_wording(dir):
    result = set()
    for root, _, files in os.walk(dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    result.update(parse_line(line))
    return result


wording = find_all_wording(args.game_data / client / 'gamedata' / 'excel')
wording.update(set([chr(x) for x in range(33, 127)]))
output_dir = args.output_dir / client
os.makedirs(output_dir, exist_ok=True)

all_context = '\n'.join(wording)
with open(os.path.join(output_dir, 'wording.txt'), 'w', encoding='utf-8') as f:
    f.write(all_context)

keys = set()
for k in all_context:
    if ord(k) <= 32:
        continue
    keys.add(k)
with open(args.keys_dir / f'{client}.txt', 'r', encoding='utf-8') as f:
    key_text = f.read()
for k in keys:
    if k not in key_text:
        key_text += k + "\n"
with open(os.path.join(output_dir, 'keys.txt'), 'w', encoding='utf-8') as f:
    f.write(key_text)

short_context = '\n'.join([w for w in wording if len(w) < 7])
short_output_dir = os.path.join(output_dir, 'short')
os.makedirs(short_output_dir, exist_ok=True)
with open(os.path.join(short_output_dir, 'short_wording.txt'), 'w', encoding='utf-8') as f:
    f.write(short_context)

long_context = '\n'.join([w for w in wording if len(w) >= 7])
long_output_dir = os.path.join(output_dir, 'long')
os.makedirs(long_output_dir, exist_ok=True)
with open(os.path.join(long_output_dir, 'long_wording.txt'), 'w', encoding='utf-8') as f:
    f.write(long_context)

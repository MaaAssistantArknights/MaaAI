import argparse
import os
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("client", choices=("zh_CN", "zh_TW", "ja_JP", "ko_KR", "en_US"))
parser.add_argument("--game-data", type=Path, default=Path("game_data/ArknightsGamedata"))
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

# ArknightsGamedata 仓库中的语言目录名（小写）
CLIENT_DIR_MAP = {
    "zh_CN": "cn",
    "zh_TW": "tw",
    "ja_JP": "jp",
    "ko_KR": "kr",
    "en_US": "en",
}

# 最终训练使用多语言合并大词典，语言混入无需过滤（混入字符本就是目标字符）

# 遮挡/乱码字符（如未翻译文本的 ■），从字典中剔除
BLOCK_CHARS = "\u25a0\u25a1\u2588"


def is_punct_only(text):
    """整行仅含标点/符号（无任何字母数字）时丢弃，避免 '...'、'——'、'”' 等无意义样本"""
    return not any(c.isalnum() for c in text)


# en_US 的 gamedata 文本 99.5% 是纯 ASCII，旧逻辑（仅提取含非 ASCII 的字符串）会
# 丢弃几乎全部英文文本。对 en_US 额外提取"含空格的英文句子"，并过滤非文本字段。
NON_TEXT_RE = re.compile(
    r'\.(png|jpe?g|webp|json|txt|wav|ogg|mp3|zip|tar)$|'
    r'^\d[\d:,.-]{7,}$|'          # 时间戳/日期/数字串
    r'^[a-zA-Z0-9_]{32,}$',       # 超长纯词/下划线 key（如 hash、id）
    re.IGNORECASE)
CONTROL_WORDS = {"null", "true", "false", "undefined", "none"}


def is_extractable_string(text):
    """该字符串是否为可提取的文本（en_US 放宽条件用）"""
    if not any(c.isalpha() for c in text):
        return False
    if text.lower() in CONTROL_WORDS:
        return False
    if NON_TEXT_RE.search(text):
        return False
    # 干员名等专有名词：首字母大写的单词（如 Ayerscarpe、W），与全小写的
    # JSON 字段名（name、code）区分开
    if re.fullmatch(r"[A-Za-z][A-Za-z'\-]{0,30}", text) and text[0].isupper():
        return True
    return " " in text or any(ord(c) > 127 for c in text)

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
                if has_non_ascii or client == "en_US":
                    wording = line[string_start:string_end]
                    # 小火龙档案里有段乱码，屏蔽掉
                    if r'■■■■■■■■■■■■■■■■■■\n■■■■■■■■■■\n■■■■■\n\n' in wording:
                        break
                    if client == "en_US" and not is_extractable_string(wording):
                        in_string = not in_string
                        has_non_ascii = False
                        pre_char = char
                        continue
                    wording = re.sub(r"<.*?>", "", wording)
                    wording = re.sub(r"{.*?}", "", wording)
                    wording = wording.replace("\\\\", "\\")
                    wording = wording.replace("\\\"", "\"")
                    wording = wording.replace("\\n", "\n")
                    wording = wording.replace("\\t", "\n")
                    wording = wording.replace("\t", "\n")
                    wording = wording.replace("......", "\n")
                    wording = wording.replace("\r", "")
                    if client != "en_US":
                        # 中/日/韩文本不依赖空格分词，删除空格（原脚本行为）；
                        # en_US 保留空格，否则英文句子会粘连
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
                        if not not_support and not is_punct_only(l):
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


wording = find_all_wording(args.game_data / CLIENT_DIR_MAP[client] / 'gamedata' / 'excel')
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
# 过滤掉不属于目标客户端书写系统的字符和遮挡符
key_text = ''.join(
    c for c in key_text
    if c not in BLOCK_CHARS)
with open(os.path.join(output_dir, 'keys.txt'), 'w', encoding='utf-8') as f:
    f.write(key_text)

# 渲染用字典：仅包含字体支持的字形，否则 --strict 模式下会因不支持的字形无限重试
render_key_text = ''.join(
    k + '\n' for k in key_text.splitlines() if k and ord(k) in unicode_map)
with open(os.path.join(output_dir, 'keys_render.txt'), 'w', encoding='utf-8') as f:
    f.write(render_key_text)

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

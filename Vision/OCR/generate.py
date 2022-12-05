import re
import os


print('Please clone these third-party repositories manually into your local directory\n'
      '- https://github.com/Kengxxiao/ArknightsGameData')


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
                    wording = wording.replace(" ", "")
                    result.update(
                        set([w for w in wording.split("\n") if w and w != ' ']))
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


# for root, dirs, _ in os.walk('ArknightsGameData'):
#     for client in dirs:

client = 'zh_CN'
wording = find_all_wording(os.path.join(
    'ArknightsGameData', client, 'gamedata', 'excel'))
wording.update(set([chr(x) for x in range(33, 127)]))
output_dir = os.path.join('output', client)
os.makedirs(output_dir, exist_ok=True)
context = '\n'.join(wording)
with open(os.path.join(output_dir, 'wording.txt'), 'w', encoding='utf-8') as f:
    f.write(context)
keys = set()
for k in context:
    if k == '\n' or k == ' ':
        continue
    keys.add(k)
with open('raw_keys.txt', 'r', encoding='utf-8') as f:
    key_text = f.read()
for k in keys:
    if k not in key_text:
        key_text += k + "\n"
with open(os.path.join(output_dir, 'keys.txt'), 'w', encoding='utf-8') as f:
    f.write(key_text)

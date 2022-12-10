import os
import json
import random
import sys

client = sys.argv[1]

corpus_dir = f'output/{client}/number/'
os.makedirs(corpus_dir, exist_ok=True)


def generate_stages():
    with open(f'ArknightsGameData/{client}/gamedata/excel/stage_table.json',
              'r',
              encoding="utf-8") as f:
        stages_json = json.loads(f.read())['stages']

    all_stages_code = set()
    for _, v in stages_json.items():
        code = v['code']
        cn_code = False
        for k in code:
            if ord(k) > 127:
                cn_code = True
                break
        if not cn_code:
            all_stages_code.add(code)

    with open(corpus_dir + 'numbers.txt', 'w', encoding="utf-8") as f:
        f.write('\n'.join(all_stages_code) + '\n')


def generate_numbers():
    numbers = []
    W_map = {
        "zh_CN": "万",
        "zh_TW": "萬",
        "ja_JP": "万",
        "ko_KR": "만"
    }
    unit = W_map[client]
    for i in range(1, 10000):
        if (1.0 / i * 3) > random.random():
            numbers.append(str(i) + unit)
        if (1.0 / i * 3) > random.random():
            for d in range(1, 10):
                if random.random() < 0.1:
                    numbers.append(str(i) + '.' + str(d) + unit)
                    break

    E_map = {
        "zh_CN": "亿",
        "zh_TW": "億",
        "ja_JP": "億",
        "ko_KR": "억"
    }
    unit = E_map[client]
    for i in range(1, 100):
        if random.random() < 0.01:
            numbers.append(str(i) + unit)
        for d in range(1, 10):
            if random.random() < 0.01:
                numbers.append(str(i) + '.' + str(d) + unit)
                break

    # for i in range(1, 1000):
    #     for unit in ['K', 'M']:
    #         numbers.append(str(i) + unit)
    #         for d in range(1, 10):
    #             numbers.append(str(i) + '.' + str(d) + unit)

    numbers += [str(x) for x in range(1, 400)]
    with open(corpus_dir + 'numbers.txt', 'a', encoding="utf-8") as f:
        f.write('\n'.join(numbers) + '\n')


def generate_other():
    # For Public Recruitment
    numbers = ['0' + str(x) for x in range(10)]
    numbers += [str(x) for x in range(10, 60)]

    # All Chars
    numbers += [chr(x) for x in range(33, 127)]

    with open(corpus_dir + 'numbers.txt', 'a', encoding="utf-8") as f:
        f.write('\n'.join(numbers) + '\n')


generate_stages()
generate_other()
generate_numbers()

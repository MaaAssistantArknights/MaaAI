import os
import json
import random
import sys
from typing import Literal, Union, Tuple
import argparse as A
from pathlib import Path

ClientLang = Union[Literal['zh_CN'], Literal['en_US'], Literal['ja_JP'],
                   Literal['ko_KR'], Literal['zh_TW'], ]
client = sys.argv[1]

OUTPUT_DIR = f'output/{client}/number/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def uniform_exponent_range(base: float, lo: float, hi: float, size: int):
    for _ in range(size):
        exponent = random.uniform(lo, hi)
        yield base**exponent


def generate_stages(stages: Union[dict, str]):
    # open stage json
    if isinstance(stages, (str, Path)):
        with open(stages, 'r', encoding="utf-8") as f:
            stages = json.loads(f.read())['stages']
    # Iterate through all the data
    all_stages_code = [code for code in stages.values() if code.is_ascii()]
    return all_stages_code


def generate_numbers(lang: ClientLang, counts: Tuple = (10000, 20)):
    numbers = []

    UNITS_BY_LANG = {
        "zh_CN": "万亿",
        "zh_TW": "萬億",
        "ja_JP": "万億",
        "ko_KR": "만억",
    }
    for i, count in enumerate(counts):
        unit = UNITS_BY_LANG[lang][i]
        rng = uniform_exponent_range(10, 1, 5, count)
        for v in rng:
            if random.random() < 0.1:
                numbers.append(f"{v:.1}{unit}")
            else:
                numbers.append(f"{int(v)}{unit}")

    numbers += [str(x) for x in range(1, 400)]
    return numbers


def generate_other():
    # For Public Recruitment
    numbers = ['0' + str(x) for x in range(10)]
    numbers += [str(x) for x in range(10, 60)]

    # All Chars
    numbers += [chr(x) for x in range(33, 127)]
    return numbers


def main(args):
    output_dir = Path(args.output_dir) / args.lang
    f = open(output_dir / 'numbers.txt', 'w', encoding="utf-8")
    # Write stages
    stages = generate_stages(args.game_data / args.lang / "gamedata" /
                             "excel" / "stage_table.json")
    f.writelines(stages)

    # write others
    others = generate_other()
    f.writelines(others)
    # generate numbers
    numbers_size = (args.total, int(args.total * args.ratio_100m))
    numbers = generate_numbers(args.lang, numbers_size)
    f.writelines(numbers)

    f.close()


def parse_args():
    parser = A.ArgumentParser()
    parser.add_argument("--lang",
                        "-l",
                        choices=("zh_CN", "zh_TW", "ja_JP", "ko_KR"),
                        help="target language, default to \"zh_CN\"",
                        default="zh-CN")
    parser.add_argument(
        "--game_data",
        "-g",
        default="ArknightsGameData",
        type=Path,
        help="path to game_data, default to \"ArknightsGameData\"")
    parser.add_argument("--output_dir", "-o", default='./output', type=Path)
    parser.add_argument(
        "--ratio_100m",
        "-r",
        default=20 / 10000,
        type=float,
        help="proportation for numbers >= 100m, default to 2/1000")
    parser.add_argument("--total",
                        "-t",
                        default=10000,
                        type=int,
                        help="total numbers to be generated, default to 10000")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

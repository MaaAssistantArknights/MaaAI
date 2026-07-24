import os
import argparse
from collections import defaultdict
from pathlib import Path


def as_line(input_dir, output_file):
    txt_context = ''
    for path, _, file_list in os.walk(input_dir):
        for file_name in file_list:
            if not (file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.bmp')):
                continue
            full_path = os.path.join(path, file_name)
            stem = file_name[:-4]
            txt_context += full_path + '\t' + stem + '\n'

    txt_context = txt_context.replace('\\', '/')
    with open(output_file, mode='w', encoding='utf-8') as fd:
        fd.write(txt_context)


def as_array(input_dir, output_file, offline_path=None, offline_prefix=None):
    text_set = defaultdict(list)
    for path, _, file_list in os.walk(input_dir):
        for file_name in file_list:
            if not file_name.endswith('.png'):
                continue
            full_path = os.path.join(path, file_name)
            stem = file_name[:-len('.png')]
            text_set[stem].append(full_path)

    if offline_path and offline_prefix:
        with open(offline_path, mode='r', encoding='utf-8') as fd:
            for l in fd.readlines():
                pos = l.find(' ')
                full_path = offline_prefix + l[:pos] + ".jpg"
                stem = l[pos+1:-1]
                text_set[stem].append(full_path)

    txt_context = ''
    for name, imgs in text_set.items():
        line = '['
        for i in imgs:
            line += f'"{i}", '
        line = line[:-2] + "]"
        line += '\t' + name + '\n'
        txt_context += line

    txt_context = txt_context.replace('\\', '/')
    with open(output_file, mode='w', encoding='utf-8') as fd:
        fd.write(txt_context)


def restruct_render(input_file, output_file):
    txt_context = ''
    with open(input_file, mode='r', encoding='utf-8') as fd:
        for l in fd.readlines():
            txt_context += os.path.dirname(input_file) + \
                "/" + l.replace(' ', '.jpg\t')

    txt_context = txt_context.replace('\\', '/')

    with open(output_file, mode='a', encoding='utf-8') as fd:
        fd.write(txt_context)


parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=Path)
parser.add_argument("output_dir", type=Path)
parser.add_argument("region")
parser.add_argument("--custom-dir", type=Path, default=Path("datasets/custom"))
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
region = args.region

output_train_file = os.path.join(output_dir, 'rec_gt_train.txt')
output_test_file = os.path.join(output_dir, 'rec_gt_test.txt')
os.makedirs(output_dir, exist_ok=True)

# Rebuild labels from scratch so repeated runs do not duplicate generated data.
for output_file in (output_train_file, output_test_file):
    with open(output_file, mode='w', encoding='utf-8'):
        pass

if os.path.exists(args.custom_dir / region / 'train'):
    as_line(args.custom_dir / region / 'train', output_train_file)

if os.path.exists(args.custom_dir / region / 'test'):
    as_line(args.custom_dir / region / 'test', output_test_file)

for path, _, files in os.walk(input_dir):
    for f in files:
        if f == 'train.txt':
            restruct_render(os.path.join(path, f), output_train_file)
        elif f == 'test.txt':
            restruct_render(os.path.join(path, f), output_test_file)

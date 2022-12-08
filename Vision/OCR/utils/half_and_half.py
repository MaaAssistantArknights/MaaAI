import os
import sys
import random

input_file = sys.argv[1]
output_dir = sys.argv[2]


def simple_count():
    lines = 0
    for _ in open(input_file):
        lines += 1
    return lines


train_str = ''
test_str = ''
with open(input_file, mode='r', encoding="utf-8") as f:
    for line in f.readlines():
        if random.random() < 0.8:
            train_str += line
        else:
            test_str += line

with open(os.path.join(output_dir, 'train.txt'), mode='w', encoding="utf-8") as f:
    f.write(train_str)

with open(os.path.join(output_dir, 'test.txt'), mode='w', encoding="utf-8") as f:
    f.write(test_str)

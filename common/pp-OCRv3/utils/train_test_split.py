import os
import sys
import random
import argparse as A
from typing import Tuple, List


def parse_args() -> A.Namespace:
    """
    Get command line arguments

    Returns:
        parsed arguments
    """
    parser = A.ArgumentParser()
    parser.add_argument("--output",
                        "-o",
                        help="path to output folder",
                        required=True)
    parser.add_argument("--train_ratio",
                        "-t",
                        type=float,
                        default=0.8,
                        help="train sample ratios")
    parser.add_argument(
        "input_labels",
        help="path to the input file, containing all the samples")

    return parser.parse_args()


def train_test_split(labels: List[str],
                     train_ratio=0.8) -> Tuple[List[str], List[str]]:
    """
    Partitioning the labels into train labels and test labels
    Args:
        labels (List[str]): labels to be partitioned
        train_ratio: proportion of training samples, between 0-1.0, default to 0.8
    Returns:
        A tuple of training samples and test samples
    """
    train_files, test_files = [], []
    for sample in labels:
        if random.random() < train_ratio:
            train_files.append(sample)
        else:
            test_files.append(sample)
    return train_files, test_files


def main(args):
    train_samples = None
    test_samples = None

    with open(args.input_labels, mode='r', encoding="utf-8") as f:
        train_samples, test_samples = train_test_split(f.readlines(),
                                                       args.train_ratio)
    with open(os.path.join(args.output, 'train.txt'),
              mode='w',
              encoding="utf-8") as f:
        f.write(''.join(train_samples))

    with open(os.path.join(args.output, 'test.txt'),
              mode='w',
              encoding="utf-8") as f:
        f.write(''.join(test_samples))


if __name__ == "__main__":
    args = parse_args()
    main(args)

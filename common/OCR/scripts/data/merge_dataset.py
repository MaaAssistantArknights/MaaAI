import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge per-language keys and labels into a single multilingual dataset")
    parser.add_argument("--langs", nargs="+", required=True)
    parser.add_argument("--generated", type=Path, default=Path("datasets/generated"))
    return parser.parse_args()


def main():
    args = parse_args()

    keys = set()
    train_lines: list[str] = []
    test_lines: list[str] = []

    for lang in args.langs:
        lang_dir = args.generated / lang
        keys.update(k for k in lang_dir.joinpath("keys.txt").read_text(encoding="utf-8").splitlines() if k)
        for name, target in (("rec_gt_train.txt", train_lines), ("rec_gt_test.txt", test_lines)):
            label_file = lang_dir.joinpath(name)
            if label_file.exists():
                target += label_file.read_text(encoding="utf-8").splitlines()
            else:
                print(f"warning: {label_file} not found, skipped")

    args.generated.joinpath("keys.txt").write_text(
        "\n".join(sorted(keys)) + "\n", encoding="utf-8")
    args.generated.joinpath("rec_gt_train.txt").write_text(
        "\n".join(train_lines) + "\n", encoding="utf-8")
    args.generated.joinpath("rec_gt_test.txt").write_text(
        "\n".join(test_lines) + "\n", encoding="utf-8")

    print(f"merged keys: {len(keys)}, train: {len(train_lines)}, test: {len(test_lines)}")


if __name__ == "__main__":
    main()

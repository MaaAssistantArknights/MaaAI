import argparse
import os
import shutil

from PIL import Image, UnidentifiedImageError

from core.data_loader import DEFAULT_CLASS_NAMES, resolve_canonical_class_name


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS


class BKTreeNode:
    def __init__(self, hash_value, sample):
        self.hash_value = hash_value
        self.sample = sample
        self.children = {}


class BKTree:
    def __init__(self):
        self.root = None

    def add(self, hash_value, sample):
        if self.root is None:
            self.root = BKTreeNode(hash_value, sample)
            return

        node = self.root
        while True:
            distance = hamming_distance(hash_value, node.hash_value)
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKTreeNode(hash_value, sample)
                return
            node = child

    def query(self, hash_value, max_distance):
        if self.root is None:
            return None, None

        best_node = None
        best_distance = max_distance + 1
        stack = [self.root]

        while stack:
            node = stack.pop()
            distance = hamming_distance(hash_value, node.hash_value)
            if distance <= max_distance and distance < best_distance:
                best_node = node
                best_distance = distance

            lower = distance - max_distance
            upper = distance + max_distance
            for child_distance, child in node.children.items():
                if lower <= child_distance <= upper:
                    stack.append(child)

        return best_node, best_distance if best_node is not None else None


def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate similar screenshots with perceptual hash')
    parser.add_argument('--input_dir', default='datasets/train', type=str, help='输入数据目录，支持 c/n/y 和 *_err')
    parser.add_argument('--duplicates_dir', default='datasets/duplicates/train', type=str, help='重复图片输出目录')
    parser.add_argument('--distance_threshold', default=4, type=int, help='感知哈希汉明距离阈值，默认 4')
    parser.add_argument('--hash_size', default=8, type=int, help='dHash 尺寸，默认 8 生成 64 位哈希')
    parser.add_argument('--clear_duplicates', action='store_true', help='运行前清空重复图片输出目录')
    return parser.parse_args()


def iter_image_files(folder_path):
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            yield filename


def hamming_distance(left, right):
    return (left ^ right).bit_count()


def compute_dhash(image_path, hash_size):
    with Image.open(image_path) as image:
        grayscale = image.convert('L')
        resized = grayscale.resize((hash_size + 1, hash_size), RESAMPLE)
        pixels = list(resized.getdata())

    hash_value = 0
    row_width = hash_size + 1
    for row in range(hash_size):
        offset = row * row_width
        for col in range(hash_size):
            left = pixels[offset + col]
            right = pixels[offset + col + 1]
            hash_value = (hash_value << 1) | int(left > right)
    return hash_value


def collect_samples(input_dir, class_names):
    samples_by_class = {class_name: [] for class_name in class_names}
    skipped_folders = []

    for folder_name in sorted(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        canonical_name, is_hard_case = resolve_canonical_class_name(folder_name, class_names)
        if canonical_name is None:
            skipped_folders.append(folder_name)
            continue

        for filename in iter_image_files(folder_path):
            samples_by_class[canonical_name].append({
                'canonical_name': canonical_name,
                'source_folder': folder_name,
                'source_path': os.path.join(folder_path, filename),
                'filename': filename,
                'is_hard_case': is_hard_case,
            })

    return samples_by_class, skipped_folders


def ensure_duplicates_dir(duplicates_dir, clear_duplicates):
    if clear_duplicates and os.path.isdir(duplicates_dir):
        shutil.rmtree(duplicates_dir)
    os.makedirs(duplicates_dir, exist_ok=True)


def build_duplicate_path(duplicates_dir, sample):
    target_dir = os.path.join(duplicates_dir, sample['source_folder'])
    os.makedirs(target_dir, exist_ok=True)

    stem, ext = os.path.splitext(sample['filename'])
    candidate_path = os.path.join(target_dir, sample['filename'])
    suffix = 1
    while os.path.exists(candidate_path):
        candidate_path = os.path.join(target_dir, f'{stem}_{suffix}{ext}')
        suffix += 1
    return candidate_path


def sample_priority(sample):
    return (0 if sample['is_hard_case'] else 1, sample['source_folder'], sample['filename'])


def deduplicate_dataset(input_dir, duplicates_dir, distance_threshold, hash_size, clear_duplicates):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f'输入目录不存在: {input_dir}')
    if distance_threshold < 0:
        raise ValueError(f'distance_threshold 不能小于 0，当前为 {distance_threshold}')
    if hash_size < 4:
        raise ValueError(f'hash_size 不能小于 4，当前为 {hash_size}')

    ensure_duplicates_dir(duplicates_dir, clear_duplicates)
    class_names = list(DEFAULT_CLASS_NAMES)
    samples_by_class, skipped_folders = collect_samples(input_dir, class_names)

    summary = []
    total_kept = 0
    total_duplicates = 0
    invalid_images = []
    report_rows = []

    for class_name in class_names:
        tree = BKTree()
        kept_count = 0
        duplicate_count = 0
        samples = sorted(samples_by_class[class_name], key=sample_priority)

        for sample in samples:
            try:
                hash_value = compute_dhash(sample['source_path'], hash_size)
            except (OSError, UnidentifiedImageError) as exc:
                invalid_images.append((sample['source_path'], str(exc)))
                continue

            matched_node, distance = tree.query(hash_value, distance_threshold)
            if matched_node is None:
                tree.add(hash_value, sample)
                kept_count += 1
                continue

            duplicate_path = build_duplicate_path(duplicates_dir, sample)
            shutil.move(sample['source_path'], duplicate_path)
            duplicate_count += 1
            report_rows.append(
                f"{sample['source_path']}\t{duplicate_path}\t{matched_node.sample['source_path']}\t{distance}"
            )

        total_kept += kept_count
        total_duplicates += duplicate_count
        summary.append({
            'class_name': class_name,
            'source_total': len(samples),
            'kept_count': kept_count,
            'duplicate_count': duplicate_count,
        })

    report_path = os.path.join(duplicates_dir, 'dedup_report.tsv')
    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write('source_path\tduplicate_path\tkept_reference\thamming_distance\n')
        for row in report_rows:
            report_file.write(f'{row}\n')

    invalid_report_path = None
    if invalid_images:
        invalid_report_path = os.path.join(duplicates_dir, 'invalid_images.txt')
        with open(invalid_report_path, 'w', encoding='utf-8') as invalid_file:
            for path, error_message in invalid_images:
                invalid_file.write(f'{path}\t{error_message}\n')

    return {
        'summary': summary,
        'skipped_folders': skipped_folders,
        'total_kept': total_kept,
        'total_duplicates': total_duplicates,
        'report_path': report_path,
        'invalid_report_path': invalid_report_path,
    }


def main():
    args = parse_args()
    result = deduplicate_dataset(
        input_dir=args.input_dir,
        duplicates_dir=args.duplicates_dir,
        distance_threshold=args.distance_threshold,
        hash_size=args.hash_size,
        clear_duplicates=args.clear_duplicates,
    )

    print(
        f"完成去重 | 输入目录: {args.input_dir} | 重复图目录: {args.duplicates_dir} | "
        f"阈值: {args.distance_threshold} | 保留: {result['total_kept']} | 去重: {result['total_duplicates']}"
    )
    for item in result['summary']:
        print(
            f"类别 {item['class_name']} | 原始样本数: {item['source_total']} | "
            f"保留: {item['kept_count']} | 去重移出: {item['duplicate_count']}"
        )

    if result['skipped_folders']:
        print(f"已跳过未识别目录: {result['skipped_folders']}")

    print(f"去重报告: {result['report_path']}")
    if result['invalid_report_path']:
        print(f"异常图片报告: {result['invalid_report_path']}")


if __name__ == '__main__':
    main()
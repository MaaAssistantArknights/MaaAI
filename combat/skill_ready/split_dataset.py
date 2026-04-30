import argparse
import os
import random
import shutil

from core.data_loader import DEFAULT_CLASS_NAMES, resolve_canonical_class_name


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def parse_args():
    parser = argparse.ArgumentParser(description='Split part of training set into validation set')
    parser.add_argument('--train_dir', default='datasets/train', type=str, help='训练集目录，支持 c/n/y 和 *_err')
    parser.add_argument('--val_dir', default='datasets/val', type=str, help='验证集目录，只会写入 c/n/y')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='验证集占比，默认 0.1')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--move', action='store_true', help='移动文件到验证集；默认是复制')
    parser.add_argument('--clear_val', action='store_true', help='划分前清空验证集目录')
    return parser.parse_args()


def iter_image_files(folder_path):
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            yield filename


def collect_samples(train_dir, class_names):
    samples_by_class = {class_name: [] for class_name in class_names}
    skipped_folders = []

    for folder_name in sorted(os.listdir(train_dir)):
        folder_path = os.path.join(train_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        canonical_name, is_hard_case = resolve_canonical_class_name(folder_name, class_names)
        if canonical_name is None:
            skipped_folders.append(folder_name)
            continue

        for filename in iter_image_files(folder_path):
            samples_by_class[canonical_name].append({
                'source_folder': folder_name,
                'source_path': os.path.join(folder_path, filename),
                'filename': filename,
                'is_hard_case': is_hard_case,
            })

    return samples_by_class, skipped_folders


def ensure_empty_or_create_val_dir(val_dir, class_names, clear_val):
    if clear_val and os.path.isdir(val_dir):
        shutil.rmtree(val_dir)

    os.makedirs(val_dir, exist_ok=True)
    existing_files = []
    for class_name in class_names:
        class_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        existing_files.extend(list(iter_image_files(class_dir)))

    if existing_files and not clear_val:
        raise FileExistsError(
            f'验证集目录 {val_dir} 已存在图片，请先清空，或使用 --clear_val 重新划分。'
        )


def build_destination_path(val_dir, class_name, source_folder, filename):
    target_dir = os.path.join(val_dir, class_name)
    stem, ext = os.path.splitext(filename)
    candidate_path = os.path.join(target_dir, filename)

    if source_folder.endswith('_err'):
        candidate_path = os.path.join(target_dir, f'{stem}__from_{source_folder}{ext}')

    suffix = 1
    while os.path.exists(candidate_path):
        base_name = f'{stem}__from_{source_folder}_{suffix}{ext}'
        candidate_path = os.path.join(target_dir, base_name)
        suffix += 1

    return candidate_path


def split_dataset(train_dir, val_dir, val_ratio, seed, copy_files, clear_val):
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f'训练集目录不存在: {train_dir}')

    if not 0 < val_ratio < 1:
        raise ValueError(f'val_ratio 必须在 0 和 1 之间，当前为 {val_ratio}')

    class_names = list(DEFAULT_CLASS_NAMES)
    ensure_empty_or_create_val_dir(val_dir, class_names, clear_val)

    samples_by_class, skipped_folders = collect_samples(train_dir, class_names)
    rng = random.Random(seed)

    summary = []
    total_selected = 0
    for class_name in class_names:
        samples = samples_by_class[class_name]
        rng.shuffle(samples)
        selected_count = int(len(samples) * val_ratio)
        selected_samples = samples[:selected_count]

        hard_case_count = 0
        for sample in selected_samples:
            destination_path = build_destination_path(
                val_dir,
                class_name,
                sample['source_folder'],
                sample['filename']
            )
            if copy_files:
                shutil.copy2(sample['source_path'], destination_path)
            else:
                shutil.move(sample['source_path'], destination_path)

            if sample['is_hard_case']:
                hard_case_count += 1

        total_selected += selected_count
        summary.append({
            'class_name': class_name,
            'source_total': len(samples),
            'selected_count': selected_count,
            'hard_case_count': hard_case_count,
        })

    return summary, skipped_folders, total_selected


def main():
    args = parse_args()

    summary, skipped_folders, total_selected = split_dataset(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        copy_files=not args.move,
        clear_val=args.clear_val,
    )

    operation = '移动' if args.move else '复制'
    print(f'完成数据集划分 | 操作: {operation} | 验证集占比: {args.val_ratio:.2%} | 总转移样本数: {total_selected}')
    for item in summary:
        print(
            f"类别 {item['class_name']} | 原始样本数: {item['source_total']} | "
            f"划入验证集: {item['selected_count']} | 其中来自 *_err: {item['hard_case_count']}"
        )

    if skipped_folders:
        print(f'已跳过未识别目录: {skipped_folders}')


if __name__ == '__main__':
    main()
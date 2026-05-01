import argparse
import os
import shutil

from core.data_loader import DEFAULT_CLASS_NAMES, resolve_canonical_class_name


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def parse_args():
    parser = argparse.ArgumentParser(description='Remove normal-folder images that already exist in *_err folders')
    parser.add_argument('--input_dir', default='datasets/train', type=str, help='输入数据目录，支持 c/n/y 和 *_err')
    parser.add_argument('--removed_dir', type=str, help='如果指定，则将删除的普通目录图片移动到该目录；默认直接删除')
    parser.add_argument('--report_path', type=str, help='清理报告输出路径，默认写到 input_dir/err_conflict_report.tsv')
    parser.add_argument('--clear_removed', action='store_true', help='运行前清空 removed_dir')
    parser.add_argument('--dry_run', action='store_true', help='只生成报告，不实际删除或移动文件')
    return parser.parse_args()


def iter_image_files(folder_path):
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            yield filename


def collect_samples(input_dir, class_names):
    samples = []
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
            samples.append({
                'canonical_name': canonical_name,
                'source_folder': folder_name,
                'source_path': os.path.join(folder_path, filename),
                'filename': filename,
                'is_hard_case': is_hard_case,
            })

    return samples, skipped_folders


def ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def ensure_removed_dir(removed_dir, clear_removed):
    if not removed_dir:
        return
    if clear_removed and os.path.isdir(removed_dir):
        shutil.rmtree(removed_dir)
    os.makedirs(removed_dir, exist_ok=True)


def build_removed_path(removed_dir, sample):
    target_dir = os.path.join(removed_dir, sample['source_folder'])
    os.makedirs(target_dir, exist_ok=True)

    stem, ext = os.path.splitext(sample['filename'])
    candidate_path = os.path.join(target_dir, sample['filename'])
    suffix = 1
    while os.path.exists(candidate_path):
        candidate_path = os.path.join(target_dir, f'{stem}_{suffix}{ext}')
        suffix += 1
    return candidate_path


def clean_err_conflicts(input_dir, removed_dir=None, report_path=None, clear_removed=False, dry_run=False):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f'输入目录不存在: {input_dir}')

    report_path = report_path or os.path.join(input_dir, 'err_conflict_report.tsv')
    ensure_parent_dir(report_path)

    ensure_removed_dir(removed_dir, clear_removed)
    class_names = list(DEFAULT_CLASS_NAMES)
    samples, skipped_folders = collect_samples(input_dir, class_names)

    err_name_map = {}

    for sample in samples:
        if not sample['is_hard_case']:
            continue
        err_name_map.setdefault(sample['filename'], []).append(sample)

    summary = []
    total_removed = 0
    report_rows = []

    for class_name in class_names:
        class_samples = [sample for sample in samples if sample['canonical_name'] == class_name and not sample['is_hard_case']]
        removed_count = 0

        for sample in class_samples:
            matched_err_samples = err_name_map.get(sample['filename'])
            if not matched_err_samples:
                continue

            if dry_run:
                action_target = 'DRY_RUN'
            elif removed_dir:
                action_target = build_removed_path(removed_dir, sample)
                shutil.move(sample['source_path'], action_target)
            else:
                os.remove(sample['source_path'])
                action_target = 'DELETED'

            removed_count += 1
            total_removed += 1
            matched_paths = ';'.join(matched_sample['source_path'] for matched_sample in matched_err_samples)
            report_rows.append(
                f"{sample['source_path']}\t{sample['source_folder']}\t{sample['filename']}\t{action_target}\t{matched_paths}"
            )

        summary.append({
            'class_name': class_name,
            'source_total': len(class_samples),
            'removed_count': removed_count,
        })

    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write('normal_path\tnormal_folder\tfilename\taction\tmatched_err_paths\n')
        for row in report_rows:
            report_file.write(f'{row}\n')

    return {
        'summary': summary,
        'skipped_folders': skipped_folders,
        'total_removed': total_removed,
        'err_reference_count': sum(len(samples) for samples in err_name_map.values()),
        'unique_err_reference_count': len(err_name_map),
        'report_path': report_path,
    }


def main():
    args = parse_args()
    result = clean_err_conflicts(
        input_dir=args.input_dir,
        removed_dir=args.removed_dir,
        report_path=args.report_path,
        clear_removed=args.clear_removed,
        dry_run=args.dry_run,
    )

    action_text = '仅报告' if args.dry_run else ('移动到 removed_dir' if args.removed_dir else '直接删除')
    print(
        f"完成 _err 冲突清理 | 输入目录: {args.input_dir} | 动作: {action_text} | "
        f"_err 参考图: {result['err_reference_count']} | 唯一文件名: {result['unique_err_reference_count']} | "
        f"清理数量: {result['total_removed']}"
    )
    for item in result['summary']:
        print(
            f"类别 {item['class_name']} | 普通目录样本数: {item['source_total']} | "
            f"清理数量: {item['removed_count']}"
        )

    if result['skipped_folders']:
        print(f"已跳过未识别目录: {result['skipped_folders']}")

    print(f"清理报告: {result['report_path']}")


if __name__ == '__main__':
    main()
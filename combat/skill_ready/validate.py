import os
import shutil
import yaml
import torch
import time
import argparse
from core.data_loader import SkillIconDataset, get_dataset_class_counts, get_dataset_class_weights, get_dataset_sample_weights
from core.model_builder import create_model
from utils.logger import MetricLogger
from torch.utils.data import DataLoader
from utils.path_manager import get_model_paths


def build_hard_example_path(hard_example_root, true_class_name, pred_class_name, image_path):
    target_dir = os.path.join(hard_example_root, f"{true_class_name}_err")
    os.makedirs(target_dir, exist_ok=True)

    stem, ext = os.path.splitext(os.path.basename(image_path))
    base_name = f"{stem}__pred_{pred_class_name}"
    candidate_path = os.path.join(target_dir, f"{base_name}{ext}")
    suffix = 1

    while os.path.exists(candidate_path):
        candidate_path = os.path.join(target_dir, f"{base_name}_{suffix}{ext}")
        suffix += 1

    return candidate_path


def parse_args():
    parser = argparse.ArgumentParser(description='Validate Pytorch format model weights')
    parser.add_argument('--config', help='Specify model configuration file',
                        default='configs/mobilenetv4_conv_small.yaml', type=str, required=True)
    parser.add_argument('--weights', help='Specify pth path', type=str)
    parser.add_argument('--val_path', help='Specify validation set path', default='datasets/val', type=str)
    parser.add_argument('--hard_example_dir', help='将误判图片按真实类别导出到训练目录，如 datasets/train', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 1. 加载config
    try:
        with open(args.config, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("Config加载成功")
    except FileNotFoundError:
        print(f"Error: {args.config} 不存在")
        return

    # 2. 初始化设备
    device = torch.device("cpu")
    print(f"Using device: {device}")
    val_batchsize = int(config['training']['batch_size']) * 2  # 检测两次，如果1个错误则认为错误
    # 获取模型路径
    paths = get_model_paths(config)

    # 3. 创建dataloader
    try:
        model_wrapper = create_model(config)
        # 从指定路径创建验证集dataloader
        val_dataset = SkillIconDataset(
            root_dir=args.val_path,
            transform=model_wrapper.val_transform
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batchsize,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        val_class_counts = get_dataset_class_counts(val_dataset, len(model_wrapper.class_names))
        val_class_weights = get_dataset_class_weights(val_dataset, len(model_wrapper.class_names))
        val_sample_weights = get_dataset_sample_weights(
            val_dataset,
            len(model_wrapper.class_names)
        )
        print(
            f"Data loader统计 | 指定验证集路径: {args.val_path} | 验证样本数: {len(val_loader.dataset)} | 批次数: {len(val_loader)}")
        print(
            f"验证集类别计数 | "
            f"{' | '.join(f'{name}: {int(count)}' for name, count in zip(model_wrapper.class_names, val_class_counts.tolist()))}"
        )
        print(
            f"验证集评估权重 | "
            f"{' | '.join(f'{name}: {weight:.4f}' for name, weight in zip(model_wrapper.class_names, val_class_weights.tolist()))}"
        )
        print("验证集评估策略 | 仅按类别数量加权，不对 _err 额外加权")
    except KeyError as e:
        print(f"Data loading配置错误: {str(e)}")
        return

    # 4. 初始化模型
    try:
        if not args.weights:
            checkpoint_path = paths['checkpoint_export']
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pytorch模型文件 {checkpoint_path} 不存在")

        model = create_model(config).to(device)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"加载最佳模型权重: {checkpoint_path}")
    except RuntimeError as e:
        print(f"模型加载失败: {str(e)}")
        return

    # 5. 初始化记录器
    logger = MetricLogger(
        log_dir=os.path.join("pytorch_val"),
        class_names=['c', 'n', 'y'],
        model_name=config['model']['name'],
        class_weights=val_class_weights,
        val_sample_weights=val_sample_weights
    )
    print(f"验证报告目录: {logger.log_dir}")

    # 6. 推理性能测试（修复尺寸问题）
    try:
        input_size = config['data']['input_size']

        # 自动处理不同输入格式
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
            print(f"Warning: 自动转换input_size为 {input_size}")
        elif isinstance(input_size, (list, tuple)) and len(input_size) == 1:
            input_size = list(input_size) * 2
            print(f"Warning: 自动扩展input_size为 {input_size}")

        dummy_input = torch.randn(1, 3, *input_size).to(device)

        # 速度测试逻辑
        warmup_iters = 10
        test_iters = 100

        print("开始推理速度测试...")
        with torch.no_grad():
            # 预热
            for _ in range(warmup_iters):
                _ = model(dummy_input)

            # 正式速度测试
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            for _ in range(test_iters):
                _ = model(dummy_input)
            torch.cuda.synchronize() if device.type == 'cuda' else None

            avg_latency = (time.time() - start_time) * 1000 / test_iters
            print(f"推理速度 | 平均延迟: {avg_latency:.2f}ms | FPS: {1000 / avg_latency:.1f}")
    except KeyError:
        print("Error: config文件中缺少input_size定义，使用默认尺寸64x64")

    # 7. 完整验证过程
    print("\n开始完整验证...")
    logger.new_epoch()
    total_samples = 0
    hard_example_count = 0

    # 输出文件路径
    output_file = None
    if args.val_path:
        output_file = open(f"{paths['checkpoints_dir']}/results_{time.strftime('%Y%m%d_%H%M%S')}.txt", "w",
                           encoding="utf-8")

    try:
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                logger.log_val(0, outputs, labels)
                total_samples += labels.size(0)

                if batch_idx == 0:
                    print(f"第一批数据 | 输入shape: {images.shape} | 输出shape: {outputs.shape}")

                # 输出结果到文件
                if output_file:
                    probs = torch.softmax(outputs, dim=1)
                    for i in range(images.size(0)):
                        index = batch_idx * val_batchsize + i
                        image_path = val_loader.dataset.samples[index][0]
                        pred_class = torch.argmax(probs[i]).item()
                        true_class = labels[i].item()
                        class_probs = probs[i].tolist()
                        output_file.write(f"图片路径: {image_path}\n")
                        output_file.write(f"预测类别: {pred_class}\n")
                        output_file.write(f"真实类别: {true_class}\n")
                        output_file.write(f"类别置信度: {[f'{p:.8f}' for p in class_probs]}\n")
                        output_file.write("-" * 50 + "\n")

                        if args.hard_example_dir and pred_class != true_class:
                            target_path = build_hard_example_path(
                                args.hard_example_dir,
                                logger.class_names[true_class],
                                logger.class_names[pred_class],
                                image_path
                            )
                            shutil.copy2(image_path, target_path)
                            hard_example_count += 1

        print(f"总验证样本数: {total_samples}")
        if args.hard_example_dir:
            print(f"已导出难例样本数: {hard_example_count} -> {args.hard_example_dir}")
    except RuntimeError as e:
        print(f"验证过程异常: {str(e)}")
        return
    finally:
        if output_file:
            output_file.close()
            print(f"验证结果保存至 {paths['checkpoints_dir']}/results_{time.strftime('%Y%m%d_%H%M%S')}.txt")

    # 8. 生成最终报告
    logger.finalize_val()
    if 'accuracy' in logger.val_metrics:
        print(f"\n验证结果 | 准确率: {logger.val_metrics['accuracy']:.2%}")
    else:
        print("Warning: 验证指标未计算，请检查数据记录")
    report_paths = logger.get_report_paths()
    print(f"分类报告: {report_paths['classification_report']}")
    print(f"混淆矩阵: {report_paths['confusion_matrix']}")
    print(f"分类指标图: {report_paths['classification_metrics']}")
    logger.writer.close()


if __name__ == "__main__":
    main()
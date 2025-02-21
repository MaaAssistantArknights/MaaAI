import os
import yaml
import torch
import time
import numpy as np
import argparse
from core.data_loader import create_loaders, SkillIconDataset
from core.model_builder import create_model
from utils.logger import MetricLogger
from torch.utils.data import DataLoader
from utils.path_manager import get_model_paths

def parse_args():
    parser = argparse.ArgumentParser(description='Validate Pytorch format model weights')
    parser.add_argument('--config', help='指定模型配置文件', default='configs/mobilenetv4_conv_small.yaml', type=str, required=True)
    parser.add_argument('--weights', help='指定pth路径', type=str)
    parser.add_argument('--val_path', help='指定验证集路径', default='datasets/val', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 1. 加载配置文件
    try:
        with open(args.config, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ 配置文件加载完成")
    except FileNotFoundError:
        print(f"❌ 错误：配置文件 {args.config} 不存在")
        return

    # 2. 初始化设备
    device = torch.device("cpu")
    print(f"⚙️  使用设备: {device}")
    val_batchsize = int(config['training']['batch_size']) * 2  # 检测2遍，有1次错就当错
    # 获取路径
    paths = get_model_paths(config)

    # 3. 创建数据加载器
    try:
        model_wrapper = create_model(config)
        # 从指定路径创建 val_loader
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
        print(f"📊 数据加载器统计 | 指定验证集路径: {args.val_path} | 验证集样本: {len(val_loader.dataset)} | 批次: {len(val_loader)}")
    except KeyError as e:
        print(f"❌ 数据加载配置错误: {str(e)}")
        return

    # 4. 初始化模型
    try:
        if not args.weights:
            checkpoint_path = paths['checkpoint_export']
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pytorch 模型文件 {checkpoint_path} 不存在")

        model = create_model(config).to(device)
            
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"🔧 已加载最佳模型权重: {checkpoint_path}")
    except RuntimeError as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return

    # 5. 初始化日志记录器
    logger = MetricLogger(
        log_dir=os.path.join("pytorch_val"),
        class_names=['c', 'n', 'y'],
        model_name=config['model']['name'],
        class_weights=None
    )

    # 6. 推理性能测试（修复尺寸问题）
    try:
        input_size = config['data']['input_size']
        
        # 自动处理不同输入格式
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
            print(f"⚠️  已自动转换 input_size 为 {input_size}")
        elif isinstance(input_size, (list, tuple)) and len(input_size) == 1:
            input_size = list(input_size) * 2
            print(f"⚠️  已自动扩展 input_size 为 {input_size}")
            
        dummy_input = torch.randn(1, 3, *input_size).to(device)
        
        # 测速逻辑
        warmup_iters = 10
        test_iters = 100
        
        print("⏱️  开始推理速度测试...")
        with torch.no_grad():
            # Warmup
            for _ in range(warmup_iters):
                _ = model(dummy_input)
            
            # 正式测速
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            for _ in range(test_iters):
                _ = model(dummy_input)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            
            avg_latency = (time.time() - start_time) * 1000 / test_iters
            print(f"🚀 推理速度 | 平均延迟: {avg_latency:.2f}ms | FPS: {1000/avg_latency:.1f}")
    except KeyError:
        print("❌ 配置文件中缺少 input_size 定义，使用默认尺寸 64x64")
        dummy_input = torch.randn(1, 3, 64, 64).to(device)
        avg_latency = 0

    # 7. 完整验证流程
    print("\n🔍 开始完整验证...")
    logger.new_epoch()
    total_samples = 0

    # 输出文件路径
    output_file = None
    if args.val_path:
        output_file = open(f"{paths['checkpoints_dir']}/results_{time.strftime('%Y%m%d_%H%M%S')}.txt", "w", encoding="utf-8")

    try:
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                logger.log_val(0, outputs, labels)
                total_samples += labels.size(0)

                if batch_idx == 0:
                    print(f"📦 首批次数据 | 输入形状: {images.shape} | 输出形状: {outputs.shape}")

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
                        output_file.write(f"标签类别: {true_class}\n")
                        output_file.write(f"类别置信度: {[f'{p:.8f}' for p in class_probs]}\n")
                        output_file.write("-" * 50 + "\n")

        print(f"📥 已验证样本总数: {total_samples}")
    except RuntimeError as e:
        print(f"❌ 验证过程异常: {str(e)}")
        return
    finally:
        if output_file:
            output_file.close()
            print(f"📝 验证结果已保存到 {paths['checkpoints_dir']}/results_{time.strftime('%Y%m%d_%H%M%S')}.txt")

    # 8. 生成最终报告
    logger.finalize_val()
    if 'accuracy' in logger.val_metrics:
        print(f"\n📈 验证结果 | 准确率: {logger.val_metrics['accuracy']:.2%}")
    else:
        print("⚠️ 未计算验证指标，请检查数据记录")

if __name__ == "__main__":
    main()

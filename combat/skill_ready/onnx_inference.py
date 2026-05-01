import os
import shutil
import yaml
import torch
import time
import numpy as np
import argparse
from core.data_loader import SkillIconDataset, get_dataset_class_counts, get_dataset_class_weights, get_dataset_sample_weights
from core.model_builder import create_model
from utils.logger import MetricLogger
from torch.utils.data import DataLoader
from utils.path_manager import get_model_paths
import onnxruntime as ort
from torchvision.utils import save_image
from utils.img_process import unnormalize


GPU_PROVIDER_PRIORITY = (
    'CUDAExecutionProvider',
    'DmlExecutionProvider',
    'CoreMLExecutionProvider',
    'ROCMExecutionProvider',
    'TensorrtExecutionProvider',
)


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
    parser = argparse.ArgumentParser(description='Validate ONNX format model weights')
    parser.add_argument('--config', help='指定模型配置文件',
                        default='configs/mobilenetv4_conv_small.yaml', type=str, required=True)
    parser.add_argument('--weights', help='指定onnx路径', type=str)
    parser.add_argument('--val_path', help='指定验证集路径', default='datasets/val', type=str)
    parser.add_argument('--hard_example_dir', help='将误判图片按真实类别导出到训练目录，如 datasets/train', type=str)
    parser.add_argument('--use_gpu', help='优先使用可用的 GPU Execution Provider 进行 ONNX 推理', action='store_true')
    args = parser.parse_args()
    return args


def resolve_onnx_providers(use_gpu):
    if not use_gpu:
        return ['CPUExecutionProvider']

    available_providers = ort.get_available_providers()
    for provider_name in GPU_PROVIDER_PRIORITY:
        if provider_name in available_providers:
            return [provider_name, 'CPUExecutionProvider']

    available_text = ', '.join(available_providers) if available_providers else '无'
    raise RuntimeError(f"未找到可用的 GPU Execution Provider，可用 Providers: {available_text}")

def main():
    args = parse_args()

    # 1. 加载配置文件
    try:
        with open(args.config, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置文件加载完成")
    except FileNotFoundError:
        print(f"错误：配置文件 {args.config} 不存在")
        return

    # 2. 确认 ONNX Runtime provider 选择
    available_providers = ort.get_available_providers()
    provider_mode = 'GPU优先' if args.use_gpu else 'CPU'
    available_provider_text = ', '.join(available_providers) if available_providers else '无'
    providers = resolve_onnx_providers(args.use_gpu)
    tensor_device = torch.device('cpu')
    print(f"ONNX Runtime 请求模式: {provider_mode} | 当前可用 Providers: {available_provider_text}")
    print(f"ONNX Runtime 计划使用 Providers: {', '.join(providers)}")
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
        val_class_counts = get_dataset_class_counts(val_dataset, len(model_wrapper.class_names))
        val_class_weights = get_dataset_class_weights(val_dataset, len(model_wrapper.class_names))
        val_sample_weights = get_dataset_sample_weights(
            val_dataset,
            len(model_wrapper.class_names)
        )
        print(f"数据加载器统计 | 指定验证集路径: {args.val_path} | 验证集样本: {len(val_loader.dataset)} | 批次: {len(val_loader)}")
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
        print(f"数据加载配置错误: {str(e)}")
        return

    # 4. 初始化 ONNX 模型
    try:
        onnx_path = args.weights or paths['onnx_export']
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 模型文件 {onnx_path} 不存在")

        ort_session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"ONNX 模型加载完成: {onnx_path} | Execution Providers: {', '.join(ort_session.get_providers())}")
    except Exception as e:
        print(f"ONNX 模型加载失败: {str(e)}")
        return

    # 5. 初始化日志记录器
    logger = MetricLogger(
        log_dir=os.path.join("onnx_val"),
        class_names=['c', 'n', 'y'],
        model_name=config['model']['name'],
        class_weights=val_class_weights,
        val_sample_weights=val_sample_weights
    )
    print(f"验证报告目录: {logger.log_dir}")

    # 6. 推理性能测试
    try:
        input_size = config['data']['input_size']
        
        # 自动处理不同输入格式
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
            print(f"已自动转换 input_size 为 {input_size}")
        elif isinstance(input_size, (list, tuple)) and len(input_size) == 1:
            input_size = list(input_size) * 2
            print(f"已自动扩展 input_size 为 {input_size}")
            
        dummy_input = torch.randn(1, 3, *input_size).numpy().astype(np.float32)
        
        # 测速逻辑
        warmup_iters = 10
        test_iters = 100
        
        print("开始推理速度测试...")
        
        # Get input and output names
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        # Warmup
        for _ in range(warmup_iters):
            ort_session.run([output_name], {input_name: dummy_input})
        
        # 正式测速
        start_time = time.time()
        for _ in range(test_iters):
            ort_session.run([output_name], {input_name: dummy_input})
        avg_latency = (time.time() - start_time) * 1000 / test_iters
        print(f"ONNX 推理速度 | 平均延迟: {avg_latency:.2f}ms | FPS: {1000/avg_latency:.1f}")
    except KeyError:
        print("配置文件中缺少 input_size 定义，使用默认尺寸 64x64")
        dummy_input = torch.randn(1, 3, 64, 64).numpy().astype(np.float32)
        avg_latency = 0

    # 7. 完整验证流程
    print("\n开始完整验证...")
    logger.new_epoch()
    total_samples = 0
    hard_example_count = 0

    # 创建保存错误图片的 debug 目录
    debug_base_dir = os.path.join("debug", f"debug_{config['model']['name']}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(debug_base_dir, exist_ok=True)

    # 输出文件路径
    output_file = None
    if args.val_path:
        output_file = open(f"{paths['onnx_dir']}/results_{time.strftime('%Y%m%d_%H%M%S')}.txt", "w", encoding="utf-8")

    try:
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(tensor_device), labels.to(tensor_device)
                
                # ONNX 推理
                input_name = ort_session.get_inputs()[0].name
                raw_outputs = ort_session.run(None, {input_name: images.to(torch.device('cpu')).numpy()})[0]
                raw_outputs = torch.from_numpy(raw_outputs).to(tensor_device)
                outputs = torch.softmax(raw_outputs, dim=1)
                
                logger.log_val(0, outputs, labels)
                total_samples += labels.size(0)

                if batch_idx == 0:
                    print(f"首批次数据 | 输入形状: {images.shape} | 输出形状: {outputs.shape}")

                # 输出结果到文件
                if output_file:
                    probs = outputs
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

                        # 如果预测错误，则保存图片到 debug 目录下对应的标签子目录中
                        if pred_class != true_class:
                            # 根据预测类别获取对应的子目录 (例如 'c', 'n', 'y')
                            label_dir = os.path.join(debug_base_dir, logger.class_names[pred_class])
                            os.makedirs(label_dir, exist_ok=True)
                            # 从原始图片路径提取文件名
                            filename = os.path.basename(image_path)
                            save_path = os.path.join(label_dir, filename)
                            save_image(unnormalize(images[i].cpu(), tensor_device), save_path)
                            score_text = ' | '.join(
                                f"{class_name}: {score:.6f}"
                                for class_name, score in zip(logger.class_names, class_probs)
                            )
                            print(f"保存错误分类图片: {save_path} | {score_text}")

                            if args.hard_example_dir:
                                target_path = build_hard_example_path(
                                    args.hard_example_dir,
                                    logger.class_names[true_class],
                                    logger.class_names[pred_class],
                                    image_path
                                )
                                shutil.copy2(image_path, target_path)
                                hard_example_count += 1

        print(f"已验证样本总数: {total_samples}")
        if args.hard_example_dir:
            print(f"已导出难例样本数: {hard_example_count} -> {args.hard_example_dir}")
    except RuntimeError as e:
        print(f"验证过程异常: {str(e)}")
        return
    finally:
        if output_file:
            output_file.close()
            print(f"验证结果已保存到 {paths['onnx_dir']}/results_{time.strftime('%Y%m%d_%H%M%S')}.txt")

    # 8. 生成最终报告
    logger.finalize_val()
    if 'accuracy' in logger.val_metrics:
        print(f"\n验证结果 | 准确率: {logger.val_metrics['accuracy']:.2%}")
        print(f"验证结果 | 准确率(c/n不区分): {logger.val_metrics['accuracy_nc_merged']:.2%}")
    else:
        print("未计算验证指标，请检查数据记录")
    report_paths = logger.get_report_paths()
    print(f"分类报告: {report_paths['classification_report']}")
    print(f"混淆矩阵: {report_paths['confusion_matrix']}")
    print(f"分类指标图: {report_paths['classification_metrics']}")
    logger.writer.close()

if __name__ == "__main__":
    main()

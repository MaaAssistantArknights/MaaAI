import os
import yaml
import torch
import torch.optim as optim
import argparse

from torch import nn

from utils.loss import FocalLoss
from core.data_loader import create_loaders
from core.model_builder import create_model
from utils.logger import MetricLogger
from torch.amp import autocast, GradScaler
from utils.path_manager import get_model_paths

def parse_args():
    parser = argparse.ArgumentParser(description='Train model scripts')
    parser.add_argument('--config', help='Specify model configuration file',
                        default='configs/mobilenetv4_conv_small.yaml', type=str, required=True)
    args = parser.parse_args()
    return args


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

    # 2. 初始化训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    paths = get_model_paths(config)

    # 3. 创建dataloader
    try:
        model_wrapper = create_model(config)
        print("模型包装器创建完成")
        train_loader, val_loader = create_loaders(
            config,
            train_transform=model_wrapper.train_transform,
            val_transform=model_wrapper.val_transform
        )
        print("数据加载器创建完成")
    except KeyError as e:
        print(f"数据加载配置错误: {str(e)}")
        return
    
    # 4. 初始化模型
    try:
        model = create_model(config).to(device)
        print("模型初始化成功")
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        return

    # 5. 初始化优化器
    try:
        try:
            optim_type = config['training']['optimizer']['type']
            optimizer_cls = getattr(optim, optim_type)
        except AttributeError:
            print(f"优化器 {optim_type} 不存在，使用默认AdamW")
            optimizer_cls = optim.AdamW

        optimizer = optimizer_cls(
            model.parameters(),                                                  # 传入模型数据
            lr=float(config['training']['optimizer']['lr']),                     # 学习率
            weight_decay=float(config['training']['optimizer']['weight_decay'])  # 权重衰减系数
        )
        print("优化器初始化成功")
    except Exception as e:
        print(f"优化器初始化失败: {str(e)}")
        return

    # 6. 初始化损失函数
    loss_type = config['training']['loss']['type']
    # 如果特指使用FocalLoss，其余情况一律使用CrossEntropyLoss
    if loss_type == 'focal':
        criterion = FocalLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=None).to(device)

    # 7. 混合精度训练
    try:
        use_cuda_amp = config['training']['use_cuda_amp']
        if use_cuda_amp:
            scaler = GradScaler('cuda')
            print("使用混合精度训练")
    except Exception as e:
        print(f"不支持混合精度训练或启动错误: {str(e)}")
        return

    # 8. 记录
    logger = MetricLogger(  # Create MetricLogger object for recording training and validation metrics
        log_dir="train",  # Specify log directory
        class_names=['c', 'n', 'y'],  # Specify class names
        model_name=config['model']['name'], # Specify model name
        class_weights=None # class_weights.cpu().numpy() if torch.cuda.is_available() else class_weights.numpy()
    )

    # 9. 训练
    best_acc = 0.0
    for epoch in range(config['training']['epochs']):
        print()
        print(f"Epoch {epoch+1} 开始")
        # 训练阶段
        model.train()  # 将模型设置为训练模式
        logger.new_epoch()  # 开始新的 epoch

        # 训练
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            if use_cuda_amp:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels.long())
            else:
                outputs = model(images)
                loss = criterion(outputs, labels.long())

            if use_cuda_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()

            # 记录指标
            logger.log_train(loss.item(), outputs, labels)  # Record training loss, outputs, and labels
            if (i+1) % 10 == 0:
                print(f"    Batch {i+1}/{len(train_loader)} completed")

        # 验证
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                if use_cuda_amp:
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels.long())
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels.long())

                # 记录指标
                logger.log_val(loss.item(), outputs, labels)

        # 10. 检查点
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 11. 保存最好的模型
        logger.finalize_epoch()
        if logger.val_metrics['accuracy'] > best_acc:
            best_acc = logger.val_metrics['accuracy']
            torch.save(model.state_dict(), paths["checkpoint_export"])
        
        # 生成报告
        print(f"Epoch {epoch+1} 完成，验证集准确率: {logger.val_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()

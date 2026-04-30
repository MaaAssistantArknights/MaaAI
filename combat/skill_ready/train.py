import os
import math
import yaml
import torch
import torch.optim as optim
import argparse

from torch import nn

from utils.loss import CostSensitiveCrossEntropyLoss, FocalLoss
from core.data_loader import create_loaders, get_dataset_class_counts, get_dataset_class_weights, get_dataset_sample_weights
from core.model_builder import create_model
from utils.logger import MetricLogger
from torch.amp import autocast, GradScaler
from utils.path_manager import get_model_paths


def build_confusion_cost(class_names, loss_config):
    penalty_config = loss_config.get('confusion_penalty', {})
    if not penalty_config.get('enabled', False):
        return None, 0.0

    class_to_idx = {name: index for index, name in enumerate(class_names)}
    confusion_cost = torch.zeros(len(class_names), len(class_names), dtype=torch.float32)

    c_index = class_to_idx.get('c')
    y_index = class_to_idx.get('y')
    if c_index is not None and y_index is not None:
        confusion_cost[c_index, y_index] = float(penalty_config.get('c_to_y', 0.0))

    return confusion_cost, float(penalty_config.get('weight', 0.0))


def build_loss_class_weights(model, loss_config, device):
    class_weight_config = loss_config.get('class_weight', {})
    if not class_weight_config.get('enabled', True):
        return None

    return model.class_weights.to(device)


def build_lr_scheduler_state(config, optimizer):
    scheduler_config = config['training'].get('lr_scheduler', {})
    scheduler_type = scheduler_config.get('type')
    if not scheduler_type:
        return None

    scheduler_type = scheduler_type.lower()
    if scheduler_type != 'cosine':
        raise ValueError(f"不支持的学习率调度类型: {scheduler_type}")

    return {
        'type': scheduler_type,
        'epochs': int(config['training']['epochs']),
        'warmup_epochs': max(0, int(scheduler_config.get('warmup_epochs', 0))),
        'min_lr': float(scheduler_config.get('min_lr', 0.0)),
        'base_lrs': [group['lr'] for group in optimizer.param_groups],
    }


def apply_epoch_learning_rate(epoch, optimizer, scheduler_state):
    if scheduler_state is None:
        return optimizer.param_groups[0]['lr']

    total_epochs = max(1, scheduler_state['epochs'])
    warmup_epochs = min(scheduler_state['warmup_epochs'], total_epochs)

    for base_lr, param_group in zip(scheduler_state['base_lrs'], optimizer.param_groups):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            lr = base_lr * float(epoch + 1) / warmup_epochs
        else:
            cosine_epochs = max(1, total_epochs - warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, cosine_epochs - 1)
            progress = min(max(progress, 0.0), 1.0)
            min_lr = min(scheduler_state['min_lr'], base_lr)
            lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

        param_group['lr'] = lr

    return optimizer.param_groups[0]['lr']


def build_early_stopping_state(config):
    early_stopping_config = config['training'].get('early_stopping', {})
    if not early_stopping_config.get('enabled', False):
        return None

    return {
        'patience': max(1, int(early_stopping_config.get('patience', 5))),
        'min_delta': float(early_stopping_config.get('min_delta', 0.0)),
        'epochs_without_improvement': 0,
    }


def format_class_stats(class_names, values, precision=None):
    parts = []
    for name, value in zip(class_names, values):
        if precision is None:
            parts.append(f"{name}: {int(value)}")
        else:
            parts.append(f"{name}: {value:.{precision}f}")
    return ' | '.join(parts)


def evaluate_accuracy(model, data_loader, device, use_cuda_amp=False, sample_weights=None):
    was_training = model.training
    model.eval()
    total_correct = 0
    total_samples = 0
    predictions_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            if use_cuda_amp and device.type == 'cuda':
                with autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)

            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            if sample_weights is not None:
                predictions_list.append(predictions.detach().cpu())
                labels_list.append(labels.detach().cpu())

    if was_training:
        model.train()

    if total_samples == 0:
        return 0.0
    if sample_weights is not None:
        predictions = torch.cat(predictions_list)
        labels = torch.cat(labels_list)
        weights = sample_weights.detach().cpu()
        weighted_correct = weights * (predictions == labels).float()
        return (weighted_correct.sum() / weights.sum().clamp_min(1e-12)).item()
    return total_correct / total_samples

def parse_args():
    parser = argparse.ArgumentParser(description='Train model scripts')
    parser.add_argument('--config', help='Specify model configuration file',
                        default='configs/mobilenetv4_conv_small.yaml', type=str, required=True)
    parser.add_argument('--weights', help='Specify pth path for fine-tuning start point', type=str)
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

    if args.weights:
        config['model']['pretrained'] = False
        print(f"微调模式 | 起始权重: {args.weights}")

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
        err_folder_weight = float(config['training'].get('sampler', {}).get('err_folder_weight', 1.0))
        val_class_counts = get_dataset_class_counts(val_loader.dataset, len(model_wrapper.class_names))
        val_class_weights = get_dataset_class_weights(val_loader.dataset, len(model_wrapper.class_names))
        val_sample_weights = get_dataset_sample_weights(
            val_loader.dataset,
            len(model_wrapper.class_names),
            err_folder_weight=err_folder_weight
        )
        print(
            f"数据加载器创建完成 | 训练样本数: {len(train_loader.dataset)} | 验证样本数: {len(val_loader.dataset)} | "
            f"训练采样器: {type(train_loader.sampler).__name__}"
        )
        print(f"验证集类别计数 | {format_class_stats(model_wrapper.class_names, val_class_counts.tolist())}")
        print(f"验证集评估权重 | {format_class_stats(model_wrapper.class_names, val_class_weights.tolist(), precision=4)}")
        print(f"验证集难例加权系数 | _err: {err_folder_weight:.2f}")
    except KeyError as e:
        print(f"数据加载配置错误: {str(e)}")
        return
    
    # 4. 初始化模型
    try:
        model = model_wrapper.to(device)
        if args.weights:
            if not os.path.exists(args.weights):
                raise FileNotFoundError(f"微调权重文件 {args.weights} 不存在")
            model.load_state_dict(torch.load(args.weights, map_location=device))
            print(f"已加载微调起点权重: {args.weights}")
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

    try:
        scheduler_state = build_lr_scheduler_state(config, optimizer)
        if scheduler_state is not None:
            print(
                f"启用学习率调度 | 类型: {scheduler_state['type']} | "
                f"warmup_epochs: {scheduler_state['warmup_epochs']} | min_lr: {scheduler_state['min_lr']:.2e}"
            )
    except ValueError as e:
        print(f"学习率调度配置错误: {str(e)}")
        return

    early_stopping_state = build_early_stopping_state(config)
    if early_stopping_state is not None:
        print(
            f"启用 Early stopping | patience: {early_stopping_state['patience']} | "
            f"min_delta: {early_stopping_state['min_delta']:.4f}"
        )

    # 6. 初始化损失函数
    loss_config = config['training']['loss']
    loss_type = loss_config['type']
    confusion_cost, confusion_weight = build_confusion_cost(model.class_names, loss_config)
    class_weights = build_loss_class_weights(model, loss_config, device)

    print(f"训练集类别计数 | {format_class_stats(model.class_names, model.class_counts.tolist())}")

    if class_weights is not None:
        print(
            f"启用自动类别权重 | "
            f"{format_class_stats(model.class_names, class_weights.detach().cpu().tolist(), precision=4)}"
        )
    else:
        print("未启用自动类别权重")

    if confusion_cost is not None and confusion_weight > 0:
        print(f"启用非对称误判惩罚 | c->y 额外权重: {confusion_cost[0, 2].item():.2f} | 损失系数: {confusion_weight:.2f}")

    # 如果特指使用FocalLoss，其余情况一律使用CrossEntropyLoss
    if loss_type == 'focal':
        criterion = FocalLoss(
            weight=class_weights,
            confusion_cost=confusion_cost,
            confusion_weight=confusion_weight
        ).to(device)
    else:
        criterion = CostSensitiveCrossEntropyLoss(
            weight=class_weights,
            confusion_cost=confusion_cost,
            confusion_weight=confusion_weight
        ).to(device)

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
        class_weights=val_class_weights,
        val_sample_weights=val_sample_weights
    )
    print(f"训练报告目录: {logger.log_dir}")

    # 9. 训练
    best_acc = 0.0
    if args.weights:
        best_acc = evaluate_accuracy(model, val_loader, device, use_cuda_amp, val_sample_weights)
        print(f"微调起点验证集准确率: {best_acc:.4f}")

    for epoch in range(config['training']['epochs']):
        print()
        current_lr = apply_epoch_learning_rate(epoch, optimizer, scheduler_state)
        print(f"Epoch {epoch+1} 开始 | lr: {current_lr:.2e}")
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
        current_val_acc = logger.val_metrics['accuracy']
        if current_val_acc > best_acc + (early_stopping_state['min_delta'] if early_stopping_state else 0.0):
            best_acc = current_val_acc
            torch.save(model.state_dict(), paths["checkpoint_export"])
            if early_stopping_state is not None:
                early_stopping_state['epochs_without_improvement'] = 0
            print(f"保存最佳模型 | epoch: {epoch+1} | 验证集准确率: {best_acc:.4f}")
        elif early_stopping_state is not None:
            early_stopping_state['epochs_without_improvement'] += 1
            print(
                f"Early stopping 计数 | {early_stopping_state['epochs_without_improvement']}/"
                f"{early_stopping_state['patience']} | 当前最佳: {best_acc:.4f}"
            )
        
        # 生成报告
        print(f"Epoch {epoch+1} 完成，验证集准确率: {current_val_acc:.4f}")

        if early_stopping_state is not None and \
                early_stopping_state['epochs_without_improvement'] >= early_stopping_state['patience']:
            print(f"触发 Early stopping | 停止于 epoch {epoch+1} | 最佳验证集准确率: {best_acc:.4f}")
            break

    report_paths = logger.get_report_paths()
    print(f"分类报告: {report_paths['classification_report']}")
    print(f"混淆矩阵: {report_paths['confusion_matrix']}")
    print(f"分类指标图: {report_paths['classification_metrics']}")
    print(f"训练曲线图: {report_paths['training_curves']}")
    logger.writer.close()

if __name__ == "__main__":
    main()

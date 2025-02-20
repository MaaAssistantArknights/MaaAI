import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from core.data_loader import create_loaders  # 导入数据加载器
from core.base_model import BaseModel  # 导入基础模型类
from core.model_builder import create_model  # 导入模型创建函数
from utils.logger import MetricLogger  # 导入日志记录器
from torch.amp import autocast, GradScaler

def main():
    # 1. 加载配置文件
    config_path = os.environ.get('CONFIG_PATH', "configs/mobilenetv4.yaml")  # 从环境变量中获取配置文件路径，默认为 configs/mobilenetv4.yaml
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)  # 使用 PyYAML 加载配置文件
    print("配置文件加载完成")

    # 2. 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 如果 GPU 可用，则使用 GPU，否则使用 CPU
    print(f"使用设备: {device}")

    # 3. 创建数据加载器
    model_wrapper = create_model(config) # 创建模型包装器
    print("创建模型包装器完成")
    train_loader, val_loader = create_loaders(  # 创建训练集和验证集的数据加载器
        config,  # 传递数据相关的配置
        train_transform=model_wrapper.train_transform,  # 传递训练集的数据增强
        val_transform=model_wrapper.val_transform  # 传递验证集的数据增强
    )
    print("数据加载器创建完成")

    # 4. 初始化模型
    model = create_model(config).to(device)  # 使用模型创建函数创建模型，并将其移动到指定的设备上
    print("模型初始化完成")

    # 5. 初始化优化器和损失函数
    optimizer = optim.AdamW(  # 使用 AdamW 优化器
        model.parameters(),  # 传递模型的所有参数
        lr=float(config['training']['optimizer']['lr']),  # 传递学习率
        weight_decay=float(config['training']['optimizer']['weight_decay'])  # 传递权重衰减系数
    )
    print("优化器和损失函数初始化完成")

    # 6. 处理类别不平衡
    criterion = nn.CrossEntropyLoss(weight=None)  # 使用交叉熵损失函数，并传递类别权重

    # 7. 混合精度训练
    use_cuda_amp = config['training']['use_cuda_amp']
    if use_cuda_amp:
        scaler = GradScaler('cuda')  # 创建 GradScaler 对象，用于混合精度训练
        print("使用混合精度训练")

    # 8. 日志记录
    logger = MetricLogger(  # 创建 MetricLogger 对象，用于记录训练和验证指标
        log_dir="logs",  # 指定日志目录
        class_names=['c', 'n', 'y'],  # 指定类别名称
        model_name=config['model']['name'], # 指定模型名称
        class_weights=None # class_weights.cpu().numpy() if torch.cuda.is_available() else class_weights.numpy()
    )
    print("日志记录器初始化完成")

    # 9. 训练循环
    best_acc = 0.0  # 初始化最佳验证集准确率
    for epoch in range(config['training']['epochs']):  # 遍历所有 epoch
        print(f"Epoch {epoch+1} 开始")
        # 训练阶段
        model.train()  # 将模型设置为训练模式
        logger.new_epoch()  # 开始新的 epoch

        for i, (images, labels) in enumerate(train_loader):  # 遍历训练集的数据加载器
            images, labels = images.to(device), labels.to(device)  # 将图像和标签移动到指定的设备上

            if use_cuda_amp:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels.long())
            else:
                outputs = model(images)
                loss = criterion(outputs, labels.long())

            if use_cuda_amp:
                scaler.scale(loss).backward()  # 反向传播损失
                scaler.step(optimizer)  # 更新优化器
                scaler.update()  # 更新 GradScaler
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()  # 清空梯度

            # 记录指标
            logger.log_train(loss.item(), outputs, labels)  # 记录训练损失、输出和标签
            if (i+1) % 10 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} 完成")

        # 验证阶段
        model.eval()  # 将模型设置为评估模式
        with torch.no_grad():  # 禁用梯度计算
            for images, labels in val_loader:  # 遍历验证集的数据加载器
                images, labels = images.to(device), labels.to(device)  # 将图像和标签移动到指定的设备上

                if use_cuda_amp:
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels.long())
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels.long())

                # 记录指标
                logger.log_val(loss.item(), outputs, labels)  # 记录验证损失、输出和标签

        # 10. 创建 checkpoints 目录（如果不存在）
        checkpoint_dir = "checkpoints"  # 指定 checkpoints 目录
        os.makedirs(checkpoint_dir, exist_ok=True)  # 创建目录，如果目录已存在则不报错

        # 11. 保存最佳模型
        logger.finalize_epoch()
        if logger.val_metrics['accuracy'] > best_acc:  # 如果当前验证集准确率大于最佳验证集准确率
            best_acc = logger.val_metrics['accuracy']  # 更新最佳验证集准确率
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_{config['model']['name']}.pth"))  # 保存模型的状态字典

        # 生成报告
        print(f"Epoch {epoch+1} 完成，验证集准确率: {logger.val_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()  # 如果是主程序，则执行 main 函数

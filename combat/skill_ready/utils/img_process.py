import torch

def unnormalize(tensor, device):
    """
    反归一化：将 tensor 中每个通道的数据从归一化状态恢复到[0,1]范围。
    tensor: shape (C, H, W)
    mean, std: list或tensor，每个通道的均值和标准差
    """
    # 这里我们假设 tensor 已经是一个单独的图像
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

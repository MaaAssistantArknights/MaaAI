import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class SkillIconDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        instances = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(class_dir, filename)
                    instances.append((path, self.class_to_idx[class_name]))
        return instances

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def create_loaders(config, train_transform, val_transform):
    # 创建完整数据集
    full_dataset = SkillIconDataset(
        root_dir=config['data']['dataset_path'],
        transform=train_transform
    )

    # 按配置文件比例拆分数据集
    split_ratio = config['data']['split_ratio']
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 设置验证集转换
    val_dataset.dataset.transform = val_transform

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config['training']['batch_size']) * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

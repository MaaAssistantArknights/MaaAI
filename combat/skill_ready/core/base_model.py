import os
import torch
import torch.nn as nn
from torchvision import transforms
from abc import ABC, abstractmethod
from core.data_loader import get_class_counts

class BaseModel(ABC, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.class_names = ['c', 'n', 'y']
        self.setup_transforms()
        self.setup_class_weights()

    @abstractmethod
    def build_model(self):
        pass

    def setup_class_weights(self):
        # 动态计算类别权重
        dataset_path = self.config['data']['dataset_path']
        class_counts = get_class_counts(dataset_path, self.class_names).clamp_min(1.0)
        self.class_counts = class_counts
        sqrt_counts = torch.sqrt(class_counts)
        self.class_weights = class_counts.sum() / (sqrt_counts.sum() * sqrt_counts)

    def setup_transforms(self):
        # 训练集增强流程
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.config['data']['input_size'],
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.12, 0.12),
                scale=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3)
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 验证集转换流程
        self.val_transform = transforms.Compose([
            transforms.Resize(
                (self.config['data']['input_size'], self.config['data']['input_size']),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

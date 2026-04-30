import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from abc import ABC, abstractmethod
from core.data_loader import get_class_counts


class MaaRuntimeValTransform:
    def __init__(self, input_size, resize_size, mean, std):
        self.input_size = int(input_size)
        self.resize_size = int(resize_size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image):
        image_array = np.asarray(image, dtype=np.uint8)
        resized = cv2.resize(
            image_array,
            (self.resize_size, self.resize_size),
            interpolation=cv2.INTER_CUBIC
        )

        crop_start = max(0, (self.resize_size - self.input_size) // 2)
        crop_end = crop_start + self.input_size
        cropped = resized[crop_start:crop_end, crop_start:crop_end]

        tensor = torch.from_numpy(cropped).permute(2, 0, 1).float().div(255.0)
        return (tensor - self.mean) / self.std

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

    def _resolve_square_input_size(self):
        input_size = self.config['data']['input_size']
        if isinstance(input_size, int):
            return input_size
        if isinstance(input_size, (list, tuple)):
            if len(input_size) == 1:
                return int(input_size[0])
            if len(input_size) == 2 and int(input_size[0]) == int(input_size[1]):
                return int(input_size[0])
        raise ValueError(f"仅支持方形 input_size，当前配置为: {input_size}")

    def setup_transforms(self):
        input_size = self._resolve_square_input_size()
        runtime_resize_size = int(round(input_size * 72 / 64))

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

        # 与 Maa 实跑对齐：使用 OpenCV INTER_CUBIC 先放大到 72，再中心裁到 64。
        self.val_transform = MaaRuntimeValTransform(
            input_size=input_size,
            resize_size=runtime_resize_size,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

import os
import torch
import torch.nn as nn
from torchvision import transforms
from abc import ABC, abstractmethod

class BaseModel(ABC, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_config = config['model']
        self.data_config = config['data']
        self.class_names = ['c', 'n', 'y']
        self.setup_transforms()
        self.setup_class_weights()

    @abstractmethod
    def build_model(self):
        pass

    def setup_class_weights(self):
        # 动态计算类别权重
        dataset_path = self.data_config['dataset_path']
        class_names = os.listdir(dataset_path)
        class_counts = []
        for class_name in class_names:
            class_dir = os.path.join(dataset_path, class_name)
            class_counts.append(len(os.listdir(class_dir)))
        class_counts = torch.tensor(class_counts).float()
        self.class_weights = 1.0 / class_counts
        self.class_weights /= self.class_weights.sum()

    def setup_transforms(self):
        # 训练集增强流程
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.data_config['input_size'],
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # 验证集转换流程
        self.val_transform = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(self.data_config['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def export_onnx(self, model_path="base_model.onnx"):
        pass

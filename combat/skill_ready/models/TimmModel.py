import timm
import torch
from core.base_model import BaseModel

class TimmModelWrapper(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build_model()
        
    def build_model(self):
        # 使用timm库的官方实现
        self.model = timm.create_model(
            self.config['model']['name'],
            pretrained=self.config['model']['pretrained'],
            num_classes=self.config['model']['num_classes'],
            in_chans=3,
            drop_path_rate=self.config['model']['drop_path_rate'],
        )

    def forward(self, x):
        return self.model(x)

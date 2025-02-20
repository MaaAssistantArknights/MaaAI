import timm
import torch
from core.base_model import BaseModel

class ShuffleNetV2Wrapper(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build_model()
        
    def build_model(self):
        # 使用timm库的官方实现
        self.model = timm.create_model(
            'shufflenetv2_x0_5',  # timm中的ShuffleNetV2模型
            pretrained=self.config.pretrained,
            num_classes=self.config.num_classes,
            in_chans=3,
            drop_path_rate=self.config.drop_path_rate,
        )
        
        # 修改分类头适应小数据集
        in_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features, self.config.num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def export_onnx(self, output_path="shufflenetv2.onnx"):
        dummy_input = torch.randn(1, 3, 64, 64)
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}},
        )

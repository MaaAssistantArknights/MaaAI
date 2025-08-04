import timm
import torch
from core.base_model import BaseModel

class CustomModelWrapper(BaseModel):
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

        # 修改分类头适应数据集
        in_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features, self.config['model']['num_classes'])
        )

    def forward(self, x):
        return self.model(x)

    def export_onnx(self, output_path, checkpoint_path, load_checkpoint=True):
        """
        如果 load_checkpoint 为 True，则加载 checkpoint，否则认为模型已经加载好权重
        """
        if load_checkpoint:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model.eval()
        
        input_size = self.config['data']['input_size']
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        dummy_input = torch.randn(1, 3, *input_size)
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

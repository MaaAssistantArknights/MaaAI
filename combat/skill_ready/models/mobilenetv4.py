import timm
import torch
from core.base_model import BaseModel

class MobileNetV4Wrapper(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build_model()
        
    def build_model(self):
        # 使用timm库的官方实现
        self.model = timm.create_model(
            'mobilenetv4_conv_small',  # timm中的最新实现
            pretrained=self.model_config['pretrained'],
            num_classes=self.model_config['num_classes'],
            in_chans=3,
            drop_path_rate=self.model_config['drop_path_rate'],
        )

        # 修改分类头适应小数据集
        in_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features, self.model_config['num_classes'])
        )

    def forward(self, x):
        return self.model(x)

    def export_onnx(self, output_path="mobilenetv4.onnx", load_checkpoint=True, checkpoint_path="checkpoints/best_mobilenetv4.pth"):
        """
        如果 load_checkpoint 为 True，则加载 checkpoint，否则认为模型已经加载好权重，导出一定要加载好权重啊。。。。
        """
        if load_checkpoint:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model.eval()
        
        input_size = self.data_config['input_size']
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

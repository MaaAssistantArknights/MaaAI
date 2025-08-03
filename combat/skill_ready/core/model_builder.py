from models.TimmModel import TimmModelWrapper
from models.CustomModel import CustomModelWrapper

def create_model(config):
    model_type = config['model']['type']  # 默认timm类型

    if model_type == 'timm':
        return TimmModelWrapper(config)
    elif model_type == 'custom':
        return CustomModelWrapper(config)  # 预留自定义模型接口
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

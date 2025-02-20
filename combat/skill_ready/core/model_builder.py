from models.mobilenetv4 import MobileNetV4Wrapper
from models.shufflenetv2 import ShuffleNetV2Wrapper

from models.mobilenetv4 import MobileNetV4Wrapper
from models.shufflenetv2 import ShuffleNetV2Wrapper

def create_model(config):
    model_name = config['model']['name']
    if model_name == 'mobilenetv4':
        return MobileNetV4Wrapper(config)
    elif model_name == 'shufflenetv2':
        return ShuffleNetV2Wrapper(config)
    else:
        raise ValueError(f"Unsupported model name: {config['model']['name']}")

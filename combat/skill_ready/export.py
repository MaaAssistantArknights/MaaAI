import argparse
import os 
import yaml
import torch
from core.model_builder import create_model
from utils.path_manager import get_model_paths

def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX format')
    parser.add_argument('--config', help='指定模型配置文件',
                        default='configs/mobilenetv4_conv_small.yaml', type=str)
    args = parser.parse_args()
    return args

def main():
    # 1. 加载配置文件
    args = parse_args()
    try:
        with open(args.config, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置文件加载完成")
    except FileNotFoundError:
        print(f"错误：配置文件 {args.config} 不存在")
        return

    # 2. 初始化设备
    device = torch.device("cpu")
    print(f"使用设备: {device}")

    # 3. 获取目标路径
    paths = get_model_paths(config)

    # 4. 初始化模型
    try:
        model = create_model(config).to(device)
        checkpoint_path = paths['checkpoint_export']
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件 {checkpoint_path} 不存在")
        # 加载模型 
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"已加载最佳模型权重: {checkpoint_path}")
    except RuntimeError as e:
        print(f"模型加载失败: {str(e)}")
        return

    try:
        input_size = config['data']['input_size']
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        # 6. 创建虚拟输入
        dummy_input = torch.randn(1, 3, *input_size)
        torch.onnx.export(
            model,
            dummy_input,
            paths['onnx_export'],
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}},
        )
        print(f"模型已导出到 {paths['onnx_export']}")
    except Exception as e:
        print(f"导出 ONNX 模型失败: {str(e)}")

if __name__ == "__main__":
    main()

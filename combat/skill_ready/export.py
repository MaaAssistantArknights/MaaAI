import os 
import yaml
import torch
from core.model_builder import create_model

def export_model(output_path="checkpoints/mobilenetv4.onnx"):
    # 1. 加载配置文件
    config_path = os.environ.get('CONFIG_PATH', "configs/mobilenetv4.yaml")
    try:
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ 配置文件加载完成")
    except FileNotFoundError:
        print(f"❌ 错误：配置文件 {config_path} 不存在")
        return

    # 2. 初始化设备
    device = torch.device("cpu")
    print(f"⚙️ 使用设备: {device}")

    # 3. 初始化模型
    try:
        model = create_model(config).to(device)
        checkpoint_dir = "checkpoints"
        checkpoint_path = os.path.join(checkpoint_dir, f"best_{config['model']['name']}.pth")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件 {checkpoint_path} 不存在")
            
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"🔧 已加载最佳模型权重: {checkpoint_path}")
    except RuntimeError as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return

    try:
        # 调用 export_onnx 时，不再重复加载权重
        model.export_onnx(output_path, load_checkpoint=False)
        print(f"模型已导出到 {output_path}")
    except Exception as e:
        print(f"❌ 导出 ONNX 模型失败: {str(e)}")

if __name__ == "__main__":
    export_model()

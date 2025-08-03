import datetime
import os

def get_model_paths(config):
    """生成所有相关路径"""
    base_name = config['model']['name']
    checkpoints_dir = f"checkpoints/{base_name}"
    onnx_dir = f"onnx/{base_name}"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    
    return {
        'checkpoints_dir': checkpoints_dir,
        'checkpoint_export': f"checkpoints/{base_name}/skill_ready_cls.pth",
        'onnx_dir': onnx_dir,
        'onnx_export': f"onnx/{base_name}/skill_ready_cls.onnx",
    }

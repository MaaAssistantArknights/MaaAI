"""Shared v5/v6 export entry. One local export-only decomposition for Paddle 3.4.
Paddle2ONNX 2.1.0 has no linear_v2 mapper; preserve x @ weight + bias explicitly.
No installed package or training source is modified.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np,paddle,paddle2onnx,yaml
from workspace import RT
sys.path.insert(0,str(RT/'vendor/PaddleOCR'))

def export_linear(x,weight,bias=None,name=None):
    """在导出时保持线性层参数语义，避免转换改变输出。"""
    y=paddle.matmul(x,weight)
    if bias is not None:y=paddle.add(y,bias)
    return y

def main():
 """从指定模型配置和检查点生成 Paddle 导出中间产物。"""
 p=argparse.ArgumentParser();p.add_argument('--no-rep',action='store_true');p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('choose a new model export directory')
 if paddle.__version__!='3.4.0' or paddle2onnx.__version__!='2.1.0':raise RuntimeError('Revalidate this compatibility decomposition for changed versions')
 paddle.set_device('cpu');paddle.seed(20260917)
 reference=paddle.nn.functional.linear;diffs=[]
 for shape in [[7,120],[2,40,120],[1,1,384]]:
  x=paddle.randn(shape);w=paddle.randn([shape[-1],96]);b=paddle.randn([96]);r=reference(x,w,b).numpy();s=export_linear(x,w,b).numpy();np.testing.assert_allclose(s,r,atol=1e-4,rtol=1e-3);diffs.append(float(abs(r-s).max()))
 # Patch only this short-lived export process. Component equivalence is a
 # prerequisite; the exported full model must still pass the downstream gate.
 paddle.nn.functional.linear=export_linear
 from ppocr.utils import export_model as exporter
 if a.no_rep:
  original=exporter.dynamic_to_static
  def preserve_training_structure(model,*args,**kwargs):
   """在导出阶段维持训练结构所需的临时模型设置。"""
   changed=[]
   for layer in model.sublayers():
    if hasattr(layer,'rep'):
     changed.append(layer);layer.rep=lambda *a,**kw:None
   try:return original(model,*args,**kwargs)
   finally:
    for layer in changed:delattr(layer,'rep')
  exporter.dynamic_to_static=preserve_training_structure
 export=exporter.export
 cfg=yaml.safe_load(a.config.read_text());cfg['Global'].update({'pretrained_model':str(a.checkpoint),'save_inference_dir':str(a.output),'use_gpu':False,'export_with_pir':True})
 export(cfg)
 (a.output/'compatibility.json').write_text(json.dumps({'paddle':paddle.__version__,'paddle2onnx':paddle2onnx.__version__,'reason':'linear_v2 is absent in pinned converter mapper registry','change':'export-process-only F.linear decomposition to MatMul + Add','component_max_abs_differences':diffs,'training_package_modified':False,'reparameterization_disabled':a.no_rep,'old_ir_attempt':'unsupported multiply Variable/Value mixing; not used'},indent=2)+'\n')
if __name__=='__main__':main()

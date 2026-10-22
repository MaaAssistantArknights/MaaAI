"""Offline, portable BASIC folding only; never serialize NCHWc/extended CPU kernels."""
import argparse,hashlib,json
from pathlib import Path
import onnx
import onnxruntime as ort

def main():
 """仅执行可移植的基础 ONNX 折叠并记录输出身份。"""
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 if a.output.exists() or a.source.resolve()==a.output.resolve():raise ValueError('preserve source and choose a new output')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 options=ort.SessionOptions();options.intra_op_num_threads=3;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC;options.optimized_model_filepath=str(a.output)
 try:
  ort.InferenceSession(str(a.source),sess_options=options,providers=['CPUExecutionProvider'])
  model=onnx.load(a.output)
  if any(n.domain not in {'','ai.onnx'} for n in model.graph.node):raise ValueError('nonstandard backend operator in portable graph')
  onnx.checker.check_model(model)
 except BaseException:
  a.output.unlink(missing_ok=True);raise
 report={'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'optimizer':ort.__version__,'level':'ORT_ENABLE_BASIC','nodes':len(model.graph.node),'standard_onnx_only':True,'status':'requires full equivalence, released-runtime quality and latency recheck'}
 a.output.with_name('offline-optimization.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()

"""Opt-in standard ONNX 1x1 Conv -> 2-D MatMul export for released CPU ORT.

Avoid NCHWc accumulation on pointwise convolutions. Flatten NHW explicitly:
ORT 1.19.2 MatMulBnFusion incorrectly assumes rank 2 for a 4-D MatMul path.
This is algebraically equivalent, not a promise of identical floating arithmetic;
run the frozen equivalence/latency gates on every exported candidate.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper

def transform(model):
 """把可识别的逐点计算改写为标准 ONNX 算子组合。"""
 m=copy.deepcopy(model);weights={x.name:numpy_helper.to_array(x) for x in m.graph.initializer}
 for node in m.graph.node:
  if node.op_type=='Constant':
   value=next((a for a in node.attribute if a.name=='value'),None)
   if value is not None:weights[node.output[0]]=numpy_helper.to_array(value.t)
  elif node.op_type=='Identity' and node.input[0] in weights:weights[node.output[0]]=weights[node.input[0]]
 nodes=[];changed=[];existing={x for n in m.graph.node for x in list(n.input)+list(n.output)}
 for i,node in enumerate(m.graph.node):
  attrs={a.name:helper.get_attribute_value(a) for a in node.attribute};w=weights.get(node.input[1]) if node.op_type=='Conv' else None
  eligible=(node.domain=='' and w is not None and w.ndim==4 and tuple(w.shape[2:])==(1,1) and attrs.get('group',1)==1 and attrs.get('strides',[1,1])==[1,1] and attrs.get('dilations',[1,1])==[1,1] and not any(attrs.get('pads',[0,0,0,0])) and attrs.get('auto_pad',b'NOTSET') in [b'NOTSET',b'VALID'])
  if not eligible:nodes.append(node);continue
  # Flatten spatial positions before MatMul, then restore dynamic N/H/W.
  # A rank-4 MatMul is mathematically valid but triggers the pinned ORT fusion bug.
  prefix=f'ocr_stable_pointwise_{i}'
  if any(x.startswith(prefix) for x in existing):raise ValueError('generated ONNX name collision')
  for suffix,array in [('weight',w[:,:,0,0].T.copy()),('flatshape',np.array([-1,w.shape[1]],dtype='int64')),('start',np.array([0],dtype='int64')),('end',np.array([3],dtype='int64')),('outch',np.array([w.shape[0]],dtype='int64'))]:m.graph.initializer.append(numpy_helper.from_array(array,name=prefix+'_'+suffix))
  def op(kind,inputs,output,**kw):
      """为当前逐点子图生成带唯一前缀的 ONNX 节点。"""
      return helper.make_node(kind,[prefix+'_'+s for s in inputs],[prefix+'_'+output],**kw)
  nodes.extend([helper.make_node('Transpose',[node.input[0]],[prefix+'_nhwc'],perm=[0,2,3,1]),op('Shape',['nhwc'],'shape'),op('Slice',['shape','start','end'],'nhw'),op('Concat',['nhw','outch'],'outshape',axis=0),op('Reshape',['nhwc','flatshape'],'flat'),op('MatMul',['flat','weight'],'mm'),op('Reshape',['mm','outshape'],'out')]);value=prefix+'_out'
  if len(node.input)>2:
   nodes.append(helper.make_node('Add',[value,node.input[2]],[prefix+'_bias']));value=prefix+'_bias'
  nodes.append(helper.make_node('Transpose',[value],list(node.output),perm=[0,3,1,2]));changed.append(node.name or node.output[0])
 # Remove dead Constant/Identity nodes and weight initializers introduced by replacement.
 needed={o.name for o in m.graph.output};kept=[]
 for node in reversed(nodes):
  if any(name in needed for name in node.output):kept.append(node);needed.update(node.input)
 del m.graph.node[:];m.graph.node.extend(reversed(kept));keep=[x for x in m.graph.initializer if x.name in needed];del m.graph.initializer[:];m.graph.initializer.extend(keep)
 onnx.checker.check_model(m);return m,changed

def main():
 """读取导出模型并写出经过受限改写的新模型。"""
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 if a.source.resolve()==a.output.resolve() or a.output.exists():raise ValueError('preserve input; choose a new output file')
 m,changed=transform(onnx.load(a.source))
 if not changed:raise ValueError('no eligible pointwise convolutions')
 a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,a.output)
 report={'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'rewritten_count':len(changed),'nodes':changed,'status':'must pass equivalence and released-runtime latency gates'}
 a.output.with_name('pointwise-transform.json').write_text(json.dumps(report,indent=2)+'\n');print('rewritten pointwise convolutions',len(changed))
if __name__=='__main__':main()

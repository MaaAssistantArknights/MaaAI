from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import helper as h,numpy_helper,TensorProto as T
import onnxruntime as ort
import pytest

from stabilize_pointwise import transform

def make_model(kernel=1,group=1,stride=1,constant=False):
 rng=np.random.default_rng(7);w=rng.normal(size=(8,4//group,kernel,kernel)).astype('float32');b=rng.normal(size=(8,)).astype('float32');wi=numpy_helper.from_array(w,name='w');bi=numpy_helper.from_array(b,name='b');nodes=[];initializers=[wi,bi]
 if constant:nodes=[h.make_node('Constant',[],['w0'],value=wi),h.make_node('Identity',['w0'],['w'])];initializers=[bi]
 nodes.append(h.make_node('Conv',['x','w','b'],['y'],kernel_shape=[kernel,kernel],group=group,strides=[stride,stride]))
 m=h.make_model(h.make_graph(nodes,'conv',[h.make_tensor_value_info('x',T.FLOAT,['N',4,'H','W'])],[h.make_tensor_value_info('y',T.FLOAT,['N',8,'YH','YW'])],initializer=initializers),opset_imports=[h.make_opsetid('',17)]);m.ir_version=8;return m

@pytest.mark.parametrize('constant',[False,True])
def test_dynamic_pointwise_preserves_values(constant):
 m=make_model(constant=constant);converted,changed=transform(m);assert len(changed)==1
 assert sum(n.op_type=='Conv' for n in m.graph.node)==1
 options=ort.SessionOptions();options.intra_op_num_threads=1
 original=ort.InferenceSession(m.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider']);actual=ort.InferenceSession(converted.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
 for shape in [(1,4,3,5),(2,4,7,11),(1,4,1,1)]:
  x=np.random.default_rng(8).normal(size=shape).astype('float32');np.testing.assert_allclose(actual.run(None,{'x':x})[0],original.run(None,{'x':x})[0],atol=1e-5,rtol=1e-5)

@pytest.mark.parametrize('kwargs',[{'kernel':3},{'group':2},{'stride':2}])
def test_unsupported_convolution_is_not_rewritten(kwargs):
 m=make_model(**kwargs);converted,changed=transform(m);assert not changed;assert any(n.op_type=='Conv' for n in converted.graph.node)

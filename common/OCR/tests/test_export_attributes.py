import copy
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper,numpy_helper,TensorProto
from repair_onnx_attributes import repair


def fixture(epsilon=1e-5):
    pir={'ops':[{'#':'p','A':[0,1,1,'scale'],'O':{'%':1}},{'#':'p','A':[0,1,1,'bias'],'O':{'%':2}},{'#':'1.layer_norm','A':[{'N':'epsilon','AT':{'#':'0.a_f64','D':epsilon}},{'N':'begin_norm_axis','AT':{'#':'0.a_i32','D':2}}],'I':[{'%':0},{'%':1},{'%':2}]}]}
    node=helper.make_node('LayerNormalization',['x','scale','bias'],['y'],axis=2,epsilon=-3e33,name='norm')
    graph=helper.make_graph([node],'test',[helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,2,4])],[helper.make_tensor_value_info('y',TensorProto.FLOAT,[1,2,4])],[numpy_helper.from_array(np.ones(4,dtype='float32'),'scale'),numpy_helper.from_array(np.zeros(4,dtype='float32'),'bias')])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);model.ir_version=10
    return model,pir


@pytest.mark.parametrize('epsilon',[1e-5,1e-6])
def test_source_epsilon_recovers_finite_near_constant_normalization(epsilon):
    original,pir=fixture(epsilon);model,changes=repair(original,pir)
    assert next(a.f for a in original.graph.node[0].attribute if a.name=='epsilon')<0
    opts=ort.SessionOptions();opts.intra_op_num_threads=1
    session=ort.InferenceSession(model.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider'])
    x=np.array([[[0,.001,.002,.003],[1,1,1,1]]],dtype='float32')
    expected=(x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+epsilon)
    actual=session.run(None,{'x':x})[0]
    assert np.isfinite(actual).all();np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3)
    assert changes[0]['source_epsilon']==epsilon


def test_mismatched_parameter_mapping_and_unknown_scalar_are_rejected():
    model,pir=fixture();model.graph.node[0].input[1]='wrong-scale'
    with pytest.raises(ValueError,match='unmatched'):repair(model,pir)
    model,pir=fixture();pir['ops'].append({'#':'1.unreviewed','A':[{'N':'value','AT':{'#':'0.a_f64','D':.1}}]})
    with pytest.raises(ValueError,match='unreviewed'):repair(model,pir)


def test_axis_mismatch_cannot_be_silently_patched():
    model,pir=fixture();next(a for a in model.graph.node[0].attribute if a.name=='axis').i=1
    with pytest.raises(ValueError,match='axis differs'):repair(model,pir)

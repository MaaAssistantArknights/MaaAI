"""Restore LayerNorm attributes from PIR, with exact parameter-name mapping.

Paddle 3.4 serializes LayerNorm epsilon as f64. The pinned Paddle2ONNX
float attribute reader marks f64 as found without assigning its value.
Never infer epsilon from a desired prediction or use a single default.
"""
import argparse,copy,json,math
from pathlib import Path
import onnx
from file_utils import sha

def operations(value):
    """解析 PIR 节点参数并保留导出所需的属性身份。"""
    if isinstance(value,dict):
        if '#' in value and 'A' in value:yield value
        for child in value.values():yield from operations(child)
    elif isinstance(value,list):
        for child in value:yield from operations(child)

def source_norms(pir):
    """从源 PIR 中定位每个归一化算子的原始属性。"""
    ops=list(operations(pir));parameters={};norms={}
    for op in ops:
        if op['#']=='p':
            if not isinstance(op['A'][-1],str):raise ValueError('unknown PIR parameter layout')
            parameters[op['O']['%']]=op['A'][-1]
        for attr in op['A']:
            if isinstance(attr,dict) and attr.get('AT',{}).get('#')=='0.a_f64':
                if (op['#'],attr['N']) not in [('1.layer_norm','epsilon'),('1.full','value')]:raise ValueError('unreviewed f64 attribute: '+str((op['#'],attr['N'])))
    for op in ops:
        if op['#']!='1.layer_norm':continue
        attrs={a['N']:a['AT']['D'] for a in op['A']};epsilon=float(attrs['epsilon'])
        if not math.isfinite(epsilon) or epsilon<=0:raise ValueError('invalid source epsilon')
        key=tuple(parameters[x['%']] for x in op['I'][1:3])
        if key in norms:raise ValueError('ambiguous LayerNorm parameter mapping')
        norms[key]={'epsilon':epsilon,'axis':int(attrs['begin_norm_axis'])}
    if not norms:raise ValueError('no source LayerNorm; compatibility review required')
    return norms

def repair(model,pir):
    """按源节点身份修复 ONNX 属性，拒绝无法匹配的算子。"""
    result=copy.deepcopy(model);norms=source_norms(pir);matched=set();changes=[]
    for node in result.graph.node:
        if node.op_type!='LayerNormalization':continue
        # Match by source scale/bias identity, not graph order or a global epsilon.
        # Unmapped layers must fail rather than inherit a guessed normalization.
        key=tuple(node.input[1:3])
        if key not in norms or key in matched:raise ValueError('unmatched/duplicated ONNX LayerNorm')
        matched.add(key);expected=norms[key];attrs={a.name:a for a in node.attribute}
        if 'axis' not in attrs or attrs['axis'].i!=expected['axis']:raise ValueError('LayerNorm axis differs from PIR')
        before=float(attrs['epsilon'].f) if 'epsilon' in attrs else None
        if before is not None and not math.isfinite(before):before=repr(before)
        if 'epsilon' in attrs:node.attribute.remove(attrs['epsilon'])
        node.attribute.append(onnx.helper.make_attribute('epsilon',expected['epsilon']))
        changes.append({'node':node.name,'weight':key[0],'bias':key[1],'axis':expected['axis'],'before':before,'source_epsilon':expected['epsilon'],'after':next(a.f for a in node.attribute if a.name=='epsilon')})
    if matched!=set(norms):raise ValueError('not all PIR LayerNorm nodes mapped')
    onnx.checker.check_model(result)
    return result,changes

def main():
    """读取 PIR 与 ONNX，写出修复模型及逐项报告。"""
    p=argparse.ArgumentParser();p.add_argument('--pir',required=True,type=Path);p.add_argument('--onnx',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--report',required=True,type=Path);a=p.parse_args()
    if a.output.exists() or a.onnx.resolve()==a.output.resolve():raise ValueError('preserve original conversion output')
    model,changes=repair(onnx.load(str(a.onnx)),json.loads(a.pir.read_text()));onnx.save(model,str(a.output))
    a.report.write_text(json.dumps({'version':1,'source_pir_sha256':sha(a.pir),'converter_output_sha256':sha(a.onnx),'repaired_onnx_sha256':sha(a.output),'mapping':'PIR scale/bias parameter identities; axis checked; epsilon copied from source','changes':changes,'numeric_equivalence_still_required':True},indent=2,allow_nan=False)+'\n');print('mapped LayerNorm source attributes',len(changes))
if __name__=='__main__':main()

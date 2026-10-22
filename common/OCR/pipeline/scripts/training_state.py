"""Atomic, bounded training recovery state. Only trusted local checkpoints are loaded."""
import json,os,random,shutil,uuid
from pathlib import Path
import numpy as np
import paddle
from file_utils import sha
from file_utils import atomic_json

def save_state(output,model,optimizer,step,position,order,rng,history,identity,elapsed,keep=2):
    """原子保存模型、优化器、随机数和采样位置供恢复。"""
    parent=output/'recovery';parent.mkdir(exist_ok=True)
    final=parent/f'step-{step:07d}'
    if final.exists():raise ValueError('checkpoint already committed')
    temp=parent/('.pending-'+uuid.uuid4().hex);temp.mkdir()
    paddle.save(model.state_dict(),str(temp/'model.pdparams'))
    paddle.save(optimizer.state_dict(),str(temp/'optimizer.pdopt'))
    paddle.save({'python':random.getstate(),'numpy':np.random.get_state(),'generator':rng.bit_generator.state,'paddle_cpu':paddle.get_rng_state('cpu'),'paddle_gpu':paddle.get_cuda_rng_state(),'order':order,'history':history},str(temp/'control.pdstate'))
    meta={'version':1,'step':step,'position':position,'identity':identity,'elapsed':elapsed,'files':{p.name:sha(p) for p in temp.iterdir()},'learning_rate_schedule':'constant; bound by run identity'}
    # Commit the complete model/optimizer/random-state directory before publishing
    # the resume pointer. An interrupted save must not replace the last valid pointer.
    atomic_json(temp/'checkpoint.json',meta);os.replace(temp,final)
    atomic_json(output/'resume.json',{'checkpoint':str(final.resolve()),'metadata_sha256':sha(final/'checkpoint.json')})
    # Never touch candidate exports or the separate selected model.
    for old in sorted(parent.glob('step-*'))[:-keep]:shutil.rmtree(old)
    return final

def inspect_state(output,identity):
    """查找与当前冻结身份匹配的可恢复检查点。"""
    pointer=json.loads((output/'resume.json').read_text());folder=Path(pointer['checkpoint'])
    if not folder.resolve().is_relative_to((output/'recovery').resolve()):raise ValueError('checkpoint outside run')
    if sha(folder/'checkpoint.json')!=pointer['metadata_sha256']:raise ValueError('checkpoint metadata changed')
    meta=json.loads((folder/'checkpoint.json').read_text())
    if meta['identity']!=identity:raise ValueError('resume run identity differs')
    if any(sha(folder/name)!=h for name,h in meta['files'].items()):raise ValueError('checkpoint file changed')
    return folder,meta

def restore_state(folder,model,optimizer,rng):
    """把检查点状态恢复到模型、优化器和随机数发生器。"""
    # Load after the caller selects the target device, including non-contiguous parameters.
    weights=paddle.load(str(folder/'model.pdparams'));model.set_state_dict(weights)
    optimizer_state=paddle.load(str(folder/'optimizer.pdopt'));optimizer.set_state_dict(optimizer_state)
    def same_tree(expected,actual):
        """逐层比较恢复状态的键和张量结构。"""
        if isinstance(expected,dict):return set(expected)==set(actual) and all(same_tree(v,actual[k]) for k,v in expected.items())
        if isinstance(expected,(list,tuple)):return len(expected)==len(actual) and all(same_tree(x,y) for x,y in zip(expected,actual))
        if isinstance(expected,paddle.Tensor):expected=expected.numpy()
        if isinstance(actual,paddle.Tensor):actual=actual.numpy()
        return np.array_equal(expected,actual)
    if not same_tree(weights,model.state_dict()) or not same_tree(optimizer_state,optimizer.state_dict()):raise ValueError('loaded recovery tensors differ from saved state')
    state=paddle.load(str(folder/'control.pdstate'),return_numpy=True)
    random.setstate(state['python']);np.random.set_state(state['numpy']);rng.bit_generator.state=state['generator']
    paddle.set_rng_state(state['paddle_cpu'],'cpu');paddle.set_cuda_rng_state(state['paddle_gpu'])
    state['restoration_validation']={'all_model_tensors_exact':True,'all_optimizer_tensors_exact':True,'model_entries':len(weights),'optimizer_entries':len(optimizer_state),'random_policy':'full state restored; planned stochastic layers use step-derived seeds'}
    return state

def validate_plan(args,plan):
    """训练前确认冻结计划的文件哈希和质量门槛仍有效。"""
    if plan['purpose'] not in ['readiness_probe','data_scale_comparison','formal_training']:raise ValueError('unknown run purpose')
    if plan['purpose']=='formal_training' and (not plan.get('actual_maa_evaluation') or not plan.get('stop_on_gate_failure')):raise ValueError('formal training requires actual MAA evaluation and stop gates')
    route=args.language or 'char'
    if route!=plan['route']:raise ValueError('route not admitted')
    for key in ['steps','batch','learning_rate','eval_every','sampling','input_policy','seed']:
        if getattr(args,key)!=plan[key]:raise ValueError('run parameter differs: '+key)
    if not 1<=args.steps<=plan['max_steps']<=5000:raise ValueError('explicit finite budget required')
    if plan.get('random_policy')!='step_seed_v1':raise ValueError('unknown stochastic-layer policy')
    if not 1<=args.eval_every<=args.steps:raise ValueError('invalid evaluation interval')
    if args.sampling is None:raise ValueError('planned training requires an explicit sampling policy')
    if not 1<=args.batch<=32:raise ValueError('batch budget exceeded')
    if not plan.get('allow_training',False):raise ValueError('plan is not enabled for execution')
    if plan['input_policy']!=('fixed' if route=='char' else 'released'):raise ValueError('unverified training input')
    if args.dataset.resolve()!=Path(plan['dataset']).resolve():raise ValueError('dataset not admitted')
    # Freeze source bytes too, including comments. Resume from the recorded code
    # revision; create a new plan for edited code rather than rewriting old hashes.
    for name,h in plan['files'].items():
        if sha(Path(name))!=h:raise ValueError('run dependency changed: '+name)
    if sha(args.dataset/'manifest.jsonl')!=plan['dataset_manifest_sha256']:raise ValueError('training manifest changed')
    if sha(args.training_profile)!=plan['profile_sha256']:raise ValueError('training profile changed')

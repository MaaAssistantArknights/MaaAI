import random
from pathlib import Path
import numpy as np
import paddle
import pytest
from training_state import save_state,inspect_state,restore_state
from checkpoint_select import assess

def setup():
    paddle.set_device('cpu');paddle.seed(33);random.seed(33);np.random.seed(33)
    model=paddle.nn.Sequential(paddle.nn.Linear(3,5),paddle.nn.Dropout(.2),paddle.nn.Linear(5,1));opt=paddle.optimizer.Adam(learning_rate=.003,parameters=model.parameters());return model,opt,np.random.default_rng(33)

def step(model,opt):
    x=paddle.randn([4,3]);loss=(model(x)**2).mean();loss.backward();opt.step();opt.clear_grad();return float(loss)

def test_recovery_restores_optimizer_and_random_streams(tmp_path):
    # Separate guards model independent processes, including lazy optimizer names.
    with paddle.utils.unique_name.guard():
        m,o,r=setup();step(m,o);order=np.arange(8)
        folder=save_state(tmp_path,m,o,1,4,order,r,[],{'test':1},0)
        expected=step(m,o);weights={k:v.numpy().copy() for k,v in m.state_dict().items()};rngs=(random.random(),np.random.rand(),r.random())
    with paddle.utils.unique_name.guard():
        m,o,r=setup();f,meta=inspect_state(tmp_path,{'test':1});restore_state(f,m,o,r)
        assert step(m,o)==expected
        for k,v in m.state_dict().items():np.testing.assert_array_equal(v.numpy(),weights[k])
        assert (random.random(),np.random.rand(),r.random())==rngs
    with pytest.raises(ValueError,match='identity'):inspect_state(tmp_path,{'test':2})
    with (folder/'optimizer.pdopt').open('ab') as handle:handle.write(b'corrupt')
    with pytest.raises(ValueError,match='file changed'):inspect_state(tmp_path,{'test':1})

def test_task_regression_cannot_be_hidden_by_raw_total():
    rows=[{'id':i,'expected_output':s,'label':s,'subset':'new_recipe','scene':'item_name','evidence_kind':'synthetic'} for i,s in [('rules/a','A'),('raw/a','A'),('raw/b','B')]]
    def predictions(values):return [{'id':r['id'],'results':[{'text':v}]} for r,v in zip(rows,values)]
    prior=predictions(['A','X','Y']);candidate=predictions(['X','A','B'])
    result=assess(rows,candidate,prior,prior)
    assert result['totals']['raw']==2
    assert not result['quality_eligible']
    assert result['new_rule_errors_vs_current']==['rules/a']


def test_training_lock_prevents_competing_initializations(tmp_path):
    from file_utils import acquire_training_lock
    (tmp_path/'runs').mkdir()
    first=acquire_training_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError,match='training lock'):acquire_training_lock(tmp_path)
    finally:first.close()
    acquire_training_lock(tmp_path).close()


def test_retention_keeps_two_recovery_points_and_selected_model(tmp_path):
    with paddle.utils.unique_name.guard():
        m,o,r=setup();step(m,o)
        chosen=tmp_path/'selected';chosen.mkdir();(chosen/'model.pdparams').write_bytes(b'kept')
        for n in [1,2,3]:save_state(tmp_path,m,o,n,n,np.arange(4),r,[],{'fixed':1},0)
        assert sorted(p.name for p in (tmp_path/'recovery').iterdir())==['step-0000002','step-0000003']
        assert (chosen/'model.pdparams').read_bytes()==b'kept'
        _,meta=inspect_state(tmp_path,{'fixed':1});assert meta['step']==3


def test_run_plan_rejects_changed_data_and_unapproved_parameters(tmp_path):
    from types import SimpleNamespace
    from training_state import validate_plan
    from file_utils import sha
    dataset=tmp_path/'data';dataset.mkdir();manifest=dataset/'manifest.jsonl';manifest.write_text('{}\n')
    profile=tmp_path/'profile.json';profile.write_text('{}\n')
    shared=dict(steps=1000,batch=16,learning_rate=5e-6,eval_every=500,sampling='balanced',input_policy='released',seed=123)
    args=SimpleNamespace(language='zh-CN',dataset=dataset,training_profile=profile,**shared)
    plan=dict(purpose='formal_training',actual_maa_evaluation=True,stop_on_gate_failure=True,route='zh-CN',max_steps=1000,allow_training=True,random_policy='step_seed_v1',dataset=str(dataset),dataset_manifest_sha256=sha(manifest),profile_sha256=sha(profile),files={str(profile):sha(profile)},**shared)
    validate_plan(args,plan)
    for key in ['actual_maa_evaluation','stop_on_gate_failure']:
        plan[key]=False
        with pytest.raises(ValueError,match='requires actual MAA'):validate_plan(args,plan)
        plan[key]=True
    args.steps=1001
    with pytest.raises(ValueError,match='parameter differs'):validate_plan(args,plan)
    args.steps=1000;manifest.write_text('{"changed":true}\n')
    with pytest.raises(ValueError,match='manifest changed'):validate_plan(args,plan)
    manifest.write_text('{}\n');profile.write_text('{"changed":true}\n')
    with pytest.raises(ValueError,match='dependency changed'):validate_plan(args,plan)


def test_previous_candidate_cannot_hide_an_official_task_regression():
    rows=[{'id':'rules/a','expected_output':'A','label':'A','subset':'guard','scene':'operator'}]
    def pred(s):return [{'id':'rules/a','results':[{'text':s}]}]
    scored=assess(rows,pred('wrong'),pred('A'),pred('wrong'))
    assert not scored['quality_eligible']
    assert scored['new_rule_errors_vs_official']==['rules/a']


def test_rule_gain_does_not_allow_lower_raw_total():
    rows=[{'id':i,'expected_output':s,'label':s,'subset':'guard','scene':'operator','evidence_kind':'synthetic'} for i,s in [('rules/a','A'),('raw/a','A')]]
    def pred(values):return [{'id':r['id'],'results':[{'text':s}]} for r,s in zip(rows,values)]
    scored=assess(rows,pred(['A','wrong']),pred(['wrong','A']),pred(['wrong','A']))
    assert not scored['quality_eligible']
    assert not scored['raw_non_decrease_vs_current']


def test_literal_confusable_errors_remain_protected_when_total_improves():
    rows=[{'id':i,'expected_output':s,'label':s,'subset':'guard','scene':scene,'evidence_kind':'synthetic'} for i,s,scene in [('raw/confusable','O0Il','confusable'),('raw/a','A','ascii'),('raw/b','B','ascii')]]
    def pred(values):return [{'id':r['id'],'results':[{'text':s}]} for r,s in zip(rows,values)]
    base=pred(['O0Il','X','X']);candidate=pred(['00Il','A','B']);scored=assess(rows,candidate,base,base)
    assert not scored['quality_eligible'];assert scored['new_confusable_errors']==['raw/confusable']


def test_real_regression_is_protected_when_id_prefix_changes():
    rows=[{'id':'raw/real_stage_crop/a','expected_output':'10-3','label':'10-3',
           'subset':'real_stage_crop','scene':'stage_code','evidence_kind':'reviewed_real'},
          {'id':'raw/new_synthetic/b','expected_output':'20-1','label':'20-1',
           'subset':'development','scene':'stage_code','evidence_kind':'synthetic'}]
    def pred(values):
        return [{'id':row['id'],'results':[{'text':value}]} for row,value in zip(rows,values)]
    official=pred(['10-3','wrong']);candidate=pred(['wrong','20-1'])
    result=assess(rows,candidate,official,official)
    assert result['raw_non_decrease_vs_current']
    assert not result['quality_eligible']
    assert result['new_real_errors_vs_official']==['raw/real_stage_crop/a']
    rows[0]['id']='raw/another_name/a'
    assert not assess(rows,pred(['wrong','20-1']),pred(['10-3','wrong']),pred(['10-3','wrong']))['quality_eligible']


def test_unclassified_raw_case_and_false_legacy_provenance_fail_closed():
    row={'id':'raw/real_stage_crop/a','expected_output':'A','label':'A',
         'subset':'new','scene':'stage_code'}
    prediction=[{'id':row['id'],'results':[{'text':'A'}]}]
    with pytest.raises(ValueError,match='needs evidence_kind'):
        assess([row],prediction,prediction,prediction)
    row.update(id='raw/real/a',evidence_kind='synthetic');prediction[0]['id']=row['id']
    with pytest.raises(ValueError,match='conflicting'):
        assess([row],prediction,prediction,prediction)

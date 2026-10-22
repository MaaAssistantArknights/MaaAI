"""Portable contracts use synthetic fixtures; no external game assets or GPU needed."""
import json,sys
from pathlib import Path
import numpy as np
import pytest

PIPELINE=Path(__file__).resolve().parents[1]/'pipeline'
@pytest.fixture
def modules(tmp_path,monkeypatch):
    monkeypatch.setenv('MAA_OCR_RUNTIME',str(tmp_path));monkeypatch.syspath_prepend(str(PIPELINE/'scripts'))
    names=['workspace','asset_utils','core_training_data','text_targets','prepare_model','source_assets','source_corpus','run_training_plan','doctor']
    for name in names:sys.modules.pop(name,None)
    def load(name):return __import__(name)
    yield load
    for name in names:sys.modules.pop(name,None)

def test_token_remap_preserves_order_and_blank_and_new_rows(modules):
    m=modules('prepare_model');a=np.arange(12,dtype=np.float32).reshape(3,4)
    new,pairs=m.remap_array(a,1,['blank','O','0','space'],['blank','0','O','space','★'],fill=-12)
    assert pairs==[(0,0),(1,2),(2,1),(3,3)]
    np.testing.assert_array_equal(new[:,:4],a[:,[0,2,1,3]])
    assert np.all(new[:,-1]==-12)
    with pytest.raises(ValueError,match='duplicate'):m.token_mapping(['0','0'],['0'])
    with pytest.raises(ValueError,match='classes'):m.remap_array(a,1,['0'],['0'])

def test_source_lock_rejects_escape_and_changed_local_file(modules,tmp_path):
    m=modules('source_assets');f=tmp_path/'asset';f.write_text('changed');lock=tmp_path/'lock.json'
    lock.write_text(json.dumps({'files':[{'path':'asset','sha256':'0'*64,'url':'https://example.invalid/never-fetched'}]}))
    with pytest.raises(ValueError,match='changed'):m.fetch(lock,offline=True)
    lock.write_text(json.dumps({'files':[{'path':'../escape','sha256':'0'*64,'url':'https://example.invalid/never-fetched'}]}))
    with pytest.raises(ValueError,match='outside'):m.fetch(lock,offline=True)

def test_planning_print_does_not_implicitly_train(modules,tmp_path):
    m=modules('run_training_plan');plan=tmp_path/'plan.json'
    plan.write_text(json.dumps({'route':'ko','dataset':'data','profile':'config','checkpoint':'weights','steps':2,'batch':1,'learning_rate':.000005,'eval_every':2,'sampling':'balanced','input_policy':'released','seed':1}))
    cmd=m.command(plan,tmp_path/'output',resume=True,pause_at=1)
    assert '--resume' in cmd and cmd[cmd.index('--language')+1]=='ko'
    assert not (tmp_path/'output').exists()

def test_visible_text_does_not_strip_meaningful_spaces(modules):
    m=modules('text_targets');assert m.transform('A\u00a0B')['label']=='A B'
    assert m.transform('스팀 펌프')['label']=='스팀 펌프'
    assert m.transform('Ⅱ')['label']=='Ⅱ'


def test_dictionary_extension_preserves_existing_order_and_appends_once(modules,tmp_path):
    model=modules('prepare_model');source=tmp_path/'source.txt';source.write_text('A\n0\nO\n')
    output=model.extend_dictionary(source,['O','★','★'],tmp_path/'keys.txt')
    assert output.read_text()=='A\n0\nO\n★\n'
    assert source.read_text()=='A\n0\nO\n'


@pytest.mark.parametrize('character',['AB','\n',' '])
def test_dictionary_extension_rejects_noncharacter_tokens(modules,tmp_path,character):
    source=tmp_path/'source.txt';source.write_text('A\n')
    with pytest.raises(ValueError,match='one visible character'):
        modules('prepare_model').extend_dictionary(source,[character],tmp_path/'keys.txt')

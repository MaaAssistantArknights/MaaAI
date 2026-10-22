import json
from pathlib import Path
import pytest
from evaluate import summarize_timing, evaluation_failed
from package_candidate import package
from checkpoint_select import equivalence_rows
from routes import CLIENT
from file_utils import sha


def test_paired_timing_rejects_changed_workload_and_settings():
    def doc(ms):return {'cpu_threads':3,'timing':[[{'id':'a','ms':ms},{'id':'b','ms':ms}]]}
    records={'official':[doc(10),doc(10),doc(10)],'candidate':[doc(11),doc(11),doc(11)]}
    r=summarize_timing(records,['a','b'],2)
    assert r['response_time_ratio']==pytest.approx(1.1) and not r['within_1_05']
    records['candidate'][0]['timing'][0][1]['id']='a'
    with pytest.raises(ValueError,match='workload'):summarize_timing(records,['a','b'],2)
    records['candidate'][0]=doc(11);records['candidate'][0]['cpu_threads']=1
    with pytest.raises(ValueError,match='settings'):summarize_timing(records,['a','b'],2)


def test_packaging_pins_every_pair_and_excludes_unrelated_resources(tmp_path):
    import zipfile
    packs={};expected={}
    for route,client in CLIENT.items():
        p=tmp_path/route/'rec';p.mkdir(parents=True)
        base=Path('resource') if client=='Official' else Path('resource/global')/client/'resource'
        base=base/('PaddleCharOCR' if route=='char' else 'PaddleOCR')/'rec'
        for name in ['inference.onnx','keys.txt']:
            f=p/name;f.write_bytes((route+name).encode());expected[str(base/name)]=sha(f)
        (p.parent/'tasks.json').write_text('must not ship')
        packs[route]=str(p.parent)
    cfg=tmp_path/'package.json';cfg.write_text(json.dumps({'packs':packs,'expected_sha256':expected}))
    index=package(cfg,tmp_path/'out')
    with zipfile.ZipFile(tmp_path/'out/candidate-recognition-resources.zip') as z:
        assert set(z.namelist())==set(expected)|{'asset-index.json'}
    assert not index['release_approved']
    (Path(packs['ko'])/'rec/keys.txt').write_text('changed')
    with pytest.raises(ValueError,match='changed'):package(cfg,tmp_path/'changed')


def test_export_equivalence_accepts_versioned_raw_prefixes():
    rows=[{'id':'raw/final/a'},{'id':'rules/final/a'},{'id':'raw/v3/b'}]
    assert [r['id'] for r in equivalence_rows(rows)]==['raw/final/a','raw/v3/b']
    with pytest.raises(ValueError,match='no raw'):
        equivalence_rows([{'id':'rules/final/a'}])


def test_timing_is_recorded_without_veto_but_quality_still_blocks():
    summary={'failed_timing_routes':['zh-CN'],'frames':None,
             'recognition':{'zh-CN':{'quality_eligible':True}}}
    assert not evaluation_failed({'purpose':'evaluation'},summary)
    assert evaluation_failed({'purpose':'evaluation','timing_policy':'enforce'},summary)
    summary['frames']={'new_truth_regressions':['bad-case']}
    assert evaluation_failed({'purpose':'evaluation'},summary)
    summary['frames']=None;summary['recognition']['zh-CN']['quality_eligible']=False
    assert evaluation_failed({'purpose':'evaluation'},summary)
    with pytest.raises(ValueError,match='timing policy'):
        evaluation_failed({'purpose':'evaluation','timing_policy':'unknown'},summary)


def test_freeze_requires_provenance_for_new_prefix(tmp_path,monkeypatch):
    import freeze_evaluation
    from PIL import Image
    source=tmp_path/'a.png';Image.new('RGB',(20,10),'white').save(source)
    mp=tmp_path/'source.jsonl';mp.write_text(json.dumps({'id':'a','image':str(source),
        'sha256':sha(source),'split':'dev','label':'0-1','scene':'stage_code'})+'\n')
    cfg=tmp_path/'freeze.json'
    spec={'routes':{'char':[{'manifest':str(mp),'prefix':'real_stage_crop',
        'subset':'real_stage_crop','selection':'all'}]}}
    cfg.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='evidence_kind'):
        freeze_evaluation.freeze(cfg,tmp_path/'no-output')
    assert not (tmp_path/'no-output').exists()
    spec['routes']['char'][0]['evidence_kind']='reviewed_real';cfg.write_text(json.dumps(spec))
    release=tmp_path/'release';(release/'resource').mkdir(parents=True)
    (release/'resource/battle_data.json').write_text('{"chars":{}}')
    monkeypatch.setattr(freeze_evaluation,'RELEASE',release)
    freeze_evaluation.freeze(cfg,tmp_path/'frozen')
    cases=[json.loads(line) for line in (tmp_path/'frozen/char.jsonl').read_text().splitlines()]
    assert {row['evidence_kind'] for row in cases}=={'reviewed_real'}
    assert cases[0]['id']=='raw/real_stage_crop/a'

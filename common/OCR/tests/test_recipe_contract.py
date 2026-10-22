import json
import subprocess
import sys
from pathlib import Path

import pytest
from dataset_contract import split
from recipe_sampling import schedule
from text_targets import normalize_text, partition, transform, validate_transform


def test_data_entry_loads_without_training_components():
    entry = Path(__file__).resolve().parents[1] / 'pipeline/scripts/recipe_dataset.py'
    result = subprocess.run(
        [sys.executable, str(entry), '--help'],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_current_profile_selects_v6_except_korean_v5():
    configs = Path(__file__).resolve().parents[1] / 'pipeline/configs'
    profile = json.loads((configs / 'training.json').read_text())
    assert {language: route['model'] for language, route in profile['languages'].items()} == {
        'zh-CN': 'v6', 'zh-TW': 'v6', 'en': 'v6', 'ja': 'v6', 'ko': 'v5-korean'
    }
    assert profile['char']['model'] == 'v6-ascii'

def test_visible_targets_preserve_source_and_compatibility_characters():
    raw='キアーヘ\u3099 Ⅱ\u00a0II';r=transform(raw)
    assert r['label']=='キアーベ Ⅱ II'
    assert r['source_label']==raw
    validate_transform(r,raw)
    with pytest.raises(ValueError):validate_transform({**r,'label':r['label'].replace('Ⅱ','II')},raw)

def test_normalization_never_promotes_old_dev_or_reserved_to_train():
    for raw in ['キアーヘ\u3099の印（中堅）','타이드\u00a0온\u00a0선셋\u00a0헤드헌팅\u00a0허가증']:
        if split(raw)!='train' or split(normalize_text(raw))!='train':assert partition(raw)!='train'

def test_word_sampling_preserves_historical_ratio_and_records_exposures():
    rows=[{'scene':'recruitment','label':'近战'},{'scene':'operator_name','label':'能天使'},{'scene':'item_name','label':'abcdefghi'},{'scene':'symbol_replay','label':'★1'},{'scene':'depot_quantity','label':'1.1万'}]
    a,summary=schedule(rows,'word','historical',1000)
    b,_=schedule(rows,'word','historical',1000)
    assert a.tolist()==b.tolist()
    assert summary['buckets']=={'short':300,'long':600,'number':100}
    assert summary['special_character_presentations']['★']>0
    with pytest.raises(ValueError):schedule(rows[:-1],'word','historical',100)

def test_char_new_formats_receive_explicit_budget():
    families=['stage_code','battle_cost','ascii_replay','timer','padded_number','spaced_ascii']
    rows=[{'scene':f,'label':'01 0O'} for f in families]
    _,s=schedule(rows,'char','balanced',1000)
    assert s['families']=={'stage_code':250,'battle_cost':150,'ascii_replay':200,'timer':150,'padded_number':100,'spaced_ascii':150}


def test_candidate_is_driven_by_maa_scenes_not_historical_length_rule():
    families=['recruitment','operator_name','item_name','depot_quantity','skill_replay','symbol_replay']
    rows=[{'scene':f,'label':'短字'} for f in families]
    _,result=schedule(rows,'word','balanced',1000)
    assert result['families']=={'recruitment':100,'operator_name':250,'item_name':300,'depot_quantity':200,'skill_replay':100,'symbol_replay':50}

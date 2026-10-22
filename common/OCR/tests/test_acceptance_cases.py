import pytest
from acceptance_cases import compare_cases,check_truth,canonical

def test_truth_is_not_replaced_by_official_prediction():
 c={'id':'x','route':'zh-CN','mode':'task_region','source_scope':'dev','truth':{'texts':['0'],'analyze_ok':True}}
 old={'id':'x','mode':'task_region','analyze_ok':True,'results':[{'text':'O','score':.9,'rect':[0,0,2,2]}]}
 result=compare_cases([c],[old],[old]);assert not result['cases'][0]['candidate_all_labeled_fields_correct'];assert result['cases'][0]['whole_output_equal']

def test_shop_priority_and_duplicate_names_matter():
 c={'truth':{'texts':['许可','许可','铁'],'ordered':True}}
 p={'results':[{'text':t} for t in ['铁','许可','许可']]};assert not check_truth(c,p)[0]['match']
 c['truth']['ordered']=False;assert check_truth(c,p)[0]['match'];p['results'].pop();assert not check_truth(c,p)[0]['match']

def test_missing_ids_cannot_pass():
 with pytest.raises(ValueError,match='ids differ'):compare_cases([{'id':'x'}],[],[])

def test_known_omission_is_reported_not_erased():
 c={'truth':{'fields':[{'id':'f','label':'6','box_xywh':[0,0,4,4]}]},'known_omissions':['f']}
 check=check_truth(c,{'results':[]})[0];assert not check['match'] and check['known_baseline_omission']

def test_business_comparison_ignores_confidence_but_keeps_decisions():
 a={'id':'x','mode':'credit_shop','analyze_ok':True,'results':[{'text':'A','score':.9,'rect':[1,2,3,4]}]}
 b={**a,'results':[{**a['results'][0],'score':.8}]};assert canonical(a)==canonical(b)
 b['analyze_ok']=False;assert canonical(a)!=canonical(b)


def test_empty_truth_or_wrong_task_cannot_pass():
 c={'id':'x','route':'zh-CN','mode':'recruit','source_scope':'dev','truth':{}}
 row={'id':'x','mode':'recruit','results':[]}
 with pytest.raises(ValueError,match='asserted truth'):compare_cases([c],[row],[row])
 c['truth']={'texts':[]}
 with pytest.raises(ValueError,match='mode differs'):compare_cases([c],[row],[dict(row,mode='depot')])
 with pytest.raises(ValueError,match='duplicate cases'):compare_cases([c,c],[row],[row])


def test_truth_checks_stage_metadata_and_drops():
    case={'id':'x','route':'zh-CN','mode':'stage_drops','source_scope':'independent',
          'truth':{'analyze_ok':True,'stage_code':'LS-1','difficulty':'Normal','stars':3,'times':-1,
                   'drops':[{'item_id':'4001','quantity':120,'type':1}]}}
    pred={'id':'x','mode':'stage_drops','analyze_ok':True,'stage_code':'LS-1','difficulty':'Normal','stars':3,'times':-1,
          'drops':[{'type':1,'quantity':120,'item_id':'4001'}]}
    result=compare_cases([case],[pred],[pred])
    assert result['cases'][0]['candidate_all_labeled_fields_correct']
    bad={**pred,'stars':2}
    result=compare_cases([case],[pred],[bad])
    assert result['new_truth_regressions']==['x']



def test_depot_quantities_are_compared_as_a_multiset():
    case={'truth':{'quantities':[20,810,20]}}
    pred={'results':[{'quantity':810},{'quantity':20},{'quantity':20}]}
    assert check_truth(case,pred)[0]['match']
    pred['results'].pop()
    assert not check_truth(case,pred)[0]['match']

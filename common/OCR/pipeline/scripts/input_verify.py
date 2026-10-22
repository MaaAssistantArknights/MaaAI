"""Strict paired prediction scorer."""
from text_targets import normalize_text

def score(rows,predictions):
    """在固定真值下统计发布接口预测的文字命中情况。"""
    truth={r['id']:r for r in rows};groups={};errors=[];pred={}
    for r in predictions:
        if r['id'] not in truth or r['id'] in pred:raise ValueError('unexpected/duplicate output ID')
        row=truth[r['id']];literal=[v['text'] for v in r['results']];texts=[normalize_text(t) for t in literal];pred[r['id']]=texts;ok=texts==[row['expected_output']]
        key=r['id'].split('/')[0]+'/'+row['subset']+'/'+row['scene'];g=groups.setdefault(key,{'count':0,'correct':0});g['count']+=1;g['correct']+=ok
        if not ok:errors.append({'id':r['id'],'label':row['label'],'expected_output':row['expected_output'],'literal_outputs':literal})
    if set(pred)!=set(truth):raise ValueError('missing output IDs')
    return {'groups':groups,'errors':errors},pred

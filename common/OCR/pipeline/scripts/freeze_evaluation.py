"""Freeze raw and task-rule development cases without looking at candidate predictions."""
import argparse,json
from pathlib import Path
from case_io import manifest,write_rows
from routes import CLIENT
from file_utils import sha
from workspace import RELEASE
from recruitment_tags import recruitment_tags
from evidence_provenance import source_evidence_kind


def target(row,characters):
    """将来源角色 ID 等信息映射到 MAA 任务实际比较的文字。"""
    if row['id'].startswith('rules/') and row.get('task_name')=='CharsNameOcrReplace':
        pointer=row['source']['json_pointer'];parts=pointer.split('/')
        if len(parts)!=3 or parts[2]!='name':raise ValueError('unexpected character source pointer')
        key=parts[1].replace('~1','/').replace('~0','~')
        return characters[key]['name'], '/chars/'+parts[1]+'/name'
    return row['label'],None


def freeze(spec,output):
    """按明确来源和真值冻结开发案例，不读取候选预测。"""
    cfg=json.loads(spec.read_text())
    # Existing frozen final/v3/real sources remain readable. New source names
    # must explicitly say whether their pixels came from reviewed real frames.
    kinds={}
    for route,sources in cfg['routes'].items():
        for number,source in enumerate(sources):
            prefix=source['prefix']
            kind=source_evidence_kind(prefix,source.get('evidence_kind'))
            kinds[(route,number)]=kind
    output.mkdir(parents=True,exist_ok=False)
    battle=RELEASE/'resource/battle_data.json';chars=json.loads(battle.read_text())['chars'];index={}
    for route,sources in cfg['routes'].items():
        if route not in CLIENT:raise ValueError('unknown route')
        rows=[];fingerprints={}
        for number,source in enumerate(sources):
            path=Path(source['manifest']);path=path if path.is_absolute() else spec.parent/path
            temp=output/f'{route}-source-{number}.jsonl';data=manifest(path,temp)
            fingerprints[str(path.resolve())]=sha(path)
            if any(r.get('split') not in ['dev','development','S-dev'] for r in data):raise ValueError('development selection must not consume training or reserved test data')
            if source['selection']=='scene-hash-and-length':
                selected={}
                for scene in sorted({r['scene'] for r in data}):
                    group=[r for r in data if r['scene']==scene]
                    for r in sorted(group,key=lambda r:r['sha256'])[:30]+sorted(group,key=lambda r:len(r['label']))[-10:]:selected[r['id']]=r
                data=list(selected.values())
            elif source['selection']!='all':raise ValueError('unknown selection policy')
            rows.extend(dict(r,id=source['prefix']+'/'+r['id'],subset=source['subset'],evidence_kind=kinds[(route,number)]) for r in data)
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate selected IDs')
        taskmap={'stage_code':'ClickStageName','battle_cost':'BattleCostData'} if route=='char' else {'recruitment':'RecruitTags','item_name':'CreditShop-ProductName','depot_quantity':'NumberOcrReplace','operator_name':'CharsNameOcrReplace','operator':'CharsNameOcrReplace'}
        cases=[]
        for r in rows:
            raw=dict(r,id='raw/'+r['id'],mode='rec',expected_output=r['label']);raw.pop('task_name',None);cases.append(raw)
            task=taskmap.get(r['scene'])
            if not task:continue
            row=dict(r,id='rules/'+r['id'],mode='task_crop',task_name=task)
            if task=='RecruitTags':row['runtime_required']=sorted(recruitment_tags(route))
            expected,pointer=target(row,chars)
            if task=='NumberOcrReplace':expected=expected.replace('萬','万').replace('億','亿').replace('만','万').replace('억','亿')
            row.update(expected_output=expected,internal_name_pointer=pointer);cases.append(row)
        dest=output/(route+'.jsonl');write_rows(dest,cases)
        index[route]={'count':len(cases),'manifest_sha256':sha(dest),'sources':fingerprints}
    (output/'index.json').write_text(json.dumps({'routes':index,'battle_data_sha256':sha(battle),'spec_sha256':sha(spec),'independent_acceptance':False,'selection_uses_model_outputs':False},ensure_ascii=False,indent=2)+'\n')
    return index

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze(a.spec,a.output)

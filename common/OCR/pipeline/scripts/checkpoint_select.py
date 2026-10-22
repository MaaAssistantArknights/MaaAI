"""Select saved checkpoints using pinned released MAA, never GPU dev score alone."""
import json,shutil
from pathlib import Path
from file_utils import atomic_json, sha
from workspace import RT
from workspace import RELEASE
from multilingual_verify import export,run
from input_verify import score
from routes import CLIENT
from evidence_provenance import raw_evidence_kind

def totals(scored):
    """汇总开发集原文和任务规则得分。"""
    return {kind:sum(v['correct'] for key,v in scored['groups'].items() if key.startswith(kind+'/')) for kind in ['raw','rules']}

def assess(rows,predictions,official,current):
    """结合参考结果评估检查点并拒绝能力回退。"""
    # Reject unclassified raw images before scoring: an ID spelling change must
    # never silently turn a protected screenshot into an ordinary synthetic row.
    real_ids = {r['id'] for r in rows if r['id'].startswith('raw/') and raw_evidence_kind(r) == 'reviewed_real'}
    # Use the same frozen cases and released task rules for all three arms.
    # An untuned v6 base is not a substitute for the current MAA baseline.
    scores,pred=score(rows,predictions);_,base=score(rows,official);current_score,prior=score(rows,current)
    new={name:[r['id'] for r in rows if reference[r['id']]==[r['expected_output']] and pred[r['id']]!=[r['expected_output']]] for name,reference in [('official',base),('current',prior)]}
    real=sorted(set(new['current']) & real_ids)
    rules=[i for i in new['current'] if i.startswith('rules/')]
    # No blanket veto for a GPU-only regression; exported actual task rules decide.
    official_rules=[i for i in new['official'] if i.startswith('rules/')]
    official_real=sorted(set(new['official']) & real_ids)
    confusable_ids={r['id'] for r in rows if r.get('scene')=='confusable' and r['id'].startswith('raw/')}
    new_confusable=sorted((set(new['official'])|set(new['current']))&confusable_ids)
    raw_non_decrease=totals(scores)['raw']>=totals(current_score)['raw']
    eligible=not real and not rules and not official_rules and not official_real and not new_confusable and raw_non_decrease
    return {'scores':scores,'totals':totals(scores),'reference_totals':totals(current_score),'new_errors':new,'new_real_errors_vs_current':real,'new_rule_errors_vs_current':rules,'new_rule_errors_vs_official':official_rules,'new_real_errors_vs_official':official_real,'raw_non_decrease_vs_current':raw_non_decrease,'new_confusable_errors':new_confusable,'selection_contract':'actual-maa-primary-and-current-v3-with-explicit-real-provenance','quality_eligible':eligible,'scope':'fixed development raw recognition and original task text rules; not full business or release acceptance'}

def equivalence_rows(rows):
    """只选原始图像案例进入导出一致性检查。"""
    selected=[r for r in rows if r.get('id','').startswith('raw/')]
    if not selected:raise ValueError('evaluation contains no raw recognition cases')
    return selected

def evaluate_checkpoint(output,checkpoint,plan,config):
    """导出单个检查点并收集发布接口与开发集评分。"""
    step=int(checkpoint.name.split('-')[-1]);route=plan['route'];parent=output/'evaluations'/checkpoint.name;parent.mkdir(parents=True,exist_ok=True)
    report=parent/'gate.json'
    if report.exists():
        cached=json.loads(report.read_text())
        if cached['checkpoint_sha256']!=sha(checkpoint/'model.pdparams'):raise ValueError('cached checkpoint differs')
        return cached
    folder=parent/f'attempt-{len(list(parent.glob("attempt-*")))+1:02d}';folder.mkdir()
    mp=Path(plan['evaluation']['manifest']);rows=[json.loads(s) for s in mp.read_text().splitlines()]
    if sha(mp)!=plan['evaluation']['manifest_sha256']:raise ValueError('evaluation manifest changed')
    for r in rows:
        if sha(Path(r['image']))!=r['sha256']:raise ValueError('evaluation image changed')
    variant=f'{output.name.lower()}-{step}-{folder.name}'
    selected=equivalence_rows(rows)
    package,gate=export(checkpoint/'model.pdparams',config,variant,folder,selected,'char' if route=='char' else 'word')
    result={'step':step,'checkpoint_sha256':sha(checkpoint/'model.pdparams'),'export_passed':gate['passed'],'export':gate,'quality_eligible':False,'new_real_errors_vs_current':[],'release_approved':False}
    if gate['passed']:
        dest=folder/'predictions.json'
        run([RT/'build/maa_ocr_probe',RELEASE,package,mp,dest,0,0,0,CLIENT[route],'char' if route=='char' else 'word'],folder/'maa.log')
        refs={name:json.loads(Path(plan['evaluation'][name]).read_text())['predictions'] for name in ['official','current']}
        result.update(assess(rows,json.loads(dest.read_text())['predictions'],refs['official'],refs['current']))
        result.update(pack=str(package),predictions=str(dest))
        current_selection=json.loads((output/'selection.json').read_text()) if (output/'selection.json').exists() else None
        best=current_selection['totals'] if current_selection else result['reference_totals']
        # Prefer task-rule correctness, then literal recognition; ties keep the
        # earlier checkpoint. A selected development checkpoint is not release approval.
        rank=lambda x:(x['rules'],x['raw'])
        if result['quality_eligible'] and rank(result['totals'])>rank(best):
            selected_dir=output/'selected';selected_dir.mkdir(exist_ok=True)
            temporary=selected_dir/'model.tmp';shutil.copy2(checkpoint/'model.pdparams',temporary);temporary.replace(selected_dir/'model.pdparams')
            shutil.copy2(config,selected_dir/'model.yml')
            atomic_json(output/'selection.json',{'step':step,'totals':result['totals'],'pack':str(package),'pack_onnx_sha256':gate['model_sha256'],'weights':str(selected_dir/'model.pdparams'),'weights_sha256':sha(selected_dir/'model.pdparams'),'gate':str(report),'release_approved':False,'reason':'no new task-rule/real-crop/literal-confusable errors vs official or prior FP32; raw total does not decrease; improves task/raw tuple; pending full acceptance'})
    atomic_json(report,result)
    return result

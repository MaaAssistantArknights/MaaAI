"""Explicitly execute one fixed plan; never discovers or starts formal plans automatically."""
import argparse,json,subprocess,sys
from pathlib import Path
from workspace import ROOT

def command(plan_path,output,resume=False,pause_at=None):
    """把冻结计划转换为训练子进程参数，不直接开始计算。"""
    p=json.loads(plan_path.read_text());route=p['route']
    args=[sys.executable,str(ROOT/'scripts/train_model.py'),'--run-plan',str(plan_path.resolve()),'--kind','char' if route=='char' else 'word','--dataset',p['dataset'],'--output',str(output.resolve()),'--training-profile',p['profile'],'--checkpoint',p['checkpoint']]
    if route!='char':args+=['--language',route]
    for key in ['steps','batch','learning_rate','eval_every','sampling','input_policy','seed']:args+=['--'+key.replace('_','-'),str(p[key])]
    if resume:args+=['--resume']
    if pause_at is not None:args+=['--pause-at',str(pause_at)]
    return args

def main():
    """检查计划和目标目录后显式启动或打印训练命令。"""
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--pause-at',type=int);p.add_argument('--print-command',action='store_true');a=p.parse_args()
    args=command(a.plan,a.output,a.resume,a.pause_at)
    if a.print_command:print(__import__('shlex').join(args));return
    subprocess.run(args,cwd=ROOT,check=True)
    if (a.output/'stopped.json').exists() and not (a.output/'result.json').exists():
        raise RuntimeError('checkpoint gate stopped training; inspect '+str(a.output/'stopped.json'))
if __name__=='__main__':main()

"""正式 OCR 训练与验收入口；耗时操作均需显式选择。"""
import argparse,os,subprocess,sys
from pathlib import Path

COMMANDS = {
    'setup': 'setup_runtime.py',
    'doctor': 'doctor.py',
    'fetch': 'source_assets.py',
    'corpus': 'source_corpus.py',
    'prepare-model': 'prepare_model.py',
    'generate': 'recipe_dataset.py',
    'char-rework': 'char_rework_dataset.py',
    'freeze-evaluation': 'freeze_evaluation.py',
    'plan': 'prepare_training_plan.py',
    'train': 'run_training_plan.py',
    'export-check': 'multilingual_verify.py',
    'evaluate': 'evaluate.py',
    'package': 'package_candidate.py',
}

def main():
    """检查外部工作目录，再将所选正式步骤交给唯一实现模块。"""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True);p.add_argument('command',choices=COMMANDS);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    runtime=a.runtime.expanduser().resolve()
    if not runtime.is_dir():p.error('runtime must be an existing external directory')
    env=dict(os.environ,MAA_OCR_RUNTIME=str(runtime),PYTHONPYCACHEPREFIX=str(runtime/'cache/pipeline-pycache'),MPLCONFIGDIR=str(runtime/'cache/matplotlib'))
    args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    result=subprocess.run([sys.executable,str(Path(__file__).parent/'scripts'/COMMANDS[a.command]),*args],env=env)
    raise SystemExit(result.returncode)
if __name__=='__main__':main()

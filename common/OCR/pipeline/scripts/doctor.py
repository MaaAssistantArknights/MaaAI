"""Read-only pinned runtime inspection; does not start training or install software."""
import argparse,json,subprocess
from pathlib import Path
from workspace import ROOT,RT
from file_utils import sha

def inspect(lock):
    """只读核对第三方源码、发布库和文件哈希是否符合锁。"""
    required=json.loads(lock.read_text());checks=[]
    for name,commit in required['git'].items():
        path=RT/name
        actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip() if (path/'.git').exists() else None
        checks.append({'path':name,'kind':'git','expected':commit,'actual':actual,'passed':actual==commit})
    for name,digest in required['files'].items():
        p=RT/name;actual=sha(p) if p.is_file() else None
        checks.append({'path':name,'kind':'sha256','actual':actual,'expected':digest,'passed':actual==digest})
    return {'passed':all(r['passed'] for r in checks),'checks':checks,'training_started':False}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,default=ROOT/'configs/runtime.lock.json');p.add_argument('--output',type=Path);a=p.parse_args();r=inspect(a.lock)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2));raise SystemExit(0 if r['passed'] else 1)

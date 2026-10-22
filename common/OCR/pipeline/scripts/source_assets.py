"""Fetch only declared, hash-pinned public files; offline validates existing assets."""
import argparse,json,urllib.request,os,uuid
from pathlib import Path
from workspace import RT
from file_utils import sha

def fetch(lock,offline=False):
    """仅下载锁内 HTTPS 素材，逐文件核验路径和 SHA-256。"""
    for row in json.loads(lock.read_text())['files']:
        dest=(RT/row['path']).resolve()
        if not dest.is_relative_to(RT):raise ValueError('asset outside runtime')
        if dest.exists():
            if sha(dest)!=row['sha256']:raise ValueError('existing asset changed: '+str(dest))
            continue
        if offline:raise FileNotFoundError(dest)
        if not row['url'].startswith('https://'):raise ValueError('HTTPS required')
        dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_name(dest.name+'.partial-'+uuid.uuid4().hex)
        try:
            with urllib.request.urlopen(row['url'],timeout=60) as src,temp.open('xb') as target:
                while block:=src.read(1024*1024):target.write(block)
            if sha(temp)!=row['sha256']:raise ValueError('download hash mismatch')
            os.replace(temp,dest)
        finally:
            if temp.exists():temp.unlink()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--offline',action='store_true');a=p.parse_args();fetch(a.lock,a.offline);print('Assets verified')

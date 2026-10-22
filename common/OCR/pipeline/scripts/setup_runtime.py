"""Provision pinned public dependencies into an explicit external runtime (Linux x64)."""
import argparse,json,os,subprocess,sys,tarfile,urllib.request
from workspace import ROOT,RT
from source_assets import fetch
from file_utils import sha

def run(args,**kw):
    """执行固定依赖安装或编译命令，失败立即停止。"""
    subprocess.run(list(map(str,args)),check=True,**kw)
def main():
    """下载并核验固定源码、权重、发布包及可选原生探针。"""
    p=argparse.ArgumentParser();p.add_argument('--native',action='store_true',help='also install pinned local compiler and build adapters');a=p.parse_args()
    lock=json.loads((ROOT/'configs/sources.lock.json').read_text());(RT/'runs').mkdir(exist_ok=True)
    for name,repo in lock['repositories'].items():
        path=RT/'vendor'/name
        if not path.exists():
            run(['git','init',path]);run(['git','-C',path,'remote','add','origin',repo['url']]);run(['git','-C',path,'-c','credential.helper=','fetch','--depth=1','origin',repo['commit']]);run(['git','-C',path,'checkout','--detach',repo['commit']])
        if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=repo['commit']:raise ValueError('existing checkout differs; choose a separate runtime')
        if subprocess.check_output(['git','-C',str(path),'diff','HEAD','--name-only'],text=True).strip():raise ValueError('existing dependency checkout has local changes')
    source=RT/'vendor/MAA-release'
    run(['git','-C',source,'-c','credential.helper=','submodule','update','--init','--depth','1','src/MaaUtils'])
    if subprocess.check_output(['git','-C',str(source/'src/MaaUtils'),'rev-parse','HEAD'],text=True).strip()!=lock['maa_utils_commit']:raise ValueError('MaaUtils commit differs')
    fetch(ROOT/'configs/weights.lock.json')
    rel=lock['release'];download_lock=RT/'cache/release-download.json';download_lock.parent.mkdir(parents=True,exist_ok=True);download_lock.write_text(json.dumps({'files':[{'url':rel['url'],'sha256':rel['sha256'],'path':'cache/'+rel['asset']}]}));fetch(download_lock)
    directory=RT/'releases'/rel['tag']
    if not directory.exists():
        directory.mkdir(parents=True)
        with tarfile.open(RT/'cache'/rel['asset']) as f:f.extractall(directory,filter='data')
    from doctor import inspect
    checks=inspect(ROOT/'configs/runtime.lock.json')
    if not checks['passed']:raise ValueError('runtime integrity check failed; do not overwrite it')
    if not a.native:return
    if not (source/'src/MaaUtils/MaaDeps/vcpkg/installed/maa-x64-linux/include/fastdeploy').exists():
        run([sys.executable,source/'tools/maadeps-download.py','--cache-asset','x64-linux'],env={k:v for k,v in os.environ.items() if k not in {'GH_TOKEN','GITHUB_TOKEN'}})
    tools=json.loads((ROOT/'configs/tools.lock.json').read_text());mm=RT/'tools/bin/micromamba'
    if not mm.exists():
        archive=RT/'cache/micromamba.tar.bz2';urllib.request.urlretrieve(tools['micromamba']['url'],archive)
        with tarfile.open(archive) as f:f.extract(f.getmember('bin/micromamba'),RT/'tools',filter='data')
    if sha(mm)!=tools['micromamba']['binary_sha256']:raise ValueError('micromamba hash differs')
    if not (RT/'toolchains/native/conda-meta/history').exists():run([mm,'--no-rc','create','-y','-p',RT/'toolchains/native','--file',ROOT/'configs/native-toolchain.lock.txt'],env=dict(os.environ,MAMBA_ROOT_PREFIX=str(RT/'cache/mamba'),CONDA_PKGS_DIRS=str(RT/'cache/conda')))
    (RT/'build').mkdir(exist_ok=True)
    run([sys.executable, ROOT/'scripts/build_native.py'])
if __name__=='__main__':main()

"""Export recognition-only evaluation resources; never patch an installed MAA."""
import argparse,hashlib,json,zipfile
from pathlib import Path,PurePosixPath
from routes import CLIENT
from file_utils import sha
from workspace import RELEASE_TAG


def package(spec,output):
    """核验六路线逐文件哈希并生成不含额外资源的交接包。"""
    cfg=json.loads(spec.read_text())
    if set(cfg['packs'])!=set(CLIENT):raise ValueError('all six recognition routes required')
    output.mkdir(parents=True,exist_ok=False)
    files={};sources={}
    for route,value in cfg['packs'].items():
        pack=Path(value);pack=(pack if pack.is_absolute() else spec.parent/pack).resolve()
        kind='PaddleCharOCR' if route=='char' else 'PaddleOCR'
        root=PurePosixPath('resource')
        if CLIENT[route]!='Official':root=root/'global'/CLIENT[route]/'resource'
        for name in ['inference.onnx','keys.txt']:
            src=pack/'rec'/name;dest=str(root/kind/'rec'/name)
            digest=sha(src)
            # The spec binds reviewed bytes to MAA's exact per-client paths.
            # Packaging does not itself certify accuracy or export equivalence.
            if cfg['expected_sha256'][dest]!=digest:raise ValueError('selected resource changed: '+dest)
            sources[dest]=src;files[dest]={'route':route,'sha256':digest,'bytes':src.stat().st_size}
    # The release delta is intentionally limited to six recognizers and six
    # dictionaries. Detection, task rules and native binaries stay upstream.
    if set(cfg['expected_sha256'])!=set(files):raise ValueError('unexpected package files')
    index={'purpose':'review candidate for the MAA OCR refresh','candidate_id':cfg.get('candidate_id'),
        'status':cfg.get('status','validation pending'),'baseline_version':RELEASE_TAG,'files':files,
        'binary_included':False,'detection_models_included':False,'task_json_changes':False,'release_approved':False}
    archive=output/'candidate-recognition-resources.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,src in sorted(sources.items()):
            content=src.read_bytes()
            if hashlib.sha256(content).hexdigest()!=files[name]['sha256']:raise ValueError('resource changed while packaging')
            info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,content)
        z.writestr('asset-index.json',json.dumps(index,ensure_ascii=False,indent=2))
    (output/'asset-index.json').write_text(json.dumps({**index,'zip_sha256':sha(archive)},ensure_ascii=False,indent=2)+'\n')
    return index

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();package(a.spec,a.output)

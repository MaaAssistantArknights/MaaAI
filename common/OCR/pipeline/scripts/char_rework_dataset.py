"""Build a bounded CharOCR rework set from a frozen synthetic base and reviewed real stage crops."""
import argparse,copy,hashlib,json,os,random
from pathlib import Path

from PIL import Image,ImageEnhance,ImageFilter,ImageOps

from label_contract import load_encoder
from file_utils import sha
from workspace import ROOT, RT
from training_preflight import audit_rows


def write(path,obj):
    """写出可供冻结和审计的 JSON 文件。"""
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def pixel_sha(image):
    """按尺寸与 RGB 像素计算图像身份，忽略文件编码差异。"""
    rgb=image.convert('RGB')
    return hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()


def inside_runtime(name):
    """将来源路径限制在外部运行目录内。"""
    path=(RT/name).resolve()
    if not path.is_relative_to(RT.resolve()):raise ValueError('source leaves runtime')
    return path


def reviewed_crop(source):
    """按已复核的 ROI 和阈值从原截图取得文字裁剪。"""
    path=inside_runtime(source['path'])
    if sha(path)!=source['sha256']:raise ValueError('reviewed source changed')
    image=Image.open(path).convert('RGB').resize(tuple(source.get('resize',[1280,720])),Image.Resampling.BILINEAR)
    x,y,w,h=source['roi_xywh'];roi=image.crop((x,y,x+w,y+h));gray=ImageOps.grayscale(roi)
    low,high=source['bin_threshold'];mask=gray.point(lambda v:255 if low<=v<=high else 0)
    box=mask.getbbox()
    if box is None:raise ValueError('reviewed source has no threshold foreground')
    pad=source.get('tight_padding',3)
    box=(max(0,box[0]-pad),max(0,box[1]-pad),min(roi.width,box[2]+pad),min(roi.height,box[3]+pad))
    # These are tight foreground crops plus padding, not the full task ROI.
    # Compare platforms using the same crop bytes, not just the same screenshot.
    return roi.crop(box),mask.crop(box)


def variant(raw,mask,index,rng):
    """生成原图、二值图及有限扰动，并记录每种变换。"""
    if index==0:return raw.copy(),{'form':'exact_raw','scale':[1,1],'padding':[0,0,0,0],'blur':0,'contrast':1}
    if index==1:return mask.convert('RGB'),{'form':'exact_binary','scale':[1,1],'padding':[0,0,0,0],'blur':0,'contrast':1}
    form='binary' if index%2 else 'threshold_gray'
    image=mask.convert('L')
    if form=='threshold_gray':
        background=rng.randint(0,24);foreground=rng.randint(225,255)
        image=image.point(lambda v:foreground if v else background)
    sx=rng.choice([.90,.94,.97,1,1.03,1.06,1.10]);sy=rng.choice([.92,.96,1,1.04,1.08])
    size=(max(2,round(image.width*sx)),max(2,round(image.height*sy)))
    image=image.resize(size,Image.Resampling.BILINEAR)
    contrast=rng.choice([.92,.97,1,1.03,1.08]);image=ImageEnhance.Contrast(image).enhance(contrast)
    blur=rng.choice([0,0,.15,.25,.35])
    if blur:image=image.filter(ImageFilter.GaussianBlur(blur))
    pads=[rng.randint(1,7),rng.randint(1,5),rng.randint(1,7),rng.randint(1,5)]
    fill=rng.randint(0,16) if form!='binary' else 0
    canvas=Image.new('L',(image.width+pads[0]+pads[2],image.height+pads[1]+pads[3]),fill)
    canvas.paste(image,(pads[0],pads[1]))
    return canvas.convert('RGB'),{'form':form,'scale':[sx,sy],'padding':pads,'blur':blur,'contrast':contrast,'background':fill}


def main():
    """从已审计基础集和已复核截图构建 Char 训练返工集。"""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();base=a.base.resolve();spec_path=a.spec.resolve();output=a.output.resolve()
    if output.exists():raise FileExistsError('choose a new output directory')
    base_audit=json.loads((base/'audit.json').read_text())
    if not base_audit['passed'] or base_audit['manifest_sha256']!=sha(base/'manifest.jsonl'):raise ValueError('invalid frozen base')
    from dataset_contract import load_contract
    spec=json.loads(spec_path.read_text())
    recipe=copy.deepcopy(json.loads((base/'recipe.json').read_text()))
    contract=ROOT/recipe['contract'];contract_sha=sha(contract);limits=load_contract(contract)
    if recipe['contract_sha256']!=contract_sha:raise ValueError('base dataset contract changed')
    if spec.get('version')!='char-real-replay-v2':raise ValueError('unknown rework spec')
    output.mkdir(parents=True);rows=[]
    base_rows=[json.loads(line) for line in (base/'manifest.jsonl').read_text().splitlines()]
    for original in base_rows:
        if original.get('route')!='CharOCR':continue
        if original['trial_contract_sha256']!=contract_sha:raise ValueError('base sample contract differs')
        source=base/original['image'];dest=output/original['image'];dest.parent.mkdir(parents=True,exist_ok=True)
        if sha(source)!=original['sha256']:raise ValueError('base image changed')
        os.link(source,dest)
        row=copy.deepcopy(original);row['trial_contract_sha256']=contract_sha
        row['inherited_from_rework']={'manifest':str((base/'manifest.jsonl').resolve()),'id':original['id']}
        rows.append(row)
    recipe['char_rework']={'version':spec['version'],'spec':str(spec_path.relative_to(RT)),'spec_sha256':sha(spec_path),'base_manifest_sha256':sha(base/'manifest.jsonl'),'policy':'reviewed real StageDrops crop; deterministic mild geometry and threshold replay; train only'}
    write(output/'recipe.json',recipe);recipe_sha=sha(output/'recipe.json')
    for row in rows:row['recipe_sha256']=recipe_sha
    rng=random.Random(spec['seed']);folder=output/'char/train';folder.mkdir(parents=True,exist_ok=True)
    for source_index,source in enumerate(spec['sources']):
        if source.get('review_status')!='confirmed' or not source.get('reviewer'):raise ValueError('source truth not reviewed')
        raw,mask=reviewed_crop(source)
        # Keep every augmentation tied to its source screenshot. These sources
        # have now influenced training and cannot count as independent acceptance.
        session='MaaTestSet@'+spec['source_revision']+':'+source['path']
        for i in range(source['variants']):
            image,trace=variant(raw,mask,i,rng);rid=f"char-train-real-stage-{source_index:02d}-{i:03d}"
            path=folder/(rid+'.png');image.save(path)
            rows.append({'id':rid,'image':str(path.relative_to(output)),'sha256':sha(path),'pixel_sha256':pixel_sha(image),'label':source['label'],
                'source':{'path':str(spec_path.relative_to(RT)),'sha256':sha(spec_path),'json_pointer':f'/sources/{source_index}/label'},
                'language':'zh-CN','client':'Official','scene':'stage_code','route':'CharOCR','split':'train','text_partition':'train',
                'source_session':session,'training_eligible':True,'bounded_trial_eligible':True,'formal_evaluation_eligible':False,
                'trial_contract_sha256':contract_sha,'appearance_status':'verified_real_crop','appearance_tier':'targeted_replay',
                'task_mapping_status':'reviewed_StageDrops-StageName','calibration_status':'verified','ctc_time_steps':limits['char_ctc_time_steps'],
                'literal_ascii':True,'missing_glyphs':[],'render':{'source_crop':source['roi_xywh'],'bin_threshold':source['bin_threshold'],'augmentation':trace},
                'recipe_sha256':recipe_sha,'leakage_group':'real-stage:'+source['sha256']})
    manifest=output/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    enc={'v6-ascii':load_encoder('v6-ascii',limits['char_encoder_max_length'])[0]}
    result,_=audit_rows(rows,output,enc,'training_trial',contract)
    write(output/'preflight.json',result)
    if not result['passed']:raise ValueError('rework data admission failed; inspect preflight.json')
    groups={}
    for row in rows:
        key=row['split']+'/'+row['scene']+'/'+row['appearance_status'];groups[key]=groups.get(key,0)+1
    audit={'passed':True,'admission':'bounded_trial_only','manifest_sha256':sha(manifest),'recipe_sha256':recipe_sha,'samples':len(rows),
           'base_samples':len([r for r in rows if 'inherited_from_rework' in r]),'real_replay_samples':len([r for r in rows if r['appearance_status']=='verified_real_crop']),
           'groups':groups,'source_spec_sha256':sha(spec_path),'source_code_sha256':sha(Path(__file__)),'formal_acceptance':False}
    write(output/'audit.json',audit);print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

"""Remap CTC and auxiliary heads by tokens, preserving all retained parameters."""
import argparse,copy,json,sys,tempfile
from pathlib import Path
import numpy as np
from workspace import RT
from file_utils import sha

def token_mapping(old,new):
    """按字符身份映射旧输出类别到新字典，拒绝重复字符。"""
    if len(old)!=len(set(old)) or len(new)!=len(set(new)):raise ValueError('duplicate dictionary tokens')
    lookup={c:i for i,c in enumerate(new)}
    return [(i,lookup[c]) for i,c in enumerate(old) if c in lookup]

def remap_array(old,axis,source_tokens,target_tokens,fill=0.,seed=20260920):
    """在指定张量轴复用已知类别权重并初始化新增类别。"""
    if old.shape[axis]!=len(source_tokens):raise ValueError('source classes differ from encoder')
    shape=list(old.shape);shape[axis]=len(target_tokens)
    value=np.random.default_rng(seed).normal(0,.02,shape).astype(old.dtype) if fill is None else np.full(shape,fill,dtype=old.dtype)
    pairs=token_mapping(source_tokens,target_tokens)
    for src,dst in pairs:
        a=[slice(None)]*old.ndim;b=a.copy();a[axis]=src;b[axis]=dst;value[tuple(b)]=old[tuple(a)]
    return value,pairs

def extend_dictionary(source, characters, output):
    """保留官方字符顺序，只在末尾追加尚未存在的可见单字符。"""
    tokens=source.read_text().splitlines()
    if len(tokens)!=len(set(tokens)) or any(len(t)!=1 for t in tokens):
        raise ValueError('invalid source dictionary')
    for character in characters:
        if len(character)!=1 or not character.isprintable() or character.isspace():
            raise ValueError('extension must be one visible character')
        if character not in tokens:tokens.append(character)
    with output.open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(tokens)+'\n')
    return output


def prepare(a,target):
    """按明确的目标字典映射预训练输出层，并核对保存后的模型。"""
    import paddle,yaml
    sys.path.insert(0,str(RT/'vendor/PaddleOCR'))
    from ppocr.modeling.architectures import build_model
    from ppocr.data.imaug.label_ops import CTCLabelEncode,NRTRLabelEncode
    cfg=yaml.safe_load(a.config.read_text());g=cfg['Global'];original=RT/'vendor/PaddleOCR'/g['character_dict_path']
    source_tokens=original.read_text().splitlines();target_tokens=target.read_text().splitlines()
    if any(len(t)!=1 for t in source_tokens+target_tokens):raise ValueError('expected character dictionary, one Unicode character per row')
    if len(set(target_tokens))!=len(target_tokens):raise ValueError('duplicate dictionary characters')
    # Encoder vocabularies include special tokens not present in keys.txt. Map
    # both heads by token identity, never by dictionary line number alone.
    length=g['max_text_length'];space=g['use_space_char']
    enc=[(CTCLabelEncode(length,str(d),space),NRTRLabelEncode(length,str(d),space)) for d in [original,target]]
    sc,sn=enc[0];tc,tn=enc[1];old_aux=sn.character+['<architecture-terminal>'];new_aux=tn.character+['<architecture-terminal>']
    cfg['Global'].update(character_dict_path=str(target),distributed=False)
    cfg['Architecture']['Head']['out_channels_list']={'CTCLabelDecode':len(tc.character),'NRTRLabelDecode':len(tn.character)}
    paddle.set_device('cpu');state=paddle.load(str(a.checkpoint));model=build_model(copy.deepcopy(cfg['Architecture']));expected=model.state_dict()
    if set(state)!=set(expected):raise ValueError('architecture parameter names differ')
    # Only vocabulary-dependent axes change. Every retained column/row is copied
    # exactly; new CTC classes start suppressed until explicitly trained.
    axes={'head.ctc_head.fc.weight':(1,sc.character,tc.character,0.),'head.ctc_head.fc.bias':(0,sc.character,tc.character,-12.),'head.gtc_head.embedding.embedding.weight':(0,old_aux,new_aux,None),'head.gtc_head.tgt_word_prj.weight':(1,old_aux,new_aux,0.)};mapped={};checks={}
    for name,tensor in state.items():
        value=tensor.numpy();pairs=None
        if name in axes:
            axis,source,dest,fill=axes[name];value,pairs=remap_array(value,axis,source,dest,fill)
        if tuple(value.shape)!=tuple(expected[name].shape):raise ValueError('unexplained parameter shape: '+name)
        mapped[name]=paddle.to_tensor(value);checks[name]={'source_shape':list(tensor.shape),'target_shape':list(value.shape),'retained_token_count':len(pairs) if pairs is not None else None,'operation':'token remap' if pairs is not None else 'unchanged'}
    model.set_state_dict(mapped);model.eval()
    with paddle.no_grad():
        x=paddle.zeros([1,3,48,320]);before=model(x).numpy()
        if not np.isfinite(before).all():raise ValueError('nonfinite initialized model')
    a.output.mkdir(parents=True);(a.output/'keys.txt').write_bytes(target.read_bytes());cfg['Global']['character_dict_path']=str((a.output/'keys.txt').resolve());cp=a.output/'initial.pdparams';paddle.save(mapped,str(cp));model.set_state_dict(paddle.load(str(cp)))
    with paddle.no_grad():after=model(x).numpy()
    if not np.array_equal(before,after):raise ValueError('saved model changed')
    cfg['Global']['pretrained_model']=str(cp.resolve().with_suffix(''));(a.output/'model.yml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    report={'source_checkpoint_sha256':sha(a.checkpoint),'source_dictionary_sha256':sha(original),'target_dictionary_sha256':sha(target),'added_characters':[c for c in target_tokens if c not in source_tokens],'removed_characters_count':len(set(source_tokens)-set(target_tokens)),'ctc_mapping':token_mapping(sc.character,tc.character),'aux_mapping':token_mapping(old_aux,new_aux),'parameter_checks':checks,'checkpoint_sha256':sha(cp),'reload_exact':True,'quality_validated':False}
    (a.output/'mapping.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('Prepared model; token mapping is not accuracy validation')
def main():
    """显式选择目标字典及可选追加字符，再准备初始权重。"""
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dictionary',type=Path,required=True)
    p.add_argument('--append-characters',nargs='+',default=[])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('output already exists')
    if not a.append_characters:
        prepare(a,a.dictionary.resolve())
        return
    # Temporary dictionary bytes are copied into the prepared model directory;
    # its YAML never points at a file that disappears when this context exits.
    with tempfile.TemporaryDirectory(prefix='ocr-dictionary-',dir=RT) as directory:
        target=extend_dictionary(a.dictionary,a.append_characters,Path(directory)/'keys.txt')
        prepare(a,target)


if __name__=='__main__':main()

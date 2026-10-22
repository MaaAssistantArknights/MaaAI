"""Lossless checks against the actual pinned PaddleOCR training label encoders."""
from pathlib import Path
import sys
import yaml
from game_text import label_issue
from file_utils import sha
from workspace import ROOT, RT

CONFIGS = {
    'v6-ascii': 'configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml',
    'v6': 'configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml',
    'v5-korean': 'configs/rec/PP-OCRv5/multi_language/korean_PP-OCRv5_mobile_rec.yml',
}


def load_encoder(model, max_text_length=None, dictionary_path=None):
    """按模型路线加载实际字典和编码器，确定标签容量。"""
    base = RT/'vendor/PaddleOCR'
    sys.path.insert(0, str(base))
    from ppocr.data.imaug.label_ops import MultiLabelEncode
    config = base/CONFIGS[model]
    cfg = yaml.safe_load(config.read_text())
    op = next(t['MultiLabelEncode'] for t in cfg['Train']['dataset']['transforms'] if 'MultiLabelEncode' in t)
    g = cfg['Global']; dictionary = (ROOT/'configs/ascii_keys.txt') if model=='v6-ascii' else base/g['character_dict_path']
    if dictionary_path is not None:dictionary=Path(dictionary_path).resolve()
    length = g['max_text_length'] if max_text_length is None else max_text_length
    if not isinstance(length,int) or isinstance(length,bool) or length<3: raise ValueError('invalid maximum text length')
    encoder = MultiLabelEncode(length, str(dictionary), g['use_space_char'], **op)
    return encoder, {'config':CONFIGS[model], 'config_sha256':sha(config),
                     'dictionary_sha256':sha(dictionary), 'configured_max_length':length, 'upstream_max_length':g['max_text_length'],
                     'model':model, 'dictionary_path':str(dictionary),
                     'label_ops_sha256':sha(base/'ppocr/data/imaug/label_ops.py')}


def inspect_label(text, encoder, time_steps=None, *, literal_ascii=False):
    """核对标签可无损编码、字符覆盖及 CTC 时间步限制。"""
    problems = []
    issue = label_issue(text)
    # Explicit synthetic literal grammar may contain '<' and '>'; never strip
    # these characters or interpret them as game rich-text markup.
    if literal_ascii and issue=='rich text or placeholder needs game formatter; source preserved' and isinstance(text,str) and all(32<=ord(c)<=126 for c in text):issue=None
    if issue: problems.append('source_text: '+issue)
    if not isinstance(text, str): return {'problems':problems, 'encoded_text':None, 'minimum_ctc_steps':None}
    missing = sorted(set(text)-set(encoder.ctc_encode.dict))
    if missing: problems.append('dictionary_missing: '+repr(''.join(missing)))
    # Paddle may omit unknown characters instead of rejecting a label. Decode
    # both training targets back to text so silent shortening cannot pass audit.
    out = encoder({'image':None, 'label':text})
    encoded = None
    if out is None:
        problems.append('training_encoder_rejected')
    else:
        length = int(out['length'])
        encoded = ''.join(encoder.ctc_encode.character[int(i)] for i in out['label_ctc'][:length])
        auxiliary = ''.join(encoder.gtc_encode.character[int(i)] for i in out['label_gtc'][1:1+length])
        if encoded != text or auxiliary != text: problems.append('encoded_label_differs_from_source')
    # A repeated adjacent character needs an intervening blank in CTC alignment.
    minimum = len(text)+sum(a==b for a,b in zip(text,text[1:]))
    if time_steps is not None:
        if isinstance(time_steps,bool) or not isinstance(time_steps,int) or time_steps < 1:
            problems.append('invalid_ctc_time_steps')
        elif minimum > time_steps: problems.append('insufficient_ctc_time_steps')
    return {'problems':problems, 'encoded_text':encoded, 'minimum_ctc_steps':minimum,
            'checked_ctc_time_steps':time_steps}

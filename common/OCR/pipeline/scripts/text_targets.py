"""Explicit source-to-visible-text contract; never alter source JSON in place."""
import unicodedata
from dataset_contract import split

POLICY='visible-text-v1'

def normalize_text(source):
    """规范游戏文字的兼容字符，同时保留有意义的空格。"""
    # Canonical composition only. Compatibility forms (II/Ⅱ, fullwidth etc.) remain distinct.
    return unicodedata.normalize('NFC',source).replace('\u00a0',' ')

def transform(source):
    """保存原标签和可见标签之间可审计的转换记录。"""
    target=normalize_text(source)
    operations=[]
    if unicodedata.normalize('NFC',source)!=source:operations.append('NFC')
    if '\u00a0' in source:operations.append('NBSP_TO_SPACE')
    return {'source_label':source,'label':target,'text_transform':{'policy':POLICY,'operations':operations}}

def validate_transform(row,source):
    """从原来源重算文字转换，拒绝被改写的标签。"""
    expected=transform(source)
    if row.get('source_label')!=source or row.get('label')!=expected['label'] or row.get('text_transform')!=expected['text_transform']:
        raise ValueError('unapproved or mismatched source-to-visible-text transform')

def partition(source):
    """即使文字规范化改变哈希，也保留原有开发与留出分区。"""
    parts={split(source),split(normalize_text(source))}
    return 'reserved' if 'reserved' in parts else 'dev' if 'dev' in parts else 'train'

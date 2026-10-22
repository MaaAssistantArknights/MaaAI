"""Explicit game text fields and RFC 6901 pointers; no guessed rich-text expansion."""
import re
import unicodedata


def pointer(*parts):
    """将 JSON 路径字段转为可追溯的转义指针。"""
    return ''.join('/'+str(p).replace('~','~0').replace('/','~1') for p in parts)


def resolve_pointer(obj, path):
    """从原始游戏数据按 JSON 指针还原文字来源。"""
    if not path:return obj
    if not path.startswith('/'):raise ValueError('invalid JSON pointer')
    for encoded in path[1:].split('/'):
        if re.search(r'~(?![01])',encoded):raise ValueError('invalid JSON pointer escape')
        part=encoded.replace('~1','/').replace('~0','~')
        if isinstance(obj,list):
            if not re.fullmatch(r'0|[1-9][0-9]*',part):raise ValueError('invalid array index')
            obj=obj[int(part)]
        elif isinstance(obj,dict):obj=obj[part]
        else:raise ValueError('pointer descends through scalar')
    return obj


def labels(table, obj):
    """从游戏表产出场景、语义类别、原文及精确来源指针。"""
    if table in {'character_table','item_table'}:
        items=obj if table=='character_table' else obj['items']
        for key,row in items.items():
            yield ('operator' if table=='character_table' else 'inventory',table,key,row.get('name'),
                   pointer(key,'name') if table=='character_table' else pointer('items',key,'name'))
    elif table=='gacha_table':
        for i,row in enumerate(obj['gachaTags']):
            yield 'recruitment','recruit_tag',str(row['tagId']),row['tagName'],pointer('gachaTags',i,'tagName')
    elif table=='building_data':
        for key,row in obj['buffs'].items():
            yield 'infrastructure','building_skill','buff:'+key,row.get('buffName'),pointer('buffs',key,'buffName')
        for key,row in obj['rooms'].items():
            yield 'infrastructure','building_room','room:'+key,row.get('name'),pointer('rooms',key,'name')
    elif table=='skill_table':
        for key,row in obj.items():
            if row.get('hidden'):continue
            seen=set()
            for i,level in enumerate(row['levels']):
                name=level.get('name')
                if name in seen:continue
                seen.add(name)
                yield 'skill','operator_skill',key+':'+str(i),name,pointer(key,'levels',i,'name')
    else:raise ValueError('unsupported source table: '+table)


def label_issue(text):
    """识别占位符、富文本或不可直接渲染的标签。"""
    if not isinstance(text,str) or not text.strip():return 'empty label'
    if '\n' in text or '\r' in text:return 'multiline needs text-box decomposition; source preserved'
    if text!=text.strip():return 'edge whitespace needs explicit UI handling; source preserved'
    if re.search(r'<[^>]*>|\{[^}]*\}',text):return 'rich text or placeholder needs game formatter; source preserved'
    if any(unicodedata.category(c) in {'Cc','Cf'} for c in text):return 'control/format character needs explicit layout; source preserved'
    if len(text)>128:return 'label exceeds current Unicode renderer capacity; source preserved'
    return None

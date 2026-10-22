"""Extract exact game fields with stable source IDs, hashes and Unicode coverage."""
import argparse,json
from pathlib import Path
from asset_utils import asset_path, cmap
from game_text import labels,label_issue
from routes import LOCALES

LANGUAGES = {locale: language for language, locale in LOCALES.items()}
def build(lock_paths,fonts_path,output):
    """从固定游戏表提取五语言文字并保留来源指针和字体信息。"""
    fonts=json.loads(fonts_path.read_text());rows=[];seen=set()
    for lp in lock_paths:
        for source in json.loads(lp.read_text())['files']:
            path=Path(source['path']);table=path.stem;locale=path.parent.name
            if table not in {'character_table','item_table','gacha_table','building_data','skill_table'} or locale not in LANGUAGES:continue
            fp=asset_path(source);language=LANGUAGES[locale];font=fonts[language];glyphs=cmap(str(asset_path(font)))
            for scene,category,key,text,pointer in labels(table,json.loads(fp.read_text())):
                identity=(language,str(path),pointer)
                if identity in seen:continue
                seen.add(identity)
                rows.append({'id':f'{locale}/{table}/{key}','label':text,'scene':scene,'category':category,'language':language,'locale':locale,'source':{'path':str(path),'sha256':source['sha256'],'json_pointer':pointer},'font':font,'label_issue':label_issue(text),'missing_glyphs':sorted({c for c in text if ord(c) not in glyphs}) if isinstance(text,str) else [],'source_kind':'game_text_only','formal_evaluation_eligible':False})
    if not rows:raise ValueError('no game fields extracted')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return len(rows)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,action='append',required=True);p.add_argument('--fonts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print('Exact fields:',build(a.lock,a.fonts,a.output))

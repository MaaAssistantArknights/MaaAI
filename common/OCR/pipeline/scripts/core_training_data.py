"""Shared deterministic rendering and provenance helpers."""
import copy,hashlib,json
import numpy as np
from functools import lru_cache
from PIL import Image,ImageFilter
from file_utils import sha
from workspace import RT
from asset_utils import asset_path
from PIL import ImageFont, ImageDraw
from asset_utils import cmap

@lru_cache(maxsize=128)
def face(path, size):
    """按固定字体路径与字号取得可缓存的字体对象。"""
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)

def raster(text, style):
    """按指定工作分辨率渲染，不暗中缩字以适应标签。"""
    path = RT/style['font']['path']
    if any(ord(c) not in cmap(str(path)) for c in text):
        raise ValueError('font missing glyph')
    forbidden = {c for pair in style['font'].get('raster_aliases_48px', []) for c in pair if c.islower()}
    if set(text) & forbidden:
        raise ValueError('ambiguous case glyph')
    scale = style['render_scale']; size = round(style['font_size']*scale)
    stroke = round(style['stroke_width']*scale)
    f = face(str(path), size)
    b = f.getbbox(text, stroke_width=stroke)
    pad = 4*scale
    im = Image.new('L', (b[2]-b[0]+2*pad, b[3]-b[1]+2*pad))
    ImageDraw.Draw(im).text((pad-b[0], pad-b[1]), text, font=f, fill=255,
                           stroke_width=stroke, stroke_fill=255)
    im = im.resize((max(1, round(im.width/scale*style['horizontal_scale'])),
                    max(1, round(im.height/scale))), Image.Resampling.BILINEAR)
    # Keep all nonzero alpha, not just a binary mask's bounds.
    bounds = im.getbbox()
    if bounds is None: raise ValueError('empty rendered text')
    return im.crop(bounds)


def pixel_hash(image):
    """按图像尺寸和 RGB 像素计算稳定哈希，用于查重与逐图重放。"""
    rgb=image.convert('RGB')
    return hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()

def write(path,data):
    """将生成配方或审计结果写为可读 JSON。"""
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

@lru_cache(maxsize=128)
def source_sha(path):
    """缓存素材哈希，避免逐图重复读取来源文件。"""
    return sha(path)

def ref(path,pointer):
    """记录来源文件哈希和 JSON 指针，支持逐图追溯。"""
    return {'path':str(path.resolve().relative_to(RT)),'sha256':source_sha(path),'json_pointer':pointer}

def sample_params(family,rng,recipe,part):
    """按场景和分区采样确定性的字体、背景及拍摄扰动。"""
    regular=family not in ['battle_cost','ascii_replay','stage_code']
    return {'style':family,'font_size_delta':rng.choice([-1,0,0,0,1]),'horizontal_scale_delta':rng.choice([-.02,0,0,.02]),
      'padding':[rng.randint(2,6) for _ in range(4)],'capture_scale':rng.choice([.75,1.,1.,1.,1.25]),
      'blur':rng.choice([0.,0.,0.,.2]),'ink':rng.randint(230,255),
      'background':rng.randint(42,56) if regular else rng.randint(12,40),
      'template':rng.choice([i for i in range(len(recipe['templates'])) if (i%5==0)==(part=='dev')]) if family=='depot_quantity' else None,
      'template_offset':[rng.randint(-3,3),rng.randint(-3,3)],'background_multiplier':.40,
      'edge_blur':.25 if family=='depot_quantity' else 0.}

def render(label,params,recipe):
    """根据已记录参数重绘 OCR 裁剪图，支持逐图重放。"""
    style=copy.deepcopy(recipe['styles'][params['style']]);style['font_size']+=params['font_size_delta'];style['horizontal_scale']+=params['horizontal_scale_delta']
    alpha=raster(label,style);left,top,right,bottom=params['padding'];w=alpha.width+left+right;h=alpha.height+top+bottom
    base=Image.new('RGB',(w,h),(params['background'],)*3)
    if params['template'] is not None:
        path=asset_path(recipe['templates'][params['template']]);raw=Image.open(path).convert('RGB').resize((144,144),Image.Resampling.BILINEAR)
        if w>140 or h>70:raise ValueError('quantity larger than supported item component')
        dx,dy=params['template_offset'];x=max(0,min(144-w,144-w-4+dx));y=max(0,min(144-h,125-h+dy))
        base=raw.crop((x,y,x+w,y+h));base=Image.fromarray(np.uint8(np.asarray(base)*params['background_multiplier']))
    mask=Image.new('L',(w,h));mask.paste(alpha,(left,top))
    if params['edge_blur']:mask=mask.filter(ImageFilter.GaussianBlur(params['edge_blur']))
    base.paste((params['ink'],)*3,(0,0,w,h),mask)
    if params['blur']:base=base.filter(ImageFilter.GaussianBlur(params['blur']))
    size=[max(1,round(v*params['capture_scale'])) for v in base.size]
    return base.resize(size,Image.Resampling.BILINEAR)

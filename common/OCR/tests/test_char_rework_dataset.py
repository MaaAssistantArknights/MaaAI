import random

from PIL import Image,ImageDraw

from char_rework_dataset import variant


def test_real_crop_variants_preserve_exact_inputs_and_are_deterministic():
    raw=Image.new('RGB',(20,12),(8,8,8));ImageDraw.Draw(raw).ellipse((2,1,10,10),outline='white',width=2)
    mask=raw.convert('L').point(lambda value:255 if value>=210 else 0)
    exact_raw,trace_raw=variant(raw,mask,0,random.Random(1))
    exact_binary,trace_binary=variant(raw,mask,1,random.Random(1))
    assert exact_raw.tobytes()==raw.tobytes() and trace_raw['form']=='exact_raw'
    assert {value for _,value in exact_binary.convert('L').getcolors()}<={0,255} and trace_binary['form']=='exact_binary'
    a,trace_a=variant(raw,mask,7,random.Random(9));b,trace_b=variant(raw,mask,7,random.Random(9))
    assert a.size==b.size and a.tobytes()==b.tobytes() and trace_a==trace_b

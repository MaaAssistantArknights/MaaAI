import numpy as np
from released_input import ReleasedInput,width_groups
from workspace import RT

def test_logical_batch_keeps_duplicates_and_every_sample_without_cross_width_padding():
    order=[3,1,3,2,0,4];widths=[320,401,320,641,401]
    groups=width_groups(order,widths)
    from collections import Counter
    assert Counter(i for g in groups for i in g)==Counter(order)
    assert all(len({widths[i] for i in g})==1 for g in groups)
    assert len(groups)==3

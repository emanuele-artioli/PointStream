import numpy as np
import pytest
from experiments.tier.receiver_replay import psnr

def test_exact_frame_denominator():
    source = np.full((3,4,5,3),100,dtype=np.uint8)
    with pytest.raises(ValueError,match='incomplete output'):
        psnr(source, source[:2])

def test_mean_frame_and_pooled_are_distinct():
    source=np.full((2,4,5,3),100,dtype=np.uint8)
    decoded=source.copy();decoded[0]+=3;decoded[1]+=10
    result=psnr(source,decoded)
    assert result['frames']==2
    assert result['mean_frame_y_psnr_db']>result['pooled_y_psnr_db']

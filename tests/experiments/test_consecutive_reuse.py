from fractions import Fraction
import numpy as np
from experiments.background.consecutive_reuse import schedule, score

def test_fractional_fps_refresh_does_not_drift():
    assert schedule(600, Fraction(60000,1001), 1)==[0,60,120,180,240,300,360,420,480,539,599]

def test_resets_deduplicate_and_stop_at_end():
    assert schedule(100, 10, 5, [0, 25, 50, 100, 101]) == [0,25,50]

def test_frame_mean_and_pooled_are_distinct():
    source=np.zeros((2,2,2),dtype=np.uint8)
    decoded=np.array([np.ones((2,2)),np.ones((2,2))*10],dtype=np.uint8)
    result=score(source,decoded)
    assert result['mse_per_frame']==[1,100]
    assert result['pooled_mse']==50.5
    assert result['mean_frame_psnr_db']>result['pooled_psnr_db']

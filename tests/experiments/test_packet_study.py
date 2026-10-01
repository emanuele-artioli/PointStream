import numpy as np
import pytest
from experiments.packet_study.run import quality


def test_quality_keeps_pooled_and_frame_mean_distinct():
    source=np.full((2,2,2,3),128,np.uint8)
    decoded=source.copy();decoded[0]-=1;decoded[1]-=20
    scores=quality(source,decoded)
    assert scores['frames']==2
    assert scores['pooled_y_psnr_db'] != scores['mean_frame_y_psnr_db']
    assert scores['pooled_mse_y']==np.mean(scores['mse_per_frame_y'])


def test_quality_rejects_missing_registered_outputs():
    source=np.zeros((3,2,2,3),np.uint8)
    with pytest.raises(ValueError,match='registered'):
        quality(source,source[:2])

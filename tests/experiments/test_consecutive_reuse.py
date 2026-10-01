from fractions import Fraction
import numpy as np
from experiments.background.consecutive_reuse import schedule, score
import json
import hashlib
import pytest
from experiments.background.manifest_receiver import decode_manifest
from experiments.background.reuse_summary import check_quality

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

def test_receiver_refuses_gap_before_any_decode(tmp_path):
    p=tmp_path/'package.json'
    p.write_text(json.dumps({'geometry':[2,2],'frames':2,'packets':[{'frame':1,'hold_until':2}]}))
    with pytest.raises(ValueError,match='placement'):decode_manifest(p)

def test_receiver_refuses_changed_charged_payload(tmp_path):
    stream=tmp_path/'native.ivf';stream.write_bytes(b'changed')
    p=tmp_path/'package.json'
    p.write_text(json.dumps({'geometry':[2,2],'frames':1,'packets':[{'frame':0,'hold_until':1,
        'stream':{'path':str(stream),'bytes':7,'sha256':hashlib.sha256(b'original').hexdigest()}}]}))
    with pytest.raises(ValueError,match='identity'):decode_manifest(p)

def test_receiver_refuses_incomplete_coverage(tmp_path):
    p=tmp_path/'package.json';p.write_text(json.dumps({'geometry':[2,2],'frames':2,'packets':[]}))
    with pytest.raises(ValueError,match='coverage'):decode_manifest(p)

def test_summary_refuses_dropped_frame_and_changed_score():
    source=np.zeros((2,2,2),dtype=np.uint8)
    decoded=np.array([np.ones((2,2)),np.ones((2,2))*10],dtype=np.uint8)
    quality=score(source,decoded)
    check_quality(quality,2)
    with pytest.raises(ValueError,match='denominator'):check_quality(quality,3)
    quality['mean_frame_psnr_db']+=0.01
    with pytest.raises(ValueError,match='PSNR'):check_quality(quality,2)

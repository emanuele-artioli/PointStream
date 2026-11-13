"""Local contract tests; no CUDA models or native workload is run."""
from types import SimpleNamespace

import pytest

from experiments.tennis import k5d_svt_profile as profile, k5d_encoder as enc


def test_named_three_point_profile_cannot_replace_default():
    original = enc.POINTS
    assert profile.select(original,None) is original
    assert profile.select(original,profile.PROFILE) == (('svt',48),('svt',55),('svt',62))
    assert len(enc.POINTS) == 6
    with pytest.raises(ValueError,match='unknown'):
        profile.select(original,'three-points')


def test_resource_snapshot_rejects_missing_memory_and_each_allowance():
    good = {'driver_used_bytes':3*2**30,'reserved_bytes':2*2**30,'peak_reserved_bytes':2*2**30,'allocated_bytes':2**30}
    profile.check_snapshot(good)
    for key,value in (('driver_used_bytes',11*2**30),('peak_reserved_bytes',9*2**30),('reserved_bytes',0),('allocated_bytes',None)):
        with pytest.raises(ValueError):
            profile.check_snapshot({**good,key:value})


def test_actual_memory_requires_sole_campaign_gpu_process(monkeypatch):
    monkeypatch.setattr(profile.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='GPU-x, 7, 2048\nGPU-x, 8, 100\n'))
    torch = SimpleNamespace(cuda=SimpleNamespace(memory_reserved=lambda _:2**30,max_memory_reserved=lambda _:2**30,memory_allocated=lambda _:1))
    setup={'gpu_uuid':'GPU-x','host_pid':7,'campaign_pgid':70}
    monkeypatch.setattr(profile.os,'getpgid',lambda _:70)
    with pytest.raises(ValueError,match='second campaign'):
        profile.snapshot(torch,setup,'phase')
    monkeypatch.setattr(profile.os,'getpgid',lambda _:80)
    assert profile.snapshot(torch,setup,'phase')['driver_used_bytes'] == 2*2**30
    monkeypatch.setattr(profile.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='GPU-x, 8, 100\n'))
    with pytest.raises(ValueError,match='missing'):
        profile.snapshot(torch,setup,'phase')


def test_subset_fit_matches_same_operating_points_in_six_decode_batch():
    import numpy as np
    from experiments.tennis import k5c_independent_ball as old
    source = np.full((1,64,64,3),100,dtype=np.uint8)
    source[0,29:35,29:35] = 240
    decodes = [np.full_like(source,v) for v in (90,95,100,105,110,115)]
    track=[(32,32)]
    all_points = old.fit_streak(track,source,decodes,(64,64))
    subset = old.fit_streak(track,source,decodes[3:],(64,64))
    assert subset == all_points[3:]

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


def test_asymmetric_moving_fixture_preserves_svt_payload_in_subset():
    """A well-conditioned example, not universal float32 batch-width invariance.

    The earlier flat square fixture had near-tied positions/appearances and
    changed on Linux BLAS when D6 became D3. Production reference comparisons
    stay exact and must reject any such mismatch on actual scientific inputs.
    """
    import numpy as np
    from experiments.tennis import k5c_independent_ball as old
    from src.pointstream import ball

    yy, xx = np.mgrid[:64, :64]
    background = np.stack((20 + xx // 4 + yy // 8, 12 + xx // 7, 8 + yy // 5), -1).astype(np.uint8)
    base = np.stack([background] * 3)
    track = [(18, 21), (25, 24), (31, 29)]
    appearances = [ball.Appearance((210, 130, 65), 3, 2, 1)] * 3
    velocity = ball.velocities(track)
    source = np.stack([ball.render(base[t], t, track, velocity, appearances) for t in range(3)])
    decodes = [np.clip(base.astype(np.int16) + offset, 0, 255).astype(np.uint8)
               for offset in (-3, -1, 0, 1, 2, 3)]
    all_points = old.fit_streak(track, source, decodes, (64, 64))
    subset = old.fit_streak(track, source, decodes[3:], (64, 64))
    for actual, expected in zip(subset, all_points[3:], strict=True):
        assert all(p is not None for p in actual['track'])
        assert all(a is not None for a in actual['appearances'])
        # delta_se is a nonserialized floating diagnostic, not a universal
        # cross-BLAS equality promise. Check exact transmitted semantics.
        assert ball.encode_frames(actual['track'], actual['appearances'], 64, 64) == ball.encode_frames(
            expected['track'], expected['appearances'], 64, 64)

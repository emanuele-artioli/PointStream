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


def admission_fixture(monkeypatch, *, initialized=False, cap_error=False):
    """Track real-allocation API ordering without importing CUDA or models."""
    from experiments.tennis import k5d_scoped_namespace as scoped

    events = []
    basis = profile.BASIS
    sharing = {'mode': 'shared', 'basis': basis, 'margin_mib': 8192}
    contention = {'policy': 'continue', 'basis': basis, 'resume_attempts': 0}
    spec = {'cpu_threads': 1, 'single_cpu_campaign': True,
            'k5d_evaluation_profile': profile.PROFILE, 'hosts': ['gpu6'],
            'gpu_models': ['RTX 6000 Ada'], 'gpu_memory_mib': 10240,
            'gpu_sharing': sharing, 'contention': contention}
    dispatch = {'gpu_sharing': sharing, 'contention_policy': contention,
                'environment': {'gpu': {'uuid': 'GPU-x'}}}
    env = {'CUDA_VISIBLE_DEVICES': 'GPU-x', 'PS_CLAIMED_GPU_UUID': 'GPU-x',
           'PS_GPU_SHARING': 'shared', 'PS_GPU_MEMORY_LIMIT_MIB': '10240',
           'PS_CPU_ALLOWANCE': '1', 'PS_STAGE': 'smoke', 'PS_STAGE_DIR': '/stage/smoke'}
    monkeypatch.setattr(profile.socket, 'gethostname', lambda: 'gpu6')
    monkeypatch.setattr(profile.os, 'sched_getaffinity', lambda _: {0}, raising=False)
    monkeypatch.setattr(scoped, 'host_group_identity', lambda _: {
        'host_pid': 7, 'host_pgid': 70, 'campaign_pgid': 70, 'start_ticks': 99})

    def cap(fraction, device):
        events.append(('cap', fraction, device))
        if cap_error:
            raise RuntimeError('cap failed')

    def empty(*shape, **kwargs):
        assert events and events[0][0] == 'cap'
        events.append(('allocate', shape, kwargs))
        return object()

    cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _: 'NVIDIA RTX 6000 Ada Generation',
        get_device_capability=lambda _: (8, 9),
        memory_allocated=lambda _: int(initialized), memory_reserved=lambda _: 0,
        max_memory_reserved=lambda _: 0,
        get_device_properties=lambda _: SimpleNamespace(total_memory=48*2**30),
        set_per_process_memory_fraction=cap,
        synchronize=lambda device: events.append(('synchronize', device)),
        reset_peak_memory_stats=lambda device: events.append(('reset_peak', device)))
    torch = SimpleNamespace(cuda=cuda, uint8=object(), empty=empty)
    return spec, dispatch, env, torch, events


def test_context_allocation_is_capped_and_synchronized_before_memory_receipt(monkeypatch):
    spec, dispatch, env, torch, events = admission_fixture(monkeypatch)
    setup = profile.admit(spec, dispatch, env, torch)
    assert [event[0] for event in events] == ['cap', 'allocate', 'synchronize', 'reset_peak']
    assert events[0] == ('cap', 8192/49152, 0)
    assert events[1][1:] == ((1,), {'dtype': torch.uint8, 'device': 'cuda:0'})
    assert setup['allocator_before'] == 0 and setup['claim_mib'] == 10240
    monkeypatch.setattr(profile.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=''))
    with pytest.raises(ValueError, match='actual GPU memory receipt missing'):
        profile.snapshot(torch, setup, 'before_models')


def test_existing_allocator_use_is_rejected_before_initialization(monkeypatch):
    spec, dispatch, env, torch, events = admission_fixture(monkeypatch, initialized=True)
    with pytest.raises(ValueError, match='before the cap'):
        profile.admit(spec, dispatch, env, torch)
    assert events == []


def test_context_allocation_never_precedes_successful_cap(monkeypatch):
    spec, dispatch, env, torch, events = admission_fixture(monkeypatch, cap_error=True)
    with pytest.raises(RuntimeError, match='cap failed'):
        profile.admit(spec, dispatch, env, torch)
    assert [event[0] for event in events] == ['cap']


def test_failed_context_allocation_does_not_return_admission(monkeypatch):
    spec, dispatch, env, torch, events = admission_fixture(monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError('CUDA allocation failed')
    torch.empty = fail
    with pytest.raises(RuntimeError, match='CUDA allocation failed'):
        profile.admit(spec, dispatch, env, torch)
    assert [event[0] for event in events] == ['cap']


def test_context_synchronization_failure_is_not_a_memory_receipt(monkeypatch):
    spec, dispatch, env, torch, events = admission_fixture(monkeypatch)
    def fail(device):
        raise RuntimeError('CUDA synchronization failed')
    torch.cuda.synchronize = fail
    with pytest.raises(RuntimeError, match='CUDA synchronization failed'):
        profile.admit(spec, dispatch, env, torch)
    assert [event[0] for event in events] == ['cap', 'allocate']

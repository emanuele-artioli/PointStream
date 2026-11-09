from types import SimpleNamespace

import pytest

from experiments.tennis import k5d_real_parity as parity
from src.codecs import dcvc_uf_worker as dcvc


def claim():
    policy = {'mode': 'shared', 'basis': parity.SHARED_BASIS, 'margin_mib': 4096}
    spec = {'k5d_shared_parity_profile': parity.SHARED_PROFILE, 'gpu_memory_mib': 16384,
            'gpu_models': ['RTX A6000'], 'gpu_sharing': policy,
            'contention': {'policy': 'continue', 'basis': parity.SHARED_BASIS, 'resume_attempts': 0}}
    env = {'PS_GPU_SHARING': 'shared', 'PS_GPU_MEMORY_LIMIT_MIB': '16384'}
    return spec, {'gpu_sharing': policy.copy()}, env


def fake_torch(*, allocated=0, total=48 * 1024**3, name='NVIDIA RTX A6000', capability=(8, 6)):
    calls = []
    cuda = SimpleNamespace(memory_allocated=lambda _index: allocated,
                           get_device_properties=lambda _index: SimpleNamespace(total_memory=total),
                           get_device_name=lambda _index: name,
                           get_device_capability=lambda _index: capability,
                           set_per_process_memory_fraction=lambda fraction, index: calls.append((fraction, index)))
    return SimpleNamespace(cuda=cuda), calls


def test_shared_claim_requires_exact_opt_in_and_unchanged_default():
    spec, dispatch, env = claim()
    parity.check_resource_profile(spec, dispatch, requested=True, environ=env)
    with pytest.raises(ValueError):
        parity.check_resource_profile(spec, dispatch, requested=False, environ=env)
    parity.check_resource_profile({}, {}, requested=False, environ={})
    for change in ({'PS_GPU_MEMORY_LIMIT_MIB': '8192'}, {'PS_GPU_SHARING': 'exclusive'}):
        with pytest.raises(ValueError):
            parity.check_resource_profile(spec, dispatch, requested=True, environ={**env, **change})
    with pytest.raises(ValueError):
        parity.check_resource_profile({**spec, 'contention': {'policy': 'stop'}}, dispatch, requested=True, environ=env)
    with pytest.raises(ValueError):
        parity.check_resource_profile(spec, {'gpu_sharing': {'mode': 'exclusive'}}, requested=True, environ=env)


def test_two_allocators_capped_before_allocation_and_child_env_propagates():
    spec, _, env = claim()
    del spec
    env['PS_K5D_SHARED_NUMERICAL'] = parity.SHARED_PROFILE
    saved = {'OTHER': 'preserved'}
    parity.configure_child_environment(saved, env)
    assert saved == {**env, 'OTHER': 'preserved'}
    parent, parent_calls = fake_torch()
    child, child_calls = fake_torch()
    parity.apply_parent_limit(parent)
    dcvc._apply_optional_shared_parity_limit(child, saved)
    assert parent_calls == child_calls == [(8192 / (48 * 1024), 0)]
    with pytest.raises(ValueError):
        parity.configure_child_environment({}, {**env, 'PS_GPU_SHARING': 'exclusive'})


def test_child_default_and_invalid_caps_fail_closed():
    torch, calls = fake_torch()
    dcvc._apply_optional_shared_parity_limit(torch, {})
    assert not calls
    _, _, env = claim()
    env['PS_K5D_SHARED_NUMERICAL'] = parity.SHARED_PROFILE
    for kwargs in ({'allocated': 1}, {'total': 8 * 1024**3}, {'name': 'wrong'}, {'capability': (8, 9)}):
        torch, _ = fake_torch(**kwargs)
        with pytest.raises(RuntimeError):
            dcvc._apply_optional_shared_parity_limit(torch, env)
    with pytest.raises(RuntimeError):
        dcvc._apply_optional_shared_parity_limit(fake_torch()[0], {**env, 'PS_GPU_MEMORY_LIMIT_MIB': '8192'})
    with pytest.raises(ValueError):
        parity.apply_parent_limit(fake_torch(allocated=1)[0])

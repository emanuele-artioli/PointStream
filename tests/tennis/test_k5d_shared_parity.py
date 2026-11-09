from types import SimpleNamespace

import pytest

from experiments.tennis import k5d_real_parity as parity
from src.codecs import dcvc_uf_worker as dcvc


def claim():
    policy = {'mode': 'shared', 'basis': parity.SHARED_BASIS, 'margin_mib': 8192}
    spec = {'k5d_shared_parity_profile': parity.SHARED_PROFILE, 'gpu_memory_mib': 16384,
            'gpu_models': ['RTX A6000'], 'gpu_sharing': policy,
            'contention': {'policy': 'continue', 'basis': parity.SHARED_BASIS, 'resume_attempts': 0}}
    env = {'PS_GPU_SHARING': 'shared', 'PS_GPU_MEMORY_LIMIT_MIB': '16384',
           'CUDA_VISIBLE_DEVICES': 'GPU-test', 'PS_CLAIMED_GPU_UUID': 'GPU-test'}
    return spec, {'gpu_sharing': policy.copy(), 'contention_policy': spec['contention'].copy(), 'environment': {'gpu': {'uuid': 'GPU-test'}}}, env


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


def test_two_allocators_capped_before_allocation_and_child_env_propagates(tmp_path, monkeypatch):
    spec, _, env = claim()
    del spec
    env['PS_K5D_SHARED_NUMERICAL'] = parity.SHARED_PROFILE
    from pathlib import Path
    from experiments.tennis import k5d_scoped_namespace
    env['PS_SCRATCH_DIR'] = str(tmp_path)
    env['PS_STAGE_DIR'] = str(tmp_path / 'smoke')
    monkeypatch.setenv('PS_STAGE_DIR', env['PS_STAGE_DIR'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-test')
    host = {'host_pid': 101, 'host_pgid': 77, 'start_ticks': 1001, 'campaign_pgid': 77}
    monkeypatch.setattr(k5d_scoped_namespace, 'host_group_identity', lambda _path: host)
    original_read = Path.read_text
    def fake_read(path, *args, **kwargs):
        if str(path) == '/proc/self/stat':
            return '102 (worker) ' + ' '.join(['S', '1', '77'] + ['0'] * 16 + ['1002'])
        if path.name == 'status.json':
            return '{\"process_group_id\": 77}'
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', fake_read)
    receipt_dir = tmp_path / 'shared-resource-receipts'
    receipt_dir.mkdir()
    env['PS_K5D_RESOURCE_RECEIPTS'] = str(receipt_dir)
    saved = {'OTHER': 'preserved'}
    parity.configure_child_environment(saved, env)
    assert saved == {key: value for key, value in env.items() if key not in ('PS_SCRATCH_DIR', 'PS_STAGE_DIR')} | {'OTHER': 'preserved'}
    saved['PS_SCRATCH_DIR'] = str(tmp_path)
    saved['PS_STAGE_DIR'] = env['PS_STAGE_DIR']
    parent, parent_calls = fake_torch()
    child, child_calls = fake_torch()
    parent_receipt = parity.apply_parent_limit(parent)
    import json
    (receipt_dir / 'parent.json').write_text(json.dumps(parent_receipt))
    plan = tmp_path / 'prepared-work/dcvc-0/plan.json'
    plan.parent.mkdir(parents=True)
    plan.write_text('{}')
    import sys
    monkeypatch.setattr(sys, 'argv', ['worker', 'encode', '--plan', str(plan), '--report', 'report'])
    dcvc._apply_optional_shared_parity_limit(child, saved)
    child_file = next(receipt_dir.glob('child-*.json'))
    record = json.loads(child_file.read_text())
    with pytest.raises(ValueError, match='cover every'):
        parity.validate_resource_receipts(receipt_dir)
    expected = {(action, f'prepared-work/dcvc-{point}/plan.json')
                for kind, point in parity.enc.POINTS if kind == 'dcvc' for action in ('encode', 'decode')}
    expected.update(('decode', f'coalesced-dcvc-{point}/receiver-plan.json')
                    for kind, point in parity.enc.POINTS if kind == 'dcvc')
    expected.update(('decode', f'prepared-work/receiver-dcvc-{point}/receiver-plan.json')
                    for kind, point in parity.enc.POINTS if kind == 'dcvc')
    expected.remove(('encode', 'prepared-work/dcvc-0/plan.json'))
    for index, (action, path) in enumerate(sorted(expected), 10000):
        value = {**record, 'pid': index, 'host_pid': index, 'start_ticks': index,
                 'action': action, 'plan': path}
        (receipt_dir / f'child-{index}.json').write_text(json.dumps(value))
    parity.validate_resource_receipts(receipt_dir)
    parent_receipt['budget_mib'] = 16384
    (receipt_dir / 'parent.json').write_text(json.dumps(parent_receipt))
    with pytest.raises(ValueError):
        parity.validate_resource_receipts(receipt_dir)
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

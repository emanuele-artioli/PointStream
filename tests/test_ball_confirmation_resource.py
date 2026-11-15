import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.tennis import ball_confirmation as confirmation
from src.codecs import dcvc_uf_worker


@pytest.mark.parametrize(('name', 'capability'), [
    ('NVIDIA RTX A6000', (8, 6)), ('NVIDIA RTX 6000 Ada Generation', (8, 9)),
])
def test_scoped_child_profile_binds_process_and_device(tmp_path, monkeypatch, name, capability):
    calls = []
    cuda = SimpleNamespace(get_device_name=lambda _: name, get_device_capability=lambda _: capability,
        memory_allocated=lambda _: 0, get_device_properties=lambda _: SimpleNamespace(total_memory=48 * 1024**3),
        set_per_process_memory_fraction=lambda fraction, index: calls.append((fraction, index)))
    plan = tmp_path / 'dcvc-0' / 'plan.json'
    plan.parent.mkdir()
    plan.write_text('{}')
    receipts = tmp_path / 'shared-resource-receipts'
    receipts.mkdir()
    env = {'PS_K5D_SHARED_NUMERICAL': confirmation.GPU_PROFILE, 'PS_GPU_SHARING': 'shared',
           'PS_GPU_MEMORY_LIMIT_MIB': '12288', 'CUDA_VISIBLE_DEVICES': 'GPU-test',
           'PS_CLAIMED_GPU_UUID': 'GPU-test', 'PS_SCRATCH_DIR': str(tmp_path),
           'PS_STAGE_DIR': str(tmp_path / 'full'), 'PS_K5D_RESOURCE_RECEIPTS': str(receipts)}
    read = Path.read_text
    def read_control(path, *args, **kwargs):
        if str(path) == '/proc/self/stat':
            return '102 (worker) ' + ' '.join(['S', '1', '77'] + ['0'] * 16 + ['1002'])
        if path.name == 'status.json':
            return '{"process_group_id": 77}'
        return read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read_control)
    monkeypatch.setattr(sys, 'argv', ['worker', 'encode', '--plan', str(plan), '--report', 'report'])
    dcvc_uf_worker._apply_optional_shared_parity_limit(SimpleNamespace(cuda=cuda), env)
    record = json.loads(next(receipts.glob('child-*.json')).read_text())
    assert calls == [(6144 / (48 * 1024), 0)]
    assert (record['profile'], record['budget_mib'], record['device_name'], record['capability']) == (
        confirmation.GPU_PROFILE, 6144, name, list(capability))
    with pytest.raises(RuntimeError):
        dcvc_uf_worker._apply_optional_shared_parity_limit(SimpleNamespace(cuda=cuda),
                                                          {**env, 'PS_GPU_MEMORY_LIMIT_MIB': '16384'})


def test_resource_receipts_require_all_six_matching_actions(tmp_path):
    root = tmp_path
    directory = root / 'resource-receipts'
    directory.mkdir()
    total = 48 * 1024**3
    parent = {'profile': confirmation.GPU_PROFILE, 'role': 'parent', 'gpu_uuid': 'GPU-a',
              'device_name': 'NVIDIA RTX A6000', 'capability': [8, 6], 'budget_mib': 4096,
              'total_bytes': total, 'fraction': 4096 * 1024**2 / total, 'allocated_before': 0,
              'host_pgid': 77, 'campaign_pgid': 77, 'start_ticks': 99}
    (directory / 'parent.json').write_text(json.dumps(parent))
    receipt = {'cuda': {'uuid': 'GPU-a', 'name': parent['device_name'], 'profile': confirmation.GPU_PROFILE,
                        'parent_allocator_cap_mib': 4096,
                        'child_allocator_cap_mib': 6144, 'parent_peak_reserved_bytes': 100}}
    assert not confirmation.validate_resource_receipts(root, receipt)

    for index, (point, action) in enumerate(((p, a) for p in (0, 21, 42) for a in ('encode', 'decode')), 1):
        child = {**parent, 'role': 'dcvc', 'budget_mib': 6144, 'fraction': 6144 * 1024**2 / total,
                 'pid': index, 'host_pid': index, 'start_ticks': index, 'action': action,
                 'plan': f'confirmation-bases/publish/bases/dcvc-{point}/plan.json'}
        (directory / f'child-{index}.json').write_text(json.dumps(child))
        destination = root / f'bases/dcvc-{point}'
        destination.mkdir(parents=True, exist_ok=True)
        report = {'peak_allocated_mib': 100, 'environment': {'device_name': parent['device_name'],
            'capability': [8, 6], 'cuda_visible_devices': 'GPU-a', 'torch': '2.10.0+cu128',
            'torch_cuda': '12.8', 'extension': {'variant': 'sm80'}}}
        (destination / f'{action}.json').write_text(json.dumps(report))
    assert confirmation.validate_resource_receipts(root, receipt)
    report['peak_allocated_mib'] = 6145
    (destination / f'{action}.json').write_text(json.dumps(report))
    assert not confirmation.validate_resource_receipts(root, receipt)


def test_acceptance_joins_canonical_full_job_and_actual_artifact_bytes():
    import hashlib

    def encode(value):
        return json.dumps(value, sort_keys=True).encode()
    job = '20261115T000000Z-1234abcd'
    stage = {'phase': 'bases', 'source_id': 'match209_000', 'frames': 64,
             'archive': {'sha256': 'c' * 64}}
    stage_raw = encode(stage)
    dispatch = {'command': ['python', '--phase', 'bases', '--source-id', 'match209_000', '--frames', '64'],
                'code': {'source_sha256': 'a' * 64}, 'staged_environment': {'sha256': confirmation.RUNTIME_SHA}}
    validation = {'passed': True, 'checks': {'native': True}, 'phase': 'bases', 'source_id': 'match209_000',
                  'frames': 64, 'stage_sha256': hashlib.sha256(stage_raw).hexdigest(),
                  'archive_sha256': 'c' * 64, 'dispatch_canonical_sha256': confirmation.canonical_sha(dispatch)}
    status = {'job_id': job, 'status': 'complete', 'gate': {'passed': True, 'source_sha256': 'a' * 64}}
    execution_raw = encode({'exit_code': 0, 'seconds': 100.})
    result = confirmation.accepted_full(job, status, stage_raw, encode(validation), execution_raw,
                                        encode(dispatch), 'c' * 64)
    assert result['canonical_complete'] and result['execution_sha256'] == hashlib.sha256(execution_raw).hexdigest()
    for bad_status in ({**status, 'job_id': '20261115T000000Z-deadbeef'}, {**status, 'status': 'running'}):
        with pytest.raises(ValueError):
            confirmation.accepted_full(job, bad_status, stage_raw, encode(validation), execution_raw,
                                       encode(dispatch), 'c' * 64)
    with pytest.raises(ValueError):
        confirmation.accepted_full(job, status, stage_raw, encode({**validation, 'frames': 32}),
                                   execution_raw, encode(dispatch), 'c' * 64)

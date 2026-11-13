import json
from pathlib import Path
import resource
import subprocess
import sys

import pytest

from experiments.jobs.inbox import validate_spec
from experiments.tennis import w8_strict_packet_job as job
from experiments.tennis import w8_strict_packet_worker as packet_worker


def test_real_spec_has_only_public_cpu_inputs():
    spec = job.make_spec('2026-11-05T03:58:00Z')
    validate_spec(spec, now=1793848684)
    assert spec['device'] == 'cpu'
    assert spec['hosts'] == ['gpu5']
    assert spec['budget_seconds'] == 240
    assert spec['full']['seconds'] == 10
    assert spec['validator_seconds'] == 40
    assert {i['name']: i['sha256'] for i in spec['staged_inputs']} == job.NAMES
    assert spec['arguments'][0] == 'packet-run'
    assert spec['validator'][-1] == 'packet-validate'


def test_exact_staged_input_binding_rejects_extra_and_tamper(tmp_path, monkeypatch):
    root = tmp_path / 'local'
    root.mkdir()
    stage = tmp_path / 'smoke'
    stage.mkdir()
    names = {}
    staged = []
    for name in job.NAMES:
        path = root / name
        path.write_text(name)
        digest = job.fixed.file_sha(path)
        names[name] = digest
        staged.append({'name': name, 'path': str(path), 'sha256': digest})
    monkeypatch.setattr(job, 'NAMES', names)
    spec = {'device': 'cpu', 'gpu_models': [], 'gpu_memory_mib': 0, 'single_cpu_campaign': True,
            'cpu_threads': 1, 'hosts': ['gpu5'], 'budget_seconds': 240,
            'environment': {'sha256': job.cost.MAIN_ENV}, 'staged_inputs': staged,
            'inputs': [{'path': x['path'], 'sha256': x['sha256']} for x in staged]}
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    (stage / 'staging.json').write_text(json.dumps({'environment': {'sha256': job.cost.MAIN_ENV},
                                                    'local_root': str(root), 'inputs': staged}))
    (stage / 'dispatch.json').write_text(json.dumps({'cpu_affinity': [0]}))
    assert set(job.bound_inputs(stage)[0]) == set(names)
    spec['inputs'].append({'path': '/undeclared-model', 'sha256': '0' * 64})
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='declared input'):
        job.bound_inputs(stage)
    spec['inputs'].pop()
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    (root / 'packet').write_text('tampered')
    with pytest.raises(ValueError, match='staged public'):
        job.bound_inputs(stage)


def test_serialization_rejects_tampered_unit(tmp_path, monkeypatch):
    units = {name: name.encode() for name in job.fixed.UNITS}
    digests = {}
    for name, contents in units.items():
        path = tmp_path / f'{name}.npz'
        path.write_bytes(contents)
        digests[name] = job.fixed.file_sha(path)
    monkeypatch.setattr(job.receiver, 'ORACLE_UNIT_SHA', digests)
    fingerprints = {'tensors': list(range(480))}
    monkeypatch.setattr(job.receiver, 'verify_serialized_match', lambda path, expected: job.CHECKPOINT_SHA if expected == fingerprints else '')
    packet = {'packet_sha256': job.fixed.WINDOW_SHA, 'profile_sha256': job.fixed.PROFILE_SHA,
              'guider_sha256': job.fixed.GUIDER_SHA, 'match_tensors': fingerprints,
              'unit_sha256': digests, 'checkpoint_sha256': job.CHECKPOINT_SHA}
    job.check_serialized(tmp_path, packet, fingerprints)
    (tmp_path / f'{next(iter(units))}.npz').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='serialization'):
        job.check_serialized(tmp_path, packet, fingerprints)


def test_packet_cpu_limit_can_raise_soft_limit_in_child():
    code = '''import resource, subprocess, sys
resource.setrlimit(resource.RLIMIT_CPU, (40, 100))
def limit(): resource.setrlimit(resource.RLIMIT_CPU, (100, 100))
subprocess.run([sys.executable, '-c', "import resource; assert resource.getrlimit(resource.RLIMIT_CPU) == (100,100)"], preexec_fn=limit, check=True)
'''
    if not hasattr(resource, 'RLIMIT_CPU'):
        pytest.skip('CPU resource limit unavailable')
    subprocess.run([sys.executable, '-c', code], check=True)


def test_worker_rejects_bad_scope_before_launch(tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'worker_kind': 'cost_probe', 'environment': {'CUDA_VISIBLE_DEVICES': ''}}))
    monkeypatch.setattr(sys, 'argv', ['worker', '--config', str(config), '--config-sha256', job.fixed.file_sha(config),
                                   '--loader', str(config), '--loader-sha256', job.fixed.file_sha(config),
                                   '--canary', str(tmp_path), '--source-root', str(tmp_path)])
    monkeypatch.setattr(packet_worker.receiver, 'launch', lambda *a, **kw: pytest.fail('launched'))
    with pytest.raises(ValueError):
        packet_worker.main()


def test_isolation_requires_all_denials_mutations_and_exact_source(monkeypatch):
    monkeypatch.setattr(job.policy, 'validate_leaf_receipt', lambda leaves, loader: None)
    paths = [f'/protected/{i}' for i in range(10)]
    config = {'worker_kind': 'packet', 'environment': {'CUDA_VISIBLE_DEVICES': ''},
              'denied_paths': {'all': paths}, 'parent_user_namespace': 'user:parent',
              'parent_network_namespace': 'net:parent'}
    module = 'experiments.tennis.w8_receiver_generate'
    relative = module.replace('.', '/') + '.py'
    receipt = {'denials': dict.fromkeys(paths, 13),
               'mutation_denials': dict.fromkeys(('write', 'truncate', 'create', 'remove', 'mkdir', 'rename', 'link'), 13),
               'landlock_abi': 3, 'handled_mask': job.policy.handled_rights(3),
               'policy': 'w8-approved-system-leaves-mutation-v1', 'network_interfaces': ['lo'],
               'pre_sandbox_threads': 1, 'extra_fds': False, 'output_write_ok': True,
               'cpu_bootstrap': {'cuda_initialized_before_isolation': False,
                                 'before_landlock_sha256': 'a', 'after_landlock_sha256': 'a'},
               'namespaces': {'user': 'user:child', 'net': 'net:child'}, 'loader_identity_sha256': 'loader',
               'approved_leaves': [], 'source_origins': {module: relative}}
    manifest = {'files': {relative: 'digest', 'src/other.py': 'digest'}}
    job.check_isolation(config, receipt, {}, manifest, 'loader')
    receipt['denials'].pop(paths[0])
    with pytest.raises(ValueError, match='isolation'):
        job.check_isolation(config, receipt, {}, manifest, 'loader')
    receipt['denials'][paths[0]] = 13
    receipt['mutation_denials']['write'] = 0
    with pytest.raises(ValueError, match='mutation'):
        job.check_isolation(config, receipt, {}, manifest, 'loader')
    receipt['mutation_denials']['write'] = 13
    receipt['source_origins'][module] = 'src/other.py'
    with pytest.raises(ValueError, match='origins'):
        job.check_isolation(config, receipt, {}, manifest, 'loader')


def test_pre_serialization_rejects_fake_cuda_and_meta():
    class Tensor:
        def __init__(self, kind):
            self.device = type('Device', (), {'type': kind})()
    state = {'reference_unet': {str(i): Tensor('cpu') for i in range(160)},
             'denoising_unet': {str(i): Tensor('cpu') for i in range(160)},
             'pose_guider': {str(i): Tensor('cpu') for i in range(160)}}
    assert packet_worker.assert_cpu_state(state, lambda x: isinstance(x, Tensor)) == 480
    for kind in ('cuda', 'meta'):
        state['pose_guider']['0'] = Tensor(kind)
        with pytest.raises(ValueError, match='non-CPU'):
            packet_worker.assert_cpu_state(state, lambda x: isinstance(x, Tensor))
    with pytest.raises(ValueError, match='non-CPU'):
        packet_worker.assert_cpu_units({'a': {'bad': Tensor('cuda')}}, ('a',), bytes)


def test_campaign_remaining_gate_rejects_underfunded(monkeypatch):
    monkeypatch.setattr(job.time, 'monotonic', lambda: 120.0)
    dispatch = {'campaign_clock': {'started': 100.0, 'phase_started': 110.0, 'budget_end': 340.0}}
    assert job.require_remaining(dispatch, 141, 190)['phase_seconds_remaining'] == 170
    with pytest.raises(ValueError, match='insufficient'):
        job.require_remaining(dispatch, 171, 190)
    dispatch['campaign_clock']['budget_end'] = 300.0
    with pytest.raises(ValueError, match='insufficient'):
        job.require_remaining(dispatch, 141, 190)


def test_fresh_host_receipt_requires_bytes_path_and_original_inode(tmp_path, monkeypatch):
    from experiments.tennis import w8_receiver_preflight
    protected = {}
    for name in ('source', 'prepared', 'fit', 'golden', 'masks'):
        path = tmp_path / name
        path.write_bytes(name.encode())
        protected[name] = (str(path), job.fixed.file_sha(path))
    monkeypatch.setattr(w8_receiver_preflight, 'PROTECTED', protected)
    monkeypatch.setattr(job, 'PROTECTED', protected)
    original = {'denied_paths': {name: [entry[0]] for name, entry in protected.items()}}
    historical = {'protected_inputs': {name: {'path': path, 'sha256': digest,
                     'device': path_device - 1, 'inode': path_inode}
                  for name, (path, digest) in protected.items()
                  for path_device, path_inode in [(job.Path(path).stat().st_dev, job.Path(path).stat().st_ino)]}}
    receipt = tmp_path / 'fresh.json'
    result = job.policy.protected_identity(original, receipt)
    assert result['sha256'] == job.fixed.file_sha(receipt)
    fresh = json.loads(receipt.read_text())
    job.reconcile_protected_identity(historical, fresh)
    fresh['protected_inputs']['source']['inode'] += 1
    with pytest.raises(ValueError, match='identity mismatch'):
        job.reconcile_protected_identity(historical, fresh)
    job.Path(protected['source'][0]).write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity mismatch'):
        job.policy.protected_identity(original, tmp_path / 'bad.json')


def test_grants_accept_json_key_order_but_reject_added_or_missing_paths(monkeypatch):
    monkeypatch.setattr(job.cost, 'strict_roots',
                        lambda root, output, runtime, files: [str(root), str(output), *(x['path'] for x in files.values())])
    config = {'output': '/output', 'allow_paths': ['/code', '/output', '/bundle', '/profile', '/guider'],
              'files': {'bundle': {'path': '/bundle'}, 'guider': {'path': '/guider'}, 'profile': {'path': '/profile'}}}
    job.check_grants(config, {})
    config['allow_paths'].append('/extra')
    with pytest.raises(ValueError, match='grants'):
        job.check_grants(config, {})
    config['allow_paths'].pop()
    removed = config['allow_paths'].pop()
    with pytest.raises(ValueError, match='grants'):
        job.check_grants(config, {})
    config['allow_paths'].append(removed)
    config['allow_paths'][-1] = '/bundle'
    with pytest.raises(ValueError, match='grants'):
        job.check_grants(config, {})


@pytest.mark.parametrize('mode', ['legacy', 'packet'])
@pytest.mark.parametrize('case', ['valid', '479', 'noncpu', 'no_restore'])
def test_actual_packet_wrapper_receipt_and_witness_forwarding(tmp_path, monkeypatch, mode, case):
    from types import SimpleNamespace
    import numpy as np
    from experiments.tennis import w8_payload
    from experiments.tennis import w8_strict_receiver_worker as receiver

    class Tensor:
        def __init__(self, device='cpu'):
            self.device = SimpleNamespace(type=device)

    units = ('unit0', 'unit1')
    fake_fixed = SimpleNamespace(UNITS=units, decode_frozen_window=lambda: (
        {name: {'pose': np.zeros(1)} for name in units}, {}, {}))
    count = 479 if case == '479' else 480
    tensors = {str(i): Tensor('cuda' if case == 'noncpu' and i == 0 else 'cpu') for i in range(count)}
    state = {'reference_unet': tensors, 'denoising_unet': {}, 'pose_guider': {}}
    monkeypatch.setattr(w8_payload, 'unpack_match', lambda *args, **kwargs: state)
    monkeypatch.setattr(w8_payload, 'restore_tensor', lambda *args, **kwargs: Tensor())
    monkeypatch.setattr(receiver, 'preload', lambda: (None, fake_fixed, None, None))
    calls = []
    def launch(*args, **kwargs):
        calls.append(kwargs)
        fake_fixed.decode_frozen_window()
        w8_payload.unpack_match(None, None, None, {'base': Tensor()})
        if case != 'no_restore':
            for _ in range(480):
                w8_payload.restore_tensor()
    monkeypatch.setattr(receiver, 'launch', launch)
    fake_torch = SimpleNamespace(is_tensor=lambda value: isinstance(value, Tensor),
        cuda=SimpleNamespace(is_initialized=lambda: False), __version__='fixture')
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    output = tmp_path / 'output'
    output.mkdir()
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'worker_kind': 'packet', 'environment': {'CUDA_VISIBLE_DEVICES': ''}, 'output': str(output)}))
    argv = ['packet', '--config', str(config), '--config-sha256', receiver.digest(config),
            '--loader', str(tmp_path / 'loader'), '--loader-sha256', 'x', '--canary', str(tmp_path / 'canary'),
            '--source-root', str(Path(packet_worker.__file__).resolve().parents[2]) ]
    if mode == 'packet':
        argv += ['--witness-mode', 'packet']
    monkeypatch.setattr(sys, 'argv', argv)
    receipt = output / 'cpu-only.json'
    if case == 'valid':
        assert packet_worker.main() == 0
        record = json.loads(receipt.read_text())
        assert record['pre_serialization'] == {'state_tensors': 480, 'restored_tensors': 480, 'unit_arrays': 2}
        assert record['cuda_initialized_after_decode'] is False
        if mode == 'packet':
            origin = record['source_origin']
            assert origin['path'] == str(Path(packet_worker.__file__).resolve())
            assert origin['sha256'] == receiver.digest(Path(origin['path']))
        else:
            assert 'source_origin' not in record
        assert calls == ([{'witness_mode': 'packet'}] if mode == 'packet' else [{}])
    else:
        with pytest.raises((ValueError, RuntimeError)):
            packet_worker.main()
        assert not receipt.exists()


def test_witness_packet_cpu_receipt_missing_and_partial_fail(tmp_path):
    receipt = tmp_path / 'cpu-only.json'
    with pytest.raises(FileNotFoundError):
        job.read_witness_packet_cpu_receipt(receipt, 'torch-fixture')
    valid = {'scope': 'CPU packet decoder only', 'cuda_visible_devices': '',
             'cuda_initialized_after_decode': False, 'torch_version': 'torch-fixture',
             'pre_serialization': {'state_tensors': 480, 'restored_tensors': 480, 'unit_arrays': 2}}
    receipt.write_text(json.dumps(valid))
    assert job.read_witness_packet_cpu_receipt(receipt, 'torch-fixture') == valid
    for key, value in (('state_tensors', 479), ('restored_tensors', 479), ('unit_arrays', 0)):
        broken = dict(valid, pre_serialization=dict(valid['pre_serialization'], **{key: value}))
        receipt.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match='pre-serialization'):
            job.read_witness_packet_cpu_receipt(receipt, 'torch-fixture')
    receipt.write_text(json.dumps(dict(valid, cuda_initialized_after_decode=True)))
    with pytest.raises(ValueError, match='pre-serialization'):
        job.read_witness_packet_cpu_receipt(receipt, 'torch-fixture')


def test_packet_source_identity_rejects_other_origin_and_digest(tmp_path):
    name = 'experiments/tennis/w8_strict_packet_worker.py'
    root = tmp_path / 'source'
    origin = {'path': str(root / name), 'relative_path': name, 'sha256': 'a' * 64}
    manifest = {'files': {name: 'a' * 64}}
    job.validate_witness_packet_source({'source_origin': origin}, root, manifest)
    for changed in (dict(origin, path=str(tmp_path / 'other.py')), dict(origin, sha256='b' * 64)):
        with pytest.raises(ValueError, match='source identity'):
            job.validate_witness_packet_source({'source_origin': changed}, root, manifest)


def test_packet_executing_source_rechecked_against_archive(tmp_path):
    import hashlib
    import tarfile

    source = tmp_path / 'worker.py'
    source.write_text('fixed worker')
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(source, arcname='experiments/tennis/w8_strict_packet_worker.py')
    cpu = {'source_origin': {'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}
    job.validate_packet_source_archive(archive, job.fixed.file_sha(archive), cpu)
    with pytest.raises(ValueError, match='archive source'):
        job.validate_packet_source_archive(archive, job.fixed.file_sha(archive), {'source_origin': {'sha256': '0' * 64}})
    with pytest.raises(ValueError, match='snapshot differs'):
        job.validate_packet_source_archive(archive, '0' * 64, cpu)

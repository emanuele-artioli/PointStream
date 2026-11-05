import copy

import pytest

from experiments.tennis import w8_strict_host_cost_job as job


def test_new_spec_limits_and_frozen_inputs():
    from experiments.jobs.inbox import validate_spec
    spec = job.make_spec('2026-11-05T04:00:00+00:00')
    validated = validate_spec(spec, now=1793830000)
    assert validated['budget_seconds'] == 600
    assert spec['scale']['steps'] == {'smoke': 1, 'full': 3}
    assert (spec['smoke']['seconds'], spec['full']['seconds'], spec['validator_seconds']) == (260, 300, 30)
    assert spec['receiver_cpu_budget_seconds'] <= 1000
    assert spec['arguments'][spec['arguments'].index('--packet-cpu') + 1] == '100'
    assert 2 * (30 + 100 + 210) + 30 + 100 <= spec['receiver_cpu_budget_seconds']
    assert spec['hosts'] == ['gpu6']
    assert spec['gpu_sharing'] == {'mode': 'exclusive'}
    assert {entry['name'] for entry in spec['staged_inputs']} >= {'packet', 'profile', 'oracle', 'preflight', 'identity'}


def test_receipt_resource_check_requires_preserved_canaries():
    summary = {'scope': 'qualification_only', 'quality_claim': False, 'actual_parent_cpu_seconds': 1,
               'uuid_api_pre_model': {'torch_version': '2.7.0+cu128', 'cuda_version': '12.8', 'uuid_attribute_type': 'property', 'cuda_initialized': False},
               'parent_cpu_limit': 30, 'packet_cpu_limit': 30, 'generator_cpu_limit': 210,
               'packet_usage': {'cpu_seconds': 1, 'wall_seconds': 1, 'status': 0, 'timed_out': False, 'canaries_preserved': True},
               'generation_usage': {'cpu_seconds': 1, 'wall_seconds': 1, 'status': 0, 'timed_out': False, 'canaries_preserved': True}}
    job.check_summary_resources(summary)
    bad = copy.deepcopy(summary)
    bad['generation_usage']['canaries_preserved'] = False
    with pytest.raises(ValueError, match='accounting'):
        job.check_summary_resources(bad)


def test_durable_archive_is_exclusive_and_detects_changed_member(tmp_path):
    source = tmp_path / 'input'
    source.write_bytes(b'fixed experiment output')
    retained = tmp_path / 'retained'
    retained.mkdir()
    names = job.durable_member_names()
    for name in names:
        job.copy_new(source, retained / name)
    record = job.seal_durable_archive(retained, names)
    job.verify_durable_archive(retained, record)
    with pytest.raises(FileExistsError):
        job.seal_durable_archive(retained, names)
    (retained / names[0]).chmod(0o644)
    (retained / names[0]).write_bytes(b'altered')
    with pytest.raises(ValueError, match='bytes differ'):
        job.verify_durable_archive(retained, record)


def test_exact_preflight_runtime_gets_only_qualified_device_grants(tmp_path):
    runtime = '/local/users/emanuele/pointstream/cache/abc/tree'
    policy = {'python': runtime + '/bin/python3.12', 'allow_paths': [runtime, '/usr/lib', '/lib', '/lib64']}
    files = {'packet': {'path': '/local/public-packet.psw8', 'sha256': 'a' * 64}}
    roots = job.strict_roots(tmp_path / 'source', tmp_path / 'output', policy, files)
    assert '/dev/nvidiactl' in roots and '/dev/nvidia-uvm' in roots
    assert '/proc/driver/nvidia/version' in roots
    assert '/local/public-packet.psw8' in roots
    altered = copy.deepcopy(policy)
    altered['allow_paths'].append('/home')
    with pytest.raises(ValueError, match='fixed profile'):
        job.strict_roots(tmp_path / 'source', tmp_path / 'output', altered, files)


def test_worker_launch_uses_authenticated_source_as_cwd(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    work = tmp_path / 'work'
    work.mkdir()
    loader = tmp_path / 'loader.json'
    loader.write_text('{}')
    captured = {}
    class FakeProcess:
        pid = 123
    def popen(command, **kwargs):
        captured['command'] = command
        captured['cwd'] = kwargs['cwd']
        captured['env'] = kwargs['env']
        return FakeProcess()
    monkeypatch.setattr(job.subprocess, 'Popen', popen)
    monkeypatch.setattr(job.receiver, 'wait_owned_worker', lambda proc, timeout: (0, False))
    result = job._worker(tmp_path / 'python', {'worker_kind': 'packet'}, work, 10, 20, source, loader)
    assert result['canaries_preserved'] is True
    assert captured['cwd'] == source and captured['env']['PYTHONPATH'] == str(source)
    command = captured['command']
    assert command[command.index('--source-root') + 1] == str(source)


def test_fresh_packet_worker_is_cpu_instrumented_and_model_worker_is_separate(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    def launch(command, **kwargs):
        calls.append((command, kwargs['env']))
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(job.subprocess, 'Popen', launch)
    monkeypatch.setattr(job.receiver, 'wait_owned_worker', lambda proc, seconds: (0, False))
    loader = tmp_path / 'loader.json'
    loader.write_text('{}')
    for kind in ('packet', 'cost_probe'):
        work = tmp_path / kind
        work.mkdir()
        job._worker(tmp_path / 'python', {'worker_kind': kind}, work, 100, 130, tmp_path, loader)
    assert 'experiments.tennis.w8_strict_packet_worker' in calls[0][0]
    assert calls[0][1]['CUDA_VISIBLE_DEVICES'] == ''
    assert 'experiments.tennis.w8_strict_host_model_worker' in calls[1][0]
    assert 'CUDA_VISIBLE_DEVICES' not in calls[1][1]


def test_new_retained_receipts_are_required():
    names = job.durable_member_names()
    assert 'packet-cpu-only.json' in names
    assert 'protected-identity.json' in names


def test_model_receipt_must_bind_claimed_physical_ada(monkeypatch):
    checked = []
    monkeypatch.setattr(job.receiver, 'check_execution_receipt', lambda receipt: checked.append(receipt))
    uuid = 'GPU-12345678-1234-4234-8234-123456789abc'
    r = {'device_uuid': uuid.removeprefix('GPU-'), 'visible_devices': uuid, 'cuda_device_count': 1,
         'name': 'NVIDIA RTX 6000 Ada Generation', 'major': 8, 'minor': 9}
    sealed = {'device': {'name': r['name'], 'major': 8, 'minor': 9}, 'kernels': {'kernel_launches': 42}}
    path = 'experiments/tennis/w8_strict_host_model_worker.py'
    files = {path: 'a' * 64}
    r['wrapper'] = {'path': path, 'sha256': 'a' * 64}
    dispatch = {'environment': {'gpu': {'uuid': uuid}}}
    job.check_gpu_binding(r, sealed, uuid, dispatch, files)
    assert checked == [sealed]
    bad = copy.deepcopy(r)
    bad['device_uuid'] = '00000000-0000-4000-8000-000000000000'
    with pytest.raises(ValueError, match='dispatched'):
        job.check_gpu_binding(bad, sealed, uuid, dispatch, files)
    bad = copy.deepcopy(r)
    bad['minor'] = 6
    with pytest.raises(ValueError, match='dispatched'):
        job.check_gpu_binding(bad, sealed, uuid, dispatch, files)
    bad = copy.deepcopy(dispatch)
    bad['environment']['gpu']['uuid'] = 'GPU-00000000-0000-4000-8000-000000000000'
    with pytest.raises(ValueError, match='dispatched'):
        job.check_gpu_binding(r, sealed, uuid, bad, files)


def test_host_cost_projection_is_advisory_and_never_admits():
    one = {'scope': 'qualification_only', 'steps': 1, 'unit': 'u', 'phase_costs': {}}
    three = {'scope': 'qualification_only', 'steps': 3, 'unit': 'u', 'phase_costs': {}}
    for key in ('model_load', 'original_probe', 'context_probe', 'sampling', 'publication'):
        one['phase_costs'][key] = {'gpu_reserved_wall_seconds': 1}
        three['phase_costs'][key] = {'gpu_reserved_wall_seconds': 3 if key == 'sampling' else 1}
    decision = job.projection(one, three, 10000, 100, 1)
    assert decision['fits'] is True
    assert decision['admission_authorized'] is False
    assert 'no quality evidence' in decision['scope']


def test_model_wrapper_rejects_packet_before_isolation_or_cuda(tmp_path, monkeypatch):
    import json
    import sys
    from experiments.tennis import w8_strict_host_model_worker as worker
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'worker_kind': 'packet'}))
    monkeypatch.setattr(sys, 'argv', ['worker', '--config', str(config), '--config-sha256', worker.receiver.digest(config),
                                     '--loader', str(config), '--loader-sha256', worker.receiver.digest(config),
                                     '--canary', str(tmp_path), '--source-root', str(tmp_path)])
    monkeypatch.setattr(worker.receiver, 'launch', lambda *a, **k: pytest.fail('incorrect model launch'))
    with pytest.raises(ValueError, match='only admits'):
        worker.main()


def test_model_wrapper_actual_entrypoint_identity_and_mutation(tmp_path):
    from experiments.tennis import w8_strict_host_model_worker as worker
    expected = tmp_path / worker.WRAPPER_RELATIVE_PATH
    expected.parent.mkdir(parents=True)
    expected.write_text('original')
    observed = worker.wrapper_identity(tmp_path, expected)
    assert observed['path'] == worker.WRAPPER_RELATIVE_PATH
    assert observed['sha256'] == worker.receiver.digest(expected)
    outsider = tmp_path / 'other.py'
    outsider.write_text('original')
    with pytest.raises(ValueError, match='outside'):
        worker.wrapper_identity(tmp_path, outsider)
    expected.write_text('mutated')
    assert worker.wrapper_identity(tmp_path, expected)['sha256'] != observed['sha256']


def test_gpu_binding_rejects_mutated_wrapper_hash(monkeypatch):
    monkeypatch.setattr(job.receiver, 'check_execution_receipt', lambda receipt: None)
    uuid = 'GPU-12345678-1234-4234-8234-123456789abc'
    path = 'experiments/tennis/w8_strict_host_model_worker.py'
    receipt = {'device_uuid': uuid, 'visible_devices': uuid, 'cuda_device_count': 1,
               'name': 'RTX 6000 Ada', 'major': 8, 'minor': 9,
               'wrapper': {'path': path, 'sha256': 'b' * 64}}
    sealed = {'device': {'name': 'RTX 6000 Ada', 'major': 8, 'minor': 9}}
    with pytest.raises(ValueError, match='wrapper origin'):
        job.check_gpu_binding(receipt, sealed, uuid, {'environment': {'gpu': {'uuid': uuid}}}, {path: 'a' * 64})


def test_uuid_api_preflight_rejects_missing_property_and_initialized_cuda():
    from types import SimpleNamespace
    cls = type('Properties', (), {'uuid': property(lambda self: 'uuid')})
    torch = SimpleNamespace(_C=SimpleNamespace(_CudaDeviceProperties=cls), __version__='2.7.0+cu128',
                            version=SimpleNamespace(cuda='12.8'), cuda=SimpleNamespace(is_initialized=lambda: False))
    assert job.require_gpu_uuid_api(torch)['cuda_initialized'] is False
    torch._C._CudaDeviceProperties = type('Missing', (), {})
    with pytest.raises(ValueError, match='UUID property'):
        job.require_gpu_uuid_api(torch)
    torch._C._CudaDeviceProperties = cls
    torch.cuda.is_initialized = lambda: True
    with pytest.raises(ValueError, match='already initialized'):
        job.require_gpu_uuid_api(torch)

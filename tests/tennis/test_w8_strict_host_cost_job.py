import copy
from pathlib import Path

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


def test_materialized_spec_is_explicit_and_default_stays_historical():
    old = job.make_spec('2026-11-05T08:10:00Z')
    new = job.make_materialized_spec('2026-11-05T08:10:00Z')
    assert '--materialize-profile' not in old['arguments']
    assert '--materialize-profile' in new['arguments']
    assert new['materialization_profile'] == job.folder_profile.PROFILE
    assert new['folder_reference_audit_sha256'] == job.FOLDER_AUDIT_SHA
    root = Path(job.__file__).resolve().parents[2]
    assert job.fixed.file_sha(root / job.FOLDER_AUDIT_PATH) == job.FOLDER_AUDIT_SHA
    assert new['budget_seconds'] == old['budget_seconds'] == 600
    assert new['receiver_cpu_budget_seconds'] == old['receiver_cpu_budget_seconds'] == 900


def test_materialization_receipt_pins_source_and_cleanup(tmp_path, monkeypatch):
    sizes = {name: 1 for name in job.folder_profile.SIZES}
    pinned = {name: 'a' * 64 for name in sizes}
    monkeypatch.setattr(job.folder_profile, 'SIZES', sizes)
    monkeypatch.setattr(job.folder_profile, 'PINNED', pinned)
    files, rows = {}, {}
    for index, name in enumerate(sizes):
        path = tmp_path / str(index)
        path.write_bytes(b'x')
        st = path.stat()
        files['weight:' + name] = {'path': str(path), 'sha256': pinned[name]}
        rows[name] = {'source_sha256': pinned[name], 'copy_sha256': pinned[name], 'bytes': 1,
                      'copy_inode': index + 1, 'source_stat': [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]}
    receipt = {'profile': job.folder_profile.PROFILE, 'cleanup_complete': True, 'files': rows, 'bytes': 4,
               'wall_seconds': 2, 'cpu_seconds': 1, 'cleanup_wall_seconds': 1, 'cleanup_cpu_seconds': 1}
    job.check_folder_receipt(receipt, files)
    bad = copy.deepcopy(receipt)
    bad['files'][next(iter(sizes))]['copy_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='identity'):
        job.check_folder_receipt(bad, files)
    receipt['cleanup_complete'] = False
    with pytest.raises(ValueError, match='cleanup'):
        job.check_folder_receipt(receipt, files)


def test_materialized_durable_archive_includes_receipt_not_weight_bodies(tmp_path):
    retained = tmp_path / 'retained'
    retained.mkdir()
    names = job.durable_member_names(True)
    assert 'materialization.json' in names
    assert all('pytorch_model' not in name and 'safetensors' not in name for name in names)
    for name in names:
        (retained / name).write_text('receipt')
    receipt = job.seal_durable_archive(retained, names, materialized=True)
    job.verify_durable_archive(retained, receipt, materialized=True)
    with pytest.raises(ValueError, match='identities'):
        job.verify_durable_archive(retained, receipt)


def test_materialized_forecast_charges_whole_stage_and_staging_margin():
    receipt = {'wall_seconds': 2.0, 'cleanup_wall_seconds': 1.0}
    result = job.materialized_three_step_forecast(100.0, 20.0, 60.0, 20.0, receipt)
    expected_extra = job.FOLDER_IO_WALL_FLOOR - 3.0
    assert result['three_step_seconds'] == pytest.approx(1.5 * (100 + 40 + expected_extra))
    assert result['stage_reserve_seconds'] == 30
    with pytest.raises(ValueError, match='full-stage'):
        job.materialized_three_step_forecast(50, 20, 60, 20, receipt)
    with pytest.raises(ValueError, match='exceeds whole worker'):
        job.materialized_three_step_forecast(100, 20, 60, 20, {'wall_seconds': 50, 'cleanup_wall_seconds': 20})


def test_materialization_flag_only_reaches_model_child(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    def launch(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(job.subprocess, 'Popen', launch)
    monkeypatch.setattr(job.receiver, 'wait_owned_worker', lambda proc, seconds: (0, False))
    loader = tmp_path / 'loader.json'
    loader.write_text('{}')
    model_work = tmp_path / 'model'
    model_work.mkdir()
    job._worker(tmp_path / 'python', {'worker_kind': 'cost_probe'}, model_work, 100, 130, tmp_path, loader,
                materialize_profile=True)
    assert '--materialize-profile' in calls[0]
    packet_work = tmp_path / 'packet'
    packet_work.mkdir()
    with pytest.raises(ValueError, match='only to model'):
        job._worker(tmp_path / 'python', {'worker_kind': 'packet'}, packet_work, 100, 130, tmp_path, loader,
                    materialize_profile=True)
    assert len(calls) == 1


def test_aa_pin_imports_declared_package_before_site_copy(tmp_path, monkeypatch):
    import importlib
    import sys
    from experiments.tennis import w8_strict_host_model_worker as worker
    opt = tmp_path / 'opt' / 'animate_anyone'
    site = tmp_path / 'site' / 'animate_anyone'
    opt.mkdir(parents=True)
    site.mkdir(parents=True)
    (opt / '__init__.py').write_text('origin = "declared"\n')
    (site / '__init__.py').write_text('origin = "foreign"\n')
    entry = opt / '__init__.py'
    config = {'files': {'code:animate_anyone': {'path': str(entry), 'sha256': worker.receiver.digest(entry)}}}
    monkeypatch.delitem(sys.modules, 'animate_anyone', raising=False)
    monkeypatch.setattr(sys, 'path', [str(tmp_path), str(site.parent), *sys.path])
    worker.pin_declared_aa_imports(config)
    module = importlib.import_module('animate_anyone')
    try:
        assert module.origin == 'declared'
        assert Path(module.__file__).resolve() == entry
        worker.pin_declared_aa_imports(config)
    finally:
        sys.modules.pop('animate_anyone', None)


def test_aa_pin_rejects_foreign_and_mixed_preloaded_origins(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from types import ModuleType
    from experiments.tennis import w8_strict_host_model_worker as worker
    package = tmp_path / 'animate_anyone'
    package.mkdir()
    init = package / '__init__.py'
    init.write_text('')
    config = {'files': {'code:animate_anyone': {'path': str(init), 'sha256': worker.receiver.digest(init)}}}
    foreign = tmp_path / 'foreign.py'
    foreign.write_text('')
    module = ModuleType('animate_anyone')
    module.__file__ = str(foreign)
    module.__spec__ = importlib.util.spec_from_file_location('animate_anyone', foreign)
    monkeypatch.setitem(sys.modules, 'animate_anyone', module)
    with pytest.raises(ValueError, match='origin or hash'):
        worker.pin_declared_aa_imports(config)
    module.__file__ = str(init)
    module.__spec__ = importlib.util.spec_from_file_location('animate_anyone', init)
    module.__path__ = [str(package), str(tmp_path)]
    with pytest.raises(ValueError, match='search path'):
        worker.pin_declared_aa_imports(config)
    module.__path__ = [str(package)]
    module.__spec__.submodule_search_locations = [str(tmp_path)]
    with pytest.raises(ValueError, match='search path'):
        worker.pin_declared_aa_imports(config)
    module.__file__ = None
    with pytest.raises(ValueError, match='not declared'):
        worker.pin_declared_aa_imports(config)
    monkeypatch.delitem(sys.modules, 'animate_anyone')
    init.write_text('mutation')
    with pytest.raises(ValueError, match='origin or hash'):
        worker.pin_declared_aa_imports(config)


def test_witness_uses_pid_namespace_without_new_host_process_group(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.tennis import w8_witness_profile as witness

    calls = []
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        receipt = Path(command[command.index('--receipt') + 1])
        receipt.write_text('{}')
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(job.subprocess, 'Popen', launch)
    monkeypatch.setattr(job.receiver, 'wait_owned_worker', lambda proc, seconds: (0, False))
    monkeypatch.setattr(job.w8_clock_profile, 'write_parent_receipt', lambda path: 'clock-sha')
    monkeypatch.setattr(witness, 'namespace_completion', lambda *_args, **_kwargs: {'children_cpu_seconds': 1.0})
    monkeypatch.setenv('PS_STAGE_DIR', str(tmp_path / 'smoke'))
    readlink = job.os.readlink
    monkeypatch.setattr(job.os, 'readlink', lambda path: 'pid:[parent]' if str(path) == '/proc/self/ns/pid' else readlink(path))
    loader = tmp_path / 'loader.json'
    loader.write_text('{}')
    work = tmp_path / 'model'
    work.mkdir()
    job._worker(tmp_path / 'python', {'worker_kind': 'context_qualification'}, work, 100, 130,
                tmp_path, loader, materialize_profile=True, clock_profile=True, witness_mode='context_qualification')
    command, kwargs = calls[0]
    assert command[:7] == ['/usr/bin/unshare', '--user', '--map-root-user', '--net', '--pid', '--fork', '--kill-child=SIGKILL']
    assert 'experiments.tennis.w8_witness_namespace' in command
    assert command[command.index('--timeout') + 1] == '120'
    assert 'start_new_session' not in kwargs
    for key in ('TMPDIR', 'TMP', 'TEMP', 'HOME'):
        assert kwargs['env'][key] == str(work)
    for key in ('XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME'):
        assert Path(kwargs['env'][key]).is_relative_to(work)
    assert kwargs['env']['HF_HUB_OFFLINE'] == '1' and kwargs['env']['TRANSFORMERS_OFFLINE'] == '1'
    assert (work / 'namespace-validation.json').is_file()

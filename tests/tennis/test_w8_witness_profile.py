import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from experiments.tennis import w8_input_job as fixed
from experiments.tennis import w8_witness_profile as witness



def first_pilot_ledger(tmp_path: Path) -> Path:
    path = tmp_path / 'first-ledger.json'
    path.write_text(json.dumps({'reservation_amount_basis': 'remaining_unspent', 'gpu_total_seconds': 6000,
        'cpu_total_seconds': 6000, 'workstreams': {'w8': {'gpu_limit_seconds': 5000, 'cpu_limit_seconds': 5000}},
        'charges': [], 'reservations': [{'phase': 'w8-full-witness-staged-20261106', 'workstream': 'w8',
                                        'gpu_seconds': 4000, 'cpu_seconds': 4000, 'status': 'remaining_unspent'}]}))
    return path


def first_pilot_admission(tmp_path: Path, proof: Path) -> Path:
    ledger = first_pilot_ledger(tmp_path)
    path = tmp_path / 'first-admission.json'
    path.write_text(json.dumps({'schema': 1, 'scope': 'root-reviewed first W8 qualification pilot',
        'attempt_nonce': 'a' * 32, 'ledger_sha256': fixed.file_sha(ledger), 'cpu_proof_sha256': fixed.file_sha(proof),
        'preparation_phases': [], 'gpu_spent_seconds': 0, 'cpu_spent_seconds': 0,
        'gpu_ceiling_seconds': 900, 'cpu_ceiling_seconds': 1000}))
    return path

def test_qualification_requires_independently_bound_both_units(tmp_path: Path, monkeypatch) -> None:
    dirs = {unit: tmp_path / str(i) for i, unit in enumerate(fixed.UNITS)}
    for path in dirs.values():
        path.mkdir()
    expected = {unit: {'unit': unit, 'checkpoint': f'checkpoint-{i}'} for i, unit in enumerate(fixed.UNITS)}
    seen = []
    def validate(path, exp):
        seen.append((path, exp))
        return {'scope': 'context_qualification_only'}
    monkeypatch.setattr(witness.generation, 'validate_context_qualification', validate)
    witness.check_generation(dirs, expected)
    assert seen == [(dirs[u], expected[u]) for u in fixed.UNITS]
    with pytest.raises(ValueError, match='both'):
        witness.check_generation({fixed.UNITS[0]: dirs[fixed.UNITS[0]]}, expected)
    swapped = {u: expected[fixed.UNITS[1-i]] for i, u in enumerate(fixed.UNITS)}
    with pytest.raises(ValueError, match='unit identity'):
        witness.check_generation(dirs, swapped)


def test_candidate_validator_rejects_context_qualification(tmp_path: Path) -> None:
    (tmp_path / 'sealed.json').write_text(json.dumps({'scope': 'context_qualification_only', 'recipe': {'steps': 30}}))
    with pytest.raises(ValueError, match='scope'):
        witness.generation.validate_sealed(tmp_path, {})


def test_qualification_finish_rejects_wrong_gpu_and_sam_source(tmp_path: Path, monkeypatch) -> None:
    import numpy as np
    from experiments.tennis import w8_receiver_finish as finish

    monkeypatch.setattr(witness, 'check_generation', lambda *_: {'bound': True})
    dirs = {unit: tmp_path / str(i) for i, unit in enumerate(fixed.UNITS)}
    for unit, directory in dirs.items():
        directory.mkdir()
        (directory / 'sealed.json').write_text('{}')
        (directory / f'{unit}@match5.npz').write_bytes(b'crop')
    base, sam, compose = (tmp_path / name for name in ('base', 'sam', 'compose'))
    for directory in (base, sam, compose):
        directory.mkdir()
    for path in (base / 'base.npy', compose / 'composite.npy'):
        a = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=(8, 1080, 1920, 3))
        del a
    gpu = '7754164a-33cd-265c-3696-c20966d228fe'
    gpu_identity = {'uuid': gpu, 'visible_uuid': gpu, 'major': 8, 'minor': 9}
    iso = {'landlock_abi': 1, 'pre_sandbox_threads': 1, 'extra_fds': False, 'network_interfaces': ['lo'],
           'namespaces': {'user': 'u', 'net': 'n'}, 'denials': {'source': 13}}
    shared = {'scope': 'context_qualification_only', 'isolation': iso, 'packet_sha256': fixed.WINDOW_SHA,
              'profile_sha256': fixed.PROFILE_SHA}
    native = {'binary_sha256': finish.FFMPEG_SHA, 'ivf_sha256': finish.IVF_SHA,
              'yuv_sha256': '7490edf9b678db805c23baae49a2a3b5e9d7b037b18dd216320aad47a061beff',
              'exit_code': 0, 'decoder': 'libdav1d', 'yuv_bytes': 240 * 480 * 272 * 3 // 2,
              'landlock_abi': 1, 'denials': {'source': 13}, 'loader_sha256': 'x', 'libraries': {'libdav1d': 'x'}}
    base_sha = fixed.file_sha(base / 'base.npy')
    (base / 'sealed.json').write_text(json.dumps({**shared, 'native': native, 'ticks': finish.TICKS[:8].tolist(),
                                                 'base_sha256': base_sha, 'gpu_identity': gpu_identity,
                                                 'input_sha256': {'yuv': native['yuv_sha256']}}))
    source = tmp_path / 'sam.py'
    source.write_text('sam source')
    gen_files = {}
    rows = []
    masks = {}
    for unit, directory in dirs.items():
        gen_files[f'crop:{unit}'] = fixed.file_sha(directory / f'{unit}@match5.npz')
        gen_files[f'generation_seal:{unit}'] = fixed.file_sha(directory / 'sealed.json')
        mask = sam / f'{unit}-masks.npz'
        np.savez_compressed(mask, t=finish.TICKS[:8], masks=np.zeros((8, 784, 512), dtype=bool))
        masks[f'mask:{unit}'] = fixed.file_sha(mask)
        rows.append({'unit': unit, 'file_sha256': fixed.file_sha(mask)})
    sam_data: dict[str, Any] = {**shared, 'files': {**gen_files, 'weights': finish.SAM_SHA, 'text': finish.SAM_SHA,
                                                  'protected_record': 'fresh-protected'},
                'rows': rows, 'sam_record_sha256': 'preflight', 'device': 'Ada', 'gpu_identity': gpu_identity,
                'sam_source_modules': {'sam3': str(source)}, 'independent_frames': True, 'flash_events': ['flash'],
                'load': {'sdpa_backend_policy': 'native_flash', 'missing_keys': [], 'unexpected_keys': []}}
    (sam / 'sealed.json').write_text(json.dumps(sam_data))
    (compose / 'sealed.json').write_text(json.dumps({**shared, 'files': {**gen_files, **masks, 'base': base_sha},
          'ticks': finish.TICKS[:8].tolist(), 'composite_sha256': fixed.file_sha(compose / 'composite.npy')}))
    trusted: dict[str, Any] = {'native_receipt': native, 'sam_record_sha256': 'preflight', 'gpu_name': 'Ada',
               'protected_record_sha256': 'fresh-protected',
               'claimed_gpu_uuid': gpu, 'sam_verified_files': {str(source): fixed.file_sha(source)}}
    args: dict[str, Any] = dict(generation_dirs=dirs, expected={}, base_dir=base, sam_dir=sam, compose_dir=compose, trusted=trusted)
    assert witness.check_finish(**args)['scope'] == 'context_qualification_only'
    sam_data['gpu_identity']['uuid'] = 'a754164a-33cd-265c-3696-c20966d228fe'
    (sam / 'sealed.json').write_text(json.dumps(sam_data))
    with pytest.raises(ValueError, match='physical GPU'):
        witness.check_finish(**args)
    sam_data['gpu_identity']['uuid'] = gpu
    (sam / 'sealed.json').write_text(json.dumps(sam_data))
    trusted['sam_verified_files'][str(source)] = '0' * 64
    with pytest.raises(ValueError, match='SAM source module'):
        witness.check_finish(**args)


def test_sam_qualification_binds_historical_and_fresh_protected_identity(monkeypatch):
    from experiments.tennis import w8_receiver_finish as finish, w8_protected_metadata as metadata

    monkeypatch.setattr(metadata, 'PROTECTED', {'source': ('/protected', 'sha')})
    old = {'source': {'path': '/protected', 'sha256': 'sha', 'inode': 4, 'device': 1}}
    fresh = {'schema': 1, 'scope': metadata.SCOPE, 'content_verified_now': False,
             'historical_receipt_sha256': 'a' * 64, 'protected_inputs': {'source': {
                 'path': '/protected', 'reference_sha256': 'sha', 'inode': 4, 'device': 2,
                 'size': 9, 'mtime_ns': 10, 'ctime_ns': 11}}}
    finish._qual_sam_identity(old, fresh, historical_sha256='a' * 64)
    with pytest.raises(ValueError):
        finish._qual_sam_identity({}, {}, historical_sha256='a' * 64)
    with pytest.raises(ValueError, match='historical receipt'):
        finish._qual_sam_identity(old, fresh, historical_sha256='b' * 64)
    fresh['protected_inputs']['source']['inode'] = 5
    with pytest.raises(ValueError, match='historical protected'):
        finish._qual_sam_identity(old, fresh, historical_sha256='a' * 64)


def test_generation_helper_rejects_mixed_or_unbounded_modes() -> None:
    base: dict[str, Any] = dict(prepared={}, dispatch={}, visible='', worker_cpu=100, worker_wall=100)
    with pytest.raises(ValueError, match='registered qualification'):
        witness.execute_generation_units(**base, mode='context_qualification', units=(fixed.UNITS[0], fixed.UNITS[0]))
    with pytest.raises(ValueError, match='registered qualification'):
        witness.execute_generation_units(**base, mode='generate', units=fixed.UNITS)
    with pytest.raises(ValueError, match='registered qualification'):
        witness.execute_generation_units(**{**base, 'worker_wall': 501}, mode='context_qualification', units=fixed.UNITS)


def test_context_orchestrator_uses_each_independent_packet_unit(tmp_path: Path, monkeypatch) -> None:
    import os
    from experiments.tennis import w8_strict_host_cost_job as host

    gpu = 'GPU-7754164a-33cd-265c-3696-c20966d228fe'
    monkeypatch.setenv('PS_CLAIMED_GPU_UUID', gpu)
    decoder = tmp_path / 'decoder'
    unitdir = decoder / 'publish' / 'units'
    unitdir.mkdir(parents=True)
    for i, unit in enumerate(fixed.UNITS):
        (unitdir / f'{unit}.npz').write_bytes(f'unit-{i}'.encode())
    monkeypatch.setattr(witness.generation, 'ORACLE_UNIT_SHA',
                        {unit: fixed.file_sha(unitdir / f'{unit}.npz') for unit in fixed.UNITS})
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    work = scratch / 'model'
    output = work / 'output'
    output.mkdir(parents=True)
    loader = tmp_path / 'loader.json'
    loader.write_text('{}')
    configs = []
    def make_config(kind, files, output, runtime, cpu, visible, *, unit):
        return {'worker_kind': kind, 'files': files, 'unit': unit, 'output': str(output)}
    def fake_worker(python, config, work, cpu, wall, root, loader, **kwargs):
        configs.append(config)
        target = Path(config['output'])
        (work / 'worker-config.json').write_text('{}')
        (target / 'sealed.json').write_text(json.dumps({'isolation': {}}))
        (target / 'materialization.json').write_text(json.dumps({'wall_seconds': 1, 'cleanup_wall_seconds': 1}))
        device = {'sealed_sha256': fixed.file_sha(target / 'sealed.json'),
                  'config_sha256': fixed.file_sha(work / 'worker-config.json'),
                  'materialization_sha256': fixed.file_sha(target / 'materialization.json')}
        (target / 'gpu-identity.json').write_text(json.dumps(device))
        assert kwargs['witness_mode'] == 'context_qualification'
        return {'wall_seconds': 3}
    monkeypatch.setattr(host, '_worker', fake_worker)
    monkeypatch.setattr(host, 'check_gpu_binding', lambda *_args: None)
    monkeypatch.setattr(host, 'check_folder_receipt', lambda *_args: None)
    monkeypatch.setattr(witness, 'check_worker_isolation', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(witness.generation, 'validate_context_qualification', lambda *_args: {'ok': True})
    prepared = {'scratch': scratch, 'decoder_out': decoder, 'model_work': work, 'model_out': output,
                'generated': {}, 'config': make_config, 'policies': {'jcv': {'python': '/python'}},
                'root': tmp_path, 'loader_receipt': loader, 'fingerprints': {}, 'code_manifest': {'files': {}}}
    dispatch = {'environment': {'gpu': {'uuid': gpu}}}
    result = witness.execute_generation_units(prepared, dispatch, visible=gpu, worker_cpu=100, worker_wall=100,
                                              mode='context_qualification', units=fixed.UNITS)
    assert list(result['outputs']) == list(fixed.UNITS)
    assert [row['unit'] for row in configs] == list(fixed.UNITS)
    assert [row['files']['unit']['sha256'] for row in configs] == [witness.generation.ORACLE_UNIT_SHA[u] for u in fixed.UNITS]
    assert os.environ['PS_CLAIMED_GPU_UUID'] == gpu


def test_native_helper_uses_fixed_pid_reaper_and_hidden_cuda(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace
    from experiments.tennis import w8_receiver_finish as finish, w8_receiver_native_bootstrap as native
    from experiments.tennis import w8_receiver_preflight as preflight, w8_witness_namespace as namespace

    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    binary = tmp_path / 'ffmpeg'
    binary.write_bytes(b'ffmpeg')
    loader = tmp_path / 'loader'
    loader.write_bytes(b'loader')
    library = tmp_path / 'libdav1d.so'
    library.write_bytes(b'library')
    inventory = tmp_path / 'inventory.json'
    mapped = {str(binary): {'sha256': 'b'}, str(loader): {'sha256': 'l'}, str(library): {'sha256': 'd'}}
    inventory.write_text(json.dumps({'ffmpeg_path': str(binary), 'elf_interpreter': str(loader), 'mapped_files': mapped}))
    inv_sha = 'cd6bb8cf2a239d94f595fd92f823cf6406c018e1fbda745e81cd0c0e91d1b59b'
    actual_sha = fixed.file_sha
    monkeypatch.setattr(fixed, 'file_sha', lambda path: inv_sha if Path(path) == inventory else actual_sha(path))
    monkeypatch.setattr(finish, '_packet', lambda _: ({'ivf': 'ivf'}, {'ivf': b'ivf'}))
    monkeypatch.setattr(fixed, 'digest', lambda _: finish.IVF_SHA)
    monkeypatch.setattr(native, 'validate_inventory', lambda *_args: {})
    monkeypatch.setattr(witness, 'namespace_completion', lambda *_args, **_kwargs: {'children_cpu_seconds': 1})
    monkeypatch.setattr(witness, 'namespace_started_absence', lambda *_args, **_kwargs: {})
    monkeypatch.setenv('PS_STAGE_DIR', str(tmp_path / 'smoke'))
    import os
    readlink = os.readlink
    monkeypatch.setattr(os, 'readlink', lambda path: ('pid:[outer]' if str(path) == '/proc/self/ns/pid'
        else 'user:[outer]' if str(path) == '/proc/self/ns/user'
        else 'net:[outer]' if str(path) == '/proc/self/ns/net' else readlink(path)))
    calls: list[Any] = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        config = json.loads(Path(command[command.index('--config') + 1]).read_text())
        output = Path(config['output'])
        (output / 'native.yuv').write_bytes(b'yuv')
        data = {'inventory_sha256': inv_sha, 'binary_sha256': finish.FFMPEG_SHA, 'ivf_sha256': finish.IVF_SHA,
                'exit_code': 0, 'yuv_sha256': actual_sha(output / 'native.yuv'),
                'denials': {value[0]: 13 for value in preflight.PROTECTED.values()}}
        (output / 'sealed.json').write_text(json.dumps(data))
        control = Path(command[command.index('--control') + 1])
        control.mkdir()
        (control / 'execution.json').write_text(json.dumps({'status': 'completed', 'returncode': 0}))
        Path(command[command.index('--receipt') + 1]).write_text('{}')
        return SimpleNamespace(pid=321, wait=lambda timeout: 0)
    monkeypatch.setattr('subprocess.Popen', popen)
    result = witness.execute_native_base({'paths': {'packet': tmp_path, 'profile': tmp_path}, 'scratch': scratch,
                                          'root': tmp_path}, inventory_path=inventory,
                                         parent_cpu=20, worker_cpu=20, child_cpu=20, wall=50)
    command, kwargs = calls[0]
    assert command[:6] == ['/usr/bin/unshare', '--user', '--map-root-user', '--pid', '--fork', '--kill-child=SIGKILL']
    assert namespace.NATIVE_TARGET in command and '--wall-seconds' in command
    assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '' and 'start_new_session' not in kwargs
    assert result['usage']['status'] == 0


def test_pilot_spec_is_separate_and_never_auto_promotes(tmp_path: Path):
    from experiments.tennis import w8_witness_job as job

    gate = tmp_path / 'cpu-gate.json'
    gate.write_text(json.dumps({'schema': 2, 'scope': job.CPU_SCOPE, 'passed': True, 'bootstrap_sha256': '9' * 64, 'input_job_sha256': '8' * 64, 'system_leaves_sha256': '7' * 64,
        'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'revision': 'c' * 40,
        'proof_campaign_environment_sha256': job.host.MAIN_ENV, 'generation_environment_sha256': job.host.JCV_ENV,
        'generation_controller_executed': False, 'cuda_qualified': False, 'host': 'gpu5', 'job_id': '20261106T000000Z-12345678',
        'spec_sha256': 'e' * 64, 'raw_receipt_sha256': dict.fromkeys(job.REQUIRED_PROOFS, 'f' * 64),
        'checks': dict.fromkeys(job.REQUIRED_CHECKS, True)}))
    spec = job.make_pilot_spec(deadline=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(), proof_local=gate,
         proof_remote='/home/itec/emanuele/Datasets/pointstream-data/tests/cpu-gate.json',
         ledger_local=first_pilot_ledger(tmp_path), ledger_remote='/home/itec/emanuele/Datasets/pointstream-data/tests/ledger.json', admission_local=first_pilot_admission(tmp_path, gate), admission_remote='/home/itec/emanuele/Datasets/pointstream-data/tests/admission.json')
    assert spec['w8_witness_profile'] == 'w8-context-pilot-v1'
    assert spec['hosts'] == ['gpu5'] and spec['witness_cpu_proof_host'] == 'gpu5'
    assert spec['scale']['phase'] == {'smoke': 'pilot', 'full': 'noop'}
    assert spec['smoke']['seconds'] <= 600 and spec['budget_seconds'] <= 900
    assert spec['witness_cpu_ceiling_seconds'] <= 1000
    changed = dict(spec, scale={'phase': {'smoke': 'pilot', 'full': 'pilot'}})
    with pytest.raises(ValueError, match='pilot spec'):
        job._pilot_spec(changed)
    with pytest.raises(ValueError, match='pilot spec'):
        job._pilot_spec(dict(spec, hosts=['gpu6']))


def test_namespace_completion_requires_started_join_and_absent_owned_pids(tmp_path: Path, monkeypatch) -> None:
    from experiments.tennis import w8_witness_namespace as namespace

    receipt = tmp_path / 'receipt.json'
    started_path = tmp_path / 'receipt.json.started'
    final: dict[str, dict[str, Any]] = {'namespace_init': {'host_pid': 99999991, 'start_ticks': 1, 'host_pgid': 1,
                                'namespace_pid': 1, 'pid_namespace': 'pid:[inner]'},
             'worker_host_identity': {'host_pid': 99999992, 'start_ticks': 2, 'host_pgid': 1,
                                      'campaign_pgid': 1}}
    monkeypatch.setattr(namespace, 'validate_receipt', lambda *_args, **_kwargs: final)
    started = {'schema': 1, 'target': 'target', 'outside_namespace': 'pid:[outer]', **final}
    started_path.write_text(json.dumps(started))
    assert witness.namespace_completion(receipt, expected_pgid=1, expected_outside='pid:[outer]',
                                        expected_target='target') == final
    started['worker_host_identity'] = {'host_pid': 99999992, 'start_ticks': 3, 'host_pgid': 1, 'campaign_pgid': 1}
    started_path.write_text(json.dumps(started))
    with pytest.raises(ValueError, match='pre-wait identity'):
        witness.namespace_completion(receipt, expected_pgid=1, expected_outside='pid:[outer]', expected_target='target')
    started['namespace_init'] = dict(final['namespace_init'], pid_namespace='pid:[outer]')
    started_path.write_text(json.dumps(started))
    with pytest.raises(ValueError, match='process binding'):
        witness.namespace_started_absence(receipt, expected_pgid=1, expected_outside='pid:[outer]',
                                          expected_target='target')
    started_path.unlink()
    with pytest.raises(FileNotFoundError):
        witness.namespace_completion(receipt, expected_pgid=1, expected_outside='pid:[outer]', expected_target='target')


def test_pilot_cpu_gate_requires_raw_audits_and_exact_source():
    from experiments.tennis import w8_witness_job as job

    source = {'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'git_head': 'c' * 40}
    gate = {'schema': 2, 'scope': job.CPU_SCOPE, 'passed': True, 'bootstrap_sha256': '9' * 64, 'input_job_sha256': '8' * 64, 'system_leaves_sha256': '7' * 64, **source,
            'revision': source['git_head'], 'proof_campaign_environment_sha256': job.host.MAIN_ENV, 'generation_environment_sha256': job.host.JCV_ENV,
        'generation_controller_executed': False, 'cuda_qualified': False, 'host': 'gpu5',
            'job_id': '20261106T000000Z-12345678', 'spec_sha256': 'e' * 64,
            'raw_receipt_sha256': {key: 'f' * 64 for key in job.REQUIRED_PROOFS},
            'checks': dict.fromkeys(job.REQUIRED_CHECKS, True)}
    job.check_cpu_gate(gate, gate_sha='1' * 64, expected_sha='1' * 64, source=source,
                       environment_sha=job.host.JCV_ENV, host_name='gpu5')
    with pytest.raises(ValueError, match='identity'):
        job.check_cpu_gate(gate, gate_sha='1' * 64, expected_sha='1' * 64, source=source,
                           environment_sha=job.host.JCV_ENV, host_name='gpu6')
    for wrong in (dict(gate, schema=1), dict(gate, proof_campaign_environment_sha256=job.host.JCV_ENV),
                  dict(gate, generation_environment_sha256=job.host.MAIN_ENV),
                  dict(gate, generation_controller_executed=True), dict(gate, cuda_qualified=True)):
        with pytest.raises(ValueError, match='identity'):
            job.check_cpu_gate(wrong, gate_sha='1' * 64, expected_sha='1' * 64, source=source,
                               environment_sha=job.host.JCV_ENV, host_name='gpu5')
    with pytest.raises(ValueError, match='identity'):
        job.check_cpu_gate(gate, gate_sha='1' * 64, expected_sha='1' * 64, source=source,
                           environment_sha=job.host.MAIN_ENV, host_name='gpu5')
    missing_policy = dict(gate)
    missing_policy.pop('system_leaves_sha256')
    with pytest.raises(ValueError, match='identity'):
        job.check_cpu_gate(missing_policy, gate_sha='1' * 64, expected_sha='1' * 64, source=source,
                           environment_sha=job.host.JCV_ENV, host_name='gpu5')
    gate['raw_receipt_sha256'].pop('native_timeout_kill')
    with pytest.raises(ValueError, match='proof references'):
        job.check_cpu_gate(gate, gate_sha='1' * 64, expected_sha='1' * 64, source=source,
                           environment_sha=job.host.JCV_ENV, host_name='gpu5')


def test_pilot_released_validation_uses_sealed_sources_not_deleted_scratch(tmp_path: Path, monkeypatch):
    from experiments.tennis import w8_strict_host_cost_job as host, w8_witness_job as job, w8_protected_metadata
    monkeypatch.setattr(w8_protected_metadata, 'reject_protected_alias', lambda _path: None)

    stage = tmp_path / 'job' / 'smoke'
    stage.mkdir(parents=True)
    source = {'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'git_head': 'c' * 40}
    gate = {'schema': 2, 'scope': job.CPU_SCOPE, 'passed': True, 'bootstrap_sha256': '9' * 64, 'input_job_sha256': '8' * 64, 'system_leaves_sha256': '7' * 64, **source, 'revision': source['git_head'],
            'proof_campaign_environment_sha256': job.host.MAIN_ENV, 'generation_environment_sha256': job.host.JCV_ENV,
        'generation_controller_executed': False, 'cuda_qualified': False, 'host': 'gpu5', 'job_id': '20261106T000000Z-12345678',
            'spec_sha256': 'e' * 64, 'raw_receipt_sha256': dict.fromkeys(job.REQUIRED_PROOFS, 'f' * 64),
            'checks': dict.fromkeys(job.REQUIRED_CHECKS, True)}
    cpu = tmp_path / 'cpu.json'
    cpu.write_text(json.dumps(gate))
    preflight = tmp_path / 'preflight.json'
    preflight.write_text('{}')
    identity = tmp_path / 'identity.json'
    identity.write_text('{}')
    oracle = tmp_path / 'oracle.json'
    oracle.write_text('{}')
    expected = {'packet': fixed.WINDOW_SHA, 'profile': fixed.PROFILE_SHA, 'oracle': host.ORACLE_SHA,
                'base': host.BASE_ARCHIVE_SHA, 'aa': host.receiver.AA_ARCHIVE_SHA,
                **{key: host.receiver.PINNED_WEIGHTS[name] for key, name in host.WEIGHT_NAMES.items()},
                'cpu_proof': fixed.file_sha(cpu), 'preflight': fixed.file_sha(preflight),
                'identity': fixed.file_sha(identity)}
    source_paths = {'cpu_proof': cpu, 'preflight': preflight, 'identity': identity, 'oracle': oracle}
    declared = [{'name': name, 'sha256': digest, 'path': str(source_paths.get(name, tmp_path / name)), 'extract': False}
                for name, digest in expected.items()]
    staged = [{'name': row['name'], 'sha256': row['sha256'], 'source': row['path'],
               'path': str(tmp_path / 'deleted-scratch' / str(row['name'])), 'extract': False} for row in declared]
    spec = {'staged_inputs': declared, 'witness_cpu_proof_sha256': fixed.file_sha(cpu), 'witness_bootstrap_sha256': '9' * 64, 'witness_input_job_sha256': '8' * 64, 'witness_system_leaves_sha256': '7' * 64}
    monkeypatch.setattr(job, 'check_bootstrap_archive', lambda *_args: None)
    (stage.parent / 'spec.json').write_text(json.dumps(spec))
    (stage / 'staging.json').write_text(json.dumps({'inputs': staged}))
    (stage / 'dispatch.json').write_text(json.dumps({'code': source, 'staged_environment': {'sha256': host.JCV_ENV},
                                                   'environment': {'alias': 'gpu5'}}))
    actual_sha = fixed.file_sha
    monkeypatch.setattr(fixed, 'file_sha', lambda p: host.ORACLE_SHA if Path(p) == oracle else actual_sha(Path(p)))
    paths, _, _ = job.load_released_inputs(stage)
    assert paths['cpu_proof'] == cpu and paths['oracle'] == oracle
    staged[0]['source'] = '/different/source'
    (stage / 'staging.json').write_text(json.dumps({'inputs': staged}))
    with pytest.raises(ValueError, match='staged asset identity'):
        job.load_released_inputs(stage)


def test_promotion_rejects_missing_cost_and_underfunded_remaining_budget():
    import time
    from experiments.tennis import w8_witness_job as job

    source = {'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64}
    components = {name: {'gpu_seconds': 8.0, 'cpu_seconds': 10.0} for name in (
        'unit1_context', 'unit0_full', 'unit1_full', 'native_sam_compose', 'postseal_score',
        'offline_validation', 'external_staging_audits', 'failure_headroom')}
    gate = {'schema': 1, 'scope': 'root-reviewed W8 remaining-work forecast', **source,
            'cpu_proof_sha256': 'c' * 64, 'prior_artifact_sha256': 'd' * 64, 'ledger_sha256': 'e' * 64,
            'raw_measurement_audit_sha256': 'f' * 64, 'safety_factor': 1.5, 'remaining_components': components,
            'gpu_remaining_seconds': 70, 'cpu_remaining_seconds': 84,
            'issued_epoch': time.time() - 1, 'expires_epoch': time.time() + 100}
    args = {'gate_sha': '1' * 64, 'expected_sha': '1' * 64, 'source': source, 'cpu_gate_sha': 'c' * 64,
            'prior_artifact_sha': 'd' * 64, 'ledger_sha': 'e' * 64, 'gpu_remaining': 70, 'cpu_remaining': 84}
    job.check_promotion(gate, **args)
    with pytest.raises(ValueError, match='budget differs'):
        job.check_promotion(gate, **{**args, 'cpu_remaining': 83})
    gate['cpu_remaining_seconds'] = 79
    with pytest.raises(ValueError, match='does not fit'):
        job.check_promotion(gate, **{**args, 'cpu_remaining': 79})
    gate['cpu_remaining_seconds'] = 84
    del components['failure_headroom']
    with pytest.raises(ValueError, match='components'):
        job.check_promotion(gate, **args)


def test_promotion_ledger_counts_remaining_once_and_rejects_overreservation():
    from experiments.tennis import w8_witness_job as job

    ledger = {'reservation_amount_basis': 'remaining_unspent', 'gpu_total_seconds': 200,
              'cpu_total_seconds': 250, 'workstreams': {'w8': {'gpu_limit_seconds': 120, 'cpu_limit_seconds': 140}},
              'charges': [{'workstream': 'w8', 'gpu_seconds': 20, 'cpu_seconds': 30}],
              'reservations': [{'phase': 'w8-full-witness-staged-20261106', 'workstream': 'w8',
                                'gpu_seconds': 70, 'cpu_seconds': 80, 'status': 'remaining_unspent'},
                               {'phase': 'other', 'workstream': 'ball', 'gpu_seconds': 50, 'cpu_seconds': 70,
                                'status': 'active'},
                               {'phase': 'old', 'workstream': 'w8', 'gpu_seconds': 9999, 'cpu_seconds': 9999,
                                'status': 'closed'}]}
    assert job.promotion_ledger_remaining(ledger) == (70, 80)
    ledger['charges'][0]['gpu_seconds'] += 10
    ledger['reservations'][0]['gpu_seconds'] -= 10
    assert job.promotion_ledger_remaining(ledger) == (60, 80)
    ledger['reservations'][1]['cpu_seconds'] = 200
    with pytest.raises(ValueError, match='ceiling'):
        job.promotion_ledger_remaining(ledger)


def test_pilot_builder_phase_and_staging_bind_exact_proof_host(tmp_path: Path, monkeypatch):
    from experiments.jobs import inbox
    from experiments.tennis import w8_strict_host_cost_job as host, w8_witness_job as job, w8_protected_metadata
    monkeypatch.setattr(w8_protected_metadata, 'reject_protected_alias', lambda _path: None)

    gate = tmp_path / 'cpu-gate.json'
    gate.write_text(json.dumps({'schema': 2, 'scope': job.CPU_SCOPE, 'passed': True, 'bootstrap_sha256': '9' * 64, 'input_job_sha256': '8' * 64, 'system_leaves_sha256': '7' * 64,
        'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'revision': 'c' * 40,
        'proof_campaign_environment_sha256': job.host.MAIN_ENV, 'generation_environment_sha256': job.host.JCV_ENV,
        'generation_controller_executed': False, 'cuda_qualified': False, 'host': 'gpu5', 'job_id': '20261106T000000Z-12345678',
        'spec_sha256': 'e' * 64, 'raw_receipt_sha256': dict.fromkeys(job.REQUIRED_PROOFS, 'f' * 64),
        'checks': dict.fromkeys(job.REQUIRED_CHECKS, True)}))
    spec = job.make_pilot_spec(deadline=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(), proof_local=gate,
        proof_remote='/home/itec/emanuele/Datasets/pointstream-data/tests/cpu-gate.json',
         ledger_local=first_pilot_ledger(tmp_path), ledger_remote='/home/itec/emanuele/Datasets/pointstream-data/tests/ledger.json', admission_local=first_pilot_admission(tmp_path, gate), admission_remote='/home/itec/emanuele/Datasets/pointstream-data/tests/admission.json')
    inbox.validate_spec(spec)
    assert inbox.phase_command(spec, 'smoke', python='/python')[:4] == ['/python', '-m', 'experiments.tennis.w8_witness_job', 'pilot']
    stage = tmp_path / 'job' / 'smoke'
    stage.mkdir(parents=True)
    (stage.parent / 'spec.json').write_text(json.dumps(spec))
    staged = []
    hashes = {}
    for item in spec['staged_inputs']:
        path = tmp_path / ('asset-' + item['name'])
        path.write_bytes(b'fixture')
        hashes[path] = item['sha256']
        staged.append({'name': item['name'], 'sha256': item['sha256'], 'source': item['path'],
                       'path': str(path), 'extract': False})
    (stage / 'staging.json').write_text(json.dumps({'inputs': staged}))
    dispatch: dict[str, Any] = {'cpu_affinity': [0], 'environment': {'alias': 'gpu5', 'gpu': {'name': 'NVIDIA RTX 6000 Ada Generation'}}}
    (stage / 'dispatch.json').write_text(json.dumps(dispatch))
    paths, _ = host.stage_inputs(stage, lambda p: hashes[Path(p)], witness=True)
    assert set(paths) == {item['name'] for item in spec['staged_inputs']}
    dispatch['environment']['alias'] = 'gpu6'
    (stage / 'dispatch.json').write_text(json.dumps(dispatch))
    with pytest.raises(ValueError, match='proof host'):
        host.stage_inputs(stage, lambda p: hashes[Path(p)], witness=True)


def test_pilot_parent_cpu_limit_allows_bounded_child_cap():
    import subprocess
    import sys

    code = '''import resource, subprocess, sys
resource.setrlimit(resource.RLIMIT_CPU, (30, 3600))
def child_limit(): resource.setrlimit(resource.RLIMIT_CPU, (100, 3600))
p = subprocess.run([sys.executable, '-c', 'import resource; assert resource.getrlimit(resource.RLIMIT_CPU) == (100, 3600)'], preexec_fn=child_limit)
assert p.returncode == 0
assert resource.getrlimit(resource.RLIMIT_CPU) == (30, 3600)
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr.decode()


def test_unit1_spec_requires_actual_ledger_and_forecast(tmp_path: Path):
    import time
    from experiments.tennis import w8_witness_job as job

    proof = tmp_path / 'proof.json'
    proof.write_text(json.dumps({'schema': 2, 'scope': job.CPU_SCOPE, 'passed': True, 'bootstrap_sha256': '9' * 64, 'input_job_sha256': '8' * 64, 'system_leaves_sha256': '7' * 64,
        'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'revision': 'c' * 40,
        'proof_campaign_environment_sha256': job.host.MAIN_ENV, 'generation_environment_sha256': job.host.JCV_ENV,
        'generation_controller_executed': False, 'cuda_qualified': False, 'host': 'gpu5', 'job_id': '20261106T000000Z-12345678',
        'spec_sha256': 'e' * 64, 'raw_receipt_sha256': dict.fromkeys(job.REQUIRED_PROOFS, 'f' * 64),
        'checks': dict.fromkeys(job.REQUIRED_CHECKS, True)}))
    ledger = tmp_path / 'ledger.json'
    ledger.write_text(json.dumps({'reservation_amount_basis': 'remaining_unspent', 'gpu_total_seconds': 3000,
        'cpu_total_seconds': 3000, 'workstreams': {'w8': {'gpu_limit_seconds': 2500, 'cpu_limit_seconds': 2500}},
        'charges': [{'workstream': 'w8', 'gpu_seconds': 20, 'cpu_seconds': 30}],
        'reservations': [{'phase': 'w8-full-witness-staged-20261106', 'workstream': 'w8',
                          'gpu_seconds': 2000, 'cpu_seconds': 2200, 'status': 'remaining_unspent'}]}))
    components = {name: {'gpu_seconds': 8.0, 'cpu_seconds': 8.0} for name in (
        'unit1_context', 'unit0_full', 'unit1_full', 'native_sam_compose', 'postseal_score',
        'offline_validation', 'external_staging_audits', 'failure_headroom')}
    components['unit1_context'] = {'gpu_seconds': 900.0, 'cpu_seconds': 1000.0}
    promotion = tmp_path / 'promotion.json'
    promotion.write_text(json.dumps({'schema': 1, 'scope': 'root-reviewed W8 remaining-work forecast',
        'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'cpu_proof_sha256': fixed.file_sha(proof),
        'prior_artifact_sha256': '1' * 64, 'ledger_sha256': fixed.file_sha(ledger),
        'raw_measurement_audit_sha256': '2' * 64, 'safety_factor': 1.5, 'remaining_components': components,
        'gpu_remaining_seconds': 2000, 'cpu_remaining_seconds': 2200,
        'qualification_charge_phases': [], 'qualification_gpu_spent_seconds': 0,
        'qualification_cpu_spent_seconds': 0, 'qualification_next_gpu_ceiling_seconds': 900,
        'qualification_next_cpu_ceiling_seconds': 1000,
        'issued_epoch': time.time() - 1, 'expires_epoch': time.time() + 100}))
    base = '/home/itec/emanuele/Datasets/pointstream-data/tests/'
    spec = job.make_unit1_spec(deadline=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(), proof_local=proof,
        proof_remote=base + 'proof.json', promotion_local=promotion, promotion_remote=base + 'promotion.json',
        ledger_local=ledger, ledger_remote=base + 'ledger.json', prior_artifact_sha='1' * 64)
    assert spec['scale']['phase'] == {'smoke': 'unit1', 'full': 'noop'}
    assert spec['witness_gpu_remaining_seconds'] == 2000 and spec['witness_cpu_remaining_seconds'] == 2200
    changed = json.loads(ledger.read_text())
    changed['reservations'][0]['cpu_seconds'] = 2199
    ledger.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='promotion evidence|budget differs'):
        job.make_unit1_spec(deadline=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(), proof_local=proof,
            proof_remote=base + 'proof.json', promotion_local=promotion, promotion_remote=base + 'promotion.json',
            ledger_local=ledger, ledger_remote=base + 'ledger.json', prior_artifact_sha='1' * 64)


def test_generated_pilot_phase_commands_reach_real_cli_dispatch(tmp_path: Path, monkeypatch):
    import sys
    from experiments.jobs import inbox
    from experiments.tennis import w8_witness_job as job
    monkeypatch.setattr(job.staged_cli, 'check_cli', lambda *args: None)

    proof = tmp_path / 'proof.json'
    proof.write_text(json.dumps({'schema': 2, 'scope': job.CPU_SCOPE, 'passed': True, 'bootstrap_sha256': '9' * 64, 'input_job_sha256': '8' * 64, 'system_leaves_sha256': '7' * 64,
        'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'revision': 'c' * 40,
        'proof_campaign_environment_sha256': job.host.MAIN_ENV, 'generation_environment_sha256': job.host.JCV_ENV,
        'generation_controller_executed': False, 'cuda_qualified': False, 'host': 'gpu5', 'job_id': '20261106T000000Z-12345678',
        'spec_sha256': 'e' * 64, 'raw_receipt_sha256': dict.fromkeys(job.REQUIRED_PROOFS, 'f' * 64),
        'checks': dict.fromkeys(job.REQUIRED_CHECKS, True)}))
    spec = job.make_pilot_spec(deadline=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(), proof_local=proof,
        proof_remote='/home/itec/emanuele/Datasets/pointstream-data/tests/proof.json',
        ledger_local=first_pilot_ledger(tmp_path), ledger_remote='/home/itec/emanuele/Datasets/pointstream-data/tests/ledger.json', admission_local=first_pilot_admission(tmp_path, proof), admission_remote='/home/itec/emanuele/Datasets/pointstream-data/tests/admission.json')
    stage = tmp_path / 'smoke'
    stage.mkdir()
    seen = []
    monkeypatch.setattr(job, 'run_pilot', lambda path, **kwargs: seen.append((path, kwargs)))
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setenv('PS_STAGE', 'smoke')
    cmd = inbox.phase_command(spec, 'smoke', python='/python')
    monkeypatch.setattr(sys, 'argv', ['w8_witness_job', *cmd[3:]])
    job.main()
    assert seen == [(stage, {})]
    monkeypatch.setenv('PS_STAGE', 'full')
    monkeypatch.setenv('PS_SCRATCH_DIR', str(tmp_path))
    cmd = inbox.phase_command(spec, 'full', python='/python')
    monkeypatch.setattr(sys, 'argv', ['w8_witness_job', *cmd[3:]])
    job.main()
    assert json.loads((tmp_path / 'publish' / 'noop.json').read_text())['scope'] == 'no additional scientific execution'


def test_candidate_spec_requires_qualified_producers_and_separate_readiness(tmp_path: Path, monkeypatch):
    import time
    import sys
    from experiments.jobs import inbox
    from experiments.tennis import w8_witness_job as job
    monkeypatch.setattr(job.staged_cli, 'check_cli', lambda *args: None)

    proof = tmp_path / 'proof.json'
    proof.write_text(json.dumps({'schema': 2, 'scope': job.CPU_SCOPE, 'passed': True, 'bootstrap_sha256': '9' * 64, 'input_job_sha256': '8' * 64, 'system_leaves_sha256': '7' * 64,
        'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'revision': 'c' * 40,
        'proof_campaign_environment_sha256': job.host.MAIN_ENV, 'generation_environment_sha256': job.host.JCV_ENV,
        'generation_controller_executed': False, 'cuda_qualified': False, 'host': 'gpu5', 'job_id': '20261106T000000Z-12345678',
        'spec_sha256': 'e' * 64, 'raw_receipt_sha256': dict.fromkeys(job.REQUIRED_PROOFS, 'f' * 64),
        'checks': dict.fromkeys(job.REQUIRED_CHECKS, True)}))
    ledger = tmp_path / 'ledger.json'
    ledger.write_text(json.dumps({'reservation_amount_basis': 'remaining_unspent', 'gpu_total_seconds': 6000,
        'cpu_total_seconds': 6000, 'workstreams': {'w8': {'gpu_limit_seconds': 5000, 'cpu_limit_seconds': 5000}},
        'charges': [{'workstream': 'w8', 'gpu_seconds': 100, 'cpu_seconds': 100}],
        'reservations': [{'phase': 'w8-full-witness-staged-20261106', 'workstream': 'w8',
                          'gpu_seconds': 4000, 'cpu_seconds': 4000, 'status': 'remaining_unspent'}]}))
    names = ('unit1_context', 'unit0_full', 'unit1_full', 'native_sam_compose', 'postseal_score',
             'offline_validation', 'external_staging_audits', 'failure_headroom')
    components = {name: {'gpu_seconds': 10.0, 'cpu_seconds': 10.0} for name in names}
    components['unit1_context'] = {'gpu_seconds': 0.0, 'cpu_seconds': 0.0}
    components['unit0_full'] = {'gpu_seconds': 1500.0, 'cpu_seconds': 1630.0}
    components['unit1_full'] = {'gpu_seconds': 1500.0, 'cpu_seconds': 1630.0}
    gate = {'schema': 1, 'scope': 'root-reviewed W8 remaining-work forecast', 'source_sha256': 'a' * 64,
            'snapshot_sha256': 'b' * 64, 'cpu_proof_sha256': fixed.file_sha(proof),
            'prior_artifact_sha256': '1' * 64, 'ledger_sha256': fixed.file_sha(ledger),
            'raw_measurement_audit_sha256': '2' * 64, 'safety_factor': 1.5, 'remaining_components': components,
            'completed_artifact_sha256': {'unit1_context': '3' * 64}, 'qualification_receipt_sha256': '4' * 64,
            'gpu_remaining_seconds': 4000, 'cpu_remaining_seconds': 4000,
            'candidate_next_limits': {'full_seconds': 1300, 'campaign_seconds': 1480,
                'gpu_ceiling_seconds': 1500, 'cpu_ceiling_seconds': 1630,
                'generator_cpu': 1100, 'generator_wall': 1100},
            'issued_epoch': time.time() - 1, 'expires_epoch': time.time() + 100}
    promotion = tmp_path / 'promotion.json'
    promotion.write_text(json.dumps(gate))
    base = '/home/itec/emanuele/Datasets/pointstream-data/tests/'
    spec = job.make_candidate_spec(deadline=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        proof_local=proof, proof_remote=base + 'proof.json', promotion_local=promotion,
        promotion_remote=base + 'promotion.json', ledger_local=ledger, ledger_remote=base + 'ledger.json',
        qualification_artifact_sha='1' * 64, unit1=False)
    assert inbox.phase_command(spec, 'smoke', python='/python')[3] == 'readiness'
    assert inbox.phase_command(spec, 'full', python='/python')[3] == 'candidate0'
    assert spec['full']['seconds'] == 1300 and spec['witness_gpu_ceiling_seconds'] == 1500
    stage = tmp_path / 'stage'
    stage.mkdir()
    calls: list[Any] = []
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setattr(job, 'run_readiness', lambda path: calls.append(('ready', path)))
    monkeypatch.setattr(job, 'run_pilot', lambda path, **kwargs: calls.append(('full', path, kwargs)))
    for phase in ('smoke', 'full'):
        command = inbox.phase_command(spec, phase, python='/python')
        monkeypatch.setattr(sys, 'argv', ['w8_witness_job', *command[3:]])
        job.main()
    assert calls == [('ready', stage), ('full', stage, {'unit1': False, 'candidate': True})]
    del gate['qualification_receipt_sha256']
    promotion.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match='qualification incomplete'):
        job.make_candidate_spec(deadline=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
            proof_local=proof, proof_remote=base + 'proof.json', promotion_local=promotion,
            promotion_remote=base + 'promotion.json', ledger_local=ledger, ledger_remote=base + 'ledger.json',
            qualification_artifact_sha='1' * 64, unit1=False)


def test_finish_reuses_exact_audited_producer_pair(tmp_path: Path, monkeypatch) -> None:
    import io
    import tarfile

    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    decoded = tmp_path / 'decoded'
    (decoded / 'publish' / 'units').mkdir(parents=True)
    unit_hashes = {}
    for unit in fixed.UNITS:
        path = decoded / 'publish' / 'units' / f'{unit}.npz'
        path.write_bytes(unit.encode())
        unit_hashes[unit] = fixed.file_sha(path)
    monkeypatch.setattr(witness.generation, 'ORACLE_UNIT_SHA', unit_hashes)
    observed = []
    monkeypatch.setattr(witness.generation, 'validate_context_qualification',
                        lambda path, expected: observed.append((path.name, expected['unit'])))
    archives, rows = {}, {}
    for index, unit in enumerate(fixed.UNITS):
        archive = tmp_path / f'{index}.tar'
        payloads = {'sealed.json': b'{}', f'{unit}@match5.npz': b'crop', f'{unit}@match5.json': b'{}',
                    'decoded-manifest.json': json.dumps({'match_tensors': {'x': 1}, 'checkpoint_sha256': 'c' * 64,
                                                         'unit_sha256': unit_hashes}).encode()}
        with tarfile.open(archive, 'w') as tar:
            for name, payload in payloads.items():
                entry = tarfile.TarInfo(name)
                entry.size = len(payload)
                tar.addfile(entry, io.BytesIO(payload))
        archives[unit] = archive
        rows[unit] = {'job_id': f'20261106T00000{index}Z-12345678', 'terminal_audit_sha256': 'd' * 64,
                      'archive_sha256': fixed.file_sha(archive)}
    audit = {'schema': 1, 'scope': 'root-reviewed W8 producer pair', 'producer_scope': 'context_qualification_only',
             'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'producers': rows}
    prepared = {'scratch': scratch, 'decoder_out': decoded, 'fingerprints': {'x': 1},
                'generated': {'checkpoint': {'sha256': 'c' * 64}, 'decoder_manifest': {'sha256': 'e' * 64}}}
    source = {'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64}
    result = witness.recover_audited_producers(prepared, audit, archives, qualification=True, source=source)
    assert set(result['outputs']) == set(fixed.UNITS)
    assert [x[1] for x in observed] == list(fixed.UNITS)
    other_scratch = tmp_path / 'other'
    other_scratch.mkdir()
    prepared['scratch'] = other_scratch
    swapped = {fixed.UNITS[0]: archives[fixed.UNITS[1]], fixed.UNITS[1]: archives[fixed.UNITS[0]]}
    with pytest.raises(ValueError, match='archive differs'):
        witness.recover_audited_producers(prepared, audit, swapped, qualification=True, source=source)
    with pytest.raises(ValueError, match='pair differs'):
        witness.recover_audited_producers(prepared, audit, archives, qualification=False, source=source)


@pytest.mark.parametrize('qualification', [True, False])
def test_finish_fixed_dispatch_and_cli(qualification: bool, monkeypatch, tmp_path: Path) -> None:
    import sys
    from experiments.jobs import inbox
    from experiments.tennis import w8_witness_job as job
    monkeypatch.setattr(job.staged_cli, 'check_cli', lambda *args: None)

    spec: dict[str, Any] = {'deadline': (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
            'w8_witness_profile': 'w8-context-qualification-v1' if qualification else 'w8-full-finish-v1',
            'entrypoint': ['-m', 'experiments.tennis.w8_witness_job'], 'arguments': ['{phase}'],
            'scale': {'phase': {'smoke': 'finish-readiness',
                                'full': 'finish-context' if qualification else 'finish-candidate'}},
            'smoke': {'seconds': 60}, 'full': {'seconds': 460 if qualification else 1700},
            'budget_seconds': 570 if qualification else 1850, 'validator_seconds': 50,
            'witness_gpu_ceiling_seconds': 600 if qualification else 1900,
            'witness_external_gpu_seconds': 30 if qualification else 50,
            'witness_cpu_ceiling_seconds': 670 if qualification else 1950,
            'witness_caps': {'parent_cpu': 30, 'packet_cpu': 100, 'native_cpu': 30, 'native_wall': 40,
                             'worker_cpu': 100 if qualification else 450,
                             'worker_wall': 88 if qualification else 460, 'external_cpu': 100},
            'witness_cpu_proof_host': 'gpu5', 'hosts': ['gpu5'], 'gpu_models': ['RTX 6000 Ada'],
            'device': 'gpu', 'gpu_sharing': {'mode': 'exclusive'}, 'single_cpu_campaign': True, 'cpu_threads': 1,
            'contention': {'policy': 'stop', 'resume_attempts': 0}, 'validator': ['{python}', '-m', 'experiments.tennis.w8_witness_job', 'finish-readiness-validate'],
            **{'witness_' + name + '_sha256': 'a' * 64 for name in ('producer_audit', 'producer0', 'producer1')}}
    job._finish_spec(spec, qualification=qualification)
    stage = tmp_path / 'full'
    stage.mkdir()
    called = []
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setattr(job, 'run_finish', lambda path, **kwargs: called.append((path, kwargs)))
    command = inbox.phase_command(spec, 'full', python='/python')
    monkeypatch.setattr(sys, 'argv', ['w8_witness_job', *command[3:]])
    job.main()
    assert called == [(stage, {'qualification': qualification})]
    spec['contention']['resume_attempts'] = 1
    with pytest.raises(ValueError, match='fixed dispatch'):
        job._finish_spec(spec, qualification=qualification)
    spec['contention']['resume_attempts'] = 0
    spec['hosts'] = ['gpu6']
    with pytest.raises(ValueError, match='fixed dispatch'):
        job._finish_spec(spec, qualification=qualification)


def test_qualification_subcap_uses_only_explicit_parent_charges_once() -> None:
    from experiments.tennis import w8_witness_job as job

    ledger = {'charges': [
        {'parent_phase': 'w8-full-witness-staged-20261106', 'phase': 'proof', 'witness_qualification': False,
         'gpu_seconds': 0, 'cpu_seconds': 100},
        {'parent_phase': 'w8-full-witness-staged-20261106', 'phase': 'pilot', 'witness_qualification': True,
         'gpu_seconds': 200, 'cpu_seconds': 300}]}
    gate = {'qualification_charge_phases': ['pilot'], 'qualification_gpu_spent_seconds': 200,
            'qualification_cpu_spent_seconds': 300}
    job.check_qualification_subcap(ledger, gate, gpu_cap=700, cpu_cap=700)
    with pytest.raises(ValueError, match='ceiling exceeded'):
        job.check_qualification_subcap(ledger, gate, gpu_cap=701, cpu_cap=700)
    ledger['charges'][1]['cpu_seconds'] = 650
    gate['qualification_cpu_spent_seconds'] = 650
    job.check_qualification_subcap(ledger, gate, gpu_cap=700, cpu_cap=550)
    job.check_qualification_subcap(ledger, gate, gpu_cap=700, cpu_cap=850)
    with pytest.raises(ValueError, match='ceiling exceeded'):
        job.check_qualification_subcap(ledger, gate, gpu_cap=700, cpu_cap=851)
    gate['qualification_cpu_spent_seconds'] = 0
    with pytest.raises(ValueError, match='ceiling exceeded'):
        job.check_qualification_subcap(ledger, gate, gpu_cap=700, cpu_cap=700)
    del ledger['charges'][0]['witness_qualification']
    with pytest.raises(ValueError, match='attribution'):
        job.check_qualification_subcap(ledger, gate, gpu_cap=700, cpu_cap=700)


def test_qualification_increased_ceiling_does_not_allow_unit1_replay(tmp_path: Path, monkeypatch) -> None:
    from experiments.tennis import w8_witness_job as job

    ledger = {'charges': [{'parent_phase': 'w8-full-witness-staged-20261106', 'phase': 'prep',
        'witness_qualification': True, 'gpu_seconds': 0, 'cpu_seconds': 650}]}
    gate = {'qualification_charge_phases': ['prep'], 'qualification_gpu_spent_seconds': 0,
        'qualification_cpu_spent_seconds': 650, 'qualification_next_gpu_ceiling_seconds': 500,
        'qualification_next_cpu_ceiling_seconds': 550, 'completed_artifact_sha256': {'unit1_context': 'a' * 64}}
    paths = {'ledger': tmp_path / 'ledger.json', 'promotion': tmp_path / 'promotion.json'}
    paths['ledger'].write_text(json.dumps(ledger))
    paths['promotion'].write_text(json.dumps(gate))
    spec = {'witness_gpu_remaining_seconds': 500, 'witness_cpu_remaining_seconds': 550,
        'witness_promotion_sha256': 'b' * 64, 'witness_cpu_proof_sha256': 'c' * 64,
        'witness_prior_artifact_sha256': 'd' * 64, 'w8_witness_profile': 'w8-context-unit1-v1',
        'witness_gpu_ceiling_seconds': 500, 'witness_cpu_ceiling_seconds': 550}
    monkeypatch.setattr(job, 'promotion_ledger_remaining', lambda _: (500, 550))
    monkeypatch.setattr(job, 'check_promotion', lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match='unit-1 context already completed; no replay'):
        job._unit1_promotion(paths, spec, {'code': {}, 'started': 0})


def test_finish_builder_binds_audit_archives_and_qualification_caps(tmp_path: Path, monkeypatch) -> None:
    from experiments.jobs import inbox
    from experiments.tennis import w8_witness_job as job

    deadline = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    monkeypatch.setattr(job, 'make_unit1_spec', lambda **_: {
        'deadline': deadline, 'entrypoint': ['-m', 'experiments.tennis.w8_witness_job'],
        'arguments': ['{phase}'], 'hosts': ['gpu5'], 'gpu_models': ['RTX 6000 Ada'],
        'witness_cpu_proof_host': 'gpu5', 'device': 'gpu', 'gpu_sharing': {'mode': 'exclusive'},
        'single_cpu_campaign': True, 'cpu_threads': 1, 'contention': {'policy': 'stop', 'resume_attempts': 0},
        'staged_inputs': [], 'inputs': []})
    monkeypatch.setattr(inbox, 'validate_spec', lambda _: None)
    gate: dict[str, Any] = {'qualification_charge_phases': ['pilot'], 'qualification_gpu_spent_seconds': 200,
            'qualification_cpu_spent_seconds': 300, 'qualification_next_gpu_ceiling_seconds': 600,
            'qualification_next_cpu_ceiling_seconds': 670,
            'completed_artifact_sha256': {'unit1_context': '3' * 64},
            'remaining_components': {'native_sam_compose': {'gpu_seconds': 600, 'cpu_seconds': 670}}}
    promotion = tmp_path / 'promotion.json'
    promotion.write_text(json.dumps(gate))
    ledger = tmp_path / 'ledger.json'
    ledger.write_text(json.dumps({'charges': [{'parent_phase': 'w8-full-witness-staged-20261106',
        'phase': 'pilot', 'witness_qualification': True, 'gpu_seconds': 200, 'cpu_seconds': 300}]}))
    audit, producer0, producer1 = (tmp_path / n for n in ('audit', 'p0', 'p1'))
    for item in (audit, producer0, producer1):
        item.write_bytes(item.name.encode())
    gate['completed_artifact_sha256']['unit1_context'] = fixed.file_sha(producer1)
    promotion.write_text(json.dumps(gate))
    base = '/home/itec/emanuele/Datasets/pointstream-data/tests/'
    spec = job.make_finish_spec(deadline=deadline, proof_local=audit, proof_remote=base + 'proof',
        promotion_local=promotion, promotion_remote=base + 'promotion', ledger_local=ledger,
        ledger_remote=base + 'ledger', producer_audit_local=audit, producer_audit_remote=base + 'audit',
        producer0_local=producer0, producer0_remote=base + 'p0', producer1_local=producer1,
        producer1_remote=base + 'p1', sam_remote='/home/itec/emanuele/Models/SAM/weights',
        native_inventory_remote=base + 'native', qualification=True)
    assert spec['witness_producer0_sha256'] == fixed.file_sha(producer0)
    assert spec['full']['seconds'] == 460 and spec['witness_caps']['worker_wall'] == 88
    gate['qualification_next_gpu_ceiling_seconds'] = 701
    gate['qualification_next_cpu_ceiling_seconds'] = 800
    promotion.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match='ceiling exceeded'):
        job.make_finish_spec(deadline=deadline, proof_local=audit, proof_remote=base + 'proof',
            promotion_local=promotion, promotion_remote=base + 'promotion', ledger_local=ledger,
            ledger_remote=base + 'ledger', producer_audit_local=audit, producer_audit_remote=base + 'audit',
            producer0_local=producer0, producer0_remote=base + 'p0', producer1_local=producer1,
            producer1_remote=base + 'p1', sam_remote='/home/itec/emanuele/Models/SAM/weights',
            native_inventory_remote=base + 'native', qualification=True)


def test_finish_publication_uses_real_fleet_archive(tmp_path: Path) -> None:
    from experiments.jobs import staging
    from experiments.tennis import w8_witness_job as job

    scratch, stage = tmp_path / 'scratch', tmp_path / 'stage'
    (scratch / 'publish').mkdir(parents=True)
    stage.mkdir()
    (scratch / 'publish' / 'summary.json').write_text(json.dumps({'scope': 'candidate'}))
    record = staging.publish_scratch(scratch, stage)
    assert record is not None
    assert job.read_finish_publication(stage / 'published.tar', maximum=1000) == {'scope': 'candidate'}
    (scratch / 'publish' / 'extra').write_text('unexpected')
    other = tmp_path / 'other'
    other.mkdir()
    staging.publish_scratch(scratch, other)
    with pytest.raises(ValueError, match='membership'):
        job.read_finish_publication(other / 'published.tar', maximum=1000)


def test_finish_rejects_contaminated_execution_and_nonterminal(tmp_path: Path) -> None:
    from experiments.tennis import w8_witness_job as job
    from experiments.jobs.inbox import digest

    root = tmp_path / '20261106T000000Z-12345678'
    stage = root / 'full'
    stage.mkdir(parents=True)
    execution = {'exit_code': 0, 'timing_contaminated': False, 'contention': [],
                 'gpu_sharing': {'mode': 'exclusive'}}
    (stage / 'execution.json').write_text(json.dumps(execution))
    (root / 'state.json').write_text(json.dumps({'status': 'complete', 'job_id': root.name}))
    (root / 'spec.json').write_text(json.dumps({'inputs': []}))
    (root / 'gate.json').write_text(json.dumps({'passed': True, 'inputs': [], 'spec_sha256': digest({'inputs': []}),
                                                'source_sha256': 'a' * 64}))
    dispatch = {'code': {'source_sha256': 'a' * 64}}
    job.check_stage_evidence(stage, dispatch, terminal=True)
    (root / 'spec.json').write_text(json.dumps({'inputs': [], 'tampered': True}))
    with pytest.raises(ValueError, match='terminal'):
        job.check_stage_evidence(stage, dispatch, terminal=True)
    (root / 'spec.json').write_text(json.dumps({'inputs': []}))
    execution['timing_contaminated'] = True
    (stage / 'execution.json').write_text(json.dumps(execution))
    with pytest.raises(ValueError, match='contaminated'):
        job.check_stage_evidence(stage, dispatch, terminal=True)
    execution['timing_contaminated'] = False
    (stage / 'execution.json').write_text(json.dumps(execution))
    (root / 'state.json').write_text(json.dumps({'status': 'contended', 'job_id': root.name}))
    with pytest.raises(ValueError, match='terminal'):
        job.check_stage_evidence(stage, dispatch, terminal=True)


def test_finish_readiness_validator_accepts_real_fleet_publication(tmp_path: Path, monkeypatch) -> None:
    from experiments.jobs import staging
    from experiments.tennis import w8_witness_job as job

    scratch, stage = tmp_path / 'scratch', tmp_path / 'smoke'
    (scratch / 'publish').mkdir(parents=True)
    stage.mkdir()
    source = 'a' * 64
    spec = {'w8_witness_profile': 'w8-context-qualification-v1',
            'witness_producer_audit_sha256': 'b' * 64,
            'witness_producer0_sha256': 'c' * 64, 'witness_producer1_sha256': 'd' * 64}
    dispatch = {'started': 1, 'code': {'source_sha256': source}}
    (scratch / 'publish' / 'summary.json').write_text(json.dumps({'scope': 'context_qualification_only',
        'source_sha256': source, 'producer_audit_sha256': 'b' * 64,
        'producer_archives': ['c' * 64, 'd' * 64]}))
    publication = staging.publish_scratch(scratch, stage)
    assert publication is not None
    (stage / 'staging.json').write_text(json.dumps({'published': publication}))
    (stage / 'execution.json').write_text(json.dumps({'exit_code': 0, 'timing_contaminated': False,
        'contention': [], 'gpu_sharing': {'mode': 'exclusive'}}))
    monkeypatch.setattr(job, 'load_released_inputs', lambda _: ({}, spec, dispatch))
    monkeypatch.setattr(job, '_finish_spec', lambda *_a, **_k: None)
    monkeypatch.setattr(job, '_unit1_promotion', lambda *_a, **_k: None)
    target = tmp_path / 'validation.json'
    job.validate_finish_readiness(stage, target)
    result = json.loads(target.read_text())
    assert result['passed'] is True and result['checks']['source_producers_and_budget'] is True


def test_finish_real_spec_staging_and_dispatch(tmp_path: Path, monkeypatch) -> None:
    import sys
    import time
    from experiments.jobs import inbox
    from experiments.tennis import w8_strict_host_cost_job as host, w8_witness_job as job, w8_protected_metadata
    monkeypatch.setattr(w8_protected_metadata, 'reject_protected_alias', lambda _path: None)

    proof = tmp_path / 'proof'
    proof.write_text(json.dumps({'schema': 2, 'scope': job.CPU_SCOPE, 'passed': True, 'bootstrap_sha256': '9' * 64, 'input_job_sha256': '8' * 64, 'system_leaves_sha256': '7' * 64,
        'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'revision': 'c' * 40,
        'proof_campaign_environment_sha256': job.host.MAIN_ENV, 'generation_environment_sha256': job.host.JCV_ENV,
        'generation_controller_executed': False, 'cuda_qualified': False, 'host': 'gpu5', 'job_id': '20261106T000000Z-12345678',
        'spec_sha256': 'e' * 64, 'raw_receipt_sha256': dict.fromkeys(job.REQUIRED_PROOFS, 'f' * 64),
        'checks': dict.fromkeys(job.REQUIRED_CHECKS, True)}))
    audit, p0, p1 = (tmp_path / name for name in ('audit', 'p0', 'p1'))
    for item in (audit, p0, p1):
        item.write_text(item.name)
    ledger = tmp_path / 'ledger'
    ledger.write_text(json.dumps({'reservation_amount_basis': 'remaining_unspent', 'gpu_total_seconds': 6000,
        'cpu_total_seconds': 6000, 'workstreams': {'w8': {'gpu_limit_seconds': 5000, 'cpu_limit_seconds': 5000}},
        'charges': [{'phase': 'pilot', 'parent_phase': 'w8-full-witness-staged-20261106',
                     'workstream': 'w8', 'gpu_seconds': 200, 'cpu_seconds': 300, 'witness_qualification': True}],
        'reservations': [{'phase': 'w8-full-witness-staged-20261106', 'workstream': 'w8',
                          'gpu_seconds': 4000, 'cpu_seconds': 4000, 'status': 'remaining_unspent'}]}))
    names = ('unit1_context', 'unit0_full', 'unit1_full', 'native_sam_compose', 'postseal_score',
             'offline_validation', 'external_staging_audits', 'failure_headroom')
    components = {name: {'gpu_seconds': 10, 'cpu_seconds': 10} for name in names}
    components['unit1_context'] = {'gpu_seconds': 0, 'cpu_seconds': 0}
    components['native_sam_compose'] = {'gpu_seconds': 600, 'cpu_seconds': 670}
    promotion = tmp_path / 'promotion'
    promotion.write_text(json.dumps({'schema': 1, 'scope': 'root-reviewed W8 remaining-work forecast',
        'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'cpu_proof_sha256': fixed.file_sha(proof),
        'prior_artifact_sha256': fixed.file_sha(audit), 'ledger_sha256': fixed.file_sha(ledger),
        'raw_measurement_audit_sha256': '2' * 64, 'safety_factor': 1.5, 'remaining_components': components,
        'completed_artifact_sha256': {'unit1_context': fixed.file_sha(p1)}, 'gpu_remaining_seconds': 4000,
        'cpu_remaining_seconds': 4000, 'issued_epoch': time.time() - 1, 'expires_epoch': time.time() + 100,
        'qualification_charge_phases': ['pilot'], 'qualification_gpu_spent_seconds': 200,
        'qualification_cpu_spent_seconds': 300, 'qualification_next_gpu_ceiling_seconds': 600,
        'qualification_next_cpu_ceiling_seconds': 670}))
    base = '/home/itec/emanuele/Datasets/pointstream-data/tests/'
    spec = job.make_finish_spec(deadline=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        proof_local=proof, proof_remote=base + 'proof', promotion_local=promotion, promotion_remote=base + 'promotion',
        ledger_local=ledger, ledger_remote=base + 'ledger', producer_audit_local=audit,
        producer_audit_remote=base + 'audit', producer0_local=p0, producer0_remote=base + 'p0',
        producer1_local=p1, producer1_remote=base + 'p1', sam_remote='/home/itec/emanuele/Models/SAM/weights',
        native_inventory_remote=base + 'native', qualification=True)
    inbox.validate_spec(spec)
    stage = tmp_path / 'job' / 'full'
    monkeypatch.setattr(job.staged_cli, 'check_cli', lambda *args: None)
    stage.mkdir(parents=True)
    rows = []
    digests = {}
    for staged_item in spec['staged_inputs']:
        path = tmp_path / ('asset-' + staged_item['name'])
        path.write_bytes(staged_item['name'].encode())
        rows.append({**staged_item, 'source': staged_item['path'], 'path': str(path)})
        digests[path] = staged_item['sha256']
    (stage.parent / 'spec.json').write_text(json.dumps(spec))
    (stage / 'staging.json').write_text(json.dumps({'inputs': rows}))
    dispatch: dict[str, Any] = {'cpu_affinity': [2], 'environment': {'alias': 'gpu5', 'gpu': {'name': 'RTX 6000 Ada'}}}
    (stage / 'dispatch.json').write_text(json.dumps(dispatch))
    actual, _ = host.stage_inputs(stage, digest_file=lambda path: digests[path], witness=True)
    assert {'producer_audit', 'producer0', 'producer1', 'sam_weights', 'native_inventory'} <= set(actual)
    called = []
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setattr(job, 'run_finish', lambda path, **kwargs: called.append((path, kwargs)))
    cmd = inbox.phase_command(spec, 'full', python='/python')
    monkeypatch.setattr(sys, 'argv', ['w8_witness_job', *cmd[3:]])
    job.main()
    assert called == [(stage, {'qualification': True})]
    dispatch['environment']['alias'] = 'gpu6'
    (stage / 'dispatch.json').write_text(json.dumps(dispatch))
    with pytest.raises(ValueError, match='dispatch differs'):
        host.stage_inputs(stage, digest_file=lambda path: digests[path], witness=True)


def test_first_pilot_rejects_prior_qualification_and_untagged_charges(tmp_path: Path) -> None:
    from experiments.tennis import w8_witness_job as job
    ledger = json.loads(first_pilot_ledger(tmp_path).read_text())
    ledger['charges'].extend([
        {'parent_phase': 'w8-full-witness-staged-20261106', 'phase': 'cpu-proof',
         'workstream': 'w8', 'gpu_seconds': 0, 'cpu_seconds': 400, 'witness_qualification': False},
        {'parent_phase': 'w8-full-witness-staged-20261106', 'phase': 'pilot-prep',
         'workstream': 'w8', 'gpu_seconds': 0, 'cpu_seconds': 50, 'witness_qualification': True,
         'qualification_charge_kind': 'pilot_preparation', 'qualification_attempt_nonce': 'a' * 32}])
    receipt = {'schema': 1, 'scope': 'root-reviewed first W8 qualification pilot', 'attempt_nonce': 'a' * 32,
               'ledger_sha256': 'b' * 64, 'cpu_proof_sha256': 'c' * 64, 'preparation_phases': ['pilot-prep'],
               'gpu_spent_seconds': 0, 'cpu_spent_seconds': 50, 'gpu_ceiling_seconds': 900,
               'cpu_ceiling_seconds': 950}
    def check() -> None:
        job.check_first_pilot_ledger(ledger, receipt, ledger_sha='b' * 64, cpu_proof_sha='c' * 64)
    check()
    ledger['charges'][1]['qualification_charge_kind'] = 'pilot_execution'
    with pytest.raises(ValueError, match='earlier execution'):
        check()
    ledger['charges'][1]['qualification_charge_kind'] = 'pilot_preparation'
    del ledger['charges'][0]['witness_qualification']
    with pytest.raises(ValueError, match='attribution'):
        check()


@pytest.mark.parametrize(('spent', 'accepted'), [(650, True), (950, True), (951, False)])
def test_first_pilot_cumulative_preparation_cpu_limit(tmp_path: Path, spent: int, accepted: bool) -> None:
    from experiments.tennis import w8_witness_job as job
    ledger = json.loads(first_pilot_ledger(tmp_path).read_text())
    ledger['charges'].append({'parent_phase': 'w8-full-witness-staged-20261106', 'phase': 'prep',
        'workstream': 'w8', 'gpu_seconds': 0, 'cpu_seconds': spent, 'witness_qualification': True,
        'qualification_charge_kind': 'pilot_preparation', 'qualification_attempt_nonce': 'a' * 32})
    receipt = {'schema': 1, 'scope': 'root-reviewed first W8 qualification pilot', 'attempt_nonce': 'a' * 32,
        'ledger_sha256': 'b' * 64, 'cpu_proof_sha256': 'c' * 64, 'preparation_phases': ['prep'],
        'gpu_spent_seconds': 0, 'cpu_spent_seconds': spent, 'gpu_ceiling_seconds': 500,
        'cpu_ceiling_seconds': 550}
    def check() -> None:
        job.check_first_pilot_ledger(ledger, receipt, ledger_sha='b' * 64, cpu_proof_sha='c' * 64)
    if not accepted:
        with pytest.raises(ValueError, match='cumulative qualification ceiling'):
            check()
        return
    check()
    ledger['charges'][0]['qualification_attempt_nonce'] = 'd' * 32
    with pytest.raises(ValueError, match='mixed'):
        check()
    ledger['charges'][0]['qualification_attempt_nonce'] = 'a' * 32
    ledger['charges'][0]['qualification_charge_kind'] = 'pilot_execution'
    with pytest.raises(ValueError, match='earlier execution'):
        check()


@pytest.mark.parametrize('runtime', ['main', 'jcv'])
def test_cpu_runtime_attestation_binds_actual_leaf_to_pinned_role(runtime):
    from experiments.tennis import w8_strict_host_cost_job as host
    from experiments.tennis import w8_witness_profile as profile

    digest = host.MAIN_ENV if runtime == 'main' else host.JCV_ENV
    prefix = '/local/users/emanuele/pointstream/cache/' + digest + '/tree'
    executable = prefix + '/bin/python3.12'
    preflight = {'receipts': {runtime: {'executable': executable, 'prefix': prefix, 'torch': 'pinned-torch'}},
                 'runtimes': {runtime: {'python': executable}}}
    config = {'allow_paths': [prefix], 'environment': {'CUDA_VISIBLE_DEVICES': ''}}
    receipt = {'scope': 'cpu_isolation_proof_only', 'cuda_initialized': False, 'visible_devices': '',
               'runtime_identity': {'executable': executable, 'prefix': prefix, 'torch': 'pinned-torch'}}
    profile.validate_cpu_runtime_attestation(receipt, config, preflight, runtime)
    for key, value in (('executable', '/other/bin/python'), ('prefix', '/other'), ('torch', 'wrong')):
        broken = dict(receipt, runtime_identity=dict(receipt['runtime_identity'], **{key: value}))
        with pytest.raises(ValueError, match='runtime attestation'):
            profile.validate_cpu_runtime_attestation(broken, config, preflight, runtime)
    with pytest.raises(ValueError, match='runtime attestation'):
        profile.validate_cpu_runtime_attestation(dict(receipt, runtime_identity={}), config, preflight, runtime)
    with pytest.raises(ValueError, match='runtime attestation'):
        profile.validate_cpu_runtime_attestation(receipt, dict(config, allow_paths=[]), preflight, runtime)
    with pytest.raises(ValueError, match='runtime attestation'):
        profile.validate_cpu_runtime_attestation(receipt, dict(config, environment={'CUDA_VISIBLE_DEVICES': 'GPU'}), preflight, runtime)


def versioned_qualification_fixture(tmp_path: Path) -> tuple[dict, dict, dict]:
    ledger = json.loads(first_pilot_ledger(tmp_path).read_text())
    ledger['reservations'][0]['cpu_seconds'] = 3000
    nonce = 'a' * 32
    ledger['charges'] = [{'parent_phase': 'w8-full-witness-staged-20261106', 'phase': 'prep',
        'workstream': 'w8', 'gpu_seconds': 0, 'cpu_seconds': 1050, 'witness_qualification': True,
        'qualification_charge_kind': 'pilot_preparation', 'qualification_attempt_nonce': nonce}]
    source = {'source_sha256': '1' * 64, 'snapshot_sha256': '2' * 64}
    ledger['qualification_budget_approvals'] = [{'status': 'approved', 'approval_sha256': 'd' * 64,
        'attempt_nonce': nonce, 'cumulative_cpu_ceiling_seconds': 1600,
        'no_prior_execution_audit_sha256': 'e' * 64, **source, 'cpu_proof_sha256': 'c' * 64}]
    receipt = {'schema': 2, 'scope': 'root-reviewed first W8 qualification pilot',
        'attempt_nonce': nonce, 'qualification_attempt_nonce': nonce,
        'qualification_cumulative_cpu_ceiling_seconds': 1600,
        'qualification_budget_approval_sha256': 'd' * 64,
        'no_prior_execution_audit_sha256': 'e' * 64, **source,
        'ledger_sha256': 'b' * 64, 'cpu_proof_sha256': 'c' * 64,
        'preparation_phases': ['prep'], 'gpu_spent_seconds': 0, 'cpu_spent_seconds': 1050,
        'gpu_ceiling_seconds': 500, 'cpu_ceiling_seconds': 550}
    return ledger, receipt, source


def test_versioned_first_pilot_requires_exact_budget_approval_and_no_prior_execution(tmp_path: Path) -> None:
    import copy
    from experiments.tennis import w8_witness_job as job

    ledger, receipt, source = versioned_qualification_fixture(tmp_path)
    def check(ledger_value: dict, receipt_value: dict, source_value: dict = source) -> None:
        job.check_first_pilot_ledger(ledger_value, receipt_value, ledger_sha='b' * 64,
                                     cpu_proof_sha='c' * 64, source=source_value)
    check(ledger, receipt)
    for field, value in (('qualification_cumulative_cpu_ceiling_seconds', None),
                         ('qualification_cumulative_cpu_ceiling_seconds', True),
                         ('qualification_cumulative_cpu_ceiling_seconds', 1599),
                         ('qualification_budget_approval_sha256', 'f' * 64),
                         ('qualification_attempt_nonce', 'f' * 32),
                         ('no_prior_execution_audit_sha256', None), ('source_sha256', 'f' * 64)):
        bad = copy.deepcopy(receipt)
        bad[field] = value
        with pytest.raises(ValueError):
            check(ledger, bad)
    for alteration in ('duplicate', 'retag', 'nonce', 'overreservation', 'approval_source', 'missing_phase'):
        bad = copy.deepcopy(ledger)
        if alteration == 'duplicate':
            bad['charges'].append(copy.deepcopy(bad['charges'][0]))
        elif alteration == 'retag':
            bad['charges'][0]['qualification_charge_kind'] = 'pilot_execution'
        elif alteration == 'nonce':
            bad['charges'][0]['qualification_attempt_nonce'] = 'f' * 32
        elif alteration == 'overreservation':
            bad['reservations'][0]['cpu_seconds'] = 500
        elif alteration == 'approval_source':
            bad['qualification_budget_approvals'][0]['source_sha256'] = 'f' * 64
        else:
            bad['charges'][0]['phase'] = 'unreviewed'
        with pytest.raises(ValueError):
            check(bad, receipt)
    bad = copy.deepcopy(receipt)
    bad['schema'] = 1
    with pytest.raises(ValueError, match='legacy artifact'):
        check(ledger, bad)
    bad = copy.deepcopy(receipt)
    bad['cpu_ceiling_seconds'] = 1001
    with pytest.raises(ValueError):
        check(ledger, bad)
    bad = copy.deepcopy(receipt)
    bad['gpu_ceiling_seconds'] = 901
    with pytest.raises(ValueError):
        check(ledger, bad)


def test_versioned_promotion_subcap_uses_same_nonce_and_approved_ledger(tmp_path: Path) -> None:
    import copy
    from experiments.tennis import w8_witness_job as job

    ledger, receipt, _ = versioned_qualification_fixture(tmp_path)
    gate = {k: receipt[k] for k in ('schema', 'source_sha256', 'snapshot_sha256', 'cpu_proof_sha256',
        'qualification_attempt_nonce', 'qualification_cumulative_cpu_ceiling_seconds',
        'qualification_budget_approval_sha256')}
    gate.update(qualification_charge_phases=['prep'], qualification_gpu_spent_seconds=0,
                qualification_cpu_spent_seconds=1050)
    job.check_qualification_subcap(ledger, gate, gpu_cap=500, cpu_cap=550)
    with pytest.raises(ValueError, match='ceiling exceeded'):
        job.check_qualification_subcap(ledger, gate, gpu_cap=500, cpu_cap=551)
    bad = copy.deepcopy(ledger)
    bad['charges'][0]['qualification_attempt_nonce'] = 'f' * 32
    with pytest.raises(ValueError, match='nonce differs'):
        job.check_qualification_subcap(bad, gate, gpu_cap=500, cpu_cap=550)
    bad = copy.deepcopy(ledger)
    bad['qualification_budget_approvals'].append(copy.deepcopy(bad['qualification_budget_approvals'][0]))
    with pytest.raises(ValueError, match='duplicated'):
        job.check_qualification_subcap(bad, gate, gpu_cap=500, cpu_cap=550)


def test_versioned_first_stage_binds_root_identity_and_proof(tmp_path: Path) -> None:
    from experiments.tennis import w8_witness_job as job

    ledger, receipt, source = versioned_qualification_fixture(tmp_path)
    paths = {'ledger': tmp_path / 'ledger.json', 'admission': tmp_path / 'admission.json',
             'cpu_proof': tmp_path / 'proof.json'}
    paths['ledger'].write_text(json.dumps(ledger))
    paths['admission'].write_text(json.dumps(receipt))
    paths['cpu_proof'].write_text(json.dumps(source))
    spec = {'witness_ledger_sha256': 'b' * 64, 'witness_cpu_proof_sha256': 'c' * 64,
            'witness_qualification_budget_approval_sha256': 'd' * 64,
            'witness_gpu_ceiling_seconds': 500, 'witness_cpu_ceiling_seconds': 550}
    job.check_first_pilot_stage(paths, spec)
    spec['witness_qualification_budget_approval_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='approval identity'):
        job.check_first_pilot_stage(paths, spec)
    spec['witness_qualification_budget_approval_sha256'] = 'd' * 64
    paths['cpu_proof'].write_text(json.dumps({**source, 'source_sha256': 'f' * 64}))
    with pytest.raises(ValueError, match='source or nonce'):
        job.check_first_pilot_stage(paths, spec)


def test_versioned_promotion_requires_explicit_allowance_and_pinned_root_digest():
    import time
    from experiments.tennis import w8_witness_job as job

    source = {'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64}
    components = {name: {'gpu_seconds': 8.0, 'cpu_seconds': 10.0} for name in (
        'unit1_context', 'unit0_full', 'unit1_full', 'native_sam_compose', 'postseal_score',
        'offline_validation', 'external_staging_audits', 'failure_headroom')}
    gate = {'schema': 2, 'scope': 'root-reviewed W8 remaining-work forecast', **source,
        'cpu_proof_sha256': 'c' * 64, 'prior_artifact_sha256': 'd' * 64, 'ledger_sha256': 'e' * 64,
        'qualification_budget_approval_sha256': 'f' * 64, 'qualification_attempt_nonce': 'a' * 32,
        'qualification_cumulative_cpu_ceiling_seconds': 1600,
        'raw_measurement_audit_sha256': 'f' * 64, 'safety_factor': 1.5, 'remaining_components': components,
        'gpu_remaining_seconds': 70, 'cpu_remaining_seconds': 84,
        'issued_epoch': time.time() - 1, 'expires_epoch': time.time() + 100}
    args = {'gate_sha': '1' * 64, 'expected_sha': '1' * 64, 'source': source, 'cpu_gate_sha': 'c' * 64,
            'prior_artifact_sha': 'd' * 64, 'ledger_sha': 'e' * 64, 'gpu_remaining': 70, 'cpu_remaining': 84}
    job.check_promotion(gate, **args)
    with pytest.raises(ValueError, match='promotion evidence'):
        job.check_promotion(gate, **{**args, 'expected_sha': '2' * 64})
    for key in ('qualification_cumulative_cpu_ceiling_seconds', 'qualification_budget_approval_sha256',
                'qualification_attempt_nonce'):
        bad = dict(gate)
        del bad[key]
        with pytest.raises(ValueError, match='allowance required'):
            job.check_promotion(bad, **args)
    gate['schema'] = 1
    with pytest.raises(ValueError, match='legacy artifact'):
        job.check_promotion(gate, **args)


def test_versioned_pilot_builder_pins_approval_and_staged_artifacts(tmp_path: Path) -> None:
    from experiments.tennis import w8_witness_job as job

    ledger, receipt, source = versioned_qualification_fixture(tmp_path)
    proof = tmp_path / 'proof.json'
    proof.write_text(json.dumps({'schema': 2, 'scope': job.CPU_SCOPE, 'passed': True,
        'bootstrap_sha256': '9' * 64, 'input_job_sha256': '8' * 64, 'system_leaves_sha256': '7' * 64,
        **source, 'revision': 'c' * 40, 'proof_campaign_environment_sha256': job.host.MAIN_ENV,
        'generation_environment_sha256': job.host.JCV_ENV, 'generation_controller_executed': False,
        'cuda_qualified': False, 'host': 'gpu5', 'job_id': '20261106T000000Z-12345678',
        'spec_sha256': 'e' * 64, 'raw_receipt_sha256': dict.fromkeys(job.REQUIRED_PROOFS, 'f' * 64),
        'checks': dict.fromkeys(job.REQUIRED_CHECKS, True)}))
    digest = fixed.file_sha(proof)
    receipt['cpu_proof_sha256'] = digest
    ledger['qualification_budget_approvals'][0]['cpu_proof_sha256'] = digest
    ledger_path = tmp_path / 'ledger-v2.json'
    ledger_path.write_text(json.dumps(ledger))
    receipt['ledger_sha256'] = fixed.file_sha(ledger_path)
    admission = tmp_path / 'admission-v2.json'
    admission.write_text(json.dumps(receipt))
    base = '/home/itec/emanuele/Datasets/pointstream-data/tests/'
    spec = job.make_pilot_spec(deadline=(datetime.now(timezone.utc) + timedelta(minutes=20)).isoformat(),
        proof_local=proof, proof_remote=base + 'proof.json', ledger_local=ledger_path,
        ledger_remote=base + 'ledger.json', admission_local=admission, admission_remote=base + 'admission.json')
    assert spec['witness_qualification_budget_approval_sha256'] == 'd' * 64
    assert spec['witness_admission_sha256'] == fixed.file_sha(admission)
    assert spec['witness_ledger_sha256'] == fixed.file_sha(ledger_path)
    assert spec['witness_cpu_proof_sha256'] == digest
    assert spec['witness_cpu_ceiling_seconds'] == 550 and spec['witness_gpu_ceiling_seconds'] == 500


def test_versioned_allowance_does_not_allow_completed_unit_replay(tmp_path: Path, monkeypatch) -> None:
    from experiments.tennis import w8_witness_job as job

    ledger, receipt, _ = versioned_qualification_fixture(tmp_path)
    gate = {k: receipt[k] for k in ('schema', 'source_sha256', 'snapshot_sha256', 'cpu_proof_sha256',
        'qualification_attempt_nonce', 'qualification_cumulative_cpu_ceiling_seconds',
        'qualification_budget_approval_sha256')}
    gate.update(qualification_charge_phases=['prep'], qualification_gpu_spent_seconds=0,
        qualification_cpu_spent_seconds=1050, qualification_next_gpu_ceiling_seconds=500,
        qualification_next_cpu_ceiling_seconds=550, completed_artifact_sha256={'unit1_context': 'a' * 64})
    paths = {'ledger': tmp_path / 'ledger.json', 'promotion': tmp_path / 'promotion.json'}
    paths['ledger'].write_text(json.dumps(ledger))
    paths['promotion'].write_text(json.dumps(gate))
    spec = {'witness_gpu_remaining_seconds': 4000, 'witness_cpu_remaining_seconds': 3000,
        'witness_promotion_sha256': 'b' * 64, 'witness_cpu_proof_sha256': 'c' * 64,
        'witness_prior_artifact_sha256': 'd' * 64, 'w8_witness_profile': 'w8-context-unit1-v1',
        'witness_qualification_budget_approval_sha256': 'd' * 64,
        'witness_gpu_ceiling_seconds': 500, 'witness_cpu_ceiling_seconds': 550}
    monkeypatch.setattr(job, 'check_promotion', lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match='unit-1 context already completed; no replay'):
        job._unit1_promotion(paths, spec, {'code': {}, 'started': 0})

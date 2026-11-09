import json
from pathlib import Path
from typing import Any

import pytest

from experiments.tennis import w8_input_job as fixed
from experiments.tennis import w8_witness_profile as witness


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
                                                 'base_sha256': base_sha, 'gpu_identity': gpu_identity}))
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
    from experiments.tennis import w8_receiver_finish as finish, w8_receiver_preflight as preflight

    monkeypatch.setattr(preflight, 'PROTECTED', {'source': ('/protected', 'sha')})
    old = {'source': {'path': '/protected', 'sha256': 'sha', 'inode': 4, 'device': 1}}
    fresh = {'source': {'path': '/protected', 'sha256': 'sha', 'inode': 4, 'device': 2}}
    finish._qual_sam_identity(old, fresh)
    with pytest.raises(ValueError, match='protected identities'):
        finish._qual_sam_identity({}, {})
    with pytest.raises(ValueError, match='protected identities'):
        finish._qual_sam_identity(old, {'source': {**fresh['source'], 'sha256': 'altered'}})
    fresh['source']['inode'] = 5
    with pytest.raises(ValueError, match='protected identities'):
        finish._qual_sam_identity(old, fresh)


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
    calls = []
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
    gate.write_text(json.dumps({'schema': 1, 'scope': job.CPU_SCOPE, 'passed': True,
        'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'revision': 'c' * 40,
        'environment_sha256': 'd' * 64, 'host': 'gpu5', 'job_id': '20261106T000000Z-12345678',
        'spec_sha256': 'e' * 64, 'raw_receipt_sha256': dict.fromkeys(job.REQUIRED_PROOFS, 'f' * 64),
        'checks': dict.fromkeys(job.REQUIRED_CHECKS, True)}))
    spec = job.make_pilot_spec(deadline='2099-01-01T00:00:00Z', proof_local=gate,
         proof_remote='/home/itec/emanuele/Datasets/pointstream-data/tests/cpu-gate.json')
    assert spec['w8_witness_profile'] == 'w8-context-pilot-v1'
    assert spec['scale']['phase'] == {'smoke': 'pilot', 'full': 'noop'}
    assert spec['smoke']['seconds'] <= 600 and spec['budget_seconds'] <= 900
    assert spec['witness_cpu_ceiling_seconds'] <= 1000
    changed = dict(spec, scale={'phase': {'smoke': 'pilot', 'full': 'pilot'}})
    with pytest.raises(ValueError, match='pilot spec'):
        job._pilot_spec(changed)


def test_namespace_completion_requires_started_join_and_absent_owned_pids(tmp_path: Path, monkeypatch) -> None:
    from experiments.tennis import w8_witness_namespace as namespace

    receipt = tmp_path / 'receipt.json'
    started_path = tmp_path / 'receipt.json.started'
    final = {'namespace_init': {'host_pid': 99999991, 'start_ticks': 1, 'host_pgid': 1},
             'worker_host_identity': {'host_pid': 99999992, 'start_ticks': 2, 'host_pgid': 1}}
    monkeypatch.setattr(namespace, 'validate_receipt', lambda *_args, **_kwargs: final)
    started = {'schema': 1, 'target': 'target', 'outside_namespace': 'pid:[outer]', **final}
    started_path.write_text(json.dumps(started))
    assert witness.namespace_completion(receipt, expected_pgid=1, expected_outside='pid:[outer]',
                                        expected_target='target') == final
    started['worker_host_identity'] = {'host_pid': 99999992, 'start_ticks': 3, 'host_pgid': 1}
    started_path.write_text(json.dumps(started))
    with pytest.raises(ValueError, match='pre-wait identity'):
        witness.namespace_completion(receipt, expected_pgid=1, expected_outside='pid:[outer]', expected_target='target')
    started_path.unlink()
    with pytest.raises(FileNotFoundError):
        witness.namespace_completion(receipt, expected_pgid=1, expected_outside='pid:[outer]', expected_target='target')


def test_pilot_cpu_gate_requires_raw_audits_and_exact_source():
    from experiments.tennis import w8_witness_job as job

    source = {'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'git_head': 'c' * 40}
    gate = {'schema': 1, 'scope': job.CPU_SCOPE, 'passed': True, **source,
            'revision': source['git_head'], 'environment_sha256': 'd' * 64, 'host': 'gpu5',
            'job_id': '20261106T000000Z-12345678', 'spec_sha256': 'e' * 64,
            'raw_receipt_sha256': {key: 'f' * 64 for key in job.REQUIRED_PROOFS},
            'checks': dict.fromkeys(job.REQUIRED_CHECKS, True)}
    job.check_cpu_gate(gate, gate_sha='1' * 64, expected_sha='1' * 64, source=source,
                       environment_sha='d' * 64, host_name='gpu5')
    gate['raw_receipt_sha256'].pop('native_timeout_kill')
    with pytest.raises(ValueError, match='proof references'):
        job.check_cpu_gate(gate, gate_sha='1' * 64, expected_sha='1' * 64, source=source,
                           environment_sha='d' * 64, host_name='gpu5')

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
    sam_data: dict[str, Any] = {**shared, 'files': {**gen_files, 'weights': finish.SAM_SHA, 'text': finish.SAM_SHA},
                'rows': rows, 'sam_record_sha256': 'preflight', 'device': 'Ada', 'gpu_identity': gpu_identity,
                'sam_source_modules': {'sam3': str(source)}, 'independent_frames': True, 'flash_events': ['flash'],
                'load': {'sdpa_backend_policy': 'native_flash', 'missing_keys': [], 'unexpected_keys': []}}
    (sam / 'sealed.json').write_text(json.dumps(sam_data))
    (compose / 'sealed.json').write_text(json.dumps({**shared, 'files': {**gen_files, **masks, 'base': base_sha},
          'ticks': finish.TICKS[:8].tolist(), 'composite_sha256': fixed.file_sha(compose / 'composite.npy')}))
    trusted: dict[str, Any] = {'native_receipt': native, 'sam_record_sha256': 'preflight', 'gpu_name': 'Ada',
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


def test_generation_helper_rejects_mixed_or_unbounded_modes() -> None:
    base: dict[str, Any] = dict(prepared={}, dispatch={}, visible='', worker_cpu=100, worker_wall=100)
    with pytest.raises(ValueError, match='qualification pair'):
        witness.execute_generation_units(**base, mode='context_qualification', units=(fixed.UNITS[0],))
    with pytest.raises(ValueError, match='qualification pair'):
        witness.execute_generation_units(**base, mode='generate', units=fixed.UNITS)
    with pytest.raises(ValueError, match='qualification pair'):
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

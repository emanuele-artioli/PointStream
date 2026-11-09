"""Fail-closed checks for the distinct W8 24-frame qualification output.

No job submission, resource release, source access or quality claim occurs here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import w8_input_job as fixed
from experiments.tennis import w8_receiver_finish as finish
from experiments.tennis import w8_receiver_generate as generation


def check_worker_isolation(config: dict[str, Any], isolation: dict[str, Any], *, runtime: dict[str, Any],
                           source_manifest: dict[str, Any], loader: dict[str, Any], loader_sha: str) -> None:
    """Recheck protected grants and clock/source bindings from independent inputs."""
    from experiments.tennis import w8_clock_profile as clock, w8_cuda_system_leaves as policy
    from experiments.tennis import w8_strict_host_cost_job as host

    expected_roots = host.strict_roots(Path(config['allow_paths'][0]), Path(config['output']), runtime, config['files'])
    if (len(config['allow_paths']) != len(expected_roots) or len(set(config['allow_paths'])) != len(expected_roots)
            or set(config['allow_paths']) != set(expected_roots)):
        raise ValueError('qualification strict grants differ')
    denied = {p for group in config['denied_paths'].values() for p in group}
    denials, mutation = isolation.get('denials', {}), isolation.get('mutation_denials', {})
    if (len(denied) != 10 or set(denials) != denied or any(x not in (1, 13) for x in denials.values())
            or isolation.get('landlock_abi', 0) < 3
            or isolation.get('handled_mask') != policy.handled_rights(isolation['landlock_abi'])
            or isolation.get('policy') != 'w8-approved-system-leaves-mutation-v1'
            or isolation.get('loader_identity_sha256') != loader_sha
            or isolation.get('output_write_ok') is not True or isolation.get('network_interfaces') != ['lo']
            or isolation.get('pre_sandbox_threads') != 1 or isolation.get('extra_fds') is not False
            or isolation.get('cpu_bootstrap', {}).get('cuda_initialized_before_isolation') is not False
            or isolation['cpu_bootstrap'].get('before_landlock_sha256') != isolation['cpu_bootstrap'].get('after_landlock_sha256')
            or isolation.get('namespaces', {}).get('user') == config['parent_user_namespace']
            or isolation['namespaces'].get('net') == config['parent_network_namespace']):
        raise ValueError('qualification strict isolation differs')
    if (set(mutation) != {'write', 'truncate', 'create', 'remove', 'mkdir', 'rename', 'link'}
            or any(v not in ((1, 13, 18) if k in ('rename', 'link') else (1, 13)) for k, v in mutation.items())):
        raise ValueError('qualification mutation controls differ')
    files, origins = source_manifest['files'], isolation.get('source_origins')
    if (not isinstance(origins, dict) or not origins
            or any(not isinstance(v, str) or v not in files
                   or v not in (k.replace('.', '/') + '.py', k.replace('.', '/') + '/__init__.py')
                   for k, v in origins.items())):
        raise ValueError('qualification executing source origins differ')
    policy.validate_leaf_receipt(isolation['approved_leaves'], loader)
    host.check_clock_source_binding(config, origins, files)
    clock.validate_receipt(isolation.get('clock_profile'))


def check_generation(outputs: dict[str, Path], expected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Bind each short-context output to independently supplied packet decoding identities."""
    if set(outputs) != set(fixed.UNITS) or set(expected) != set(fixed.UNITS):
        raise ValueError('both qualified units and independent expected identities required')
    result = {}
    for unit in fixed.UNITS:
        if expected[unit].get('unit') != unit:
            raise ValueError('unit identity differs')
        result[unit] = generation.validate_context_qualification(outputs[unit], expected[unit])
    return result


def check_finish(*, generation_dirs: dict[str, Path], expected: dict[str, dict[str, Any]],
                 base_dir: Path, sam_dir: Path, compose_dir: Path,
                 trusted: dict[str, Any]) -> dict[str, Any]:
    """Verify the short smoke separately; it cannot be a full-candidate seal."""
    generations = check_generation(generation_dirs, expected)
    ticks = finish.TICKS[:8]
    base = np.load(base_dir / 'base.npy', mmap_mode='r', allow_pickle=False)
    composite = np.load(compose_dir / 'composite.npy', mmap_mode='r', allow_pickle=False)
    if any(x.shape != (8, 1080, 1920, 3) or x.dtype != np.uint8 for x in (base, composite)):
        raise ValueError('qualification base or composite coverage differs')
    receipts = [json.loads((directory / 'sealed.json').read_text()) for directory in (base_dir, sam_dir, compose_dir)]
    for receipt in receipts:
        if receipt.get('scope') != 'context_qualification_only':
            raise ValueError('qualification finish scope differs')
        finish._isolation(receipt['isolation'])
        if (receipt.get('packet_sha256') != fixed.WINDOW_SHA
                or receipt.get('profile_sha256') != fixed.PROFILE_SHA):
            raise ValueError('qualification packet or profile differs')
    base_seal, sam_seal, compose_seal = receipts
    native = trusted.get('native_receipt')
    if (not isinstance(native, dict) or native != base_seal.get('native')
            or native.get('binary_sha256') != finish.FFMPEG_SHA or native.get('ivf_sha256') != finish.IVF_SHA
            or native.get('yuv_sha256') != '7490edf9b678db805c23baae49a2a3b5e9d7b037b18dd216320aad47a061beff'
            or native.get('exit_code') != 0 or native.get('decoder') != 'libdav1d'
            or native.get('yuv_bytes') != 240 * 480 * 272 * 3 // 2
            or native.get('landlock_abi', 0) < 1 or not native.get('denials')
            or any(v not in (1, 13) for v in native['denials'].values())
            or not native.get('loader_sha256') or not any('libdav1d' in p for p in native.get('libraries', {}))):
        raise ValueError('independent native receipt differs')
    if (sam_seal.get('sam_record_sha256') != trusted.get('sam_record_sha256')
            or not trusted.get('sam_record_sha256')
            or sam_seal.get('files', {}).get('weights') != finish.SAM_SHA
            or sam_seal['files'].get('text') != finish.SAM_SHA
            or sam_seal.get('device') != trusted.get('gpu_name') or not trusted.get('gpu_name')
            or sam_seal.get('load', {}).get('sdpa_backend_policy') != 'native_flash'
            or sam_seal['load'].get('missing_keys') or sam_seal['load'].get('unexpected_keys')):
        raise ValueError('independent SAM or device identities differ')
    from experiments.tennis.w8_cuda_system_leaves import canonical_gpu_uuid

    claimed = canonical_gpu_uuid(trusted.get('claimed_gpu_uuid', ''))
    for receipt in (base_seal, sam_seal):
        observed = receipt.get('gpu_identity', {})
        if (canonical_gpu_uuid(observed.get('uuid', '')) != claimed
                or canonical_gpu_uuid(observed.get('visible_uuid', '')) != claimed
                or (observed.get('major'), observed.get('minor')) != (8, 9)):
            raise ValueError('qualification physical GPU differs from dispatch claim')
    modules, verified = sam_seal.get('sam_source_modules'), trusted.get('sam_verified_files')
    if not isinstance(modules, dict) or not modules or not isinstance(verified, dict) or not verified:
        raise ValueError('independent SAM source mapping missing')
    for path in modules.values():
        if path not in verified or fixed.file_sha(Path(path)) != verified[path]:
            raise ValueError('SAM source module differs from preflight')
    if (base_seal.get('ticks') != ticks.tolist() or compose_seal.get('ticks') != ticks.tolist()
            or base_seal.get('base_sha256') != fixed.file_sha(base_dir / 'base.npy')
            or compose_seal.get('composite_sha256') != fixed.file_sha(compose_dir / 'composite.npy')
            or compose_seal.get('files', {}).get('base') != base_seal['base_sha256']
            or sam_seal.get('independent_frames') is not True or not sam_seal.get('flash_events')):
        raise ValueError('qualification output or SAM seal differs')
    sam_rows = {row['unit']: row for row in sam_seal['rows']}
    if set(sam_rows) != set(fixed.UNITS) or len(sam_seal['rows']) != 2:
        raise ValueError('qualification SAM units differ')
    for unit in fixed.UNITS:
        crop = generation_dirs[unit] / f'{unit}@match5.npz'
        seal = generation_dirs[unit] / 'sealed.json'
        mask = sam_dir / f'{unit}-masks.npz'
        with np.load(mask, allow_pickle=False) as archive:
            if (set(archive.files) != {'t', 'masks'} or not np.array_equal(archive['t'], ticks)
                    or archive['masks'].shape != (8, 784, 512) or archive['masks'].dtype != bool):
                raise ValueError('qualification mask coverage differs')
        for receipt in (sam_seal, compose_seal):
            if (receipt.get('files', {}).get(f'crop:{unit}') != fixed.file_sha(crop)
                    or receipt['files'].get(f'generation_seal:{unit}') != fixed.file_sha(seal)):
                raise ValueError('qualification generation artifact join differs')
        if (sam_rows[unit].get('file_sha256') != fixed.file_sha(mask)
                or compose_seal.get('files', {}).get(f'mask:{unit}') != fixed.file_sha(mask)):
            raise ValueError('qualification SAM mask join differs')
    return {'scope': 'context_qualification_only', 'ticks': ticks.tolist(), 'units': generations,
            'base_sha256': base_seal['base_sha256'], 'composite_sha256': compose_seal['composite_sha256']}


def execute_generation_units(prepared: dict[str, Any], dispatch: dict[str, Any], *,
                             visible: str, worker_cpu: int, worker_wall: int,
                             mode: str, units: tuple[str, ...]) -> dict[str, Any]:
    """Run explicitly selected contexts or one full unit after authenticated setup.

    The caller must enforce the whole-stage budget and same-source CPU proof;
    this function is not a fleet entrypoint or a release decision.
    """
    import os
    from experiments.tennis import w8_strict_host_cost_job as host
    from experiments.tennis import w8_cuda_system_leaves as policy

    if ((mode == 'context_qualification' and units != fixed.UNITS)
            or (mode == 'generate' and (len(units) != 1 or units[0] not in fixed.UNITS))
            or mode not in ('context_qualification', 'generate')
            or type(worker_cpu) is not int or not 1 <= worker_cpu <= 1800
            or type(worker_wall) is not int or not 11 <= worker_wall <= 1800
            or (mode == 'context_qualification' and (worker_cpu > 500 or worker_wall > 500))):
        raise ValueError('exact qualification pair or one bounded full candidate required')
    if (not visible or visible != os.environ.get('PS_CLAIMED_GPU_UUID')
            or policy.canonical_gpu_uuid(visible) != policy.canonical_gpu_uuid(
                (dispatch.get('environment', {}).get('gpu') or {}).get('uuid', ''))):
        raise ValueError('qualification claim and dispatch GPU differ')
    outputs, expected, usage = {}, {}, {}
    scratch, decoder = prepared['scratch'], prepared['decoder_out']
    for index, unit in enumerate(units):
        if index == 0:
            work, output = prepared['model_work'], prepared['model_out']
        else:
            work = scratch / 'witness-model-unit1'
            work.mkdir()
            output = work / 'output'
            output.mkdir()
        generated = dict(prepared['generated'])
        unit_path = decoder / 'publish' / 'units' / f'{unit}.npz'
        if fixed.file_sha(unit_path) != generation.ORACLE_UNIT_SHA[unit]:
            raise ValueError('decoded qualification unit differs from oracle')
        generated['unit'] = host._item(unit_path)
        config = prepared['config'](mode, generated, output, prepared['policies']['jcv'],
                                    worker_cpu, visible, unit=unit)
        usage[unit] = host._worker(Path(prepared['policies']['jcv']['python']), config, work, worker_cpu,
                                   worker_wall, prepared['root'], prepared['loader_receipt'],
                                   materialize_profile=True, clock_profile=True,
                                   witness_mode=mode)
        expected[unit] = {'unit': unit, 'packet_sha256': fixed.WINDOW_SHA, 'profile_sha256': fixed.PROFILE_SHA,
                          'tensor_fingerprints': prepared['fingerprints'],
                          'files': {k: v['sha256'] for k, v in generated.items()}}
        if mode == 'context_qualification':
            generation.validate_context_qualification(output, expected[unit])
        else:
            generation.validate_sealed(output, expected[unit])
        device = json.loads((output / 'gpu-identity.json').read_text())
        sealed = json.loads((output / 'sealed.json').read_text())
        check_worker_isolation(config, sealed['isolation'], runtime=prepared['policies']['jcv'],
                               source_manifest=prepared['code_manifest'],
                               loader=json.loads(prepared['loader_receipt'].read_text()),
                               loader_sha=fixed.file_sha(prepared['loader_receipt']))
        if (device.get('sealed_sha256') != fixed.file_sha(output / 'sealed.json')
                or device.get('config_sha256') != fixed.file_sha(work / 'worker-config.json')):
            raise ValueError('qualification device identity differs from worker')
        host.check_gpu_binding(device, sealed, visible, dispatch, prepared['code_manifest']['files'])
        materialized = json.loads((output / 'materialization.json').read_text())
        host.check_folder_receipt(materialized, config['files'])
        if (device.get('materialization_sha256') != fixed.file_sha(output / 'materialization.json')
                or materialized['wall_seconds'] + materialized['cleanup_wall_seconds'] > usage[unit]['wall_seconds']):
            raise ValueError('qualification materialization receipt differs')
        outputs[unit] = output
    checked = (check_generation(outputs, expected) if mode == 'context_qualification'
               else {units[0]: generation.validate_sealed(outputs[units[0]], expected[units[0]])})
    return {'outputs': outputs, 'expected': expected, 'usage': usage, 'validation': checked}

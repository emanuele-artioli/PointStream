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

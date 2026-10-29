"""W5 fleet job specifications for the Medvedev window (python -m experiments.tennis.w5_specs OUT_DIR TARGET_KBPS).

The masks and CRF-search encode jobs reuse W1's modules with `--window med`; the scoring job needs the
published archives of the others, so it is written separately (`score_spec`) once their sha256 are known.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from experiments.tennis import w1_specs as w1
from experiments.tennis.w1_roi import MED

DEADLINE = '2026-10-31T12:00:00+00:00'
MED_FILE = (w1.DATA + '/tennis/k6b-confirm-20261024-execution/sources/medvedev_struff__014.mp4', MED.source_sha)
FILES = {**w1.FILES, 'med': MED_FILE}


def _base(module: str, arguments: list[str], names: list[str], extra: dict[str, tuple[str, str]] | None = None) -> dict[str, Any]:
    spec = w1.base(module, arguments, names, {**FILES, **(extra or {})})
    spec['deadline'] = DEADLINE
    return spec


def masks() -> dict[str, Any]:
    spec = _base('experiments.tennis.w1_masks', ['--anchor', '{staged:anchor}', '--transport', '{staged:transport}',
                                                 '--window', 'med', '--judged', '{judged}'], ['anchor', 'transport'])
    spec.update({'device': 'cpu', 'hosts': w1.CPU_HOSTS, 'gpu_models': [], 'gpu_memory_mib': 0, 'cpu_threads': 2,
                 'local_storage_gib': 20, 'scale': {'judged': {'smoke': 3, 'full': 0}},
                 'smoke': {'seconds': 400, 'representative_basis': 'First three saved ticks of Med014@194 rebuilt with the full K6 drawing path and compared with the K6 window hashes of R and S.'},
                 'full': {'seconds': 1500}, 'validator_seconds': 120, 'budget_seconds': 2400, 'stall_seconds': 600.0,
                 'contention': {'policy': 'stop', 'resume_attempts': 0}})
    return spec


def encode(scale: int, target_kbps: float, *, controls: bool, outside: bool, threads: int, full_seconds: int = 2600) -> dict[str, Any]:
    """Plain (offset 0) or outside-R (+20/+40) CRF search at one resolution, rate-only, 240 frames."""
    arguments = ['--source', '{staged:med}', '--window', 'med', '--transport', '{staged:transport}', '--scale', str(scale),
                 '--frames', '{frames}', '--max-encodes', '{max_encodes}', '--target-kbps', repr(target_kbps)]
    arguments += ['--convention', 'outside', '--offsets', '-20', '-40'] if outside else ['--offsets', '0']
    arguments += ['--controls'] if controls else []
    spec = _base('experiments.tennis.w1_encode', arguments, ['med', 'transport'])
    spec.update({'device': 'cpu', 'hosts': w1.CPU_HOSTS, 'gpu_models': [], 'gpu_memory_mib': 0, 'cpu_threads': threads,
                 'local_storage_gib': 30,
                 'scale': {'frames': {'smoke': 24, 'full': 240}, 'max_encodes': {'smoke': 2, 'full': 12 if outside else 8}},
                 'smoke': {'seconds': 560, 'representative_basis': f'Med014@194 scale {scale}, 24 frames: the same conversion, ROI map, parallel offset lanes, two encodes per offset, independent dav1d decode and packaging as the full stage; 1.35x projection of the longest chain.'},
                 'full': {'seconds': full_seconds}, 'validator_seconds': 120, 'budget_seconds': full_seconds + 1100,
                 'stall_seconds': 900.0, 'contention': {'policy': 'stop', 'resume_attempts': 0}})
    return spec


def score_spec(masks_archive: tuple[str, str], hybrid: tuple[str, str], encodes: list[tuple[str, str]], *,
               primary: str = 'kps2', full_seconds: int = 2400) -> dict[str, Any]:
    """GPU scoring of every selected encode of the Med window; `encodes[0]` must carry the plain controls."""
    extra = {'masks': masks_archive, 'hybrid': hybrid, 'encode_half': encodes[0]}
    names = ['med', 'masks', 'hybrid', 'lpips', 'dists', 'encode_half']
    arguments = ['--source', '{staged:med}', '--window', 'med', '--masks', '{staged:masks}', '--hybrid', '{staged:hybrid}',
                 '--lpips', '{staged:lpips}', '--dists', '{staged:dists}', '--encode-half', '{staged:encode_half}',
                 '--primary', primary, '--judged', '{judged}']
    if len(encodes) > 1:
        arguments += ['--encode-extra'] + [f'{{staged:encode_x{i}}}' for i in range(len(encodes) - 1)]
        for i, item in enumerate(encodes[1:]):
            extra[f'encode_x{i}'] = item
            names.append(f'encode_x{i}')
    spec = _base('experiments.tennis.w1_score', arguments, names, extra)
    spec.update({'device': 'gpu', 'hosts': ['gpu3'], 'gpu_models': ['RTX A6000'], 'gpu_memory_mib': 8000, 'cpu_threads': 4,
                 'local_storage_gib': 30, 'scale': {'judged': {'smoke': 3, 'full': 0}},
                 'gpu_sharing': {'mode': 'shared', 'basis': 'Saved-pixel scoring; scores do not depend on wall time', 'margin_mib': 8192},
                 'contention': {'policy': 'continue', 'basis': 'Deterministic scoring of saved streams does not depend on foreign workload timing.', 'resume_attempts': 0},
                 'smoke': {'seconds': 560, 'representative_basis': 'Three saved ticks (0/3/6) of every selected stream and the three plain CRF 63 controls, with decode hashes, S masks, all four metrics and the per-tick reproduction of the step 1 job controls within 1e-4.'},
                 'full': {'seconds': full_seconds}, 'validator_seconds': 120, 'budget_seconds': full_seconds + 1000, 'stall_seconds': 900.0})
    spec['staged_inputs'] = [x for x in spec['staged_inputs']]
    return spec


def encode_specs(target_kbps: float) -> dict[str, dict[str, Any]]:
    """Four CPU searches: plain with the controls at half, plain at quarter, outside +20/+40 at both."""
    return {'encode-half': encode(2, target_kbps, controls=True, outside=False, threads=8),
            'encode-quarter': encode(4, target_kbps, controls=False, outside=False, threads=8),
            'encode-half-out': encode(2, target_kbps, controls=False, outside=True, threads=16),
            'encode-quarter-out': encode(4, target_kbps, controls=False, outside=True, threads=16)}


def main() -> None:
    out = Path(sys.argv[1])
    jobs = {'masks': masks()}
    if len(sys.argv) > 2:  # the encode searches need the hybrid's rate, known after the step 1 smoke
        jobs.update(encode_specs(float(sys.argv[2])))
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in jobs.items():
        (out / f'{name}.json').write_text(json.dumps(spec, indent=1) + '\n')


if __name__ == '__main__':
    main()

"""Write the W1 fleet job specifications (python -m experiments.tennis.w1_specs OUT_DIR)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from experiments.tennis.w1_roi import (ANCHOR_SHA, HYBRID_ARCHIVE_SHA, SOURCE_SHA, TRANSPORT_SHA, WEIGHTS)

DATA = '/home/itec/emanuele/Datasets/pointstream-data'
ENVIRONMENT = {'path': DATA + '/environments/pointstream-20261009T214002Z.tar.gz',
               'sha256': '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'}
FILES = {
    'al': (DATA + '/tennis/k6b-confirm-20261024-execution/sources/alcaraz_hurkacz__031.mp4', SOURCE_SHA),
    'transport': (DATA + '/jobs/fleet/inbox/20261025T082944Z-29f021a5/full/published.tar', TRANSPORT_SHA),
    'anchor': (DATA + '/jobs/fleet/inbox/20261024T201059Z-29586984/full/published.tar', ANCHOR_SHA),
    'hybrid': (DATA + '/jobs/fleet/inbox/20261027T211451Z-f6a754eb/full/published.tar', HYBRID_ARCHIVE_SHA),
    'lpips': (DATA + '/visor/h3-2026-10-10/inputs/lpips/alexnet-owt-7be5be79.pth', WEIGHTS['lpips']),
    'dists': (DATA + '/visor/h3b-2026-10-11/inputs/dists/vgg16-397923af.pth', WEIGHTS['dists']),
}
DEADLINE = '2026-10-30T12:00:00+00:00'
CPU_HOSTS = ['gpu2', 'gpu3', 'gpu6']  # gpu1 and gpu5 workers had no fresh heartbeat on 2026-10-29


def staged(names: list[str], extra: dict[str, tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    table = {**FILES, **(extra or {})}
    return [{'name': n, 'path': table[n][0], 'sha256': table[n][1], 'extract': False} for n in names]


def base(module: str, arguments: list[str], names: list[str], extra: dict[str, tuple[str, str]] | None = None) -> dict[str, Any]:
    items = staged(names, extra)
    return {'schema': 1, 'entrypoint': ['-m', module], 'arguments': arguments, 'staged_inputs': items,
            'inputs': [{'path': x['path'], 'sha256': x['sha256']} for x in items], 'environment': ENVIRONMENT,
            'required_commands': ['ffmpeg'], 'validator': ['{python}', '-m', module, '--validate'],
            'deadline': DEADLINE, 'verify_staged_inputs': True}


def masks() -> dict[str, Any]:
    spec = base('experiments.tennis.w1_masks', ['--anchor', '{staged:anchor}', '--transport', '{staged:transport}', '--judged', '{judged}'],
                ['anchor', 'transport'])
    spec.update({'device': 'cpu', 'hosts': CPU_HOSTS, 'gpu_models': [], 'gpu_memory_mib': 0, 'cpu_threads': 2,
                 'local_storage_gib': 20, 'scale': {'judged': {'smoke': 3, 'full': 0}},
                 'smoke': {'seconds': 400, 'representative_basis': 'First three saved ticks of Al031@129 rebuilt with the full K6 drawing path and compared with the K6 window hashes of R and S.'},
                 'full': {'seconds': 1500}, 'validator_seconds': 120, 'budget_seconds': 2400, 'stall_seconds': 600.0,
                 'contention': {'policy': 'stop', 'resume_attempts': 0}})
    return spec


def encode(scale: int, controls: bool, threads: int, full_seconds: int, outside: bool = False) -> dict[str, Any]:
    arguments = ['--al', '{staged:al}', '--transport', '{staged:transport}', '--scale', str(scale), '--frames', '{frames}',
                 '--max-encodes', '{max_encodes}'] + (['--controls'] if controls else [])
    if outside:
        arguments += ['--convention', 'outside', '--offsets', '-20', '-40']
    spec = base('experiments.tennis.w1_encode', arguments, ['al', 'transport'])
    spec.update({'device': 'cpu', 'hosts': CPU_HOSTS, 'gpu_models': [], 'gpu_memory_mib': 0, 'cpu_threads': threads,
                 'local_storage_gib': 30, 'scale': {'frames': {'smoke': 24, 'full': 240}, 'max_encodes': {'smoke': 2, 'full': 12 if outside else 8}},
                 'smoke': {'seconds': 560, 'representative_basis': f'Scale {scale}, 24 frames: the same conversion, ROI map, parallel offset lanes, two encodes per offset (same first CRF, so the ROI rate ordering is checked), independent dav1d decode and packaging as the full stage; 1.35x projection of the longest 8-encode chain.'},
                 'full': {'seconds': full_seconds}, 'validator_seconds': 120, 'budget_seconds': full_seconds + 1100,
                 'stall_seconds': 900.0, 'contention': {'policy': 'stop', 'resume_attempts': 0}})
    return spec


def maxbg() -> dict[str, Any]:
    spec = base('experiments.tennis.w1_maxbg', ['--al', '{staged:al}', '--transport', '{staged:transport}', '--frames', '{frames}',
                                                '--max-encodes', '{max_encodes}'], ['al', 'transport'])
    spec.update({'device': 'cpu', 'hosts': CPU_HOSTS, 'gpu_models': [], 'gpu_memory_mib': 0, 'cpu_threads': 24,
                 'local_storage_gib': 40, 'scale': {'frames': {'smoke': 24, 'full': 240}, 'max_encodes': {'smoke': 2, 'full': 14}},
                 'smoke': {'seconds': 560, 'representative_basis': 'Mode probe (two modes x offsets 0/-20/-40 at half resolution), then two encodes per scale and the three controls, decode, PSNR and packaging as in the full stage, on 24 frames.'},
                 'full': {'seconds': 2400}, 'validator_seconds': 120, 'budget_seconds': 3500, 'stall_seconds': 900.0,
                 'contention': {'policy': 'stop', 'resume_attempts': 0}})
    return spec


def roicheck() -> dict[str, Any]:
    spec = base('experiments.tennis.w1_roicheck', ['--al', '{staged:al}', '--transport', '{staged:transport}', '--frames', '{frames}'],
                ['al', 'transport'])
    spec.update({'device': 'cpu', 'hosts': CPU_HOSTS, 'gpu_models': [], 'gpu_memory_mib': 0, 'cpu_threads': 16,
                 'local_storage_gib': 30, 'scale': {'frames': {'smoke': 24, 'full': 240}},
                 'smoke': {'seconds': 560, 'representative_basis': 'Same conversion, ROI maps, overlays, eight encodes at half and at quarter resolution, decode and region PSNR as the full stage, on 24 frames.'},
                 'full': {'seconds': 2000}, 'validator_seconds': 120, 'budget_seconds': 3200, 'stall_seconds': 900.0,
                 'contention': {'policy': 'stop', 'resume_attempts': 0}})
    return spec


def score(half: tuple[str, str], other: tuple[str, str] | None, masks_archive: tuple[str, str],
          prior: tuple[str, str] | None = None) -> dict[str, Any]:
    extra = {'encode_half': half, 'masks': masks_archive}
    if other:
        extra['encode_full'] = other
    if prior:
        extra['prior'] = prior
    arguments = ['--al', '{staged:al}', '--masks', '{staged:masks}', '--hybrid', '{staged:hybrid}', '--lpips', '{staged:lpips}',
                 '--dists', '{staged:dists}', '--encode-half', '{staged:encode_half}',
                 '--judged', '{judged}'] + (['--encode-full', '{staged:encode_full}'] if other else []) + (['--prior', '{staged:prior}'] if prior else [])
    spec = base('experiments.tennis.w1_score', arguments,
                ['al', 'masks', 'hybrid', 'lpips', 'dists', 'encode_half'] + (['encode_full'] if other else []) + (['prior'] if prior else []), extra)
    spec.update({'device': 'gpu', 'hosts': ['gpu3'], 'gpu_models': ['RTX A6000'], 'gpu_memory_mib': 8000, 'cpu_threads': 4,
                 'local_storage_gib': 30, 'scale': {'judged': {'smoke': 3, 'full': 0}},
                 'gpu_sharing': {'mode': 'shared', 'basis': 'Saved-pixel scoring; scores do not depend on wall time', 'margin_mib': 8192},
                 'contention': {'policy': 'continue', 'basis': 'Deterministic scoring of saved streams does not depend on foreign workload timing.', 'resume_attempts': 0},
                 'smoke': {'seconds': 560, 'representative_basis': 'Three saved ticks (0/3/6) of every selected stream and the three plain CRF 63 controls, with decode hashes, S masks, all four metrics and the per-tick reproduction of the hybrid job controls within 1e-4.'},
                 'full': {'seconds': 2400}, 'validator_seconds': 120, 'budget_seconds': 3400, 'stall_seconds': 900.0})
    return spec


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    specs = {'masks': masks(), 'encode-half': encode(2, True, 16, 2600), 'encode-full': encode(1, False, 24, 2600),
             'encode-quarter': encode(4, False, 12, 2600), 'roicheck': roicheck(), 'maxbg': maxbg(),
             'encode-half-out': encode(2, True, 16, 2600, True), 'encode-quarter-out': encode(4, False, 12, 2600, True)}
    for name, spec in specs.items():
        (out / f'{name}.json').write_text(json.dumps(spec, indent=1) + '\n')


if __name__ == '__main__':
    main()

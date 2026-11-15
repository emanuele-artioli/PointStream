"""Build, but never submit, the bounded VISOR human-review SAM proposal job."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.segmentation.sam31 import DEFAULT_CHECKPOINT_SHA256

DATA = '/home/itec/emanuele/Datasets/pointstream-data'
RUNTIME_SHA = '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'


def spec(archive: str, archive_sha: str, deadline: str) -> dict[str, Any]:
    if not Path(archive).is_absolute() or not re.fullmatch(r'[0-9a-f]{64}', archive_sha):
        raise ValueError('absolute archive path and lowercase SHA256 required')
    if deadline != '2026-11-15T11:07:34Z':
        raise ValueError('deadline differs from authorized ceiling')
    inputs = [
        {'name': 'archive', 'path': archive, 'sha256': archive_sha, 'extract': False},
        {'name': 'checkpoint', 'path': DATA + '/background/g1-2026-10-08/inputs/sam3.1_multiplex.pt',
         'sha256': DEFAULT_CHECKPOINT_SHA256, 'extract': False},
    ]
    basis = 'Independent source-only review proposals; shared GPU timings are not scientific speed evidence.'
    return {
        'schema': 1, 'device': 'gpu', 'hosts': ['gpu3', 'gpu5', 'gpu6'],
        'gpu_models': ['RTX A6000', 'RTX 6000 Ada'], 'gpu_memory_mib': 14336, 'cpu_threads': 1,
        'single_cpu_campaign': True, 'cpu_affinity_index': 0,
        'gpu_sharing': {'mode': 'shared', 'margin_mib': 8192, 'basis': basis},
        'contention': {'policy': 'continue', 'basis': basis, 'resume_attempts': 0},
        'local_storage_gib': 30, 'entrypoint': ['-m', 'experiments.visor.human_sam_proposals'],
        'arguments': ['--archive', '{staged:archive}', '--checkpoint', '{staged:checkpoint}', '--frames', '{frames}'],
        'staged_inputs': inputs, 'inputs': [{'path': x['path'], 'sha256': x['sha256']} for x in inputs],
        'environment': {'path': DATA + '/environments/pointstream-20261009T214002Z.tar.gz', 'sha256': RUNTIME_SHA},
        'verify_staged_inputs': True, 'required_commands': [],
        'scale': {'frames': {'smoke': 3, 'full': 18}},
        'smoke': {'seconds': 180, 'representative_basis': 'One source frame per video, all three prompts and real polygon-derived box guidance, same model and conversion.'},
        'full': {'seconds': 600},
        'validator': ['{python}', '-m', 'experiments.visor.human_sam_proposals', '--validate'],
        'validator_seconds': 60, 'budget_seconds': 960, 'stall_seconds': 180,
        'deadline': deadline,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True)
    p.add_argument('--archive-sha256', required=True)
    p.add_argument('--deadline', required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    with args.output.open('x') as out:
        json.dump(spec(args.archive, args.archive_sha256, args.deadline), out, indent=2)
        out.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

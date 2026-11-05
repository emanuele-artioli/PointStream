"""Build a test_2-only source-judge fleet spec after verifying the reviewed join."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_opentt_source as worker

ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / 'docs/experiments/k5d-inputs/opentt-source-selection-20261104.json'
SELECTION_SHA = 'b104b87aefae92e3a3ea7766bc51e3b34f6c3c7a1f071bbbbca7eeb218984877'
WASB_TREE = 'd456047c70e9dbd3a2f7196e2d1991a069393ab68d7466d0ee7ecfd23360cab0'
DATA = '/home/itec/emanuele/Datasets/'


def build(receipt: Path, evidence: Path, receipt_remote: str, evidence_remote: str,
          selection_remote: str, deadline: str) -> dict[str, Any]:
    if worker.identity.sha(SELECTION) != SELECTION_SHA:
        raise ValueError('selection changed')
    clip = next(x for x in json.loads(SELECTION.read_text())['items'] if x['id'] == 'test_2')
    if (clip['start_zero_based_provisional'], clip['visible'], clip['absent'], clip['unknown']) != (1066, 85, 6, 149):
        raise ValueError('unexpected source context')
    receipt_sha, evidence_sha = worker.identity.sha(receipt), worker.identity.sha(evidence)
    worker.join_witness(receipt, receipt_sha, clip, evidence)
    for path in (receipt_remote, evidence_remote, selection_remote):
        if not path.startswith(DATA) or '..' in Path(path).parts:
            raise ValueError('staged receipt, evidence and selection must have Dataset paths')
    parsed = datetime.fromisoformat(deadline)
    if parsed.tzinfo is None:
        raise ValueError('deadline needs an explicit timezone')
    join = json.loads((ROOT / 'docs/experiments/k5d-inputs/opentt-join-pilot-spec-20261105.json').read_text())
    calibration = json.loads((ROOT / 'docs/experiments/k5d-inputs/calibration-fleet-spec-20261104.json').read_text())
    inputs = [dict(x) for x in join['staged_inputs']]
    inputs += [dict(x) for x in calibration['staged_inputs'] if x['name'] in {'wasb_code', 'wasb_weights', 'wasb_config'}]
    inputs += [{'name': 'selection', 'path': selection_remote, 'sha256': SELECTION_SHA},
               {'name': 'join_witness', 'path': receipt_remote, 'sha256': receipt_sha},
               {'name': 'join_evidence', 'path': evidence_remote, 'sha256': evidence_sha}]
    args = ['--clip', 'test_2', '--frames', '{frames}']
    for name in ('selection', 'video', 'labels', 'join_witness', 'join_evidence', 'wasb_code', 'wasb_weights', 'wasb_config'):
        args += ['--' + name.replace('_', '-'), '{staged:' + name + '}']
    for name, digest in [('selection', SELECTION_SHA), ('join-witness', receipt_sha), ('wasb-code', WASB_TREE),
                         ('wasb-weights', worker.WASB_SHA), ('wasb-config', worker.CONFIG_SHA)]:
        args += ['--' + name + '-sha256', digest]
    return {'schema': 1, 'device': 'gpu', 'hosts': join['hosts'], 'gpu_models': calibration['gpu_models'],
            'gpu_memory_mib': 8000, 'cpu_threads': 1, 'single_cpu_campaign': True,
            'entrypoint': ['-m', 'experiments.tennis.k5d_opentt_source'], 'arguments': args,
            'scale': {'frames': {'smoke': 16, 'full': 240}}, 'staged_inputs': inputs,
            'inputs': [{k: x[k] for k in ('path', 'sha256')} for x in inputs],
            'environment': calibration['environment'], 'local_storage_gib': 16, 'verify_staged_inputs': True,
            'required_commands': [], 'smoke': {'seconds': 80, 'representative_basis': '16 consecutive full source images and the frozen tennis WASB judge; source I/O, initialization and inference are timed separately; smoke is not a qualification result.'},
            'full': {'seconds': 120}, 'validator': ['{python}', '-m', 'experiments.tennis.k5d_opentt_source', '--validate-smoke'],
            'validator_seconds': 10, 'budget_seconds': 220, 'deadline': deadline, 'stall_seconds': 120}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('receipt', 'evidence'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('receipt-remote', 'evidence-remote', 'selection-remote', 'deadline'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.receipt, args.evidence, args.receipt_remote, args.evidence_remote,
                           args.selection_remote, args.deadline), indent=2))


if __name__ == '__main__':
    main()

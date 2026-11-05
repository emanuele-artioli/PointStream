"""Bounded 16/188 control confirmation; not a corpus or model qualification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.tennis import k5d_controls as controls
from experiments.tennis import k5d_fleet as fleet
from experiments.tennis import k5d_source_oracle as oracle

PILOT_SHA = {
    'pilot_smoke': 'e05684c829f65804c5317040fa5c6c04bc62299fb18f960b8fcd5219548b56be',
    'pilot_full': '8a877a2f8d424dd886b44ed002be606ee29c11607c95ae82fb1ee029f0504e8e',
    'cost_audit': 'f438382b94d44dcd81bc3a3bc17405d4ad2d78fd6866691ab20655ef11f5c0ad',
}
CALIBRATION_SHA = {
    'calibration_smoke': 'a39643080425a1b05da7bade7a74c0a96bbf14848434c121ae7f5ab8353e6873',
    'calibration_full': '622fd68d5380b97449a09872bc15402117dd9c3e7a2576a757cd6aff8fd090d7',
}
DATA = Path('/home/itec/emanuele/Datasets/pointstream-data')


def make_spec(audit_path: str, audit_copy: Path, deadline: str) -> dict:
    if not Path(audit_path).is_absolute() or not Path(audit_path).is_relative_to(DATA):
        raise ValueError('audit receipt must be an absolute data-root path')
    if oracle.sha(audit_copy) != PILOT_SHA['cost_audit']:
        raise ValueError('actual pilot audit SHA mismatch')
    audit = json.loads(audit_copy.read_text())
    if (audit['job_id'] != '20261105T021434Z-3ec159d8'
            or audit['stages']['smoke']['archive_sha256'] != PILOT_SHA['pilot_smoke']
            or audit['stages']['full']['archive_sha256'] != PILOT_SHA['pilot_full']):
        raise ValueError('audit is not the accepted pilot')
    base = fleet.ROOT / 'docs/experiments/k5d-inputs/calibration-controls-pilot-spec-20261105.json'
    spec = json.loads(base.read_text())
    wanted = {'archive', 'balltrack_code', 'balltrack_weight', 'wasb_code', 'wasb_weights', 'wasb_config',
              'lpips_backbone', 'dists_backbone', 'calibration_smoke', 'calibration_full'}
    declared = spec['staged_inputs']
    if len(declared) != len(wanted) or {item['name'] for item in declared} != wanted:
        raise ValueError('base control input set differs')
    old_full = next(item for item in declared if item['name'] == 'calibration_full')
    if old_full['sha256'] != controls.CALIBRATION_SHA['calibration_full']:
        raise ValueError('base pilot calibration differs')
    old_pair = {'path': old_full['path'], 'sha256': old_full['sha256']}
    if spec['inputs'].count(old_pair) != 1:
        raise ValueError('base calibration input binding differs')
    new_pair = {'path': str(DATA / 'jobs/fleet/inbox/20261105T003756Z-cc450baf/full/published.tar'),
                'sha256': CALIBRATION_SHA['calibration_full']}
    spec['inputs'][spec['inputs'].index(old_pair)] = new_pair
    old_full.update(new_pair)
    if next(item['sha256'] for item in declared if item['name'] == 'calibration_smoke') != CALIBRATION_SHA['calibration_smoke']:
        raise ValueError('base smoke calibration differs')
    inputs = [
        {'name': 'pilot_smoke', 'path': str(DATA / 'jobs/fleet/inbox/20261105T021434Z-3ec159d8/smoke/published.tar'), 'sha256': PILOT_SHA['pilot_smoke']},
        {'name': 'pilot_full', 'path': str(DATA / 'jobs/fleet/inbox/20261105T021434Z-3ec159d8/full/published.tar'), 'sha256': PILOT_SHA['pilot_full']},
        {'name': 'cost_audit', 'path': audit_path, 'sha256': PILOT_SHA['cost_audit']},
    ]
    spec['staged_inputs'].extend(inputs)
    spec['inputs'].extend([{k: item[k] for k in ('path', 'sha256')} for item in inputs])
    spec['entrypoint'] = ['-m', 'experiments.tennis.k5d_controls_confirmation']
    spec['scale']['frames'] = {'smoke': 16, 'full': 188}
    spec['smoke'] = {'seconds': 100, 'representative_basis': 'Exact five-arm 16-frame parity with independently audited 3ec159d8; slowdown-scaled affine cost gate before 188 frames.'}
    spec['full'] = {'seconds': 400}
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.k5d_controls_confirmation', '--validate-smoke']
    spec['validator_seconds'] = 30
    spec['budget_seconds'] = 550
    spec['deadline'] = deadline
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, choices=(16, 188))
    parser.add_argument('--validate-smoke', action='store_true')
    parser.add_argument('--make-spec', action='store_true')
    parser.add_argument('--audit-path')
    parser.add_argument('--audit-copy', type=Path)
    parser.add_argument('--deadline')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.make_spec:
        if not all((args.audit_path, args.audit_copy, args.deadline, args.output)):
            parser.error('actual audit path, copy, deadline and new output are required')
        spec = make_spec(args.audit_path, args.audit_copy, args.deadline)
        from experiments.jobs.inbox import validate_spec
        validate_spec(spec)
        with args.output.open('x') as stream:
            json.dump(spec, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.validate_smoke:
        controls.validate_smoke(stage, Path(os.environ['PS_VALIDATION_PATH']), full_frames=188,
                                calibration_sha=CALIBRATION_SHA, extra_sha=PILOT_SHA)
    elif args.frames in (16, 188):
        controls.run(stage, Path(os.environ['PS_SCRATCH_DIR']), args.frames, full_frames=188,
                     calibration_sha=CALIBRATION_SHA, extra_sha=PILOT_SHA)
    else:
        parser.error('16 or 188 frames required')


if __name__ == '__main__':
    main()

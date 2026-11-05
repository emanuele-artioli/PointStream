"""Bounded coalesced 16/48 checkpoint equivalence adapter; does not dispatch jobs."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_checkpoint as cp, k5d_checkpoint_fleet as cf
from experiments.tennis import k5d_coalesced as co, k5d_cost_pilot as cost
from experiments.tennis import k5d_fleet as fleet, k5d_source_oracle as oracle

FULL_SHA = 'f0dd118a5d64ca148be675a097606aa6dacb8ae49ffcac639ad30911ae74a727'
FULL_PATH = str(cf.DATA_ROOT / 'jobs/fleet/inbox/20261105T000550Z-d8b5c76b/full/published.tar')


def score_spec(base: dict[str, Any], *, small: dict[str, str], full: dict[str, str],
               reference16_local: Path, reference48_local: Path, deadline: str) -> dict[str, Any]:
    a = cf.checkpoint_archive('prepare16', small['remote'], small['archive_sha256'], Path(small['local']), small['checkpoint_sha256'], 16)
    b = cf.checkpoint_archive('prepare48', full['remote'], full['archive_sha256'], Path(full['local']), full['checkpoint_sha256'], 48)
    r16 = cf.asset('reference16', cf.D8_SMOKE_PATH, cf.D8_SMOKE_SHA, reference16_local, extract=False)
    r48 = cf.asset('reference48', FULL_PATH, FULL_SHA, reference48_local, extract=False)
    spec = cf.common_spec(base, [a, b, r16, r48], deadline)
    spec['entrypoint'] = ['-m', 'experiments.tennis.k5d_coalesced_fleet']
    spec['arguments'] = ['score', *base['arguments'], '--prepare16', '{staged:prepare16}', '--prepare48', '{staged:prepare48}',
                         '--checkpoint16-sha256', small['checkpoint_sha256'], '--checkpoint48-sha256', full['checkpoint_sha256'],
                         '--reference16', '{staged:reference16}', '--reference48', '{staged:reference48}']
    spec['smoke'] = {'seconds': 200, 'representative_basis': 'All six complete 16-frame receiver replays, shared judges and exact accepted-output comparison.'}
    spec['full'] = {'seconds': 420}
    spec['budget_seconds'] = 650
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.k5d_coalesced_fleet', 'validate']
    return spec


def run(args: argparse.Namespace) -> None:
    if {'smoke': 16, 'full': 48}.get(os.environ.get('PS_STAGE', '')) != args.frames:
        raise ValueError('stage frame scope differs')
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    identity = cp.execution_identity()
    if oracle.sha(fleet.SOURCE) != fleet.SOURCE_SHA or oracle.sha(fleet.EXPOSURE) != fleet.EXPOSURE_SHA:
        raise ValueError('source manifest or provenance changed')
    data = vars(args).copy()
    entries = {item['name']: item for item in json.loads((stage / 'staging.json').read_text())['inputs']}
    for name in ('wasb_code', 'balltrack_code'):
        archive = Path(data[name]).parent / Path(entries[name]['source']).name
        if oracle.sha(archive) != entries[name]['sha256'] or oracle.tree_sha(Path(data[name])) != fleet.archive_tree_sha(archive):
            raise ValueError('model source archive or installed tree differs')
    which = str(args.frames)
    data.update(checkpoint=str(Path(getattr(args, 'prepare' + which)) / 'publish'), output=str(scratch / 'publish'))
    checkpoint_sha = getattr(args, 'checkpoint' + which + '_sha256')
    co.score_all(argparse.Namespace(**data), checkpoint_sha, identity)
    sha = cf.D8_SMOKE_SHA if args.frames == 16 else FULL_SHA
    with cost.unpack(Path(getattr(args, 'reference' + which)), sha) as prior:
        comparison = cp.compare_equivalence(prior, scratch / 'publish' / 'assembled', oracle.sha(prior / 'pilot.json'))
    cp.write_new(scratch / 'publish' / 'comparison.json', comparison)


def validate() -> None:
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        stage = Path(os.environ['PS_STAGE_DIR'])
        entries = {x['name']: x for x in json.loads((stage / 'staging.json').read_text())['inputs']}
        with (cost.unpack(Path(entries['reference16']['path']), cf.D8_SMOKE_SHA) as prior,
              cost.unpack(stage / 'published.tar', json.loads((stage / 'staging.json').read_text())['published']['sha256']) as output):
            result = cp.compare_equivalence(prior, output / 'assembled', oracle.sha(prior / 'pilot.json'))
            if result != json.loads((output / 'comparison.json').read_text()) or result['frames'] != 16:
                raise ValueError('coalesced smoke comparison differs')
        projection = cf.forecast(stage, 420, 3, time.monotonic())
        target.write_text(json.dumps({'passed': True, 'checks': {'all_six_exact': True, 'forecast': True}, 'forecast': projection}) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'coalesced': False}, 'error': str(exc)}) + '\n')
        raise


def main() -> None:
    if sys.argv[1:] == ['validate']:
        validate()
        return
    if sys.argv[1:2] == ['--make-spec']:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        opts = parser.parse_args(sys.argv[2:])
        data = json.loads(opts.config.read_text())
        for name in ('reference16_local', 'reference48_local'):
            data[name] = Path(data[name])
        spec = score_spec(json.loads(cf.BASE.read_text()), **data)
        from experiments.jobs.inbox import validate_spec
        validate_spec(spec)
        cp.write_new(opts.output, spec)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('score',))
    for name in ('archive', 'balltrack-code', 'balltrack-weight', 'wasb-code', 'wasb-weights', 'wasb-config',
                 'image-ckpt', 'video-ckpt', 'lpips-backbone', 'dists-backbone', 'prepare16', 'prepare48',
                 'checkpoint16-sha256', 'checkpoint48-sha256', 'reference16', 'reference48'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--frames', required=True, type=int)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

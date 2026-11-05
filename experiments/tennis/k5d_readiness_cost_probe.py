"""CPU-only cost probe for a full source-header and asset readiness check.

No model, GPU, codec, image decode or production-admission decision is run.
Fleet hashes staged archives in each phase; the probe runs the production asset
checks and hashes the source once immediately before the full header scan. Historical checkpoint verification
uses its actual original execution identity.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import tarfile
import time
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_checkpoint as cp, k5d_checkpoint_fleet as cf
from experiments.tennis import k5d_control_readiness as readiness
from experiments.tennis import k5d_fleet as fleet, k5d_source_oracle as oracle

CLIP = 'tracknet/game10/Clip12'
FRAMES = 586
CHECKPOINT_SHA = 'c71b409f2fa3e4a17b424fbbaa145de453dd102286b398d48e352f3a4882f9b1'
CHECKPOINT_IDENTITY = {'source_sha256': '8a3b51c487cf9b456e304fdf24300f230ece3f3c7b2c8569ab6681fbd8166f13',
                       'snapshot_sha256': 'c3ff5419167504afcd84d25931791eddfe8bae16b5e492ed124df31e43f66f67',
                       'environment_sha256': '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'}


def staged_identities(stage: Path) -> dict[str, str]:
    data = json.loads((stage / 'staging.json').read_text())
    if data['environment']['sha256'] != CHECKPOINT_IDENTITY['environment_sha256']:
        raise ValueError('packed runtime identity differs')
    records = data['inputs']
    if len({r['name'] for r in records}) != len(records):
        raise ValueError('duplicate staged input')
    return {r['name']: r['sha256'] for r in records}


def verify_smoke(stage: Path) -> dict[str, Any]:
    smoke = stage.parent / 'smoke'
    staging = json.loads((smoke / 'staging.json').read_text())
    archive = smoke / 'published.tar'
    if oracle.sha(archive) != staging['published']['sha256']:
        raise ValueError('smoke publication SHA differs')
    with tarfile.open(archive) as tar:
        matches = [m for m in tar if m.name == 'publish/readiness-assets.json']
        if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 500_000:
            raise ValueError('assets smoke receipt absent')
        stream = tar.extractfile(matches[0])
        if stream is None:
            raise ValueError('assets smoke receipt unreadable')
        result = json.load(stream)
    if (result.get('scope') != 'readiness-cost only; no hardware or production admission'
            or result.get('staged_inputs') != staged_identities(smoke)):
        raise ValueError('assets smoke receipt differs from fleet staging')
    return result


def execute(args: argparse.Namespace) -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    identity = cp.execution_identity()
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if (dispatch['cpu_affinity'] != [0] or len(os.sched_getaffinity(0)) != 1
            or sorted(os.sched_getaffinity(0)) != [0] or dispatch['environment']['gpu'] is not None
            or identity['environment_sha256'] != CHECKPOINT_IDENTITY['environment_sha256']):
        raise ValueError('requires the intended gpu2 CPU0 and packed runtime without GPU')
    if {'smoke': 'assets', 'full': 'source'}.get(os.environ['PS_STAGE']) != args.mode:
        raise ValueError('stage mode differs')
    started = time.monotonic()
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    staged = staged_identities(stage)
    if args.mode == 'assets':
        details = readiness.asset_inputs(args, stage)
    else:
        prior = verify_smoke(stage)
        if prior['staged_inputs'] != staged:
            raise ValueError('source stage inputs differ from verified smoke')
        record = json.loads((stage / 'staging.json').read_text())
        archive = next(r for r in record['inputs'] if r['name'] == 'archive')
        if archive['sha256'] != fleet.EXPECTED['archive'] or Path(archive['path']) != Path(args.archive):
            raise ValueError('source staged archive differs')
        details = {'source': readiness.source(Path(args.archive), CLIP, FRAMES),
                   'checkpoint': cp.verify(Path(args.checkpoint) / 'publish', CHECKPOINT_SHA, CHECKPOINT_IDENTITY)}
        if details['checkpoint']['clip'] != 'tracknet/game10/Clip3' or details['checkpoint']['frames'] != 48:
            raise ValueError('historical full48 checkpoint scope differs')
    after, after_children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {'schema': 1, 'scope': 'readiness-cost only; no hardware or production admission',
               'mode': args.mode, 'staged_inputs': staged, 'execution_identity': identity,
               'checkpoint_execution_identity': CHECKPOINT_IDENTITY, 'details': details,
               'command_wall_seconds': time.monotonic() - started,
               'command_cpu_seconds': (after.ru_utime + after.ru_stime - own.ru_utime - own.ru_stime
                                       + after_children.ru_utime + after_children.ru_stime - children.ru_utime - children.ru_stime),
               'measurement_note': 'staging hash/I/O and environment setup are recorded separately by fleet'}
    output = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
    output.mkdir()
    cp.write_new(output / ('readiness-assets.json' if args.mode == 'assets' else 'readiness-source.json'), receipt)


def validate() -> None:
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        stage = Path(os.environ['PS_STAGE_DIR'])
        result = verify_smoke(stage)
        if (result.get('mode') != 'assets' or result.get('execution_identity') != cp.execution_identity()
                or not result['details'].get('source_hash_deferred_to_scan')):
            raise ValueError('assets readiness validation incomplete')
        target.write_text(json.dumps({'passed': True, 'checks': {'staged_assets_and_trees': True,
                                      'strict_source_identity': True}, 'scope': result['scope']}) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'readiness_cost': False}, 'error': str(exc)}) + '\n')
        raise


def make_spec(base: dict[str, Any], *, checkpoint_remote: str, checkpoint_sha256: str,
              checkpoint_archive_sha256: str, checkpoint_local: Path, deadline: str) -> dict[str, Any]:
    if base.get('arguments', [])[-2:] != ['--frames', '{frames}']:
        raise ValueError('unexpected calibration argument layout')
    if checkpoint_sha256 != CHECKPOINT_SHA:
        raise ValueError('exact fe451 checkpoint SHA required')
    item = cf.checkpoint_archive('checkpoint', checkpoint_remote, checkpoint_archive_sha256,
                                 checkpoint_local, checkpoint_sha256, 48)
    result = json.loads(json.dumps(base))
    result.update(device='cpu', hosts=['gpu2'], gpu_models=[], gpu_memory_mib=0, cpu_threads=1,
                  single_cpu_campaign=True, cpu_affinity_index=0, deadline=deadline,
                  entrypoint=['-m', 'experiments.tennis.k5d_readiness_cost_probe'])
    result.pop('gpu_sharing', None)
    result['staged_inputs'].append(item)
    result['inputs'].append({k: item[k] for k in ('path', 'sha256')})
    result['scale'] = {'mode': {'smoke': 'assets', 'full': 'source'}}
    result['arguments'] = [*base['arguments'][:-2], '--mode', '{mode}', '--checkpoint', '{staged:checkpoint}']
    result['smoke'] = {'seconds': 90, 'representative_basis': 'Real staged source and model assets, packed environment and pinned model trees, on gpu2 CPU0.'}
    result['full'] = {'seconds': 90}
    result['validator_seconds'] = 20
    result['budget_seconds'] = 240
    result['required_commands'] = []
    result['validator'] = ['{python}', '-m', 'experiments.tennis.k5d_readiness_cost_probe', '--validate']
    from experiments.jobs.inbox import validate_spec
    validate_spec(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true')
    for name in ('archive', 'balltrack-code', 'balltrack-weight', 'wasb-code', 'wasb-weights', 'wasb-config',
                 'image-ckpt', 'video-ckpt', 'lpips-backbone', 'dists-backbone', 'checkpoint', 'mode'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.validate:
        validate()
    elif any(getattr(args, name.replace('-', '_')) is None for name in
             ('archive', 'balltrack-code', 'balltrack-weight', 'wasb-code', 'wasb-weights', 'wasb-config',
              'image-ckpt', 'video-ckpt', 'lpips-backbone', 'dists-backbone', 'checkpoint', 'mode')):
        parser.error('all staged assets and mode are required')
    else:
        execute(args)


if __name__ == '__main__':
    main()

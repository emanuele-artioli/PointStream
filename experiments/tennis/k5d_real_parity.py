"""Proposed real 16-frame K5d parity proof; not a dispatch or admission.

The fixed worker reuses the frozen numerical producer and shared scorer, but
runs them with a new strict execution identity under the namespace launcher.
It compares the complete real path to the pre-existing 16-frame d8 reference.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

from experiments.tennis import k5d_checkpoint as cp, k5d_checkpoint_fleet as cf
from experiments.tennis import k5d_coalesced as co, k5d_cost_pilot as cost
from experiments.tennis import k5d_encoder as enc, k5d_fleet as fleet, k5d_source_oracle as oracle


def spec(base: dict, *, assets: dict[str, dict[str, str]], deadline: str, unshare_sha256: str,
         identity: dict[str, str]) -> dict:
    """Build a bounded candidate only after actual assets are available.

    This does not reserve resources or approve a fleet submission.
    """
    if set(assets) != {'fidelity_gate', 'accepted16', 'native_cpu_receipt'}:
        raise ValueError('fidelity gate, accepted reference and audited CPU receipt required')
    if not cf.digest_string(unshare_sha256) or assets['accepted16']['sha256'] != cf.D8_SMOKE_SHA:
        raise ValueError('unshare or historical reference identity differs')
    extra = [cf.asset(name, item['remote'], item['sha256'], Path(item['local']), extract=False)
             for name, item in assets.items()]
    cpu = json.loads(Path(assets['native_cpu_receipt']['local']).read_text())
    from experiments.tennis import k5d_scoped_namespace as namespace
    if (cpu.get('audited') is not True or cpu.get('passed') is not True
            or cpu.get('scope') != 'native CPU PID containment and accounting' or cpu.get('gpu_used') is not False
            or cpu.get('consumer_execution_identity') != identity
            or cpu.get('unshare_sha256') != unshare_sha256
            or cpu.get('namespace_wrapper_sha256') != oracle.sha(Path(namespace.__file__))):
        raise ValueError('audited actual CPU proof does not match namespace')
    result = cf.common_spec(base, extra, deadline)
    result['entrypoint'] = ['-m', 'experiments.tennis.k5d_scoped_namespace']
    result['arguments'] = ['outer', '--unshare-sha', unshare_sha256, '--timeout', '{namespace_timeout}', '--', '{proof_mode}',
                           *base['arguments'], '--fidelity-gate', '{staged:fidelity_gate}',
                           '--fidelity-gate-sha256', assets['fidelity_gate']['sha256'], '--accepted16', '{staged:accepted16}',
                           '--native-cpu-receipt', '{staged:native_cpu_receipt}',
                           '--native-cpu-receipt-sha256', assets['native_cpu_receipt']['sha256'],
                           '--unshare-sha256', unshare_sha256]
    result['scale']['frames'] = {'smoke': 16, 'full': 16}
    result['scale']['proof_mode'] = {'smoke': 'parity16', 'full': 'parity-noop'}
    result['scale']['namespace_timeout'] = {'smoke': 580, 'full': 4}
    result.update(gpu_models=['RTX A6000'], gpu_sharing={'mode': 'exclusive'}, budget_seconds=800,
                  validator_seconds=30, smoke={'seconds': 600, 'representative_basis':
                  'Actual 16-frame preparation and all six native replay, metric and judge paths with exact saved parity.'},
                  full={'seconds': 5}, validator=['{python}', '-m', 'experiments.tennis.k5d_real_parity', 'validate'])
    from experiments.jobs.inbox import validate_spec
    validate_spec(result)
    return result


def run(args: argparse.Namespace) -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    if os.environ['PS_STAGE'] != 'smoke' or args.frames != 16:
        raise ValueError('only the 16-frame calibration clip is admitted')
    identity = cp.execution_identity()
    cpu = json.loads(Path(args.native_cpu_receipt).read_text())
    if (oracle.sha(Path(args.native_cpu_receipt)) != args.native_cpu_receipt_sha256
            or cpu.get('schema') != 1 or cpu.get('audited') is not True or cpu.get('passed') is not True
            or cpu.get('scope') != 'native CPU PID containment and accounting' or cpu.get('gpu_used') is not False
            or cpu.get('consumer_execution_identity') != identity
            or not isinstance(cpu.get('job_id'), str) or not cpu['job_id'].startswith('2026')
            or cpu.get('all_owned_pids_absent') is not True or cpu.get('independent_cpu_match') is not True
            or cpu.get('cases') != ['normal', 'timeout', 'sigterm', 'sigkill']):
        raise ValueError('audited native CPU proof required')
    from experiments.tennis import k5d_scoped_namespace as namespace
    if (cpu.get('namespace_wrapper_sha256') != oracle.sha(Path(namespace.__file__))
            or cpu.get('unshare_sha256') != args.unshare_sha256):
        raise ValueError('CPU proof does not bind this exact namespace wrapper')
    import torch
    if (torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA RTX A6000'
            or tuple(torch.cuda.get_device_capability(0)) != (8, 6)):
        raise ValueError('exclusive A6000 CUDA device required')
    if oracle.sha(cf.AUDIT_PATH) != cf.AUDIT_SHA or oracle.sha(fleet.SOURCE) != fleet.SOURCE_SHA or oracle.sha(fleet.EXPOSURE) != fleet.EXPOSURE_SHA:
        raise ValueError('historical audit, source manifest or exposure differs')
    data = vars(args).copy()
    for name, digest in fleet.EXPECTED.items():
        data[name + '_sha256'] = digest
    data.update(clip=cost.CLIP, start_frame=0, manifest=str(fleet.SOURCE), manifest_sha256=fleet.SOURCE_SHA,
                exposure=str(fleet.EXPOSURE), exposure_sha256=fleet.EXPOSURE_SHA,
                code_revision=fleet.snapshot_head(stage), output=scratch / 'prepared',
                feasibility_no_metrics=False, cost_pilot=True)
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    data['snapshot_identity'] = {key: dispatch['code'].get(key) for key in
                                 ('git_head', 'source_sha256', 'snapshot_sha256', 'tracked_worktree_patch_sha256', 'included_untracked_sha256')}
    entries = {item['name']: item for item in json.loads((stage / 'staging.json').read_text())['inputs']}
    for name in ('wasb_code', 'balltrack_code'):
        archive = Path(data[name]).parent / Path(entries[name]['source']).name
        if oracle.sha(archive) != entries[name]['sha256']:
            raise ValueError('model code archive differs')
        expected = fleet.archive_tree_sha(archive)
        if oracle.tree_sha(Path(data[name])) != expected:
            raise ValueError('model code tree differs')
        data[name + '_sha256'] = expected
    for name, binary in (('ffmpeg', 'ffmpeg'), ('encoder', 'SvtAv1EncApp'), ('decoder', 'dav1d')):
        data[name] = str(Path(sys.prefix) / 'bin' / binary)
    cf.prepare_baseline()
    enc.fidelity_gate(Path(args.fidelity_gate), args.fidelity_gate_sha256, fleet.EXPECTED['archive'],
                      purpose='48-frame-cost-pilot', clip=cost.CLIP, frames=16)
    with cost.unpack(Path(args.accepted16), cf.D8_SMOKE_SHA) as prior:
        cp.prepare(argparse.Namespace(**data), identity)
        prepared = Path(data['output'])
        cf.prepare_parity(prepared, prior, identity)
        checkpoint_sha = oracle.sha(prepared / 'checkpoint.json')
        data.update(checkpoint=str(prepared), output=str(scratch / 'scored'))
        co.score_all(argparse.Namespace(**data), checkpoint_sha, identity)
        scored = Path(data['output'])
        comparison = cp.compare_equivalence(prior, scored / 'assembled', oracle.sha(prior / 'pilot.json'))
    publish = scratch / 'publish'
    publish.mkdir()
    shutil.move(str(prepared), publish / 'prepared')
    shutil.move(str(scored), publish / 'scored')
    cp.write_new(publish / 'parity.json', {'comparison': comparison, 'checkpoint_sha256': checkpoint_sha,
                 'execution_identity': identity, 'native_cpu_receipt_sha256': args.native_cpu_receipt_sha256,
                 'scope': 'exact 16-frame calibration parity; not an independent sample'})


def verify_scored(root: Path, identity: dict[str, str]) -> None:
    """Rebuild the output from all six saved points and require all arms ran."""
    import tempfile

    prepared, proof = root / 'prepared', root / 'scored'
    checkpoint_sha = oracle.sha(prepared / 'checkpoint.json')
    cp.verify(prepared, checkpoint_sha, identity)
    record = json.loads((proof / 'coalesced-validation.json').read_text())
    points = record['points']
    expected = [f'point-{codec}-{point}.json' for codec, point in enc.POINTS]
    if (record['checkpoint_sha256'] != checkpoint_sha or record['execution_identity'] != identity
            or [item['name'] for item in points] != expected):
        raise ValueError('six scored checkpoint proofs differ')
    files = []
    for item in points:
        path = proof / item['name']
        if oracle.sha(path) != item['sha256']:
            raise ValueError('point proof hash differs')
        files.append({'path': str(path), 'sha256': item['sha256']})
    with tempfile.TemporaryDirectory(prefix='k5d-parity-assemble-', dir=os.environ.get('TMPDIR')) as folder:
        assembled = Path(folder) / 'assembled'
        result = cp.assemble(prepared, checkpoint_sha, files, assembled, identity)
        if result != record['result'] or oracle.sha(assembled / 'pilot.json') != oracle.sha(proof / 'assembled' / 'pilot.json'):
            raise ValueError('saved assembly differs from scored points')
    report = json.loads((proof / 'assembled' / 'pilot.json').read_text())
    source = json.loads((proof / 'assembled' / 'source-qualification.json').read_text())
    if (len(source['rows']) != 16 or any(row['label_kind'] != 1 for row in source['rows'])
            or any(point['streak']['guard_off'] is not False for point in report['points'])):
        raise ValueError('calibration source or guard differs')
    events = json.loads((proof / 'timing.json').read_text())['events']
    required = {f'{kind}.{codec}.{point}.{arm}' for codec, point in enc.POINTS
                for kind in ('metrics', 'judges') for arm in ('baseline', 'streak')}
    if any(len(found := [e for e in events if e['name'] == name]) != 1 or found[0]['completed'] is not True
           for name in required):
        raise ValueError('metric or judge timing event missing or duplicated')


def validate() -> None:
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        stage = Path(os.environ['PS_STAGE_DIR'])
        entries = {item['name']: item for item in json.loads((stage / 'staging.json').read_text())['inputs']}
        published_sha = json.loads((stage / 'staging.json').read_text())['published']['sha256']
        with (cost.unpack(Path(entries['accepted16']['path']), cf.D8_SMOKE_SHA) as prior,
              cost.unpack(stage / 'published.tar', published_sha) as output):
            identity = cp.execution_identity()
            saved = json.loads((output / 'parity.json').read_text())
            verify_scored(output, identity)
            cf.prepare_parity(output / 'prepared', prior, identity)
            comparison = cp.compare_equivalence(prior, output / 'scored' / 'assembled', oracle.sha(prior / 'pilot.json'))
            if (saved['comparison'] != comparison or saved['execution_identity'] != identity
                    or saved['checkpoint_sha256'] != oracle.sha(output / 'prepared' / 'checkpoint.json')
                    or saved['native_cpu_receipt_sha256'] != entries['native_cpu_receipt']['sha256']):
                raise ValueError('published parity and execution identity differ')
        prefix = 'K5D_NAMESPACE_CANDIDATE '
        lines = [json.loads(line.removeprefix(prefix)) for line in (stage / 'command.log').read_text().splitlines()
                 if line.startswith(prefix)]
        if (len(lines) != 1 or lines[0].get('reason') != 'complete' or lines[0].get('status') != 0
                or lines[0].get('complete_descendant_wait') is not True
                or lines[0]['namespace_init']['namespace_pid'] != 1):
            raise ValueError('actual namespace execution proof missing')
        values = [lines[0]['children_cpu_seconds'], lines[0]['init_cpu_seconds']]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
            raise ValueError('actual namespace CPU accounting invalid')
        target.write_text(json.dumps({'passed': True, 'checks': {'actual_preparation_reference': True,
                          'all_six_native_and_pixel_parity': True, 'all_metric_and_judge_arrays': True,
                          'new_strict_checkpoint_identity': True}}) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'actual_parity': False}, 'error': str(exc)}) + '\n')
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'validate', 'noop'))
    for name in ('archive', 'balltrack-code', 'balltrack-weight', 'wasb-code', 'wasb-weights', 'wasb-config',
                 'image-ckpt', 'video-ckpt', 'lpips-backbone', 'dists-backbone', 'fidelity-gate',
                 'fidelity-gate-sha256', 'accepted16', 'native-cpu-receipt', 'native-cpu-receipt-sha256',
                 'unshare-sha256'):
        parser.add_argument('--' + name)
    parser.add_argument('--frames', type=int)
    args = parser.parse_args()
    if args.action == 'validate':
        validate()
    elif args.action == 'noop':
        if os.environ.get('PS_STAGE') != 'full':
            raise ValueError('noop is only permitted in full stage')
        output = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
        output.mkdir()
        cp.write_new(output / 'no-op.json', {'scope': 'no repeated parity work', 'stage': 'full'})
    else:
        if args.frames != 16 or any(getattr(args, name) is None for name in (
                'archive', 'balltrack_code', 'balltrack_weight', 'wasb_code', 'wasb_weights', 'wasb_config',
                'image_ckpt', 'video_ckpt', 'lpips_backbone', 'dists_backbone', 'fidelity_gate',
                'fidelity_gate_sha256', 'accepted16', 'native_cpu_receipt', 'native_cpu_receipt_sha256', 'unshare_sha256')):
            parser.error('all pinned 16-frame source, model and CPU-proof inputs required')
        run(args)


if __name__ == '__main__':
    main()

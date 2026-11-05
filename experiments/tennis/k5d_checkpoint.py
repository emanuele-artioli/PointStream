"""Checkpoint the full-clip K5d path; requires separately reviewed fleet admission.

No stage is admitted by this module. Preparation retains six complete native
contexts and one joint fit. Each scoring stage replays one immutable packet over
the complete clip, without re-encoding or fitting. An assembled report still
requires the normal K5d validator and the separately authorized fidelity gate.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k5d_encoder as enc, k5d_source_oracle as oracle
from experiments.tennis.k5d_timing import Timer


def write_new(path: Path, value: Any) -> None:
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def manifest_files(root: Path) -> dict[str, str]:
    """Bind every retained regular file; reject links and unexpected file kinds."""
    files = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink() or not (p.is_dir() or p.is_file()):
            raise ValueError('checkpoint contains link or special file')
        if p.is_file() and p != root / 'checkpoint.json':
            files[p.relative_to(root).as_posix()] = oracle.sha(p)
    return files


def execution_identity() -> dict[str, str]:
    stage = Path(os.environ['PS_STAGE_DIR'])
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    code = dispatch['code']
    return {'source_sha256': code['source_sha256'], 'snapshot_sha256': code['snapshot_sha256'],
            'environment_sha256': dispatch['staged_environment']['sha256']}


def validate_prepared(root: Path) -> dict[str, Any]:
    report = json.loads((root / 'pilot.json').read_text())
    qualification = json.loads((root / 'source-qualification.json').read_text())
    oracle.validate_report(qualification)
    gate = report.get('fidelity_gate')
    if gate:
        enc.fidelity_gate(root / 'fidelity-gate.json', gate['sha256'], qualification['identities']['archive']['sha256'],
                          purpose=gate.get('purpose', 'full-retrospective-benchmark'), clip=report['clip'], frames=report['frames'])
    elif not qualification.get('qualified'):
        raise ValueError('unqualified source lacks fidelity gate')
    if not report.get('checkpoint_prepare') or [(p['codec'], p['point']) for p in report['points']] != list(enc.POINTS):
        raise ValueError('prepare is not the complete six-point joint-fit path')
    if any(p[a]['metrics'] or p[a]['judges'] for p in report['points'] for a in ('baseline', 'streak')):
        raise ValueError('unexpected partial scores during preparation')
    for point in report['points']:
        folder = root / f"{point['codec']}-{point['point']}"
        if (oracle.sha(folder / 'native.bin') != point['native_sha256']
                or oracle.sha(folder / 'stream.psbl') != point['wire_sha256']
                or oracle.sha(folder / 'baseline.psbl') != point['baseline']['zero_ball_wire_sha256']
                or not point['decode_parity'] or not point['render_parity']):
            raise ValueError('prepare packet or fresh receiver parity failed')
    return report


def seal(root: Path, identity: dict[str, str]) -> dict[str, Any]:
    enc.validate(root, prepared=True)
    report = validate_prepared(root)
    data = {'schema': 1, 'execution_identity': identity, 'clip': report['clip'],
            'frames': report['frames'], 'points': [list(p) for p in enc.POINTS],
            'files': manifest_files(root), 'scope': 'full native contexts and joint fit; no scored output'}
    write_new(root / 'checkpoint.json', data)
    return data


def verify(root: Path, expected_sha: str, identity: dict[str, str]) -> dict[str, Any]:
    if oracle.sha(root / 'checkpoint.json') != expected_sha:
        raise ValueError('checkpoint manifest SHA256 mismatch')
    data = json.loads((root / 'checkpoint.json').read_text())
    if (data.get('schema') != 1 or data.get('execution_identity') != identity
            or data.get('points') != [list(p) for p in enc.POINTS]
            or data.get('files') != manifest_files(root)):
        raise ValueError('checkpoint content, source or runtime mismatch')
    report = validate_prepared(root)
    if (report['clip'] != data['clip'] or report['frames'] != data['frames']
            or report['full_metrics'] is not False):
        raise ValueError('checkpoint report scope mismatch')
    return data


def verify_score_inputs(args: argparse.Namespace, qualification: dict[str, Any], report: dict[str, Any]) -> None:
    """Rebind source, judge code and weights; never trust relocated filenames."""
    ids = qualification['identities']
    for name in ('archive', 'balltrack_weight', 'wasb_config'):
        if oracle.sha(Path(getattr(args, name))) != ids[name]['sha256']:
            raise ValueError('score input identity mismatch: ' + name)
    for name in ('balltrack_code', 'wasb_code'):
        if oracle.tree_sha(Path(getattr(args, name))) != ids[name]['tree_sha256']:
            raise ValueError('score code identity mismatch: ' + name)
    if oracle.sha(Path(args.wasb_weights) / 'wasb_tennis_best.pth.tar') != ids['wasb_weights']['sha256']:
        raise ValueError('score WASB weight identity mismatch')
    oracle.bind_config(Path(args.wasb_code), ids['wasb_config']['sha256'])
    for name in ('lpips_backbone', 'dists_backbone'):
        if oracle.sha(Path(getattr(args, name))) != report['identities'][name]['sha256']:
            raise ValueError('score metric weight identity mismatch: ' + name)


def score_point(args: argparse.Namespace, checkpoint_sha: str, identity: dict[str, str]) -> None:
    import torch
    from experiments.tennis import k5b_trackers as trk, windows as tw
    from src.codecs import quality

    root, output = Path(args.checkpoint), Path(args.output)
    verify(root, checkpoint_sha, identity)
    if len(os.sched_getaffinity(0)) != 1 or os.environ.get('PS_CPU_ALLOWANCE') != '1' or torch.cuda.device_count() != 1:
        raise ValueError('one CPU and one CUDA device required')
    report = json.loads((root / 'pilot.json').read_text())
    source_report = json.loads((root / 'source-qualification.json').read_text())
    oracle.validate_report(source_report)
    if list(torch.cuda.get_device_capability(0)) != source_report['device']['capability']:
        raise ValueError('GPU class differs from prepared source qualification')
    verify_score_inputs(args, source_report, report)
    key = (args.codec, args.point)
    if key not in enc.POINTS:
        raise ValueError('unregistered point')
    p = next(p for p in report['points'] if (p['codec'], p['point']) == key)
    game = int(report['clip'].split('/')[1][4:])
    clips = [c for c in trk.tracknet_clips(Path(args.archive), [game]) if c['id'] == report['clip']]
    if len(clips) != 1:
        raise ValueError('clip missing or ambiguous')
    clip = {**clips[0], 'rows': clips[0]['rows'][:report['frames']]}
    labels = trk.tracknet_labels(clip)
    if labels != [(r['label_kind'], r['label_x'], r['label_y']) for r in source_report['rows']]:
        raise ValueError('source labels or frame ordering changed')
    if [f"{clip['dir']}/{r[0]}" for r in clip['rows']] != [r['member'] for r in source_report['rows']]:
        raise ValueError('source frame join changed')
    output.mkdir(parents=True, exist_ok=False)
    timer = Timer(output / 'timing.json')
    images = timer.measure('source.decode', lambda: trk.tracknet_frames(Path(args.archive), clip))
    work = Path(os.environ['PS_SCRATCH_DIR']) / ('score-' + args.codec + '-' + str(args.point))
    wire = root / f'{args.codec}-{args.point}' / 'stream.psbl'
    cmd = [sys.executable, '-m', 'experiments.tennis.k5d_worker', 'decode', '--wire', str(wire),
           '--resources', str(root / 'resources.json'), '--work-dir', str(work)]
    timer.measure('receiver.decode', lambda: subprocess.run(cmd, check=True, capture_output=True, timeout=2400))
    receipt = json.loads((work / 'receiver.json').read_text())
    if (receipt['decoded_yuv_sha256'] != p['encoder_decoded_sha256']
            or receipt['rendered_rgb_sha256'] != p['encoder_rendered_sha256']
            or receipt['wire_sha256'] != p['wire_sha256']
            or receipt['resource_map_sha256'] != report['resource_map_sha256']):
        raise ValueError('fresh decoded or rendered pixels differ from prepare')
    n = report['frames']
    w, h = report['size']
    decoded = timer.measure('receiver.rgb', lambda: tw.rgb_frames(work / 'native.yuv', w, h, n, work / 'native-rgb.npy'))
    rendered = np.memmap(work / 'rendered.rgb', dtype=np.uint8, mode='r', shape=(n, h, w, 3))
    bt = timer.measure('model.balltrack', lambda: trk.BallTrack(Path(args.balltrack_code), Path(args.balltrack_weight)))
    zoo = timer.measure('model.wasb', lambda: trk.Zoo(Path(args.wasb_code), Path(args.wasb_weights), names=('wasb-step1',)))
    nets = (timer.measure('model.lpips', lambda: quality.load_lpips(Path(args.lpips_backbone), 'cuda')),
            timer.measure('model.dists', lambda: quality.load_dists(Path(args.dists_backbone), 'cuda')))
    if next(bt.net.parameters()).device.type != 'cuda' or next(zoo.settings['wasb-step1']['net'].parameters()).device.type != 'cuda':
        raise ValueError('judge device mismatch')
    def judges(data: np.ndarray) -> dict[str, Any]:
        return {'wasb': enc.prediction_rows(labels, zoo.run('wasb-step1', data)),
                'balltrack': enc.prediction_rows(labels, bt.run(data))}
    bm, bj, sm, sj, reuse = enc.score_arms(images, decoded, rendered, labels, nets, judges,
                                            p['streak']['guard_off'], timer, f'{args.codec}.{args.point}')
    p['baseline'].update(metrics=bm, judges=bj)
    p['streak'].update(metrics=sm, judges=sj, reuse=reuse)
    # Scoring's separately recreated receiver may have a different command path;
    # its exact pixel identities are bound to the original prepare receipt above.
    write_new(output / 'point.json', {'schema': 1, 'checkpoint_sha256': checkpoint_sha,
                                      'execution_identity': identity, 'point': p, 'receiver_replay': receipt})


def assemble(checkpoint: Path, checkpoint_sha: str, point_files: list[dict[str, str]], output: Path,
             identity: dict[str, str]) -> dict[str, Any]:
    verify(checkpoint, checkpoint_sha, identity)
    report = json.loads((checkpoint / 'pilot.json').read_text())
    expected = {(p['codec'], p['point']): p for p in report['points']}
    scores = {}
    for item in point_files:
        file = Path(item['path'])
        if oracle.sha(file) != item['sha256']:
            raise ValueError('point checkpoint SHA256 mismatch')
        record = json.loads(file.read_text())
        p = record['point']
        key = (p['codec'], p['point'])
        if (record['checkpoint_sha256'] != checkpoint_sha or record['execution_identity'] != identity
                or key not in expected or key in scores):
            raise ValueError('missing, duplicate or mismatched point checkpoint')
        baseline = {**p['baseline'], 'metrics': [], 'judges': {}}
        streak = {**p['streak'], 'metrics': [], 'judges': {}, 'reuse': None}
        if {**p, 'baseline': baseline, 'streak': streak} != expected[key]:
            raise ValueError('point metadata changed after preparation')
        replay = record['receiver_replay']
        if replay['decoded_yuv_sha256'] != p['encoder_decoded_sha256'] or replay['rendered_rgb_sha256'] != p['encoder_rendered_sha256']:
            raise ValueError('point replay differs from prepared pixels')
        scores[key] = p
    if set(scores) != set(expected):
        raise ValueError('all six point checkpoints required')
    if output.exists():
        raise ValueError('assemble output already exists')
    shutil.copytree(checkpoint, output)
    (output / 'checkpoint.json').unlink()
    report.pop('checkpoint_prepare')
    report['full_metrics'] = True
    report['scope'] = ('16/48-frame bounded cost pilot; filesystem denial not established'
                       if report.get('cost_pilot') else 'six-point calibration; filesystem denial not established')
    report['points'] = [scores[k] for k in enc.POINTS]
    (output / 'pilot.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for p in report['points']:
        (output / f"{p['codec']}-{p['point']}" / 'point.json').write_text(json.dumps(p, indent=2, allow_nan=False) + '\n')
    return enc.validate(output)


def compare_equivalence(prior: Path, new: Path, expected_prior_report_sha: str) -> dict[str, Any]:
    """Compare accepted monolithic and assembled scores without repeating inference."""
    if oracle.sha(prior / 'pilot.json') != expected_prior_report_sha:
        raise ValueError('accepted monolithic report identity mismatch')
    left = json.loads((prior / 'pilot.json').read_text())
    right = json.loads((new / 'pilot.json').read_text())
    enc.validate(prior)
    enc.validate(new)
    for key in ('frames', 'clip', 'size', 'fps', 'full_metrics'):
        if left[key] != right[key]:
            raise ValueError('monolithic contract differs: ' + key)
    for key in ('yuv_sha256', 'floor_psnr'):
        if left['conversion'][key] != right['conversion'][key]:
            raise ValueError('monolithic converted pixels differ: ' + key)
    old_source = json.loads((prior / 'source-qualification.json').read_text())
    new_source = json.loads((new / 'source-qualification.json').read_text())
    if old_source['rows'] != new_source['rows'] or old_source['scores'] != new_source['scores']:
        raise ValueError('monolithic source qualification differs')
    for a, b in zip(left['points'], right['points']):
        for key in ('codec', 'point', 'native_sha256', 'wire_sha256', 'encoder_decoded_sha256', 'encoder_rendered_sha256'):
            if a[key] != b[key]:
                raise ValueError('monolithic native or pixel mismatch: ' + key)
        for arm in ('baseline', 'streak'):
            if a[arm]['metrics'] != b[arm]['metrics'] or a[arm]['judges'] != b[arm]['judges']:
                raise ValueError('monolithic per-frame metrics or judge outputs differ')
        if (a['streak']['guard_off'] != b['streak']['guard_off']
                or a['baseline']['zero_ball_wire_sha256'] != b['baseline']['zero_ball_wire_sha256']):
            raise ValueError('monolithic guard or disabled packet differs')
    return {'passed': True, 'frames': right['frames'], 'clip': right['clip'],
            'prior_report_sha256': expected_prior_report_sha, 'assembled_report_sha256': oracle.sha(new / 'pilot.json'),
            'scope': 'exact frozen-output parity, not an independent scientific sample'}


def prepare(args: argparse.Namespace, identity: dict[str, str]) -> None:
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    out = Path(args.output).resolve()
    if not out.is_relative_to(scratch):
        raise ValueError('output must be inside task scratch')
    if getattr(args, 'feasibility_no_metrics', False):
        raise ValueError('checkpoint study requires full scoring contract')
    if getattr(args, 'cost_pilot', False):
        enc.fidelity_gate(Path(args.fidelity_gate), args.fidelity_gate_sha256, args.archive_sha256,
                          purpose='48-frame-cost-pilot', clip=args.clip, frames=args.frames)
    args.checkpoint_prepare = True
    args.timer = Timer(scratch / 'checkpoint-prepare-timing.json')
    enc.run(args)
    if getattr(args, 'cost_pilot', False):
        gate_path = out / 'fidelity-gate.json'
        if not gate_path.exists():
            shutil.copyfile(args.fidelity_gate, gate_path)
        if oracle.sha(gate_path) != args.fidelity_gate_sha256:
            raise ValueError('prepared fidelity gate identity changed')
        report_path = out / 'pilot.json'
        report = json.loads(report_path.read_text())
        report['fidelity_gate'] = {'sha256': args.fidelity_gate_sha256, 'scope': 'fidelity-only',
                                   'purpose': '48-frame-cost-pilot'}
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    seal(out, identity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'score', 'assemble', 'verify', 'compare'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    if oracle.sha(args.config) != args.config_sha256:
        raise ValueError('stage config identity mismatch')
    data = json.loads(args.config.read_text())
    if not isinstance(data, dict):
        raise ValueError('stage config must be a JSON object')
    identity = execution_identity()
    stage = Path(os.environ['PS_STAGE_DIR'])
    spec = json.loads((stage.parent / 'spec.json').read_text())
    phase = os.environ['PS_STAGE']
    wall_limit = data['wall_budget_seconds']
    cpu_limit = data['cpu_budget_seconds']
    if (type(wall_limit) is not int or type(cpu_limit) is not int
            or not 0 < wall_limit <= 2700 or not 0 < cpu_limit <= 2700
            or spec['cpu_threads'] != 1 or spec.get('single_cpu_campaign') is not True
            or spec[phase]['seconds'] > wall_limit or spec['budget_seconds'] > wall_limit
            or wall_limit > cpu_limit):
        raise ValueError('bounded single-core stage budget required')
    started = time.monotonic()
    own = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    if args.action == 'prepare':
        data['output'] = Path(data['output'])
        prepare(argparse.Namespace(**data), identity)
    elif args.action == 'score':
        if not Path(data['output']).resolve().is_relative_to(scratch):
            raise ValueError('score output must be inside task scratch')
        score_point(argparse.Namespace(**data), data['checkpoint_sha256'], identity)
    elif args.action == 'verify':
        verify(Path(data['checkpoint']), data['checkpoint_sha256'], identity)
    elif args.action == 'compare':
        output = Path(data['output'])
        if not output.resolve().is_relative_to(scratch):
            raise ValueError('comparison output must be inside task scratch')
        result = compare_equivalence(Path(data['prior']), Path(data['assembled']), data['prior_report_sha256'])
        write_new(output, result)
    else:
        output = Path(data['output'])
        if not output.resolve().is_relative_to(scratch):
            raise ValueError('assembly output must be inside task scratch')
        result = assemble(Path(data['checkpoint']), data['checkpoint_sha256'],
                          data['point_files'], output, identity)
        write_new(output / 'assembly-validation.json', result)
    elapsed = time.monotonic() - started
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    used = a.ru_utime + a.ru_stime - own.ru_utime - own.ru_stime + b.ru_utime + b.ru_stime - child.ru_utime - child.ru_stime
    if elapsed > wall_limit or used > cpu_limit:
        raise ValueError('measured stage cost exceeds bound')
    receipt = {'action': args.action, 'argv': sys.argv, 'config_sha256': args.config_sha256,
               'execution_identity': identity, 'wall_seconds': elapsed, 'cpu_seconds': used,
               'wall_budget_seconds': wall_limit, 'cpu_budget_seconds': cpu_limit,
               'scope': 'own and waited-child CPU; fleet enforces one-core wall budget'}
    write_new(scratch / ('checkpoint-' + args.action + '-stage-receipt.json'), receipt)


if __name__ == '__main__':
    main()

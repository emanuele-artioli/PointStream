"""CPU-only saved-input diagnosis for K6b 5b; no model loading or fitting.

Run with staged K6 archives and the original eight clips. Outputs new JSON
under PS_STAGE_DIR. The validator checks actual decoded scene measurements,
input hashes, registration indices and reference membership, not a smoke flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from typing import Any

import numpy as np
import zstandard


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def spans(indices: list[int]) -> list[list[int]]:
    result: list[list[int]] = []
    for t in sorted(set(indices)):
        if result and result[-1][-1] + 1 == t:
            result[-1][-1] = t
        else:
            result.append([t, t])
    return result


def parse_scenes(text: str) -> list[float]:
    return [float(x) for x in re.findall(r'lavfi.scene_score=([0-9.eE+-]+)', text)]


def shot_id(t: int, cuts: list[int]) -> int:
    return sum(c <= t for c in cuts)


def pose_distance(a: Any, b: Any, height: float) -> float | None:
    """Translation-invariant joint displacement in median mask-height units."""
    a, b = np.asarray(a), np.asarray(b)
    use = (a[:, 2] >= .5) & (b[:, 2] >= .5)
    if use.sum() < 4:
        return None
    x, y = a[use, :2], b[use, :2]
    return float(np.linalg.norm((x - x.mean(0)) - (y - y.mean(0)), axis=1).mean() / height)


def diagnose(row: dict[str, Any], poses: dict[str, Any], fits: dict[str, Any], scenes: list[float]) -> dict[str, Any]:
    n = row['frames']
    registered = row['scored_registered']
    scored = row['scored']
    cuts = [t for t, value in enumerate(scenes) if value > .3]
    players = []
    both = 0
    for t in registered:
        both += all(poses[p][t] and poses[p][t].get('fitted') for p in row['players'])
    for p in row['players']:
        fitted = [t for t in scored if poses[p][t] and poses[p][t].get('fitted')]
        reference = fitted[0] if fitted else None
        targets = [t for t in row['judged'] if poses[p][t] and poses[p][t].get('fitted')]
        distances = [pose_distance(poses[p][reference]['kps'], poses[p][t]['kps'], row['median_height'][p]) for t in targets] if reference is not None else []
        valid_distances = [x for x in distances if x is not None]
        players.append({'player': p, 'height_px': row['median_height'][p],
                        'fitted': len(fitted), 'miss_share_registered': 1 - len(fitted) / len(registered),
                        'reference_t': reference, 'target_range': [min(targets), max(targets)] if targets else [],
                        'reference_in_registered': reference in registered,
                        'cross_shot_targets': sum(shot_id(t, cuts) != shot_id(reference, cuts) for t in targets) if reference is not None else None,
                        'reference_time_gap_s_median': float(np.median([abs(t - reference) / row['fps'] for t in targets])) if targets else None,
                        'reference_pose_distance_median': float(np.median(valid_distances)) if valid_distances else None})
    # Motion of court centre on the 960x540 registration plane, converted to 1080p px.
    centre = np.asarray([10.97 / 2, 23.77 / 2, 1.])
    projected = {}
    for t in registered:
        if str(t + row['start']) in fits:
            xy = np.asarray(fits[str(t + row['start'])]) @ centre
            projected[t] = xy[:2] / xy[2] * 2
    speed = [float(np.linalg.norm(projected[t] - projected[t - 1])) for t in projected if t - 1 in projected]
    return {'window': row['id'], 'frames': n, 'fps': row['fps'], 'registered': len(registered),
            'registered_share': len(registered) / n, 'registered_spans': spans(registered),
            'judged': row['judged'], 'both_fitted_share_registered': both / len(registered),
            'players': players, 'scene_scores': scenes,
            'cuts': cuts, 'cuts_by_threshold': {str(c): [t for t, v in enumerate(scenes) if v > c] for c in (.15, .3, .45)},
            'camera_centre_step_px_median': float(np.median(speed)) if speed else None,
            'camera_centre_step_px_p95': float(np.percentile(speed, 95)) if speed else None}


def analyze(args: argparse.Namespace) -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise ValueError('ffmpeg missing')
    inputs: list[dict[str, str]] = []
    records = {}
    for source in args.k6_stage:
        source = Path(source)
        inputs.append({'path': str(source), 'sha256': sha(source)})
        with tarfile.open(source) as archive:
            for member in archive.getmembers():
                if member.name.endswith('/window.json'):
                    stream = archive.extractfile(member)
                    assert stream is not None
                    row = json.load(stream)
                    parent = str(Path(member.name).parent)
                    parts = {}
                    for name in ('poses.json.zst', 'fits.json'):
                        stream = archive.extractfile(parent + '/' + name)
                        assert stream is not None
                        data = stream.read()
                        if sha_bytes(data) != row['files'][name]:
                            raise ValueError(f'changed {name}')
                        parts[name] = json.loads(zstandard.ZstdDecompressor().decompress(data) if name.endswith('.zst') else data)
                    records[row['id']] = (row, parts)
    rows = []
    chosen = list(zip(args.windows, args.sources))
    if args.max_windows:
        chosen = chosen[:args.max_windows]
    for wid, source_name in chosen:
        source = Path(source_name)
        command = [ffmpeg, '-v', 'info', '-nostdin', '-threads', '2', '-i', str(source), '-frames:v', '240',
                   '-vf', "trim=end_frame=240,scale=320:180,select='gte(scene,0)',metadata=print", '-an', '-f', 'null', '-']
        done = subprocess.run(command, capture_output=True, text=True, check=True)
        scenes = parse_scenes(done.stderr)
        row, parts = records[wid]
        value = diagnose(row, parts['poses.json.zst'], parts['fits.json'], scenes)
        value['source'] = {'path': str(source), 'sha256': sha(source)}
        value['command'] = command
        sheet = stage / ('sheet-' + wid.replace('/', '__').replace('@', '_at') + '.jpg')
        subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-threads', '2', '-i', str(source),
                        '-vf', "trim=end_frame=240,select='eq(n,0)+eq(n,60)+eq(n,109)+eq(n,153)+eq(n,180)+eq(n,239)',scale=480:270,tile=3x2",
                        '-frames:v', '1', '-q:v', '3', str(sheet)], check=True)
        value['sheet'] = {'path': sheet.name, 'sha256': sha(sheet), 'frames': [0, 60, 109, 153, 180, 239]}
        rows.append(value)
        print(f'{wid}: {len(scenes)} frames; cuts {value["cuts"]}', flush=True)
    inventory = json.loads((Path(args.inputs) / 'inputs.json').read_text())
    inventory = {w: r['window'] for w, r in inventory['windows'].items()}
    output = {'rows': rows, 'inventory': inventory, 'inputs': inputs, 'ffmpeg': ffmpeg,
              'ffmpeg_version': subprocess.check_output([ffmpeg, '-version'], text=True).splitlines()[0],
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'expected_windows': len(chosen)}
    (stage / 'scope.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate(directory: Path) -> dict[str, Any]:
    d = json.loads((directory / 'scope.json').read_text())
    checks = {'cpu_only': d['cuda_visible_devices'] == '', 'windows': len(d['rows']) == d['expected_windows'] > 0}
    for row in d['rows']:
        checks[row['window'] + ':decoded_frames'] = len(row['scene_scores']) == row['frames'] == 240
        checks[row['window'] + ':scene_finite'] = all(np.isfinite(x) and 0 <= x <= 1 for x in row['scene_scores'])
        checks[row['window'] + ':reference_registered'] = all(p['reference_in_registered'] for p in row['players'])
        checks[row['window'] + ':source_hash'] = len(row['source']['sha256']) == 64
    return {'passed': all(checks.values()), 'checks': checks}


def summarize(scope: dict[str, Any], reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply an encoder-only inclusion rule to identical saved method/anchor rows.

    The single-shot rule is the prespecified first hypothesis; the size rule is
    exploratory. Missing generated windows never enter a reported denominator.
    """
    rules = {
        'pooled': lambda r: True,
        'single_shot': lambda r: not r['cuts'],
        'court_95pct': lambda r: r['registered_share'] >= .95,
        'single_shot_height_115': lambda r: not r['cuts'] and min(p['height_px'] for p in r['players']) >= 115,
        'court_95pct_height_115': lambda r: r['registered_share'] >= .95 and min(p['height_px'] for p in r['players']) >= 115,
    }
    by_window = {r['window']: r for r in scope['rows']}
    output: dict[str, Any] = {}
    for name, rule in rules.items():
        included = sorted(w for w, r in by_window.items() if rule(r))
        steps = []
        for report in reports:
            step = report.get('step', 2)
            variant = {2: 'gen-original', 3: 'gen-fit', 4: 'gen-match', 5: 'gen-match5'}[step]
            rows = [r for r in report['rows'] if r['window'] in included]
            result: dict[str, Any] = {'step': step, 'windows': [r['window'] for r in rows],
                                     'missing': sorted(set(included) - {r['window'] for r in rows})}
            for method in [variant, 'warp@q4-velocity-e4', 'ref@q4-velocity-e4']:
                for billed in (False, True):
                    cells = [r[method]['billed'] if billed else r[method] for r in rows
                             if method in r and (not billed or 'billed' in r[method])]
                    if not cells:
                        continue
                    wins = sum(c[m]['won'] for c in cells for m in ('lpips', 'dists'))
                    result[method + (' billed' if billed else '')] = {'won': wins, 'cells': 2 * len(cells), 'score': wins / (2 * len(cells))}
            steps.append(result)
        output[name] = {'included': included, 'excluded': sorted(set(by_window) - set(included)), 'steps': steps}
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    run = sub.add_parser('analyze')
    run.add_argument('--inputs', required=True)
    run.add_argument('--k6-stage', action='append', required=True)
    run.add_argument('--windows', nargs='+', required=True)
    run.add_argument('--sources', nargs='+', required=True)
    run.add_argument('--max-windows', type=int, default=0)
    report = sub.add_parser('report')
    report.add_argument('--scope', type=Path, required=True)
    report.add_argument('--reports', type=Path, nargs='+', required=True)
    report.add_argument('--out', type=Path, required=True)
    check = sub.add_parser('validate')
    check.add_argument('--directory', type=Path, default=Path(os.environ.get('PS_STAGE_DIR', '.')))
    args = parser.parse_args()
    if args.action == 'report':
        result = {'scores': summarize(json.loads(args.scope.read_text()), [json.loads(p.read_text()) for p in args.reports]),
                  'inputs': [{'path': str(p), 'sha256': sha(p)} for p in [args.scope, *args.reports]]}
        with args.out.open('x') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
        return 0
    if args.action == 'analyze':
        if len(args.windows) != len(args.sources):
            parser.error('each window needs one source')
        analyze(args)
        return 0
    result = validate(args.directory)
    path = Path(os.environ.get('PS_VALIDATION_PATH', str(args.directory / 'validation.json')))
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

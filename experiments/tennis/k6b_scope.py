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
        distances = [x for x in distances if x is not None]
        players.append({'player': p, 'height_px': row['median_height'][p],
                        'fitted': len(fitted), 'miss_share_registered': 1 - len(fitted) / len(registered),
                        'reference_t': reference, 'target_range': [min(targets), max(targets)] if targets else [],
                        'reference_in_registered': reference in registered,
                        'cross_shot_targets': sum(shot_id(t, cuts) != shot_id(reference, cuts) for t in targets) if reference is not None else None,
                        'reference_time_gap_s_median': float(np.median([abs(t - reference) / row['fps'] for t in targets])) if targets else None,
                        'reference_pose_distance_median': float(np.median(distances)) if distances else None})
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
                   '-vf', "scale=320:180,select='gte(scene,0)',metadata=print", '-an', '-f', 'null', '-']
        done = subprocess.run(command, capture_output=True, text=True, check=True)
        scenes = parse_scenes(done.stderr)
        row, parts = records[wid]
        value = diagnose(row, parts['poses.json.zst'], parts['fits.json'], scenes)
        value['source'] = {'path': str(source), 'sha256': sha(source)}
        value['command'] = command
        rows.append(value)
        print(f'{wid}: {len(scenes)} frames; cuts {value["cuts"]}', flush=True)
    output = {'rows': rows, 'inputs': inputs, 'ffmpeg': ffmpeg,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    run = sub.add_parser('analyze')
    run.add_argument('--k6-stage', action='append', required=True)
    run.add_argument('--windows', nargs='+', required=True)
    run.add_argument('--sources', nargs='+', required=True)
    run.add_argument('--max-windows', type=int, default=0)
    check = sub.add_parser('validate')
    check.add_argument('--directory', type=Path, default=Path(os.environ.get('PS_STAGE_DIR', '.')))
    args = parser.parse_args()
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

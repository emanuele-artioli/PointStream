"""CPU-only clip-local source audit; never loads models or opens quality scores.

Collect sequential decoded native yuv420p hashes and 64x36 luma descriptors.
The descriptor screen is deliberately not a certificate of source independence.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from typing import Any

SCOPE = 'ac640ae8f2ee0d807a085555f5504a354de9cba554aaf465cd036673c0d046d2'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_hashes(text: str) -> list[str]:
    rows = [s for s in text.splitlines() if s and not s.startswith('#')]
    result = [s.split(',')[-1].strip() for s in rows]
    if any(len(s) != 64 or any(c not in '0123456789abcdef' for c in s) for s in result):
        raise ValueError('invalid native frame hash')
    return result


def exact_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    out = []
    for r in records:
        if r['role'] == 'old' and r.get('hash_mode', 'native') == 'native':
            for t, h in enumerate(r['hashes']):
                index[(r['size'], h)].append((r['clip'], t))
    for r in records:
        if r['role'] != 'candidate' or r.get('hash_mode', 'native') != 'native':
            continue
        for t, h in enumerate(r['hashes']):
            for c, u in index.get((r['size'], h), []):
                out.append({'candidate': r['clip'], 'frame': r['start'] + t,
                            'other': c, 'other_frame': u, 'sha256': h})
        for t, h in enumerate(r['hashes']):
            index[(r['size'], h)].append((r['clip'], r['start'] + t))
    return out


def collect(args: argparse.Namespace) -> None:
    # Bound the entire attended read/CPU-decode command, not merely each child.
    def expired(signum: int, frame: Any) -> None:
        raise TimeoutError('600-second CPU audit deadline')

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(600)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    root = Path('/home/itec/emanuele/Datasets')
    scope = root / 'pointstream-data/jobs/fleet/inbox/20261024T083117Z-2e644b08/full/scope.json'
    if sha(scope) != SCOPE:
        raise ValueError('changed old input inventory')
    old = json.loads(scope.read_text())['inventory']
    candidates = json.loads(base64.b64decode(args.cohort))
    sources = {}
    for v in old.values():
        sources[v['clip']] = {'clip': v['clip'], 'sha256': v['sha256'], 'role': 'old', 'start': 0}
    for v in candidates:
        if v['clip'] in sources:
            raise ValueError('candidate clip is in old inventory')
        sources[v['clip']] = {**v, 'role': 'candidate'}
    if args.smoke:
        sources = {k: v for match in ('alcaraz_hurkacz', 'medvedev_struff')
                   for role in ('old', 'candidate')
                   for k, v in sorted(sources.items())
                   if v['role'] == role and k.startswith(match + '/')
                   and k == next(c for c, r in sorted(sources.items())
                                 if r['role'] == role and c.startswith(match + '/'))}
    expected_clips = sorted(sources)
    if args.max_clips:
        keys = sorted(sources)[args.start_clip:args.start_clip + args.max_clips]
        sources = {k: sources[k] for k in keys}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    ffmpeg = '/opt/local/bin/ffmpeg'
    ffprobe = '/opt/local/bin/ffprobe'
    result: dict[str, Any] = {'schema': 1, 'scope_sha256': SCOPE, 'code_sha256': args.code_sha256,
                              'cuda_visible_devices': '', 'ffmpeg': ffmpeg, 'ffmpeg_version':
                              subprocess.check_output([ffmpeg, '-version'], text=True).splitlines()[0],
                              'records': [], 'complete': False, 'smoke': args.smoke, 'expected_clips': expected_clips}
    with tempfile.TemporaryDirectory(prefix='k6b-recovery-', dir='/tmp') as tmp, (out / 'descriptors.bin').open('wb') as aggregate:
        offset = 0
        for clip, source in sorted(sources.items()):
            match, name = clip.split('/')
            path = root / 'archive/pre-reset-2026-10-05/tennis_games' / match / (name + '.mp4')
            if not path.exists():
                path = path.parent / 'videos' / path.name
            if sha(path) != source['sha256']:
                raise ValueError('changed source ' + clip)
            meta = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,nb_frames,r_frame_rate', '-of', 'json', str(path)], timeout=20))['streams'][0]
            n = int(meta['nb_frames'])
            expected = 240 if source['role'] == 'candidate' else n
            start = source['start']
            if start < 0 or start + expected > n:
                raise ValueError('bad window bounds')
            desc = Path(tmp) / 'descriptor.bin'
            native = (int(meta['width']), int(meta['height'])) == (1920, 1080)
            if source['role'] == 'candidate' and not native:
                raise ValueError('candidate resolution differs from frozen cohort')
            filt = f'trim=start_frame={start}:end_frame={start + expected},format=yuv420p,split=2[full][small];[small]scale=64:36:flags=area,format=gray[desc]'
            if not native:
                filt = f'trim=start_frame={start}:end_frame={start + expected},scale=64:36:flags=area,format=gray,split=2[full][desc]'
            command = [ffmpeg, '-v', 'error', '-nostdin', '-threads', '2', '-filter_complex_threads', '1',
                       '-i', str(path), '-filter_complex', filt, '-map', '[full]', '-an', '-fps_mode', 'passthrough',
                       '-f', 'framehash', '-hash', 'sha256', 'pipe:1', '-map', '[desc]', '-an',
                       '-fps_mode', 'passthrough', '-f', 'rawvideo', '-y', str(desc)]
            hashes = parse_hashes(subprocess.check_output(command, text=True, timeout=500 if not native else 120))
            data = desc.read_bytes()
            if len(hashes) != expected or len(data) != expected * 64 * 36:
                raise ValueError('decoded frame-count mismatch ' + clip)
            aggregate.write(data)
            result['records'].append({**source, 'path': str(path), 'size': f"{meta['width']}x{meta['height']}",
                'fps': meta['r_frame_rate'], 'hash_mode': 'native' if native else 'screen_only_other_dimensions',
                'frames': expected, 'source_frames': n, 'offset': offset,
                'command': command, 'hashes': hashes})
            offset += expected
            aggregate.flush()
            checkpoint = out / 'partial-next.json'
            checkpoint.write_text(json.dumps(result))
            checkpoint.replace(out / 'partial.json')
            print(json.dumps({'completed': len(result['records']), 'total': len(sources), 'clip': clip}), flush=True)
    result['complete'] = True
    result['descriptors_sha256'] = sha(out / 'descriptors.bin')
    result['exact_pairs'] = exact_pairs(result['records'])
    (out / 'audit.json').write_text(json.dumps(result))
    print(json.dumps({'complete': True, 'records': len(result['records']), 'exact_pair_count': len(result['exact_pairs']),
                      'audit_sha256': sha(out / 'audit.json'), 'out': str(out)}), flush=True)


def screen(folder: Path, extras: list[Path] | None = None) -> dict[str, Any]:
    import numpy as np

    records: list[dict[str, Any]] = []
    arrays = []
    receipts: list[dict[str, Any]] = []
    expected = None
    checkpoint_seen = False
    total = 0
    for directory in [folder] + (extras or []):
        report = directory / 'audit.json'
        if not report.exists():
            report = directory / 'partial.json'
        x = json.loads(report.read_text())
        if x.get('smoke'):
            raise ValueError('smoke input cannot be a full audit')
        checkpoint_seen |= not x.get('complete', False)
        actual_sha = sha(directory / 'descriptors.bin')
        if x.get('complete') and actual_sha != x['descriptors_sha256']:
            raise ValueError('changed descriptor collection')
        a = np.fromfile(directory / 'descriptors.bin', np.uint8).reshape(-1, 64 * 36)
        if len(a) != sum(r['frames'] for r in x['records']):
            raise ValueError('incomplete descriptor checkpoint')
        if x.get('expected_clips'):
            if expected is not None and expected != x['expected_clips']:
                raise ValueError('different expected inventories')
            expected = x['expected_clips']
        for r in x['records']:
            if r['clip'] in {v['clip'] for v in records}:
                raise ValueError('duplicate collected clip')
            records.append({**r, 'offset': r['offset'] + total})
        total += len(a)
        arrays.append(a)
        receipts.append({'report_sha256': sha(report), 'descriptors_sha256': actual_sha,
                         'checkpoint': not x.get('complete', False)})
    if checkpoint_seen and expected is None:
        raise ValueError('partial collection needs a complete expected inventory')
    if expected is not None and sorted(r['clip'] for r in records) != expected:
        raise ValueError('missing source clips: no complete audit')
    data = np.concatenate(arrays).astype(np.float32)
    x = {'records': records}
    if not np.isfinite(data).all():
        raise ValueError('nonfinite pixels')
    old = [r for r in x['records'] if r['role'] == 'old']
    result: dict[str, Any] = {'complete': True, 'collection_receipts': receipts, 'exact_pairs': exact_pairs(records),
                              'clips': len(records), 'decoded_frames': total, 'rmse_flag_threshold': 3,
              'definition': '64x36 grayscale closest-frame RMSE; screen only, not independence proof',
              'candidates': []}
    for r in x['records']:
        if r['role'] != 'candidate':
            continue
        a = data[r['offset']:r['offset'] + r['frames']]
        an = (a * a).sum(1)
        best = np.full(len(a), np.inf)
        where: list[dict[str, Any]] = [{} for _ in a]
        for other in old:
            b = data[other['offset']:other['offset'] + other['frames']]
            mse = np.maximum((an[:, None] + (b*b).sum(1)[None, :] - 2 * (a @ b.T)) / a.shape[1], 0)
            j = mse.argmin(1)
            vals = mse[np.arange(len(a)), j]
            for i in np.flatnonzero(vals < best):
                where[i] = {'other': other['clip'], 'other_frame': other['start'] + int(j[i])}
            best = np.minimum(best, vals)
        rms = np.sqrt(best)
        order = np.argsort(rms)
        result['candidates'].append({'clip': r['clip'], 'start': r['start'], 'frames': r['frames'],
            'flagged_frames': int((rms <= 3).sum()), 'minimum_rmse': float(rms.min()),
            'median_nearest_rmse': float(np.median(rms)),
            'nearest': [{'frame': r['start'] + int(i), 'rmse': float(rms[i]), **where[i]} for i in order[:3]]})
        old.append(r)  # includes distinct candidate pairs, excludes self-matches
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--cohort', required=True)
    c.add_argument('--smoke', action='store_true')
    c.add_argument('--start-clip', type=int, default=0)
    c.add_argument('--max-clips', type=int, default=0)
    c.add_argument('--out', required=True)
    c.add_argument('--code-sha256', required=True)
    s = sub.add_parser('screen')
    s.add_argument('folder', type=Path)
    s.add_argument('--extra', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.command == 'collect':
        collect(args)
    else:
        print(json.dumps(screen(args.folder, args.extra), indent=2))


if __name__ == '__main__':
    main()

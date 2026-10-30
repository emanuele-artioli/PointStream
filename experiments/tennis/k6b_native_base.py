"""Retain native whole-frame SVT streams and independently decoded sample pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import signal
import struct
import subprocess
import tarfile
import time
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

# Constrain the full workload, including imports, to one inherited CPU. The
# fleet also applies a hard stage deadline; CPU time cannot exceed wall time.
if os.environ.get('PS_STAGE') and '--validate' not in __import__('sys').argv:
    permitted = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(permitted)})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('workload time guard')))
    signal.alarm(250 if os.environ['PS_STAGE'] == 'smoke' else 1300)

import cv2
import numpy as np

from experiments.tennis.k4b_features import convert_command
from experiments.tennis.k6 import downscaled_source
from experiments.visor.h3 import rgb_memmap
from src.codecs import svtav1

SOURCES = [
    ('al', 9, 25, '054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76', 'e7b960cd2456b439c5439b337a45e47d2a0a44a2a32025a2de595af3feed4fa2'),
    ('med', 74, 50, 'a2898b5cdfb463f59b5084fddfe23a2442cf2ba5feaaddb9d5414873e52dd0be', '53633ce617b12d779bc8c5746d8d44cbe090ced6c49ba79dbe246cc6abacfb08'),
]
TOOLS = {'SvtAv1EncApp': '4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4', 'dav1d': 'e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6', 'ffmpeg': '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2'}


def sources_for(args: argparse.Namespace) -> tuple[list[Path], list[tuple[str, int, int, str, str]]]:
    """Source files and `SOURCES` tuples: the frozen Al/Med pair (`--al --med`) or a window manifest (`--windows-file` + `--source`)."""
    if args.windows_file is None:
        return [args.al, args.med], SOURCES
    from experiments.tennis import window_manifest as wm

    windows = wm.select(wm.load(args.windows_file), args.window or None)
    if len(args.source or []) != len(windows):
        raise ValueError('one --source per selected window, in order')
    return list(args.source), wm.legacy_sources(windows)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cpu() -> float:
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def ivf(path: Path, count: int, width: int, height: int, fps: int) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 32 or data[:4] != b'DKIF':
        raise ValueError('invalid IVF header')
    version, header, codec, w, h, rate, scale, declared, _ = struct.unpack('<HH4sHHIIII', data[4:32])
    if (version, header, codec, w, h, rate, scale, declared) != (0, 32, b'AV01', width, height, fps, 1, count):
        raise ValueError('incorrect IVF geometry or timing')
    offset, sizes, ticks, payload_hashes = 32, [], [], []
    while offset < len(data):
        if len(data) - offset < 12:
            raise ValueError('truncated frame header')
        n, t = struct.unpack('<IQ', data[offset:offset + 12])
        offset += 12
        if not n or offset + n > len(data):
            raise ValueError('truncated payload')
        sizes.append(n)
        ticks.append(t)
        payload_hashes.append(hashlib.sha256(data[offset:offset + n]).hexdigest())
        offset += n
    if len(sizes) != count or ticks != list(range(count)):
        raise ValueError('incorrect timestamp sequence')
    return {'bytes': len(data), 'payload_bytes': sum(sizes), 'header_bytes': 32 + 12 * count, 'timestamps': ticks, 'payload_sha256': payload_hashes, 'sha256': sha(path)}


def run(args: argparse.Namespace) -> None:
    import torch

    stage = os.environ['PS_STAGE']
    allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(allowed)})
    if len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('single-CPU budget guard unavailable')
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    began = time.monotonic()
    n = args.frames
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch / 'publish'
    publish.mkdir()
    tools = {name: svtav1.tool(name) for name in TOOLS}
    for name, expected in TOOLS.items():
        if tools[name]['sha256'] != expected:
            raise ValueError('native tool mismatch: ' + name)
    rows: list[dict[str, Any]] = []
    for source, (name, start, fps, source_hash, yuv_hash) in zip(*sources_for(args)):
        if sha(source) != source_hash:
            raise ValueError('source identity mismatch')
        raw = scratch / (name + '-source.yuv')
        command = convert_command('ffmpeg', source, raw, start, n)
        subprocess.run(command, check=True, capture_output=True)
        if raw.stat().st_size != n * 1920 * 1080 * 3 // 2:
            raise ValueError('source count mismatch')
        source_yuv_sha = sha(raw)
        if n == 240 and source_yuv_sha != yuv_hash:
            raise ValueError('source conversion mismatch')
        source_rgb = rgb_memmap(raw, scratch / (name + '-source-rgb.npy'), n, 1920, 1080, False, 'cpu')
        source_rgb_hashes = [hashlib.sha256(source_rgb[t].tobytes()).hexdigest() for t in range(0, n, 3)]
        del source_rgb
        for scale in ([2] if n == 24 else [8, 4, 2]):
            width, height = {8: (240, 136), 4: (480, 272), 2: (960, 544)}[scale]
            prefix = f'{name}-s{scale}'
            down = downscaled_source(raw, n, scale, scratch / (prefix + '-source.yuv'))
            stream = publish / (prefix + '.ivf')
            dec = publish / (prefix + '.yuv')
            enc = svtav1.encode_command('SvtAv1EncApp', down, stream, width=width, height=height, fps=Fraction(fps, 1), frames=n, crf=63, preset=4, full_range=False)
            subprocess.run(enc, check=True, capture_output=True)
            decode = svtav1.decode_command('dav1d', stream, dec, threads=1)
            subprocess.run(decode, check=True, capture_output=True)
            if dec.stat().st_size != n * width * height * 3 // 2:
                raise ValueError('decoded length mismatch')
            info = ivf(stream, n, width, height, fps)
            again = scratch / (prefix + '-second.yuv')
            subprocess.run(svtav1.decode_command('dav1d', stream, again, threads=1), check=True, capture_output=True)
            if sha(dec) != sha(again):
                raise ValueError('fresh decode mismatch')
            again.unlink()
            rgb = rgb_memmap(dec, scratch / (prefix + '-small.npy'), n, width, height, False, 'cpu')
            ticks = list(range(0, n, 3))
            pixels = np.lib.format.open_memmap(publish / (prefix + '-rgb.npy'), mode='w+', dtype=np.uint8, shape=(len(ticks), 1080, 1920, 3))
            frame_hashes = []
            for i, t in enumerate(ticks):
                pixels[i] = cv2.resize(np.asarray(rgb[t]), (1920, 1080), interpolation=cv2.INTER_CUBIC)
                frame_hashes.append(hashlib.sha256(pixels[i].tobytes()).hexdigest())
            pixels.flush()
            yuv_frames = np.memmap(dec, dtype=np.uint8, mode='r', shape=(n, height * 3 // 2, width))
            yuv_frame_hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in yuv_frames]
            del yuv_frames
            rows.append({'window': name, 'start': start, 'fps': fps, 'frames': n, 'scale': scale, 'size': [width, height], 'source_sha256': source_hash, 'source_yuv_sha256': source_yuv_sha, 'source_rgb_frame_sha256': source_rgb_hashes, 'source_command': command, 'encode_command': enc, 'decode_command': decode, 'ivf': info, 'yuv_sha256': sha(dec), 'yuv_frame_sha256': yuv_frame_hashes, 'rgb_sha256': sha(publish / (prefix + '-rgb.npy')), 'rgb_frame_sha256': frame_hashes, 'ticks': ticks})
            del pixels, rgb
            down.unlink()
        raw.unlink()
    report = {'stage': stage, 'frames': n, 'tools': tools, 'rows': rows, 'opencv_version': cv2.__version__, 'torch_version': torch.__version__, 'color': 'limited-range yuv420, bilinear chroma and native quality.yuv420_to_rgb, bicubic RGB resize', 'affinity_cpus': 1}
    (publish / 'manifest.json').write_text(json.dumps(report, indent=2))
    # The archive is built by the workload, so its CPU belongs to its budget.
    archive = scratch / 'bundle.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(publish, arcname='base')
    for child in publish.iterdir():
        child.unlink()
    shutil.move(archive, publish / 'bundle.tar')
    bundle_hash = sha(publish / 'bundle.tar')
    measured = cpu()
    wall = time.monotonic() - began
    projected = max(measured, wall) * 30 * 1.25
    result = {'frames': n, 'rows': len(rows), 'workload_cpu_seconds': measured, 'workload_wall_seconds': wall, 'single_cpu': True, 'projected_full_seconds': projected, 'bundle_sha256': bundle_hash}
    if args.windows_file is not None:
        result['source_sha256'] = [t[3] for t in sources_for(args)[1]]
    (Path(os.environ['PS_STAGE_DIR']) / 'base-result.json').write_text(json.dumps(result))
    signal.alarm(0)


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    result = json.loads((stage / 'base-result.json').read_text())
    expected_sources = result.get('source_sha256', [r[3] for r in SOURCES])
    checks = {'two_windows': result['rows'] == len(expected_sources), 'smoke_24': result['frames'] == 24, 'single_cpu': result['single_cpu'], 'projection': result['projected_full_seconds'] <= 1300, 'cpu_under_guard': result['workload_cpu_seconds'] <= 270}
    with tempfile.TemporaryDirectory(prefix='k6b-base-validator-') as tmp:
        root = Path(tmp)
        with tarfile.open(stage / 'published.tar') as outer:
            member = outer.getmember('publish/bundle.tar')
            f = outer.extractfile(member)
            if f is None:
                raise ValueError('missing bundle')
            bundle = root / 'bundle.tar'
            with bundle.open('wb') as target:
                shutil.copyfileobj(f, target)
        checks['bundle_identity'] = sha(bundle) == result['bundle_sha256']
        with tarfile.open(bundle) as tar:
            tar.extractall(root, filter='data')
        data = root / 'base'
        manifest = json.loads((data / 'manifest.json').read_text())
        checks['source_identities'] = [r['source_sha256'] for r in manifest['rows']] == expected_sources
        checks['pinned_tools'] = all(manifest['tools'][n]['sha256'] == d for n, d in TOOLS.items())
        for row in manifest['rows']:
            prefix = f"{row['window']}-s{row['scale']}"
            stream, yuv = data / (prefix + '.ivf'), data / (prefix + '.yuv')
            info = ivf(stream, 24, row['size'][0], row['size'][1], row['fps'])
            checks[prefix + ':ivf'] = info == row['ivf']
            checks[prefix + ':yuv'] = sha(yuv) == row['yuv_sha256']
            second = root / (prefix + '-fresh.yuv')
            subprocess.run(svtav1.decode_command('dav1d', stream, second, threads=1), check=True, capture_output=True)
            checks[prefix + ':fresh_decode'] = sha(second) == row['yuv_sha256']
            rgb = data / (prefix + '-rgb.npy')
            checks[prefix + ':rgb_file'] = sha(rgb) == row['rgb_sha256']
            frames = np.load(rgb, mmap_mode='r')
            checks[prefix + ':rgb_frames'] = frames.shape == (8, 1080, 1920, 3) and [hashlib.sha256(f.tobytes()).hexdigest() for f in frames] == row['rgb_frame_sha256']
            bad = root / 'corrupt.ivf'
            bad.write_bytes(stream.read_bytes()[:-1])
            try:
                ivf(bad, 24, row['size'][0], row['size'][1], row['fps'])
                checks[prefix + ':corruption_rejected'] = False
            except ValueError:
                checks[prefix + ':corruption_rejected'] = True
    out = {'passed': all(checks.values()), 'checks': checks, 'result': result}
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(out))
    if not out['passed']:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--al', type=Path)
    parser.add_argument('--med', type=Path)
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--windows-file', type=Path, help='window manifest (experiments/tennis/window_manifest.py) instead of --al/--med')
    parser.add_argument('--window', action='append', help='window name from the manifest (repeat; default all)')
    parser.add_argument('--source', type=Path, action='append', help='source clip of each selected window, in order')
    args = parser.parse_args()
    if args.validate:
        validate()
    else:
        if (args.windows_file is None and (args.al is None or args.med is None)) or args.frames not in (24, 240):
            parser.error('both sources (or --windows-file with --source) and 24 or 240 frames are required')
        run(args)


if __name__ == '__main__':
    main()

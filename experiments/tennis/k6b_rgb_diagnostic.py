"""Exact native GPU RGB control and saved-YUV base conversion diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import tarfile
import time
import tempfile
from typing import Any

import numpy as np

from experiments.tennis.k4b_features import convert_command
from experiments.visor.h3 import rgb_memmap

SOURCES = [
    ('al', 9, 25, '054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76', 'e7b960cd2456b439c5439b337a45e47d2a0a44a2a32025a2de595af3feed4fa2'),
    ('med', 74, 50, 'a2898b5cdfb463f59b5084fddfe23a2442cf2ba5feaaddb9d5414873e52dd0be', '53633ce617b12d779bc8c5746d8d44cbe090ced6c49ba79dbe246cc6abacfb08'),
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pixel_diff(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {'max': int(d.max()), 'sum': int(d.sum()), 'channel_sum': d.sum(axis=(0, 1)).tolist(),
            'different_pixels': int(np.any(d != 0, axis=2).sum()), 'pixels': int(d.shape[0] * d.shape[1]),
            'mean_absolute': float(d.mean()), 'channel_mean_absolute': d.mean(axis=(0, 1)).tolist()}


def source_numeric_probe(yuv: Path) -> dict[str, Any]:
    """Compare original bilinear chroma, pre-round math and rounded RGB on tick 0."""
    import torch
    import torch.nn.functional as functional
    from src.codecs.quality import KB, KR, read_frames, split_planes

    sample = torch.from_numpy(read_frames(yuv, 1920, 1080, 0, 1))

    def stages(device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y, u, v = split_planes(sample.to(device), 1920, 1080)
        uv = functional.interpolate(torch.stack([u, v], dim=1).float(), size=(1080, 1920), mode='bilinear', align_corners=False)
        yn = (y.float() - 16.0) / 219.0
        cb, cr = (uv[:, 0] - 128.0) / 224.0, (uv[:, 1] - 128.0) / 224.0
        r = yn + 2.0 * (1.0 - KR) * cr
        b = yn + 2.0 * (1.0 - KB) * cb
        g = (yn - KR * r - KB * b) / (1.0 - KR - KB)
        before = torch.stack([r, g, b], dim=1) * 255.0
        rounded = torch.clamp(torch.round(before), 0, 255)
        return uv.cpu().numpy(), before.cpu().numpy(), rounded.cpu().numpy()

    cpu, gpu = stages('cpu'), stages('cuda')
    out = {}
    for label, left, right in zip(('bilinear_chroma', 'pre_round_rgb', 'rounded_clipped_rgb'), cpu, gpu):
        difference = np.abs(left - right)
        out[label] = {'max_absolute': float(difference.max()), 'different_values': int(np.count_nonzero(difference)), 'values': int(difference.size)}
    return out


def run(a: argparse.Namespace) -> None:
    import cv2
    import torch

    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    if not torch.cuda.is_available() or 'A6000' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Original RTX A6000 class required')
    torch.cuda.set_per_process_memory_fraction(min(1.0, 12000 * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    root = Path(os.environ['PS_SCRATCH_DIR'])
    publish = root / 'publish'
    publish.mkdir()
    begin = time.monotonic()
    gpu_old = json.loads(a.prior.read_text())
    rows_old = {r['id']: r for r in gpu_old['rows']}
    # Outer/inner tar are SHA256-pinned by the fleet and validated again here.
    with tarfile.open(a.base) as outer:
        member = outer.extractfile('publish/bundle.tar')
        assert member is not None
        with tarfile.open(fileobj=member, mode='r:') as inner:
            inner.extractall(root, filter='data')
    base = root / 'base'
    manifest = json.loads((base / 'manifest.json').read_text())
    setup = time.monotonic() - begin
    records: list[dict[str, Any]] = []
    source_arrays: dict[str, np.ndarray] = {}
    for source, (name, start, fps, source_hash, yuv_hash) in zip([a.al, a.med], SOURCES):
        wid = 'alcaraz_hurkacz/031@129' if name == 'al' else 'medvedev_struff/014@194'
        if sha(source) != source_hash:
            raise ValueError('source identity')
        raw = root / (name + '-source.yuv')
        subprocess_run(convert_command('ffmpeg', source, raw, start, a.frames))
        if a.frames == 240 and sha(raw) != yuv_hash:
            raise ValueError('original source YUV mismatch')
        gpu = rgb_memmap(raw, root / (name + '-gpu-source.npy'), a.frames, 1920, 1080, False, 'cuda')
        cpu = rgb_memmap(raw, root / (name + '-cpu-source.npy'), a.frames, 1920, 1080, False, 'cpu')
        ticks = list(range(0, a.frames, 3))
        original = rows_old[wid]['source_rgb_frame_sha256']
        mismatches = [t for t in ticks if hashlib.sha256(gpu[t].tobytes()).hexdigest() != original[str(t)]]
        differences = [pixel_diff(cpu[t], gpu[t]) for t in ticks]
        records.append({'window': wid, 'source_matches': not mismatches, 'mismatches': mismatches,
                        'ticks': ticks, 'gpu_hashes': [hashlib.sha256(gpu[t].tobytes()).hexdigest() for t in ticks],
                        'cpu_gpu_differences': differences, 'numeric_probe_tick0': source_numeric_probe(raw), 'source_yuv_sha256': sha(raw), 'fps': fps})
        source_arrays[name] = gpu
        del cpu
    (Path(os.environ['PS_STAGE_DIR']) / 'source-control.json').write_text(json.dumps({'frames': a.frames, 'source': records}, indent=2))
    if not all(r['source_matches'] for r in records):
        raise ValueError('Original GPU source control mismatch; source-control.json preserves differences')
    base_rows: list[dict[str, Any]] = []
    for row in manifest['rows']:
        name, scale = row['window'], row['scale']
        yuv = base / f'{name}-s{scale}.yuv'
        if sha(yuv) != row['yuv_sha256']:
            raise ValueError('base decoded YUV mismatch')
        w, h = row['size']
        decoded = root / f'{name}-s{scale}-fresh.yuv'
        subprocess_run(['dav1d', '-i', str(base / f'{name}-s{scale}.ivf'), '-o', str(decoded), '--threads', '1', '--quiet'])
        if sha(decoded) != row['yuv_sha256']:
            raise ValueError('independent base IVF decode mismatch')
        decoded.unlink()
        original = np.load(base / f'{name}-s{scale}-rgb.npy', mmap_mode='r')
        first = rgb_memmap(yuv, root / f'{name}-s{scale}-first.npy', a.frames, w, h, False, 'cuda')
        second = rgb_memmap(yuv, root / f'{name}-s{scale}-second.npy', a.frames, w, h, False, 'cuda')
        ticks = list(range(0, a.frames, 3))
        output = np.lib.format.open_memmap(publish / f'{name}-s{scale}-gpu.npy', mode='w+', dtype=np.uint8, shape=(len(ticks), 1080, 1920, 3))
        diffs, hashes = [], []
        from src.codecs.quality import read_frames, yuv420_to_rgb
        sample_cpu = yuv420_to_rgb(torch.from_numpy(read_frames(yuv, w, h, 0, 1)), w, h, full_range=False)[0].permute(1, 2, 0).to(torch.uint8).numpy()
        small_diff = pixel_diff(sample_cpu, np.asarray(first[0]))
        fixed = np.asarray(first[0])
        resize_repeat_equal = np.array_equal(cv2.resize(fixed, (1920, 1080), interpolation=cv2.INTER_CUBIC), cv2.resize(fixed, (1920, 1080), interpolation=cv2.INTER_CUBIC))
        for i, t in enumerate(ticks):
            one = cv2.resize(np.asarray(first[t]), (1920, 1080), interpolation=cv2.INTER_CUBIC)
            two = cv2.resize(np.asarray(second[t]), (1920, 1080), interpolation=cv2.INTER_CUBIC)
            if not np.array_equal(one, two):
                raise ValueError('fresh native GPU conversion mismatch')
            output[i] = one
            hashes.append(hashlib.sha256(one.tobytes()).hexdigest())
            diffs.append(pixel_diff(original[i], one))
        output.flush()
        del output, first, second
        path = publish / f'{name}-s{scale}-gpu.npy'
        base_rows.append({'window': name, 'scale': scale, 'ticks': ticks, 'frame_sha256': hashes, 'sha256': sha(path), 'cpu_gpu_differences': diffs, 'pre_resize_tick0': small_diff, 'same_input_resize_repeat_equal': resize_repeat_equal})
    elapsed = time.monotonic() - begin
    # Conservative representative projection: same fixed archive/source setup plus
    # ten times measured post-setup work for 24 -> 240 frames, and a 25% margin.
    projected = (setup + (elapsed - setup) * 10) * 1.25
    result = {'frames': a.frames, 'source': records, 'base': base_rows, 'cuda': torch.cuda.get_device_name(0),
              'torch': torch.__version__, 'opencv': cv2.__version__, 'setup_seconds': setup,
              'seconds': elapsed, 'projected_full_seconds': projected,
              'attribution': 'Source tick 0 compares bilinear chroma, pre-round math and rounded RGB; base tick 0 compares pre-resize bytes and repeat bicubic on identical input. Other frames report final pixel differences only.'}
    (publish / 'diagnostic.json').write_text(json.dumps(result, indent=2))
    (Path(os.environ['PS_STAGE_DIR']) / 'diagnostic-summary.json').write_text(json.dumps({'frames': a.frames, 'setup_seconds': setup, 'seconds': elapsed, 'projected_full_seconds': projected, 'source_passed': all(r['source_matches'] for r in records), 'base_sets': len(base_rows)}))


def subprocess_run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True)


def validate() -> None:
    root = Path(os.environ['PS_STAGE_DIR'])
    data = json.loads((root / 'diagnostic-summary.json').read_text())
    checks = {'smoke_24': data['frames'] == 24, 'exact_source': data['source_passed'], 'six_base_sets': data['base_sets'] == 6,
              'full_projection_under_reservation': data['projected_full_seconds'] < 360}
    dispatch_record = json.loads((root / 'dispatch.json').read_text())
    dispatch = dispatch_record['command']
    total_smoke = time.time() - dispatch_record['started']
    publication_and_startup = max(0.0, total_smoke - data['seconds'])
    projected_with_publication = (data['setup_seconds'] + (data['seconds'] - data['setup_seconds'] + publication_and_startup) * 10) * 1.25
    checks['full_projection_with_publication'] = projected_with_publication < 360
    prior = Path(dispatch[dispatch.index('--prior') + 1])
    if sha(prior) != 'b0a55a5c1440463fff97efa975a661c2c65db9ce35e0696fade9d5648c39ccdc':
        raise ValueError('prior result identity')
    old = {r['id']: r['source_rgb_frame_sha256'] for r in json.loads(prior.read_text())['rows']}
    with tempfile.TemporaryDirectory(prefix='k6b-rgb-validate-') as temp:
        out = Path(temp)
        with tarfile.open(root / 'published.tar') as tar:
            tar.extractall(out, filter='data')
        published = out / 'publish'
        report = json.loads((published / 'diagnostic.json').read_text())
        checks['original_gpu_class'] = 'A6000' in report['cuda']
        for row in report['source']:
            checks[row['window'] + ':original_hashes'] = all(h == old[row['window']][str(t)] for t, h in zip(row['ticks'], row['gpu_hashes'])) and len(row['ticks']) == 8
        for row in report['base']:
            name = f"{row['window']}-s{row['scale']}"
            path = published / (name + '-gpu.npy')
            frames = np.load(path, mmap_mode='r')
            checks[name + ':file'] = sha(path) == row['sha256']
            checks[name + ':frames'] = frames.shape == (8, 1080, 1920, 3) and [hashlib.sha256(f.tobytes()).hexdigest() for f in frames] == row['frame_sha256']
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps({'passed': all(checks.values()), 'checks': checks, 'result': data, 'projected_with_publication': projected_with_publication}))
    if not all(checks.values()):
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--validate', action='store_true')
    p.add_argument('--base', type=Path)
    p.add_argument('--prior', type=Path)
    p.add_argument('--al', type=Path)
    p.add_argument('--med', type=Path)
    p.add_argument('--frames', type=int, default=24)
    a = p.parse_args()
    if a.validate:
        validate()
    elif any(v is None for v in (a.base, a.prior, a.al, a.med)) or a.frames not in (24, 240):
        p.error('all inputs and 24/240 frames required')
    else:
        run(a)


if __name__ == '__main__':
    main()

"""Lossless native-frame cache for bounded W7 registration batches (CPU only)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k8b, w7_select, w7_tier2_select

FRAME_BYTES = 1920 * 1080 * 3 // 2


def chunks_for(scene: str, count: int) -> list[int]:
    chunks: set[int] = set()
    for half in ('fit', 'eval'):
        for a, b in w7_tier2_select.draw_shots(scene, half)[:count]:
            chunks.update(range(a // k8b.CHUNK * k8b.CHUNK, b, k8b.CHUNK))
    return sorted(chunks)


def ranges(starts: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start in starts:
        end = min(start + k8b.CHUNK, w7_tier2_select.FRAMES)
        if out and out[-1][1] == start:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def decode_command(ffmpeg: str, video: Path, starts: list[int], output: Path) -> list[str]:
    spans = ranges(starts)
    select = '+'.join(f'between(n\\,{a}\\,{b - 1})' for a, b in spans)
    count = sum(b - a for a, b in spans)
    return [ffmpeg, '-v', 'error', '-nostdin', '-threads', '1', '-i', str(video), '-vf',
            f'select={select},scale=1920:1080:flags=area:out_range=tv,format=yuv420p',
            '-frames:v', str(count), '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', str(output)]


def register_raw(path: Path, offset: int, start: int, end: int) -> list[dict[str, Any]]:
    """Identical per-frame conversion and 200-frame warm start as k8b.register_chunk."""
    import cv2

    from experiments.tennis import court, windows

    cv2.setNumThreads(1)
    rows, previous = [], None
    with path.open('rb') as f:
        f.seek(offset)
        for frame in range(start, end):
            data = f.read(FRAME_BYTES)
            if len(data) != FRAME_BYTES:
                raise ValueError(f'truncated cached frame {frame}')
            rgb = windows.yuv_to_rgb(np.frombuffer(data, np.uint8).reshape(1080 * 3 // 2, 1920), 1920, 1080)
            row = court.register_v2(rgb, previous)
            previous = np.array(row['homography']) if row['homography'] is not None else None
            rows.append({'frame': frame, **row})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-sha', required=True)
    p.add_argument('--pilot', type=Path, required=True, help='completed 200-frame equivalence result')
    p.add_argument('--dest', type=Path, required=True)
    p.add_argument('--count', type=int, default=12)
    p.add_argument('--ffmpeg', default='/opt/local/bin/ffmpeg')
    a = p.parse_args()
    pilot = json.loads(a.pilot.read_text())
    if pilot.get('equal') is not True or pilot.get('frames') != 200:
        raise ValueError('native registration equivalence pilot has not passed')
    if a.count < 1 or w7_select.file_sha256(a.source) != a.source_sha:
        raise ValueError('invalid count or source identity')
    starts = chunks_for(a.scenes.read_text(), a.count)
    total = sum(min(k8b.CHUNK, w7_tier2_select.FRAMES - s) for s in starts) * FRAME_BYTES
    if not str(a.dest.resolve()).startswith('/local/users/emanuele/') or shutil.disk_usage(a.dest.parent).free < total * 1.2:
        raise ValueError('cache requires enough separate local disk')
    a.dest.mkdir(parents=True, exist_ok=False)
    raw = a.dest / 'frames.yuv'
    command = decode_command(a.ffmpeg, a.source, starts, raw)
    subprocess.run(command, check=True, capture_output=True, timeout=2400, env={**os.environ, 'TMPDIR': str(a.dest)})
    if raw.stat().st_size != total:
        raise ValueError('cached decoded length does not match selected frames')
    offsets = {}
    pos = 0
    for start in starts:
        offsets[str(start)] = pos
        pos += min(k8b.CHUNK, w7_tier2_select.FRAMES - start) * FRAME_BYTES
    result = {'source_sha256': a.source_sha, 'scene_sha256': w7_select.file_sha256(a.scenes), 'count': a.count,
              'pilot_sha256': w7_select.file_sha256(a.pilot), 'numpy': np.__version__, 'raw_sha256': w7_select.file_sha256(raw), 'raw_bytes': total,
              'offsets': offsets, 'command': command}
    (a.dest / 'index.json').write_text(json.dumps(result, indent=2))
    print('cached', len(starts), 'chunks', total, 'bytes', flush=True)


if __name__ == '__main__':
    main()

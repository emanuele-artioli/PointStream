"""Bounded real-native CPU workload for the unapproved K5d PID oracle.

This is not a scientific result and does not dispatch jobs. The orchestrator
must independently check executable hashes, containment and CPU accounting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from fractions import Fraction
from pathlib import Path


def run(archive: Path, archive_sha: str, output: Path, binaries: dict[str, str]) -> dict:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('CPU-only, single-affinity workload required')
    expected = json.loads(os.environ['PS_K5D_ARCHIVE_IDENTITY'])
    def stamp() -> list[int]:
        st = archive.stat()
        return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    if expected.get('sha256') != archive_sha or expected.get('stat') != stamp():
        raise ValueError('staged source identity changed')
    tools = {name: Path(sys.prefix) / 'bin' / binary for name, binary in
             (('ffmpeg', 'ffmpeg'), ('encoder', 'SvtAv1EncApp'), ('decoder', 'dav1d'))}
    for name, path in tools.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != binaries[name]:
            raise ValueError('native executable changed: ' + name)
    from experiments.tennis import k5, k5b_trackers as trk, k5d_encoder
    from src.codecs import svtav1

    clips = [c for c in trk.tracknet_clips(archive, [9]) if c['id'] == 'tracknet/game9/Clip5']
    if len(clips) != 1 or len(clips[0]['rows']) < 2:
        raise ValueError('registered source clip unavailable')
    clip = {**clips[0], 'rows': clips[0]['rows'][:2]}
    rgb = trk.tracknet_frames(archive, clip)
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    if output.resolve().parent != scratch:
        raise ValueError('native output must be directly inside task scratch')
    output.mkdir(exist_ok=False)
    yuv = output / 'source.yuv'
    conversion = k5d_encoder.convert_rgb(str(tools['ffmpeg']), rgb, yuv, Fraction(30, 1))
    h, w = rgb.shape[1:3]
    record = svtav1.code(yuv, output, width=w, height=h, fps=Fraction(30, 1), frames=2,
                         crf=48, preset=k5.PRESET, threads=1, full_range=False,
                         encoder=str(tools['encoder']), decoder=str(tools['decoder']),
                         cpus=list(os.sched_getaffinity(0)))
    if stamp() != expected['stat']:
        raise ValueError('staged source changed during native work')
    if record['decoded_frames'] != 2 or record['temporal_units'] != 2:
        raise ValueError('native output timeline differs')
    return {'source_sha256': archive_sha, 'clip': clip['id'], 'frames': 2,
            'source_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
            'yuv_sha256': hashlib.sha256(yuv.read_bytes()).hexdigest(),
            'decoded_sha256': hashlib.sha256(Path(record['decoded']).read_bytes()).hexdigest(),
            'conversion_command': conversion, 'native': record, 'tools': binaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--archive-sha', required=True)
    parser.add_argument('--output', required=True, type=Path)
    for name in ('ffmpeg', 'encoder', 'decoder'):
        parser.add_argument('--' + name + '-sha', required=True)
    args = parser.parse_args()
    result = run(args.archive, args.archive_sha, args.output,
                 {name: getattr(args, name + '_sha') for name in ('ffmpeg', 'encoder', 'decoder')})
    print('K5D_NATIVE_CPU ' + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

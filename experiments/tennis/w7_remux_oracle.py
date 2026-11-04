"""Prospective lossless IVF-to-Matroska transport oracle; no quality scoring.

Run only under a separately admitted, bounded single-CPU supervisor. All
artifacts are new; a failed run retains its directory for inspection.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_hash(path: Path, expected: str) -> str:
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('expected SHA256 must be lowercase hex')
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'identity mismatch: {path}')
    return actual


def commands(ffmpeg: str, ffprobe: str, source: Path, remux: Path,
             original_yuv: Path, remux_yuv: Path) -> dict[str, list[str]]:
    """No encoder is permitted: remux copies the single AV1 stream."""
    def probe(path: Path) -> list[str]:
        return [ffprobe, '-v', 'error', '-threads', '1', '-select_streams', 'v:0', '-show_streams',
                '-show_frames', '-show_entries',
                'stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,time_base,color_range,color_space,color_transfer,color_primaries:frame=best_effort_timestamp',
                '-of', 'json', str(path)]

    def decode(path: Path, output: Path) -> list[str]:
        return [ffmpeg, '-nostdin', '-v', 'error', '-hwaccel', 'none', '-c:v', 'libdav1d',
                '-threads', '1', '-i', str(path),
                '-map', '0:v:0', '-an', '-fps_mode', 'passthrough', '-pix_fmt', 'yuv420p',
                '-threads', '1', '-f', 'rawvideo', '-n', str(output)]

    return {'remux': [ffmpeg, '-nostdin', '-v', 'error', '-threads', '1', '-i', str(source),
                      '-map', '0:v:0', '-an', '-c:v', 'copy', '-fps_mode', 'passthrough',
                      '-f', 'matroska', '-n', str(remux)],
            'probe_original': probe(source), 'probe_remux': probe(remux),
            'decode_original': decode(source, original_yuv), 'decode_remux': decode(remux, remux_yuv)}


def validate_probes(original: dict[str, Any], remux: dict[str, Any], *, width: int,
                    height: int, frames: int, fps: Fraction) -> dict[str, Any]:
    """Check decoded-frame time points, geometry and color; time bases may differ."""
    metadata = []
    times = []
    for data in (original, remux):
        streams = data.get('streams', [])
        items = data.get('frames', [])
        if len(streams) != 1 or len(items) != frames:
            raise ValueError('wrong video stream or frame count')
        s = streams[0]
        if (s.get('codec_name') != 'av1' or s.get('width') != width or s.get('height') != height
                or s.get('pix_fmt') != 'yuv420p' or Fraction(s['r_frame_rate']) != fps):
            raise ValueError('codec, geometry, pixel format or frame rate changed')
        time_base = Fraction(s['time_base'])
        if time_base <= 0:
            raise ValueError('invalid time base')
        try:
            stamps = [int(frame['best_effort_timestamp']) * time_base for frame in items]
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError('missing frame timestamp') from exc
        if any(stamps[i] - stamps[0] != Fraction(i, 1) / fps for i in range(frames)):
            raise ValueError('frame timestamps do not preserve the exact cadence')
        color_keys = ('color_range', 'color_space', 'color_transfer', 'color_primaries')
        if any(key not in s for key in color_keys):
            raise ValueError('missing color metadata')
        metadata.append({key: s[key] for key in color_keys})
        times.append(stamps)
    if metadata[0] != metadata[1] or times[0] != times[1]:
        raise ValueError('color metadata or frame timestamps changed')
    return {'frames': frames, 'fps': str(fps), 'first_timestamp': str(times[0][0]),
            'last_timestamp': str(times[0][-1]), 'duration': str(Fraction(frames, 1) / fps),
            'color': metadata[0]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('one-CPU affinity required')
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2 or args.frames <= 0:
        raise ValueError('invalid dimensions or frame count')
    fps = Fraction(args.fps)
    if fps <= 0:
        raise ValueError('invalid FPS')
    source, ffmpeg, ffprobe = (Path(p).resolve(strict=True) for p in (args.input, args.ffmpeg, args.ffprobe))
    identities = {'input': check_hash(source, args.input_sha256), 'ffmpeg': check_hash(ffmpeg, args.ffmpeg_sha256),
                  'ffprobe': check_hash(ffprobe, args.ffprobe_sha256)}
    # This must be a new path. Do not remove it on any failure.
    work = Path(args.workdir).absolute()
    work.mkdir(mode=0o700, parents=False, exist_ok=False)
    remux, source_yuv, remux_yuv = (work / p for p in ('remux.mkv', 'original.yuv', 'remux.yuv'))
    command = commands(str(ffmpeg), str(ffprobe), source, remux, source_yuv, remux_yuv)
    (work / 'commands.json').write_text(json.dumps(command, indent=2) + '\n')
    outputs: dict[str, Any] = {}
    for name in ('remux', 'probe_original', 'probe_remux', 'decode_original', 'decode_remux'):
        completed = subprocess.run(command[name], capture_output=True, check=False, timeout=60)
        (work / f'{name}.stderr').write_bytes(completed.stderr)
        if completed.returncode:
            raise RuntimeError(f'{name} failed with exit {completed.returncode}')
        if name.startswith('probe_'):
            (work / f'{name}.json').write_bytes(completed.stdout)
            outputs[name] = json.loads(completed.stdout)
    details = validate_probes(outputs['probe_original'], outputs['probe_remux'], width=args.width,
                              height=args.height, frames=args.frames, fps=fps)
    expected_bytes = args.width * args.height * 3 // 2 * args.frames
    if source_yuv.stat().st_size != expected_bytes or remux_yuv.stat().st_size != expected_bytes:
        raise ValueError('decoded frame count or geometry mismatch')
    original_hash = sha256(source_yuv)
    if original_hash != sha256(remux_yuv):
        raise ValueError('decoded YUV mismatch')
    historical_match = None if args.original_decoded_sha256 is None else original_hash == args.original_decoded_sha256
    if historical_match is False:
        raise ValueError('original decode differs from historical decoded SHA256')
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = {'status': 'passed', 'scope': 'transport only; no quality score', 'identities': identities,
              'source_ivf_bytes': source.stat().st_size, 'remux_mkv_bytes': remux.stat().st_size,
              'bytes_saved': source.stat().st_size - remux.stat().st_size,
              'source_kbps': source.stat().st_size * 8 * float(fps) / args.frames / 1000,
              'remux_kbps': remux.stat().st_size * 8 * float(fps) / args.frames / 1000,
              'remux_sha256': sha256(remux), 'decoded_yuv_sha256': original_hash,
              'historical_decoded_sha256': args.original_decoded_sha256, 'historical_decoded_match': historical_match,
              'metadata': details, 'child_cpu_seconds': usage.ru_utime + usage.ru_stime}
    (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'input-sha256', 'ffmpeg', 'ffmpeg-sha256', 'ffprobe', 'ffprobe-sha256',
                 'workdir', 'width', 'height', 'frames', 'fps'):
        parser.add_argument('--' + name, required=True, type=int if name in ('width', 'height', 'frames') else str)
    parser.add_argument('--original-decoded-sha256')
    result = run(parser.parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

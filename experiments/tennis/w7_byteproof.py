"""Fail-closed W7 Matroska byte proofs; no GPU, scoring, or job submission.

A successful proof requires the complete reconstructed IVF to equal the
historical IVF byte for byte. A mismatch is retained as unproven; a separate
reviewed and budgeted native-YUV comparison is required before reuse.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
from typing import Any

from experiments.tennis import w7_mkv_profile as profile
from experiments.tennis import w7_remux_job as old


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ivf(path: Path) -> tuple[bytes, list[tuple[int, bytes]]]:
    """Parse exact IVF framing; never accept truncated or surplus records."""
    data = path.read_bytes()
    if len(data) < 32:
        raise ValueError('short IVF header')
    magic, version, length, codec, width, height, rate, scale, count, _ = struct.unpack('<4sHH4sHHIIII', data[:32])
    if ((magic, version, length, codec) != (b'DKIF', 0, 32, b'AV01')
            or rate <= 0 or scale <= 0 or count not in (0, 240)):
        raise ValueError('unexpected IVF header, cadence or frame count')
    if width <= 0 or height <= 0:
        raise ValueError('invalid IVF geometry')
    frames = []
    offset = 32
    for _ in range(240):
        if offset + 12 > len(data):
            raise ValueError('truncated IVF frame header')
        size, timestamp = struct.unpack_from('<IQ', data, offset)
        offset += 12
        if not size or offset + size > len(data):
            raise ValueError('truncated or empty IVF payload')
        frames.append((timestamp, data[offset:offset + size]))
        offset += size
    if offset != len(data) or [Fraction(x[0] * scale, rate) for x in frames] != [Fraction(i, 50) for i in range(240)]:
        raise ValueError('extra bytes or IVF frame timestamps changed')
    return data[:32], frames


def compare_ivf(original: Path, returned: Path) -> dict[str, Any]:
    header_a, frames_a = ivf(original)
    header_b, frames_b = ivf(returned)
    exact = original.read_bytes() == returned.read_bytes()
    return {'original_ivf_sha256': digest(original), 'reconstructed_ivf_sha256': digest(returned),
            'original_header_hex': header_a.hex(), 'reconstructed_header_hex': header_b.hex(),
            'frames': 240, 'timestamps_equal': True,
            'ordered_payloads_equal': all(a[1] == b[1] for a, b in zip(frames_a, frames_b)),
            'exact_ivf_bytes': exact}


def probe(probe_data: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    streams, frames = probe_data.get('streams', []), probe_data.get('frames', [])
    if len(streams) != 1 or len(frames) != 240:
        raise ValueError('stream or frame count differs')
    stream = streams[0]
    if (stream['codec_name'] != 'av1' or (stream['width'], stream['height']) != (width, height)
            or stream['pix_fmt'] != 'yuv420p' or Fraction(stream['r_frame_rate']) != 50
            or stream.get('color_range') != 'tv'):
        raise ValueError('native profile differs')
    base = Fraction(stream['time_base'])
    times = [int(x['best_effort_timestamp']) * base for x in frames]
    if base <= 0 or any(x - times[0] != Fraction(i, 50) for i, x in enumerate(times)):
        raise ValueError('frame PTS differ')
    colors = {name: stream.get(name) for name in profile.COLOR_FIELDS}
    for name in profile.COLOR_FIELDS[1:]:
        value = colors[name]
        if value is not None and value.lower() not in ('bt709', 'unknown', 'unspecified', 'reserved'):
            raise ValueError('color metadata contradicts immutable profile')
    return {'timestamps': [str(t) for t in times], 'colors': colors}


def commands(ffmpeg: Path, ffprobe: Path, src: Path, work: Path) -> dict[str, list[str]]:
    return profile.commands(str(ffmpeg), str(ffprobe), 'unused-dav1d', src, work)


def prove(row: dict[str, Any], src: Path, work: Path, ffmpeg: Path, ffprobe: Path) -> dict[str, Any]:
    if digest(src) != row['original_ivf_sha256'] or digest(ffmpeg) != old.FFMPEG_SHA or digest(ffprobe) != old.FFPROBE_SHA:
        raise ValueError('native or tool identity mismatch')
    ivf(src)
    work.mkdir(mode=0o700, parents=True, exist_ok=False)
    cmd = commands(ffmpeg, ffprobe, src, work)
    results = {}
    for name in ('mux', 'probe_original', 'probe_mkv', 'extract_ivf'):
        run = subprocess.run(cmd[name], capture_output=True, check=True, timeout=30)
        if name.startswith('probe_'):
            results[name] = json.loads(run.stdout)
    before = probe(results['probe_original'], row['width'], row['height'])
    after = probe(results['probe_mkv'], row['width'], row['height'])
    if before['timestamps'] != after['timestamps']:
        raise ValueError('delivery timestamps differ')
    comparison = compare_ivf(src, work / 'decoder-scratch.ivf')
    out = {'schema': 1, **row, **comparison, 'delivered_mkv_sha256': digest(work / 'delivered.mkv'),
           'delivered_mkv_bytes': (work / 'delivered.mkv').stat().st_size, 'fps': '50',
           'profile': profile.PROFILE, 'original_color_metadata': before['colors'],
           'delivered_color_metadata': after['colors'], 'commands': {k: cmd[k] for k in ('mux', 'probe_original', 'probe_mkv', 'extract_ivf')},
           'native_decoder_input_unchanged': comparison['exact_ivf_bytes'],
           'scope': 'byte proof only; source/RGB/mask/scorer equality must be separately established'}
    (work / 'proof.json').write_text(json.dumps(out, indent=2) + '\n')
    if not comparison['exact_ivf_bytes']:
        raise ValueError('IVF bytes differ; separate budgeted native-YUV decode required')
    return out




def representative_three(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    """Freeze disjoint roles before examining any new transport outcomes."""
    if (inventory.get('schema') != 1 or len(inventory.get('points', [])) != 53
            or len(inventory.get('controls', [])) != 18 or len(inventory.get('new_points', [])) != 4):
        raise ValueError('inventory must cover 53 points, 18 controls and four new points')
    requests = (('controls', 'med000', 'control_s8_crf63'),
                ('points', 'med019', 's2_o0_crf68'),
                ('new_points', 'med000', 's2_o20_crf70'))
    result = []
    for section, window, name in requests:
        rows = [x for x in inventory[section] if x.get('window') == window and x.get('name') == name]
        if len(rows) != 1:
            raise ValueError('missing or duplicate representative inventory row')
        result.append(rows[0])
    return result

def validate_proof(row: dict[str, Any], original: Path, work: Path) -> dict[str, Any]:
    record = json.loads((work / 'proof.json').read_text())
    for key, value in row.items():
        if record.get(key) != value:
            raise ValueError('proof inventory identity differs: ' + key)
    mkv = work / 'delivered.mkv'
    comparison = compare_ivf(original, work / 'decoder-scratch.ivf')
    if (digest(original) != row['original_ivf_sha256']
            or any(record.get(key) != value for key, value in comparison.items())
            or not comparison['exact_ivf_bytes'] or not comparison['ordered_payloads_equal']
            or record.get('profile') != profile.PROFILE
            or record.get('delivered_mkv_sha256') != digest(mkv)
            or record.get('delivered_mkv_bytes') != mkv.stat().st_size):
        raise ValueError('persisted packet proof or billed MKV differs')
    return record


def smoke_forecast(seconds: float, remaining: float, full_limit: float, reserve: float) -> float:
    import math
    values = (seconds, remaining, full_limit, reserve)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('invalid pilot timing or reserve')
    projected = 1.5 * seconds * (75 / 3) + reserve
    if projected > full_limit or projected > remaining:
        raise ValueError('75-role projected cost exceeds bound')
    return projected

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--row', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--ffprobe', type=Path, required=True)
    args = parser.parse_args()
    root = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    if not args.output.resolve().is_relative_to(root) or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('requires exclusive host-local scratch and one CPU')
    prove(json.loads(args.row.read_text()), args.input, args.output, args.ffmpeg, args.ffprobe)


if __name__ == '__main__':
    main()

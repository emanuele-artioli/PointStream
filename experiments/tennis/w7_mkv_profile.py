"""Prospective one-profile AV1 Matroska sensitivity oracle; not the failed remux gate.

The delivery is the entire Matroska file. Raw YUV and the IVF reconstructed from
Matroska are decoder scratch, never wire bytes. No neural scoring is performed.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from typing import Any

from experiments.tennis import w7_remux_job as old
from experiments.tennis import w7_remux_oracle as oracle

MAX_BYTES = {'med000': 25351, 'med019': 25209}
DAV1D_SHA = 'e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6'
PROFILE = 'explicit scorer interpretation: limited-range BT.709, historical cubic upscale; not inferred source metadata'
COLOR_FIELDS = ('color_range', 'color_space', 'color_transfer', 'color_primaries')


def probe_pair(original: dict[str, Any], delivered: dict[str, Any]) -> dict[str, Any]:
    timestamps = []
    colors = []
    for probe in (original, delivered):
        streams, frames = probe.get('streams', []), probe.get('frames', [])
        if len(streams) != 1 or len(frames) != 240:
            raise ValueError('wrong stream or 240-frame count')
        stream = streams[0]
        if (stream.get('codec_name') != 'av1' or stream.get('width') != 960 or stream.get('height') != 544
                or stream.get('pix_fmt') != 'yuv420p' or Fraction(stream['r_frame_rate']) != Fraction(50)):
            raise ValueError('wrong native geometry, format or cadence')
        base = Fraction(stream['time_base'])
        if base <= 0:
            raise ValueError('invalid timestamp base')
        pts = [int(f['best_effort_timestamp']) * base for f in frames]
        if any(v - pts[0] != Fraction(i, 50) for i, v in enumerate(pts)):
            raise ValueError('timestamps do not preserve all frame positions')
        timestamps.append(pts)
        colors.append({k: stream.get(k) for k in COLOR_FIELDS})
    if timestamps[0] != timestamps[1] or colors[0]['color_range'] != 'tv' or colors[1]['color_range'] != 'tv':
        raise ValueError('timestamps or known limited range differ')
    for field in ('color_space', 'color_transfer', 'color_primaries'):
        value = colors[1][field]
        if value is not None and value.lower() not in ('bt709', 'unknown', 'unspecified', 'reserved'):
            raise ValueError('delivered color metadata contradicts explicit BT.709 profile')
    return {'frames': 240, 'fps': '50', 'first': str(timestamps[0][0]),
            'last': str(timestamps[0][-1]), 'original_color_metadata': colors[0],
            'delivered_color_metadata': colors[1], 'interpretation': PROFILE}


def commands(ffmpeg: str, ffprobe: str, dav1d: str, source: Path, work: Path) -> dict[str, list[str]]:
    def probe(p: Path) -> list[str]:
        return [ffprobe, '-v', 'error', '-threads', '1', '-select_streams', 'v:0', '-show_streams',
                '-show_frames', '-show_entries',
                'stream=codec_name,width,height,pix_fmt,r_frame_rate,time_base,color_range,color_space,color_transfer,color_primaries:frame=best_effort_timestamp',
                '-of', 'json', str(p)]
    mkv, internal = work / 'delivered.mkv', work / 'decoder-scratch.ivf'
    return {
        'mux': [ffmpeg, '-nostdin', '-v', 'error', '-threads', '1', '-i', str(source), '-map', '0:v:0',
                '-an', '-c:v', 'copy', '-fps_mode', 'passthrough', '-f', 'matroska', '-n', str(mkv)],
        'probe_original': probe(source), 'probe_mkv': probe(mkv),
        'extract_ivf': [ffmpeg, '-nostdin', '-v', 'error', '-threads', '1', '-i', str(mkv), '-map', '0:v:0',
                        '-an', '-c:v', 'copy', '-fps_mode', 'passthrough', '-f', 'ivf', '-n', str(internal)],
        'decode_original': [dav1d, '-i', str(source), '-o', str(work / 'original.yuv'), '--threads', '1', '--quiet'],
        'decode_delivery': [dav1d, '-i', str(internal), '-o', str(work / 'delivery.yuv'), '--threads', '1', '--quiet'],
    }


def check_result(report: dict[str, Any], original: dict[str, Any], delivered: dict[str, Any],
                 mkv: bytes, case: str) -> None:
    metadata = probe_pair(original, delivered)
    if (case not in MAX_BYTES or report['case'] != case or report['status'] != 'passed'
            or report['scope'] != 'new transport sensitivity only; no quality or old gate result'
            or report['metadata'] != metadata or report['wire_bytes'] != len(mkv)
            or len(mkv) > MAX_BYTES[case] or report['byte_limit'] != MAX_BYTES[case]
            or not math.isclose(report['wire_kbps'], len(mkv) * 8 / 4.8 / 1000, rel_tol=1e-12)
            or report['wire_sha256'] != hashlib.sha256(mkv).hexdigest()
            or report['original_yuv_sha256'] != old.CASES[case]['decoded']
            or report['delivery_yuv_sha256'] != old.CASES[case]['decoded']
            or report['input_sha256'] != old.CASES[case]['stream']
            or report['ffmpeg_sha256'] != old.FFMPEG_SHA or report['ffprobe_sha256'] != old.FFPROBE_SHA
            or report['dav1d_sha256'] != DAV1D_SHA):
        raise ValueError('wire, profile, native decode or provenance mismatch')


def context(stage: Path, case: str) -> tuple[Path, Path, dict[str, Any]]:
    source, bins, spec = old.context(stage, case)
    if (spec['budget_seconds'] != 90 or spec['validator_seconds'] != 10
            or spec['smoke']['seconds'] != 35 or spec['full']['seconds'] != 35):
        raise ValueError('profile campaign bounds differ')
    return source, bins, spec


def run(stage: Path, scratch: Path, case: str) -> None:
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError('single CPU affinity required')
    source, bins, _ = context(stage, case)
    local_root = Path(json.loads((stage / "staging.json").read_text())["local_root"]).resolve(strict=True)
    if not scratch.resolve(strict=True).is_relative_to(local_root):
        raise ValueError("scratch must be under verified host-local root")
    ffmpeg, ffprobe, dav1d = (bins / name for name in ('ffmpeg', 'ffprobe', 'dav1d'))
    for path, digest in ((ffmpeg, old.FFMPEG_SHA), (ffprobe, old.FFPROBE_SHA), (dav1d, DAV1D_SHA)):
        oracle.check_hash(path, digest)
    work = scratch / 'new-mkv-profile'
    work.mkdir(mode=0o700, exist_ok=False)
    cmd = commands(str(ffmpeg), str(ffprobe), str(dav1d), source, work)
    (work / 'commands.json').write_text(json.dumps(cmd, indent=2) + '\n')
    outputs: dict[str, Any] = {}
    for name in ('mux', 'probe_original', 'probe_mkv', 'extract_ivf', 'decode_original', 'decode_delivery'):
        result = subprocess.run(cmd[name], capture_output=True, check=False, timeout=30)
        (work / (name + '.stderr')).write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError(f'{name} failed: {result.returncode}')
        if name.startswith('probe_'):
            (work / (name + '.json')).write_bytes(result.stdout)
            outputs[name] = json.loads(result.stdout)
        if name == 'probe_mkv':
            probe_pair(outputs['probe_original'], outputs['probe_mkv'])
            if (work / 'delivered.mkv').stat().st_size > MAX_BYTES[case]:
                raise ValueError('complete Matroska wire exceeds target byte limit')
    expected = 960 * 544 * 3 // 2 * 240
    for name in ('original.yuv', 'delivery.yuv'):
        if (work / name).stat().st_size != expected:
            raise ValueError('native decoded YUV size differs from 240 complete frames')
    mkv = (work / 'delivered.mkv').read_bytes()
    report = {'case': case, 'status': 'passed', 'scope': 'new transport sensitivity only; no quality or old gate result',
              'input_sha256': old.CASES[case]['stream'], 'ffmpeg_sha256': old.FFMPEG_SHA,
              'ffprobe_sha256': old.FFPROBE_SHA, 'dav1d_sha256': DAV1D_SHA,
              'wire_bytes': len(mkv), 'byte_limit': MAX_BYTES[case],
              'wire_kbps': len(mkv) * 8 / 4.8 / 1000, 'wire_sha256': hashlib.sha256(mkv).hexdigest(),
              'original_yuv_sha256': oracle.sha256(work / 'original.yuv'),
              'delivery_yuv_sha256': oracle.sha256(work / 'delivery.yuv'),
              'metadata': probe_pair(outputs['probe_original'], outputs['probe_mkv']),
              'decoder_scratch_only': 'decoder-scratch.ivf is internal and is not billed as wire',
              'tools': {'ffmpeg': '8.1.2', 'dav1d': '1.5.4', 'environment_sha256': old.ENV_SHA}}
    check_result(report, outputs['probe_original'], outputs['probe_mkv'], mkv, case)
    (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    out = scratch / 'publish'
    out.mkdir()
    for name in ('report.json', 'probe_original.json', 'probe_mkv.json', 'delivered.mkv', 'commands.json'):
        shutil.copyfile(work / name, out / name)


def validate(stage: Path, target: Path) -> int:
    try:
        source, bins, spec = context(stage, 'med000')
        with tarfile.open(stage / 'published.tar') as bundle:
            def read(name: str) -> bytes:
                member = bundle.getmember('publish/' + name)
                if not member.isfile() or member.size > 2_000_000:
                    raise ValueError('invalid member')
                f = bundle.extractfile(member)
                if f is None:
                    raise ValueError('missing member')
                return f.read()
            report = json.loads(read('report.json'))
            recorded = json.loads(read('commands.json'))
            work = Path(recorded['mux'][-1]).parent
            if recorded != commands(str(bins / 'ffmpeg'), str(bins / 'ffprobe'), str(bins / 'dav1d'), source, work):
                raise ValueError('published native commands differ from frozen contract')
            check_result(report, json.loads(read('probe_original.json')), json.loads(read('probe_mkv.json')),
                         read('delivered.mkv'), 'med000')
        oracle.check_hash(source, old.CASES['med000']['stream'])
        clock = json.loads((stage / 'dispatch.json').read_text())['campaign_clock']
        now = time.monotonic()
        elapsed = now - clock['phase_started']
        projected = max(5., 1.5 * elapsed)
        reserve = max(5., clock['phase_started'] - clock['started'])
        if (not clock['started'] <= clock['phase_started'] <= now < clock['budget_end']
                or projected > spec['full']['seconds'] or projected + reserve > clock['budget_end'] - now):
            raise ValueError('second stage does not fit measured remaining budget')
        target.write_text(json.dumps({'passed': True, 'checks': {'frozen_inputs': True, 'full_wire': True,
                          'all_frame_timestamps': True, 'historical_yuv': True, 'bounded_full': True},
                          'projection_seconds': projected, 'scope': 'new transport only; old gate remains failed'}) + '\n')
        return 0
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'error': f'{type(exc).__name__}: {exc}'}) + '\n')
        return 1



def make_spec(deadline: str) -> dict[str, Any]:
    """Bind the existing archived inputs and an actual future deadline."""
    when = datetime.fromisoformat(deadline)
    if when.tzinfo is None or when <= datetime.now(timezone.utc):
        raise ValueError('a future timezone-aware deadline is required')
    template = Path(__file__).resolve().parents[2] / 'docs/experiments/w7-inputs/remux-transport-oracle-20261104.json'
    spec = json.loads(template.read_text())
    if {x['name']: x['sha256'] for x in spec['staged_inputs']} != {k: v['archive'] for k, v in old.CASES.items()}:
        raise ValueError('frozen archived inputs differ')
    spec['entrypoint'] = ['-m', 'experiments.tennis.w7_mkv_profile']
    spec['smoke'] = {'seconds': 35, 'representative_basis': 'Full 240-frame med000 native stream; exact Matroska byte, timestamps, and old/new native YUV checks under explicit chosen scorer interpretation.'}
    spec['full'] = {'seconds': 35}
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.w7_mkv_profile', '--validate']
    spec['validator_seconds'] = 10
    spec['budget_seconds'] = 90
    spec['deadline'] = deadline
    spec['stall_seconds'] = 45
    return spec

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=sorted(MAX_BYTES))
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--make-spec', action='store_true')
    parser.add_argument('--deadline')
    args = parser.parse_args()
    if args.make_spec:
        if not args.deadline or args.validate or args.case:
            parser.error('--make-spec requires only --deadline')
        print(json.dumps(make_spec(args.deadline), indent=2))
        return 0
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.validate:
        return validate(stage, Path(os.environ['PS_VALIDATION_PATH']))
    if args.case is None:
        parser.error('--case required')
    run(stage, Path(os.environ['PS_SCRATCH_DIR']), args.case)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

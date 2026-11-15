"""Physical controls for new-match ball confirmation; no submission or scoring.

This module contains the variable-length CPU-side primitives.  The caller must
supply authenticated source and six freshly encoded bases and must qualify the
sender independently before using any output as evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tarfile
import time
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import ball_pixel_oracle, w1_roi
from src.codecs import svtav1
from src.pointstream import ball, ball_stream, tracked_patch

W, H = 1280, 720
SMOKE_FRAMES = 32
FRAME_BYTES = W * H * 3 // 2
POINTS = (("dcvc", 0), ("dcvc", 21), ("dcvc", 42), ("svt", 48), ("svt", 55), ("svt", 62))
SVT_POINTS = (48, 55, 62)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def accepted_full(job_id: str, status: dict[str, Any], stage_raw: bytes, validation_raw: bytes,
                  execution_raw: bytes, dispatch_raw: bytes, measured_archive_sha256: str) -> dict[str, Any]:
    """Join actual same-job full artifacts after canonical terminal inspection.

    The caller must retrieve these bytes from that job's full directory, retain
    the unmodified status response and independently hash the actual archive.
    This is an acceptance index, not a signature or proof of an untrusted host.
    """
    stage, validation, execution, dispatch = [json.loads(raw) for raw in
                                             (stage_raw, validation_raw, execution_raw, dispatch_raw)]
    source_id, phase = stage['source_id'], stage['phase']
    identity = canonical_sha(dispatch)
    command = dispatch['command']
    def argument(flag: str) -> str | None:
        return command[command.index(flag) + 1] if command.count(flag) == 1 and command.index(flag) + 1 < len(command) else None
    checks = dict(validation.get('checks', {}))
    checks.update({
        'canonical_complete': status.get('job_id') == job_id and status.get('status') == 'complete',
        'job_id': re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', job_id) is not None,
        'canonical_gate': status.get('gate', {}).get('passed') is True,
        'gate_code_identity': status.get('gate', {}).get('source_sha256') == dispatch['code']['source_sha256'],
        'full_command': argument('--phase') == phase and argument('--source-id') == source_id
            and argument('--frames') == str(stage['frames']),
        'runtime_identity': dispatch['staged_environment']['sha256'] == RUNTIME_SHA,
        'full_source_span': source_id in SOURCES and stage['frames'] == SOURCES[source_id][0],
        'full_validation_binding': validation.get('phase') == phase
            and validation.get('source_id') == source_id and validation.get('frames') == stage['frames']
            and validation.get('stage_sha256') == hashlib.sha256(stage_raw).hexdigest()
            and validation.get('archive_sha256') == measured_archive_sha256
            and validation.get('dispatch_canonical_sha256') == identity,
        'actual_archive_hash': measured_archive_sha256 == stage['archive']['sha256'],
        'outer_execution': execution.get('exit_code') == 0 and type(execution.get('seconds')) in (int, float)
            and 0 < execution['seconds'] <= 1800,
    })
    if (phase not in ('bases', 'controls') or validation.get('passed') is not True
            or not validation.get('checks') or not all(v is True for v in checks.values())):
        raise ValueError('same-job full producer acceptance failed')
    return {'schema': 'pointstream-ball-confirmation-accepted-full-v1', 'passed': True,
            'checks': checks, 'phase': phase, 'source_id': source_id, 'frames': stage['frames'],
            'producer_job_id': job_id, 'canonical_complete': True,
            'archive_sha256': measured_archive_sha256, 'stage_sha256': hashlib.sha256(stage_raw).hexdigest(),
            'execution_sha256': hashlib.sha256(execution_raw).hexdigest(),
            'raw_full_validation_sha256': hashlib.sha256(validation_raw).hexdigest(),
            'dispatch_canonical_sha256': identity, 'raw_dispatch_sha256': hashlib.sha256(dispatch_raw).hexdigest(),
            'canonical_status_sha256': canonical_sha(status), 'outer_execution_seconds': execution['seconds'],
            'scope': 'ROOT-joined full artifacts from the canonically complete job; shared timing is not throughput.'}



def decode_source(video: Path, expected_sha256: str, frames: int, fps: Fraction,
                  destination: Path) -> dict[str, Any]:
    """Sequential native OpenCV decode, BGR to RGB, then RGB area resize.

    The caller authenticates the frozen manifest and selects its source entry;
    no annotation file or annotation row is consumed here.
    """
    import cv2

    if sha(video) != expected_sha256 or frames not in (SMOKE_FRAMES, 64, 154) or fps not in (Fraction(25), Fraction(60)):
        raise ValueError('source identity or registered window mismatch')
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise ValueError('source video cannot be decoded')
    hashes: list[str] = []
    try:
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != 1920
                or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != 1080
                or abs(cap.get(cv2.CAP_PROP_FPS) - float(fps)) > 1e-6):
            raise ValueError('native source geometry or cadence mismatch')
        backend = cap.getBackendName()
        with destination.open('xb') as out:
            for _ in range(frames):
                ok, bgr = cap.read()
                if not ok or bgr.shape != (1080, 1920, 3) or bgr.dtype != np.uint8:
                    raise ValueError('native decode stopped or changed geometry')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
                data = resized.tobytes()
                out.write(data)
                hashes.append(hashlib.sha256(data).hexdigest())
    finally:
        cap.release()
    return {'source_sha256': expected_sha256, 'frames': frames,
            'fps': [fps.numerator, fps.denominator], 'size': [W, H],
            'rgb_sha256': sha(destination), 'rgb_frame_sha256': hashes,
            'backend': backend, 'opencv_version': cv2.__version__,
            'opencv_build_sha256': hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
            'transform': 'sequential native BGR decode, BGR2RGB, RGB INTER_AREA 1280x720',
            'pts': 'index zero is first decoded frame; no resampling'}

def checked_track(track: list[Any], frames: int) -> list[tuple[int, int] | None]:
    if len(track) != frames:
        raise ValueError('track frame count mismatch')
    result: list[tuple[int, int] | None] = []
    for p in track:
        if p is None:
            result.append(None)
        elif (isinstance(p, (list, tuple)) and len(p) == 2
              and all(isinstance(v, int) and not isinstance(v, bool) for v in p)
              and 0 <= p[0] < W and 0 <= p[1] < H):
            result.append((p[0], p[1]))
        else:
            raise ValueError('invalid track position')
    return result


def roi_map(track: list[Any], frames: int) -> str:
    """One ROI map row per frame; absent hints encode the ordinary native frame."""
    positions = checked_track(track, frames)
    masks = np.zeros((frames, H, W), dtype=bool)
    for i, point in enumerate(positions):
        if point is not None:
            x, y = point
            left = min(max(x - 32, 0), W - 64)
            top = min(max(y - 32, 0), H - 64)
            masks[i, top:top + 64, left:left + 64] = True
    # Missing hints require all-zero QP offsets, not the outside +40 policy.
    lines = w1_roi.roi_text(masks, W, H, -40, outside=True).splitlines()
    for i, point in enumerate(positions):
        if point is None:
            lines[i] = str(i) + ' ' + ' '.join(['0'] * (((H + 63) // 64) * ((W + 63) // 64)))
    return '\n'.join(lines) + '\n'


def patch_frames(source_yuv: Path, track: list[Any], frames: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Extract every 32x32 YUV patch, with neutral padding for missing hints."""
    positions = checked_track(track, frames)
    if source_yuv.stat().st_size != frames * FRAME_BYTES:
        raise ValueError('source YUV frame count mismatch')
    source = np.memmap(source_yuv, mode='r', dtype=np.uint8, shape=(frames, FRAME_BYTES))
    patches = np.empty((frames, 64 * 64 * 3 // 2), dtype=np.uint8)
    rows = []
    ysize, csize = W * H, W * H // 4
    for t, point in enumerate(positions):
        row = ball_pixel_oracle.placement(point)
        patch = patches[t]
        patch[:4096] = 16
        patch[4096:] = 128
        y, u, v = patch[:4096].reshape(64, 64), patch[4096:5120].reshape(32, 32), patch[5120:].reshape(32, 32)
        if point is not None:
            left, top = row['left'], row['top']
            raw = source[t]
            sy = raw[:ysize].reshape(H, W)
            su = raw[ysize:ysize + csize].reshape(H // 2, W // 2)
            sv = raw[ysize + csize:].reshape(H // 2, W // 2)
            y[16:48, 16:48] = sy[top:top + 32, left:left + 32]
            u[8:24, 8:24] = su[top // 2:top // 2 + 16, left // 2:left // 2 + 16]
            v[8:24, 8:24] = sv[top // 2:top // 2 + 16, left // 2:left // 2 + 16]
        row['frame'] = t
        row['padded_yuv_sha256'] = hashlib.sha256(patch.tobytes()).hexdigest()
        rows.append(row)
    return patches, rows


def validate_sender(sender: dict[str, Any], fps: Fraction, frames: int) -> None:
    if (sender['frames'] != frames or sender['size'] != [W, H]
            or sender['fps'] != [fps.numerator, fps.denominator]
            or [(r['codec'], r['point']) for r in sender['points']] != list(POINTS)):
        raise ValueError('sender matrix, geometry or timing mismatch')
    checked_track(sender['initial_track'], frames)
    for row in sender['points']:
        expected = checked_track(row['centres'], frames)
        payload = bytes.fromhex(row['ball_payload_hex'])
        if hashlib.sha256(payload).hexdigest() != row['ball_payload_sha256']:
            raise ValueError('sender payload hash mismatch')
        decoded, _ = ball.decode_frames(payload, frames, W, H)
        if decoded != expected:
            raise ValueError('sender payload positions mismatch')



def prepare_bases(*, source: np.ndarray, source_yuv: Path, fps: Fraction, work: Path,
                  args: Any, tracker: Any, cpus: list[int]) -> dict[str, Any]:
    """Fresh six-point baselines and sender; caller qualifies resources/device first.

    The output directory must not exist. Worker native containers, decodes and
    DCVC reports remain there; no annotation input is accepted.
    """
    from experiments.tennis import fresh_ball_sender, k5, windows

    if (source.dtype != np.uint8 or source.ndim != 4 or source.shape[1:] != (H, W, 3)
            or len(set(cpus)) != 4 or source_yuv.stat().st_size != len(source) * FRAME_BYTES):
        raise ValueError('invalid source, YUV or CPU allocation')
    work.mkdir(parents=True, exist_ok=False)
    frames = len(source)
    window = {'size': [W, H], 'fps': float(fps)}
    decoded_bases: list[tuple[tuple[str, int], np.ndarray]] = []
    records = []
    for kind, point in POINTS:
        directory = work / f'{kind}-{point}'
        if kind == 'dcvc':
            if directory.exists():
                raise ValueError('DCVC worker directory must be new')
            record = k5.dcvc_attempt(args, source_yuv, directory, window, frames, point)
            from src.codecs.dcvc_uf_worker import dcvc_command
            record['commands'] = {action: dcvc_command(Path(sys.prefix), action, directory / 'plan.json',
                                                       directory / f'{action}.json')[0]
                                  for action in ('encode', 'decode')}
            if not (record['deterministic'] and record['decoder_matches_encoder_intra']
                    and record['luma_psnr'] >= k5.SANE_FRAME_PSNR):
                raise ValueError('DCVC native decode sanity check failed')
        else:
            record = k5.code_svt(args, source_yuv, directory, window, frames, point, cpus)
        yuv = Path(record['decoded'])
        if yuv.stat().st_size != frames * FRAME_BYTES:
            raise ValueError('native decode size mismatch')
        rgb = windows.rgb_frames(yuv, W, H, frames, directory / 'decoded.npy')
        decoded_bases.append(((kind, point), rgb))
        raw = np.memmap(yuv, dtype=np.uint8, mode='r', shape=(frames, FRAME_BYTES))
        native_file = directory / ('stream.psdc' if kind == 'dcvc' else 'stream.ivf')
        records.append({'kind': kind, 'point': point, 'native_file_bytes': native_file.stat().st_size,
                        'reference_price_bytes': native_file.stat().st_size,
                        'reference_price_scope': 'raw native stream; known benchmark timing and profile are free',
                        'decoded_yuv_sha256': sha(yuv),
                        'decoded_rgb_sha256': sha(directory / 'decoded.npy'),
                        'yuv_frame_sha256': [hashlib.sha256(f.tobytes()).hexdigest() for f in raw],
                        'rgb_frame_sha256': [hashlib.sha256(f.tobytes()).hexdigest() for f in rgb], **record})
    sender = args.timer.measure('sender.derive', lambda: fresh_ball_sender.derive(source, decoded_bases, fps, tracker))
    validate_sender(sender, fps, frames)
    write_receipt(work / 'sender.json', sender)
    return {'frames': frames, 'fps': [fps.numerator, fps.denominator],
            'source_yuv_sha256': sha(source_yuv), 'bases': records, 'sender': sender}

def cpu_controls(*, source_yuv: Path, sender: dict[str, Any], bases: dict[int, Path],
                 fps: Fraction, frames: int, work: Path, encoder: str, decoder: str,
                 cpus: list[int]) -> dict[str, Any]:
    """Encode six ROI controls, three paid tiles and matching PSBL streak packets.

    ``work`` must be new and located by the caller under the fleet scratch root.
    This function does not authenticate its inputs or establish receiver isolation.
    """
    if len(set(cpus)) != 4 or fps <= 0 or set(bases) != set(SVT_POINTS):
        raise ValueError('four distinct CPUs, positive fps and three native bases required')
    validate_sender(sender, fps, frames)
    work.mkdir(parents=True, exist_ok=False)
    from experiments.tennis import k5d_worker
    resources = work / 'resources.json'
    write_receipt(resources, {'resources': {'svt': {'codec': 'svt-av1-ivf', 'decoder': decoder,
                                                   'decoder_sha256': sha(Path(decoder))}}})
    points = {row['point']: row for row in sender['points'] if row['codec'] == 'svt'}
    records: list[dict[str, Any]] = []
    common: dict[str, Any] = dict(fps=fps, frames=frames, preset=4, threads=4, full_range=False,
                  encoder=encoder, decoder=decoder, timeout=1800, cpus=cpus)
    for family in ('raw', 'fitted'):
        for crf in ((35, 48, 62) if family == 'raw' else SVT_POINTS):
            track = sender['initial_track'] if family == 'raw' else points[crf]['centres']
            directory = work / f'roi-{family}-{crf}'
            directory.mkdir()
            roi = directory / 'roi.txt'
            roi.write_text(roi_map(track, frames))
            result = svtav1.code(source_yuv, directory, width=W, height=H, crf=crf,
                                 extra=['--roi-map-file', str(roi)], **common)
            if result['decoded_frames'] != frames or result['temporal_units'] != frames:
                raise ValueError('ROI frame count mismatch')
            decoded = Path(result['decoded'])
            decoded_frames = np.memmap(decoded, dtype=np.uint8, mode='r', shape=(frames, FRAME_BYTES))
            records.append({'family': family, 'crf': crf, 'roi_sha256': sha(roi),
                            'decoded_yuv_sha256': sha(decoded),
                            'yuv_frame_sha256': [hashlib.sha256(f.tobytes()).hexdigest() for f in decoded_frames], **result})
    patches = []
    for crf in SVT_POINTS:
        row = points[crf]
        directory = work / f'pixel-{crf}'
        directory.mkdir()
        data, positions = patch_frames(source_yuv, row['centres'], frames)
        patch_input = directory / 'input.yuv'
        patch_input.write_bytes(data.tobytes())
        result = svtav1.code(patch_input, directory, width=64, height=64, crf=20, **common)
        if result['decoded_frames'] != frames or result['temporal_units'] != frames:
            raise ValueError('patch frame count mismatch')
        native = bases[crf].read_bytes()
        stream = ball_stream.Stream(W, H, fps, frames, 'svt-av1-ivf', 'svt', native,
                                    bytes.fromhex(row['ball_payload_hex']))
        packet = directory / 'streak.psbl'
        packet.write_bytes(ball_stream.pack(stream))
        if ball_stream.unpack(packet.read_bytes()) != stream:
            raise ValueError('PSBL round trip mismatch')
        streak_replay = k5d_worker.decode(packet, resources, directory / 'streak-replay')
        origins = [[p['left'], p['top']] if p['present'] else None for p in positions]
        native_meta = {'version': 1, 'profile': 'native-av1', 'width': W, 'height': H,
                       'fps': [fps.numerator, fps.denominator], 'frames': frames,
                       'full_range': False, 'decoder_sha256': sha(Path(decoder))}
        meta = {'version': 1, 'profile': 'tracked-pixel-v1', 'native': native_meta,
                'numpy_version': np.__version__, 'positions': origins, 'tile_size': 64,
                'inner_box': [16, 16, 32, 32]}
        tile = Path(result['stream']).read_bytes() if any(p is not None for p in origins) else b''
        ptp = directory / 'pixel.ptp'
        ptp.write_bytes(tracked_patch.pack(meta, native, tile))
        replay = tracked_patch.decode(ptp, directory / 'rendered.rgb', scratch=directory, decoder=decoder)
        inspection = tracked_patch.inspect(ptp.read_bytes())
        if replay['receipts'][0]['decoded_yuv_sha256'] != streak_replay['decoded_yuv_sha256']:
            raise ValueError('independent receivers disagree on native base decode')
        streak_rgb = np.memmap(directory / 'streak-replay/rendered.rgb', dtype=np.uint8,
                               mode='r', shape=(frames, W * H * 3))
        streak_frame_hashes = [hashlib.sha256(f.tobytes()).hexdigest() for f in streak_rgb]
        patches.append({'crf': crf, 'positions': positions, 'input_sha256': sha(patch_input),
                        'streak_sha256': sha(packet), 'base_sha256': sha(bases[crf]),
                        'streak_replay': streak_replay, 'streak_frame_sha256': streak_frame_hashes, 'packet': inspection, 'replay': replay, **result})
    return {'frames': frames, 'fps': [fps.numerator, fps.denominator], 'roi': records, 'pixel': patches}


def write_receipt(path: Path, record: dict[str, Any]) -> None:
    with path.open('x') as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write('\n')


# These immutable source identities are copied from the pre-outcome freeze;
# the scoring-only CSV and its coordinates are never read by either encoder.
SOURCES = {
    'match209_000': (64, Fraction(25), '48eb20bf0353fea6d7663332ec31cadbc89939df596ed801ea128e0a777427d0'),
    'match269_000': (64, Fraction(25), 'c714c168c5334cc324e6821082c74023a96b2c1478ba3840fbb5c8fb58bdf5ed'),
    'match235_000': (154, Fraction(60), 'd3918629801618f6f6a3cea88302901c577ca1852dc3c786fe0ec2f9cb318fdd'),
}
SOURCE_ARCHIVE_SHA = 'e5edc12c857658a119bf6d6cbe029ee64c09d6a860f1b975fe0708952895293d'
RUNTIME_SHA = '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'
MODEL_TREE = 'f7a5c1c6284365f84c324d62169186ef1312c4d921bc0df5dbacd16ef5361964'
MODEL_WEIGHT = 'b052479e6229d840a1a7f42dbc83a6b752a53ef8bc5841a4b957acd519563fec'
IMAGE_CKPT = 'b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302'
VIDEO_CKPT = '934bde4a12fc0b6b0c679ec6ae21d41cb7722debf8b631ee0dd750d030b1a6b3'
GPU_PROFILE = 'ball-confirmation-parent4-child6-v1'
TOOL_SHA = {'ffmpeg': '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2',
            'encoder': '4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4',
            'decoder': 'e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6'}


def publish_tree(root: Path, target: Path) -> dict[str, Any]:
    """Create a regular-files-only archive and bind every member by SHA256."""
    paths = sorted(p for p in root.rglob('*') if p.is_file())
    if not paths or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('empty or symlinked publication')
    inventory = {p.relative_to(root).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths}
    with tarfile.open(target, 'x') as tar:
        for p in paths:
            tar.add(p, arcname='publish/' + p.relative_to(root).as_posix(), recursive=False)
    return {'sha256': sha(target), 'bytes': target.stat().st_size, 'members': inventory}



def verify_archive(archive: Path, binding: dict[str, Any]) -> bool:
    if sha(archive) != binding['sha256'] or archive.stat().st_size != binding['bytes']:
        return False
    expected = binding['members']
    with tarfile.open(archive) as tar:
        entries = tar.getmembers()
        if (len(entries) != len(expected) or {m.name for m in entries} != {'publish/' + x for x in expected}
                or any(not m.isfile() for m in entries)):
            return False
        for entry in entries:
            rel = entry.name.removeprefix('publish/')
            if Path(rel).is_absolute() or '..' in Path(rel).parts or entry.size != expected[rel]['bytes']:
                return False
            handle = tar.extractfile(entry)
            if handle is None:
                return False
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
            if digest.hexdigest() != expected[rel]['sha256']:
                return False
    return True

def extract_tree(archive: Path, binding: dict[str, Any], destination: Path) -> None:
    if sha(archive) != binding['sha256'] or archive.stat().st_size != binding['bytes']:
        raise ValueError('publication archive identity mismatch')
    expected = binding['members']
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as tar:
        entries = tar.getmembers()
        if (len(entries) != len(expected) or {m.name for m in entries} != {'publish/' + x for x in expected}
                or any(not m.isfile() for m in entries)):
            raise ValueError('publication inventory mismatch')
        for member in entries:
            rel = member.name.removeprefix('publish/')
            if Path(rel).is_absolute() or '..' in Path(rel).parts:
                raise ValueError('unsafe publication member')
            info = expected[rel]
            if member.size != info['bytes']:
                raise ValueError('publication member size mismatch')
            reader = tar.extractfile(member)
            if reader is None:
                raise ValueError('unreadable publication member')
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as out:
                shutil.copyfileobj(reader, out)
            if sha(target) != info['sha256']:
                raise ValueError('publication member hash mismatch')


def tool_identity() -> dict[str, Any]:
    result = {}
    for name, binary in [('ffmpeg', 'ffmpeg'), ('encoder', 'SvtAv1EncApp'), ('decoder', 'dav1d')]:
        info = svtav1.tool(str(Path(sys.prefix) / 'bin' / binary))
        if info['sha256'] != TOOL_SHA[name]:
            raise ValueError(f'{name} differs from frozen tool')
        result[name] = info
    return result



def valid_cpu_binding(binding: dict[str, Any], cpus: list[int], *, role: str,
                      parent: dict[str, Any] | None = None) -> bool:
    from experiments.tennis import ball_confirmation_entry as entry

    try:
        allowed, selected = binding['allowed'], binding['selected']
        valid = (binding['schema'] == 1 and binding['role'] == role and binding['claim'] == 4
                 and binding['task_count_before'] == 1 and allowed == sorted(set(allowed))
                 and len(allowed) >= 4 and selected == allowed[:4] == binding['after'] == cpus
                 and binding['pid'] > 0 and binding['pgid'] > 0 and binding['start_ticks'] > 0
                 and binding['code_sha256'] == sha(Path(entry.__file__)))
        if parent is not None:
            valid = (valid and binding['pid'] == parent['host_pid'] and binding['pgid'] == parent['host_pgid']
                     and binding['start_ticks'] == parent['start_ticks'])
        return bool(valid)
    except (KeyError, TypeError, ValueError):
        return False

def stage_context() -> tuple[Path, Path, dict[str, Any], list[int]]:
    from experiments.tennis import ball_confirmation_entry

    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    cpus = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined,unused-ignore]
    binding = json.loads((stage / 'command-cpu-binding.json').read_text())
    if (not valid_cpu_binding(binding, cpus, role='command') or binding['pid'] != os.getpid()
            or binding['pgid'] != os.getpgrp()
            or binding['start_ticks'] != ball_confirmation_entry.process_start_ticks()
            or os.environ.get('PS_CPU_ALLOWANCE') != '4'
            or dispatch['staged_environment']['sha256'] != RUNTIME_SHA):
        raise ValueError('four-CPU affinity or frozen runtime mismatch')
    return stage, scratch, dispatch, cpus


def source_member(archive: Path, expected_archive: str, source_id: str, destination: Path) -> None:
    if expected_archive != SOURCE_ARCHIVE_SHA or sha(archive) != SOURCE_ARCHIVE_SHA or source_id not in SOURCES:
        raise ValueError('source archive or ID mismatch')
    name = f'sources/{source_id}/original.mp4'
    with tarfile.open(archive) as tar:
        matches = [m for m in tar.getmembers() if m.name == name and m.isfile()]
        if len(matches) != 1:
            raise ValueError('missing or duplicate source member')
        reader = tar.extractfile(matches[0])
        if reader is None:
            raise ValueError('source member unreadable')
        with destination.open('xb') as out:
            shutil.copyfileobj(reader, out)
    if sha(destination) != SOURCES[source_id][2]:
        raise ValueError('original video identity mismatch')


def run_bases(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from experiments.tennis import k5, k5b_trackers, k5d_encoder, k5d_source_oracle

    stage, scratch, dispatch, cpus = stage_context()
    n, fps, source_sha = SOURCES[args.source_id]
    frames = SMOKE_FRAMES if os.environ['PS_STAGE'] == 'smoke' else n
    if args.frames != frames:
        raise ValueError('stage span mismatch')
    device = (torch.cuda.get_device_name(0), tuple(torch.cuda.get_device_capability(0)))
    if (torch.cuda.device_count() != 1 or device != ('NVIDIA RTX A6000', (8, 6))
            or torch.__version__ != '2.10.0+cu128' or torch.version.cuda != '12.8'):
        raise ValueError('qualified A6000 CUDA runtime required')
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (not uuid.startswith('GPU-') or ',' in uuid or os.environ.get('PS_CLAIMED_GPU_UUID') != uuid
            or os.environ.get('PS_GPU_SHARING') != 'shared' or os.environ.get('PS_GPU_MEMORY_LIMIT_MIB') != '12288'
            or dispatch['gpu_sharing']['mode'] != 'shared' or dispatch['gpu_sharing']['margin_mib'] != 8192
            or dispatch['environment']['gpu']['uuid'] != uuid or torch.cuda.memory_allocated(0) != 0):
        raise ValueError('shared GPU claim or allocation mismatch')
    if (k5d_source_oracle.tree_sha(Path(args.model_code)) != MODEL_TREE or sha(Path(args.model_weight)) != MODEL_WEIGHT
            or sha(Path(args.image_ckpt)) != IMAGE_CKPT or sha(Path(args.video_ckpt)) != VIDEO_CKPT):
        raise ValueError('model resource identity mismatch')
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if total_memory < 12288 * 1024**2:
        raise ValueError('insufficient GPU capacity')
    torch.cuda.set_per_process_memory_fraction(4096 * 1024**2 / total_memory, 0)
    status = json.loads((stage.parent / 'run/status.json').read_text())
    stat = Path('/proc/self/stat').read_text()
    fields = stat[stat.rfind(')') + 2:].split()
    host_pid, host_pgid, start_ticks = int(stat[:stat.find(' ')]), int(fields[2]), int(fields[19])
    if status.get('process_group_id') != host_pgid or os.getpgrp() != host_pgid or host_pgid <= 0 or start_ticks <= 0:
        raise ValueError('parent process group differs from campaign')
    resource_dir = scratch / 'shared-resource-receipts'
    resource_dir.mkdir(exist_ok=False)
    parent = {'profile': GPU_PROFILE, 'role': 'parent', 'gpu_uuid': uuid, 'device_name': device[0],
              'capability': list(device[1]), 'budget_mib': 4096, 'total_bytes': total_memory,
              'fraction': 4096 * 1024**2 / total_memory, 'allocated_before': 0,
              'host_pid': host_pid, 'host_pgid': host_pgid, 'campaign_pgid': host_pgid, 'start_ticks': start_ticks}
    write_receipt(resource_dir / 'parent.json', parent)
    os.environ['PS_K5D_SHARED_NUMERICAL'] = GPU_PROFILE
    os.environ['PS_K5D_RESOURCE_RECEIPTS'] = str(resource_dir)
    # k5 captures its process environment at import time; explicitly propagate
    # the scoped guard and claimed device into each actual DCVC worker.
    for key in ('PS_K5D_SHARED_NUMERICAL', 'PS_K5D_RESOURCE_RECEIPTS', 'PS_SCRATCH_DIR', 'PS_STAGE_DIR',
                'PS_GPU_SHARING', 'PS_GPU_MEMORY_LIMIT_MIB', 'PS_CLAIMED_GPU_UUID', 'CUDA_VISIBLE_DEVICES'):
        k5.STAGE_ENV[key] = os.environ[key]
    tools = tool_identity()
    work = scratch / 'confirmation-bases'
    work.mkdir(exist_ok=False)
    pub = work / 'publish'
    pub.mkdir()
    source_member(Path(args.source_archive), args.source_archive_sha256, args.source_id, work / 'source.mp4')
    from experiments.tennis.k5d_timing import Timer
    timer = Timer(pub / 'timing.json')
    start = time.monotonic()
    source_record = timer.measure('source.native_decode_and_resize', lambda: decode_source(
        work / 'source.mp4', source_sha, frames, fps, pub / 'source.rgb'))
    source = np.memmap(pub / 'source.rgb', dtype=np.uint8, mode='r', shape=(frames, H, W, 3))
    convert = timer.measure('source.rgb_to_yuv', lambda: k5d_encoder.convert_rgb(
        tools['ffmpeg']['path'], source, pub / 'source.yuv', fps))
    tracker = timer.measure('sender.model_init', lambda: k5b_trackers.BallTrack(Path(args.model_code), Path(args.model_weight)))
    if next(tracker.net.parameters()).device.type != 'cuda':
        raise ValueError('BallTrack model is not on CUDA')
    worker_args = argparse.Namespace(timer=timer, encoder=tools['encoder']['path'], decoder=tools['decoder']['path'],
                                     image_ckpt=args.image_ckpt, image_sha256=IMAGE_CKPT,
                                     video_ckpt=args.video_ckpt, video_sha256=VIDEO_CKPT)
    result = prepare_bases(source=source, source_yuv=pub / 'source.yuv', fps=fps, work=pub / 'bases',
                           args=worker_args, tracker=tracker, cpus=cpus)
    shutil.copytree(resource_dir, pub / 'resource-receipts')
    receipt = {'schema': 1, 'phase': 'bases', 'source_id': args.source_id, 'frames': frames,
               'fps': [fps.numerator, fps.denominator], 'source_archive_sha256': args.source_archive_sha256,
               'source': source_record, 'transform_code_sha256': sha(Path(__file__)),
               'conversion_command': convert, 'tools': tools, 'result': result,
               'resources': {'model_tree': MODEL_TREE, 'model_weight': MODEL_WEIGHT,
                             'image_ckpt': IMAGE_CKPT, 'video_ckpt': VIDEO_CKPT},
               'dispatch': dispatch, 'cpus': cpus, 'cpu_binding': json.loads((stage / 'command-cpu-binding.json').read_text()),
               'wall_seconds': time.monotonic() - start,
               'cuda': {'name': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'parent_allocator_cap_mib': 4096, 'parent_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                        'child_allocator_cap_mib': 6144, 'profile': GPU_PROFILE, 'uuid': uuid}}
    write_receipt(pub / 'receipt.json', receipt)
    binding = publish_tree(pub, stage / 'published.tar')
    result_stage = {'phase': 'bases', 'source_id': args.source_id, 'frames': frames, 'archive': binding}
    write_receipt(stage / 'confirmation-stage.json', result_stage)
    return result_stage


def prefix_sender(sender: dict[str, Any], frames: int) -> dict[str, Any]:
    """Bind a smoke prefix to the same full-window fitted sender appearances."""
    result = {**sender, 'frames': frames, 'initial_track': sender['initial_track'][:frames]}
    rows = []
    for original in sender['points']:
        positions, appearances = ball.decode_frames(bytes.fromhex(original['ball_payload_hex']), sender['frames'], W, H)
        payload = ball.encode_frames(positions[:frames], appearances[:frames], W, H)
        row = {**original, 'centres': original['centres'][:frames], 'origins': original['origins'][:frames],
               'ball_payload_hex': payload.hex(), 'ball_payload_sha256': hashlib.sha256(payload).hexdigest()}
        rows.append(row)
    result['points'] = rows
    return result


def run_controls(args: argparse.Namespace) -> dict[str, Any]:
    stage, scratch, dispatch, cpus = stage_context()
    n, fps, _ = SOURCES[args.source_id]
    smoke = os.environ['PS_STAGE'] == 'smoke'
    frames = SMOKE_FRAMES if smoke else n
    if args.frames != frames:
        raise ValueError('stage span mismatch')
    tools = tool_identity()
    binding = json.loads(Path(args.bases_binding).read_text())
    if (sha(Path(args.bases_binding)) != args.bases_binding_sha256 or binding['phase'] != 'bases'
            or binding['source_id'] != args.source_id or binding['frames'] != n):
        raise ValueError('authenticated full baseline binding required')
    if sha(Path(args.bases_acceptance)) != args.bases_acceptance_sha256:
        raise ValueError('baseline acceptance identity differs')
    acceptance = json.loads(Path(args.bases_acceptance).read_text())
    if (acceptance.get('schema') != 'pointstream-ball-confirmation-accepted-full-v1'
            or acceptance.get('passed') is not True or acceptance.get('canonical_complete') is not True
            or acceptance.get('phase') != 'bases' or acceptance.get('source_id') != args.source_id
            or acceptance.get('frames') != n or acceptance.get('stage_sha256') != args.bases_binding_sha256
            or acceptance.get('archive_sha256') != binding['archive']['sha256']
            or not acceptance.get('checks') or not all(v is True for v in acceptance['checks'].values())):
        raise ValueError('accepted complete baseline producer required')
    work = scratch / 'confirmation-controls'
    work.mkdir(exist_ok=False)
    root = work / 'baseline'
    extract_tree(Path(args.bases_archive), binding['archive'], root)
    if not all(validate_publication(root, binding).values()):
        raise ValueError('baseline publication validation failed')
    parent = json.loads((root / 'receipt.json').read_text())
    if (parent['source_id'] != args.source_id or parent['frames'] != n
            or parent['source_archive_sha256'] != SOURCE_ARCHIVE_SHA
            or canonical_sha(parent['dispatch']) != acceptance['dispatch_canonical_sha256']):
        raise ValueError('baseline source binding mismatch')
    sender = json.loads((root / 'bases/sender.json').read_text())
    validate_sender(sender, fps, n)
    start = time.monotonic()
    pub = work / 'publish'
    pub.mkdir()
    source = root / 'source.yuv'
    bases = {p: root / f'bases/svt-{p}/stream.ivf' for p in SVT_POINTS}
    if smoke:
        sender = prefix_sender(sender, frames)
        source = pub / 'source-prefix.yuv'
        with (root / 'source.yuv').open('rb') as reader, source.open('xb') as out:
            remaining = frames * FRAME_BYTES
            while remaining:
                chunk = reader.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError('truncated source prefix')
                out.write(chunk)
                remaining -= len(chunk)
        for crf in SVT_POINTS:
            directory = pub / f'smoke-base-{crf}'
            directory.mkdir()
            svtav1.code(source, directory, width=W, height=H, fps=fps, frames=frames, crf=crf,
                        preset=4, threads=4, full_range=False, encoder=tools['encoder']['path'],
                        decoder=tools['decoder']['path'], timeout=600, cpus=cpus)
            bases[crf] = directory / 'stream.ivf'
    write_receipt(pub / 'sender.json', sender)
    result = cpu_controls(source_yuv=source, sender=sender, bases=bases, fps=fps, frames=frames,
                          work=pub / 'controls', encoder=tools['encoder']['path'], decoder=tools['decoder']['path'], cpus=cpus)
    receipt = {'schema': 1, 'phase': 'controls', 'source_id': args.source_id, 'frames': frames,
               'fps': [fps.numerator, fps.denominator], 'base_archive_sha256': binding['archive']['sha256'],
               'base_acceptance_sha256': args.bases_acceptance_sha256,
               'source_yuv_sha256': sha(source), 'tools': tools, 'dispatch': dispatch, 'cpus': cpus,
               'cpu_binding': json.loads((stage / 'command-cpu-binding.json').read_text()),
               'result': result, 'wall_seconds': time.monotonic() - start}
    write_receipt(pub / 'receipt.json', receipt)
    archive = publish_tree(pub, stage / 'published.tar')
    output = {'phase': 'controls', 'source_id': args.source_id, 'frames': frames, 'archive': archive}
    write_receipt(stage / 'confirmation-stage.json', output)
    return output



def validate_resource_receipts(root: Path, receipt: dict[str, Any]) -> bool:
    """Bind parent and all six actual worker caps to plans, reports and device."""
    directory = root / 'resource-receipts'
    files = list(directory.glob('child-*.json'))
    if len(files) != 6 or {p.name for p in directory.iterdir()} != {'parent.json', *(p.name for p in files)}:
        return False
    parent = json.loads((directory / 'parent.json').read_text())
    cuda = receipt['cuda']
    uuid = cuda['uuid']
    device = (parent['device_name'], tuple(parent['capability']))
    variants = {('NVIDIA RTX A6000', (8, 6)): 'sm80', ('NVIDIA RTX 6000 Ada Generation', (8, 9)): 'sm89'}
    if (device not in variants or parent['profile'] != GPU_PROFILE or parent['role'] != 'parent'
            or parent['budget_mib'] != 4096 or parent['allocated_before'] != 0
            or parent['gpu_uuid'] != uuid or cuda['name'] != device[0]
            or cuda['profile'] != GPU_PROFILE or cuda['parent_allocator_cap_mib'] != 4096 or cuda['child_allocator_cap_mib'] != 6144
            or cuda['parent_peak_reserved_bytes'] > 4096 * 1024**2
            or parent['host_pgid'] != parent['campaign_pgid'] or parent['start_ticks'] <= 0
            or abs(parent['fraction'] - 4096 * 1024**2 / parent['total_bytes']) > 1e-12):
        return False
    expected = {(action, f'confirmation-bases/publish/bases/dcvc-{p}/plan.json')
                for p in (0, 21, 42) for action in ('encode', 'decode')}
    seen: set[tuple[str, str]] = set()
    process_ids: set[tuple[int, int]] = set()
    for path in files:
        child = json.loads(path.read_text())
        key = (child['action'], child['plan'])
        process = (child['host_pid'], child['start_ticks'])
        if (key not in expected or key in seen or process in process_ids or child['profile'] != GPU_PROFILE
                or child['role'] != 'dcvc' or child['budget_mib'] != 6144 or child['allocated_before'] != 0
                or child['gpu_uuid'] != uuid or child['device_name'] != device[0]
                or tuple(child['capability']) != device[1] or child['total_bytes'] != parent['total_bytes']
                or child['host_pgid'] != parent['host_pgid'] or child['campaign_pgid'] != parent['campaign_pgid']
                or child['pid'] != child['host_pid'] or child['start_ticks'] <= 0
                or abs(child['fraction'] - 6144 * 1024**2 / child['total_bytes']) > 1e-12):
            return False
        seen.add(key)
        process_ids.add(process)
        point = next(p for p in (0, 21, 42) if f'dcvc-{p}/plan.json' in key[1])
        report = json.loads((root / f'bases/dcvc-{point}/{key[0]}.json').read_text())
        env = report['environment']
        if (env['device_name'] != device[0] or tuple(env['capability']) != device[1]
                or env['cuda_visible_devices'] != uuid or env['torch'] != '2.10.0+cu128'
                or env['torch_cuda'] != '12.8' or env['extension']['variant'] != variants[device]
                or not 0 <= report['peak_allocated_mib'] <= 6144):
            return False
    return seen == expected

def validate_publication(root: Path, record: dict[str, Any]) -> dict[str, bool]:
    """Check actual payload structure and every published frame, not only tar hashes."""
    from experiments.tennis import k5d_worker

    source_id = record['source_id']
    n, fps, _ = SOURCES[source_id]
    frames = record['frames']
    if frames not in (SMOKE_FRAMES, n):
        raise ValueError('unregistered frame span')
    receipt = json.loads((root / 'receipt.json').read_text())
    checks = {'phase_and_source': receipt['phase'] == record['phase'] and receipt['source_id'] == source_id
              and receipt['frames'] == frames and receipt['fps'] == [fps.numerator, fps.denominator],
              'runtime': receipt['dispatch']['staged_environment']['sha256'] == RUNTIME_SHA,
              'allocation': len(receipt['cpus']) == 4 and len(set(receipt['cpus'])) == 4}
    binding = receipt['cpu_binding']
    parent = json.loads((root / 'resource-receipts/parent.json').read_text()) if record['phase'] == 'bases' else None
    checks['cpu_binding'] = valid_cpu_binding(binding, receipt['cpus'], role='command', parent=parent)
    if record['phase'] == 'bases':
        checks['gpu_child_allocator_cap'] = validate_resource_receipts(root, receipt)
        source = root / 'source.rgb'
        raw = np.memmap(source, dtype=np.uint8, mode='r', shape=(frames, H * W * 3))
        checks['source_frames'] = (source.stat().st_size == frames * H * W * 3
            and sha(source) == receipt['source']['rgb_sha256']
            and [hashlib.sha256(f.tobytes()).hexdigest() for f in raw] == receipt['source']['rgb_frame_sha256'])
        checks['source_yuv'] = ((root / 'source.yuv').stat().st_size == frames * FRAME_BYTES
                                and sha(root / 'source.yuv') == receipt['result']['source_yuv_sha256'])
        sender = json.loads((root / 'bases/sender.json').read_text())
        validate_sender(sender, fps, frames)
        checks['sender'] = sender == receipt['result']['sender']
        rows = receipt['result']['bases']
        checks['six_ordered_bases'] = [(r['kind'], r['point']) for r in rows] == list(POINTS)
        for row in rows:
            kind, point = row['kind'], row['point']
            directory = root / 'bases' / f'{kind}-{point}'
            native = directory / ('stream.psdc' if kind == 'dcvc' else 'stream.ivf')
            yuv = directory / 'decoded.yuv'
            checks[f'{kind}-{point}-bytes'] = (sha(native) == row['stream_sha256']
                                                and native.stat().st_size == row['native_file_bytes'] == row['reference_price_bytes']
                                                and sha(yuv) == row['decoded_yuv_sha256']
                                                and yuv.stat().st_size == frames * FRAME_BYTES)
            raw_yuv = np.memmap(yuv, dtype=np.uint8, mode='r', shape=(frames, FRAME_BYTES))
            rgb = np.load(directory / 'decoded.npy', mmap_mode='r')
            checks[f'{kind}-{point}-frames'] = (rgb.shape == (frames, H, W, 3)
                and sha(directory / 'decoded.npy') == row['decoded_rgb_sha256']
                and [hashlib.sha256(f.tobytes()).hexdigest() for f in raw_yuv] == row['yuv_frame_sha256']
                and [hashlib.sha256(f.tobytes()).hexdigest() for f in rgb] == row['rgb_frame_sha256'])
            if kind == 'svt':
                checks[f'{kind}-{point}-native'] = len(svtav1.ivf_frames(native.read_bytes())) == frames
            else:
                checks[f'{kind}-{point}-native'] = bool(row['deterministic'] and row['decoder_matches_encoder_intra'])
    elif record['phase'] == 'controls':
        sender = json.loads((root / 'sender.json').read_text())
        validate_sender(sender, fps, frames)
        result = receipt['result']
        checks['roi_matrix'] = [(x['family'], x['crf']) for x in result['roi']] == [
            ('raw', 35), ('raw', 48), ('raw', 62), ('fitted', 48), ('fitted', 55), ('fitted', 62)]
        checks['pixel_matrix'] = [x['crf'] for x in result['pixel']] == list(SVT_POINTS)
        for row in result['roi']:
            directory = root / 'controls' / f"roi-{row['family']}-{row['crf']}"
            track = sender['initial_track'] if row['family'] == 'raw' else next(
                p['centres'] for p in sender['points'] if p['codec'] == 'svt' and p['point'] == row['crf'])
            checks[f"roi-{row['family']}-{row['crf']}"] = (
                (directory / 'roi.txt').read_text() == roi_map(track, frames)
                and sha(directory / 'stream.ivf') == row['stream_sha256']
                and len(svtav1.ivf_frames((directory / 'stream.ivf').read_bytes())) == frames
                and (directory / 'decoded.yuv').stat().st_size == frames * FRAME_BYTES
                and sha(directory / 'decoded.yuv') == row['decoded_yuv_sha256']
                and [hashlib.sha256(f.tobytes()).hexdigest() for f in np.memmap(
                    directory / 'decoded.yuv', dtype=np.uint8, mode='r', shape=(frames, FRAME_BYTES))]
                    == row['yuv_frame_sha256'])
        for row in result['pixel']:
            crf = row['crf']
            directory = root / 'controls' / f'pixel-{crf}'
            psbl = ball_stream.unpack((directory / 'streak.psbl').read_bytes())
            k5d_worker.validate_native_header(psbl)
            meta, native_bytes, tile = tracked_patch.unpack((directory / 'pixel.ptp').read_bytes())
            fitted = next(p for p in sender['points'] if p['codec'] == 'svt' and p['point'] == crf)
            checks[f'pixel-{crf}-binding'] = (psbl.ball_payload.hex() == fitted['ball_payload_hex']
                and psbl.native == native_bytes and meta['positions'] == fitted['origins']
                and (not tile or sha(directory / 'stream.ivf') == hashlib.sha256(tile).hexdigest())
                and tracked_patch.inspect((directory / 'pixel.ptp').read_bytes()) == row['packet'])
            checks[f'pixel-{crf}-reconstruction'] = (
                sha(directory / 'rendered.rgb') == row['replay']['decoded_sha256']
                and sha(directory / 'streak-replay/rendered.rgb') == row['streak_replay']['rendered_rgb_sha256']
                and row['replay']['receipts'][0]['decoded_yuv_sha256'] == row['streak_replay']['decoded_yuv_sha256']
                and [hashlib.sha256(f.tobytes()).hexdigest() for f in np.memmap(
                    directory / 'rendered.rgb', dtype=np.uint8, mode='r', shape=(frames, W * H * 3))]
                    == row['replay']['rgb_frame_sha256']
                and [hashlib.sha256(f.tobytes()).hexdigest() for f in np.memmap(
                    directory / 'streak-replay/rendered.rgb', dtype=np.uint8, mode='r', shape=(frames, W * H * 3))]
                    == row['streak_frame_sha256'])
    else:
        raise ValueError('unknown phase')
    return checks


def validate_stage() -> dict[str, Any]:
    import tempfile
    from experiments.tennis import ball_confirmation_entry

    stage = Path(os.environ['PS_STAGE_DIR'])
    validator_binding = json.loads((stage / 'validator-cpu-binding.json').read_text())
    validator_cpus = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined,unused-ignore]
    if (not valid_cpu_binding(validator_binding, validator_cpus, role='validator')
            or validator_binding['pid'] != os.getpid() or validator_binding['pgid'] != os.getpgrp()
            or validator_binding['start_ticks'] != ball_confirmation_entry.process_start_ticks()):
        raise ValueError('validator CPU binding mismatch')
    record = json.loads((stage / 'confirmation-stage.json').read_text())
    with tempfile.TemporaryDirectory(prefix='confirmation-check-', dir=os.environ['TMPDIR']) as temp:
        root = Path(temp) / 'publish'
        extract_tree(stage / 'published.tar', record['archive'], root)
        checks = validate_publication(root, record)
        actual_dispatch = json.loads((stage / 'dispatch.json').read_text())
        receipt = json.loads((root / 'receipt.json').read_text())
        checks['receipt_dispatch_binding'] = receipt['dispatch'] == actual_dispatch
    execution = json.loads((stage / 'execution.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    command = dispatch['command']
    def flag(name: str) -> str:
        if command.count(name) != 1:
            raise ValueError(f'invalid dispatched argument {name}')
        return command[command.index(name) + 1]
    stage_name = os.environ['PS_STAGE']
    expected = SMOKE_FRAMES if stage_name == 'smoke' else SOURCES[record['source_id']][0]
    checks['dispatch_binding'] = (stage_name in ('smoke', 'full') and record['frames'] == expected
        and flag('--frames') == str(expected) and flag('--source-id') == record['source_id']
        and flag('--phase') == record['phase'])
    limit = 600 if os.environ['PS_STAGE'] == 'smoke' else 1800
    checks['execution_success'] = execution['exit_code'] == 0 and 0 < execution['seconds'] <= limit
    forecast = None
    if os.environ['PS_STAGE'] == 'smoke':
        full_frames = SOURCES[record['source_id']][0]
        forecast = execution['seconds'] * full_frames / SMOKE_FRAMES * 1.25
        checks['bounded_full_forecast'] = forecast <= 1800
    return {'passed': all(checks.values()), 'checks': checks, 'projected_full_seconds': forecast,
            'scope': 'physical preparation, not a quality result'}


def spec(phase: str, source_id: str, deadline: str, *, bases_archive: str = '', bases_sha: str = '',
         bases_binding: str = '', binding_sha: str = '', bases_acceptance: str = '', acceptance_sha: str = '') -> dict[str, Any]:
    if phase not in ('bases', 'controls') or source_id not in SOURCES:
        raise ValueError('invalid phase or source')
    data = '/home/itec/emanuele/Datasets/pointstream-data'
    source = data + '/tennis/confirmation-source-20261115T062148Z-297a05e2/tennis-confirmation-sources.tar'
    if phase == 'bases':
        inputs = [
            ('source_archive', source, SOURCE_ARCHIVE_SHA, False),
            ('model_code', data + '/tennis/k5b-2026-10-20/inputs/RacketVision-c44af2a.tar',
             '5bec7ca535d2562ce0252a45f3dded0cfd5897571a6e29a10247a87884d961ff', True),
            ('model_weight', data + '/tennis/k5b-2026-10-20/inputs/balltrack_best.state_dict.pt', MODEL_WEIGHT, False),
            ('image_ckpt', data + '/visor/b2-2026-10-07/inputs/dcvc/cvpr2026_image.pth.tar', IMAGE_CKPT, False),
            ('video_ckpt', data + '/visor/b2-2026-10-07/inputs/dcvc/cvpr2026_video_htl.pth.tar', VIDEO_CKPT, False),
        ]
    else:
        if not all((bases_archive, bases_sha, bases_binding, binding_sha, bases_acceptance, acceptance_sha)):
            raise ValueError('controls require accepted baseline archive and binding identities')
        inputs = [('bases_archive', bases_archive, bases_sha, False), ('bases_binding', bases_binding, binding_sha, False),
                  ('bases_acceptance', bases_acceptance, acceptance_sha, False)]
    staged = [{'name': k, 'path': path, 'sha256': digest, 'extract': extract} for k, path, digest, extract in inputs]
    arguments = ['--phase', phase, '--source-id', source_id, '--frames', '{frames}']
    for name, _, _, _ in inputs:
        arguments += ['--' + name.replace('_', '-'), '{staged:' + name + '}']
    arguments += (['--source-archive-sha256', SOURCE_ARCHIVE_SHA] if phase == 'bases'
                  else ['--bases-binding-sha256', binding_sha, '--bases-acceptance-sha256', acceptance_sha])
    result: dict[str, Any] = {
        'schema': 1, 'device': 'gpu' if phase == 'bases' else 'cpu', 'hosts': ['gpu2', 'gpu3'] if phase == 'bases' else ['gpu1', 'gpu3', 'gpu5'],
        'gpu_models': ['RTX A6000'] if phase == 'bases' else [],
        'gpu_memory_mib': 12288 if phase == 'bases' else 0,
        'cpu_threads': 4, 'local_storage_gib': 40, 'entrypoint': ['-m', 'experiments.tennis.ball_confirmation_entry'],
        'arguments': arguments, 'staged_inputs': staged,
        'inputs': [{'path': p['path'], 'sha256': p['sha256']} for p in staged],
        'environment': {'path': data + '/environments/pointstream-20261009T214002Z.tar.gz', 'sha256': RUNTIME_SHA},
        'verify_staged_inputs': True, 'required_commands': [],
        'scale': {'frames': {'smoke': SMOKE_FRAMES, 'full': SOURCES[source_id][0]}},
        'smoke': {'seconds': 600, 'representative_basis': 'Actual first 32 native frames, six real bases and sender or six controls; full-run forecast gate. Historical 16/48 timings suggest fixed overhead, not new-host throughput evidence.'},
        'full': {'seconds': 1800}, 'validator': ['{python}', '-m', 'experiments.tennis.ball_confirmation_entry', '--validate'],
        'validator_seconds': 120, 'budget_seconds': 2700, 'stall_seconds': 600, 'deadline': deadline,
        'contention': {'policy': 'stop', 'resume_attempts': 0},
    }
    if phase == 'bases':
        result['gpu_sharing'] = {'mode': 'shared', 'margin_mib': 8192,
                                 'basis': 'Parent 4 GiB and single DCVC child 6 GiB allocator caps; no throughput claim.'}
        result['contention'] = {'policy': 'continue', 'resume_attempts': 0,
                                'basis': 'Shared GPU correctness with bounded allocator caps; timings are not throughput.'}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('bases', 'controls'))
    parser.add_argument('--source-id', choices=tuple(SOURCES))
    parser.add_argument('--frames', type=int)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--deadline')
    for name in ('source-archive', 'source-archive-sha256', 'model-code', 'model-weight', 'image-ckpt',
                 'video-ckpt', 'bases-archive', 'bases-binding', 'bases-binding-sha256', 'bases-sha',
                 'bases-acceptance', 'bases-acceptance-sha256'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.validate:
        try:
            result = validate_stage()
        except Exception as exc:
            result = {'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
        write_receipt(Path(os.environ['PS_VALIDATION_PATH']), result)
        if not result['passed']:
            raise SystemExit(1)
        return
    if args.spec:
        if not args.phase or not args.source_id or not args.deadline:
            parser.error('spec requires phase, source-id and deadline')
        write_receipt(args.spec, spec(args.phase, args.source_id, args.deadline,
                      bases_archive=args.bases_archive or '', bases_sha=args.bases_sha or '',
                      bases_binding=args.bases_binding or '', binding_sha=args.bases_binding_sha256 or '',
                      bases_acceptance=args.bases_acceptance or '', acceptance_sha=args.bases_acceptance_sha256 or ''))
        return
    if not args.phase or not args.source_id or not args.frames:
        parser.error('phase, source-id and frames required')
    if args.phase == 'bases':
        if not all(getattr(args, x) for x in ('source_archive', 'source_archive_sha256', 'model_code',
                                               'model_weight', 'image_ckpt', 'video_ckpt')):
            parser.error('bases requires staged source and resources')
        record = run_bases(args)
    else:
        if not all(getattr(args, x) for x in ('bases_archive', 'bases_binding', 'bases_binding_sha256',
                                               'bases_acceptance', 'bases_acceptance_sha256')):
            parser.error('controls requires accepted baseline archive and binding')
        record = run_controls(args)
    if os.environ['PS_STAGE'] == 'full':
        stage = Path(os.environ['PS_STAGE_DIR'])
        name = 'confirmation-bases' if args.phase == 'bases' else 'confirmation-controls'
        root = Path(os.environ['PS_SCRATCH_DIR']) / name / 'publish'
        checks = validate_publication(root, record)
        checks['archive_identity'] = verify_archive(stage / 'published.tar', record['archive'])
        full = {'passed': all(checks.values()), 'checks': checks, 'phase': record['phase'],
                'source_id': record['source_id'], 'frames': record['frames'],
                'archive_sha256': record['archive']['sha256'],
                'stage_sha256': sha(stage / 'confirmation-stage.json'),
                'dispatch_canonical_sha256': canonical_sha(json.loads((stage / 'dispatch.json').read_text())),
                'scope': 'full artifact correctness; outer execution status must be checked separately'}
        write_receipt(stage / 'full-validation.json', full)
        if not full['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()

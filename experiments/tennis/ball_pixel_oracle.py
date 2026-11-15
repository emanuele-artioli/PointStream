"""Finite 32px transmitted-position coded-pixel oracle; no job submission.

Placement and framing are granted for free. This is a finite-family control,
not a complete transport or a universal bound on patch methods.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import ball_roi_control as common, k5d_encoder, windows
from src.codecs import svtav1
from src.pointstream import ball, ball_stream

RATES = (48, 55, 62)
EXPECTED_PACKETS = {
    48: ('e85bbd4407c90f7fc633a8e41977f499c0b7d9da671223889f5a16b23f6524ef', 'e01110717a2e9fd04a0c9c80f16209aeaf9466d04da7c316ef53f28420d9d504', 90949),
    55: ('2536b9ed986d6ef41733a309ddf86f530b56521d4adf309282705b72f5a810f5', 'b8f8bb9cc97da4b08f5579aba79d988c20b9b9ddefb88c03587cc4c371fe95b8', 63879),
    62: (common.NATIVE62, '90e8d04c4b1b7a2e35c3d4cd528b8f583543bf872f2eeb44be732cde546a3c95', 34146),
}
PATCH = 32
PAD = 64
PATCH_FRAME_BYTES = PAD * PAD * 3 // 2


def placement(position: tuple[int, int] | None) -> dict[str, Any]:
    if position is None:
        return {'present': False, 'position': None, 'left': None, 'top': None}
    x, y = position
    if not (0 <= x < common.W and 0 <= y < common.H):
        raise ValueError('transmitted position out of bounds')
    left = min(max(2 * round((x - PATCH / 2) / 2), 0), common.W - PATCH)
    top = min(max(2 * round((y - PATCH / 2) / 2), 0), common.H - PATCH)
    return {'present': True, 'position': [x, y], 'left': left, 'top': top}


def patch_frames(source: Path, track: ball.Track, frames: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if len(track) != 48 or frames not in (16, 48):
        raise ValueError('unregistered track or prefix')
    if source.stat().st_size != common.N * common.FRAME_BYTES:
        raise ValueError('source YUV geometry mismatch')
    source_raw = np.memmap(source, mode='r', dtype=np.uint8, shape=(48, common.FRAME_BYTES))
    patches = np.empty((frames, PATCH_FRAME_BYTES), dtype=np.uint8)
    placements = []
    y_size, chroma_size = common.W * common.H, common.W * common.H // 4
    for t, point in enumerate(track[:frames]):
        row = placement(point)
        padded = patches[t]
        padded[:4096] = 16
        padded[4096:] = 128
        y = padded[:4096].reshape(64, 64)
        u = padded[4096:5120].reshape(32, 32)
        v = padded[5120:].reshape(32, 32)
        if point is not None:
            left, top = row['left'], row['top']
            frame = source_raw[t]
            sy = frame[:y_size].reshape(common.H, common.W)
            su = frame[y_size:y_size + chroma_size].reshape(common.H // 2, common.W // 2)
            sv = frame[y_size + chroma_size:].reshape(common.H // 2, common.W // 2)
            y[16:48, 16:48] = sy[top:top + 32, left:left + 32]
            u[8:24, 8:24] = su[top // 2:top // 2 + 16, left // 2:left // 2 + 16]
            v[8:24, 8:24] = sv[top // 2:top // 2 + 16, left // 2:left // 2 + 16]
        row['frame'] = t
        row['source_crop_sha256'] = common.digest(y[16:48, 16:48].tobytes() + u[8:24, 8:24].tobytes() + v[8:24, 8:24].tobytes())
        row['padded_yuv_sha256'] = common.digest(padded.tobytes())
        placements.append(row)
    return patches, placements


def compose(base: np.ndarray, patch: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    if len(base) < len(rows) or len(patch) != len(rows):
        raise ValueError('composite frame count mismatch')
    result = np.array(base[:len(rows)], copy=True)
    for index, row in enumerate(rows):
        if row['frame'] != index:
            raise ValueError('invalid placement order')
        if row['present']:
            left, top = row['left'], row['top']
            result[index, top:top + 32, left:left + 32] = patch[index, 16:48, 16:48]
    return result


def reconstruct_packet(point: dict[str, Any], native: bytes) -> ball_stream.Stream:
    stream = ball_stream.Stream(common.W, common.H, Fraction(30, 1), 48, 'svt-av1-ivf',
                                point['resource_id'], native, bytes.fromhex(point['ball_payload_hex']))
    if common.digest(ball_stream.pack(stream)) != point['wire_sha256']:
        raise ValueError('placement payload is not bound to accepted PSBL')
    return stream


def decode_pair(base: Path, patch: Path, rows: list[dict[str, Any]], decoder: str,
                scratch: Path, base_sha: str, patch_sha: str) -> dict[str, Any]:
    scratch.mkdir(parents=True, exist_ok=False)
    byuv, pyuv = scratch / 'base.yuv', scratch / 'patch.yuv'
    svtav1.run(svtav1.decode_command(decoder, base, byuv, threads=1), timeout=120)
    svtav1.run(svtav1.decode_command(decoder, patch, pyuv, threads=1), timeout=120)
    common.require_hash(byuv, base_sha)
    common.require_hash(pyuv, patch_sha)
    decoded_base = windows.rgb_frames(byuv, common.W, common.H, 48, scratch / 'base.npy')
    decoded_patch = windows.rgb_frames(pyuv, PAD, PAD, len(rows), scratch / 'patch.npy')
    rendered = compose(decoded_base, decoded_patch, rows)
    return {'base_yuv_sha256': common.k5d_source_oracle.sha(byuv),
            'patch_yuv_sha256': common.k5d_source_oracle.sha(pyuv),
            'base_rgb_sha256': common.digest(decoded_base.tobytes()),
            'patch_rgb_sha256': common.digest(decoded_patch.tobytes()),
            'composite_rgb_sha256': common.digest(rendered.tobytes()),
            'base_yuv_frame_sha256': [common.digest(np.asarray(x).tobytes()) for x in
                np.memmap(byuv, dtype=np.uint8, mode='r', shape=(48, common.FRAME_BYTES))],
            'patch_yuv_frame_sha256': [common.digest(np.asarray(x).tobytes()) for x in
                np.memmap(pyuv, dtype=np.uint8, mode='r', shape=(len(rows), PATCH_FRAME_BYTES))],
            'composite_frame_sha256': [common.digest(frame.tobytes()) for frame in rendered]}


def make_spec(deadline: str) -> dict[str, Any]:
    spec = common.cpu_spec(deadline)
    spec['entrypoint'] = ['-m', 'experiments.tennis.ball_pixel_oracle']
    spec['arguments'] = ['--fleet', '--producer', '{staged:producer}', '--archive', '{staged:archive}', '--frames', '{frames}']
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.ball_pixel_oracle', '--validate']
    spec['smoke']['representative_basis'] = 'Full authenticated 48-frame source and native62 control; three separately coded first16 patch streams and fresh receiver replay.'
    return spec


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve()
    work = (scratch / 'pixel-private').resolve()
    if work.exists():
        raise ValueError('private workspace exists')
    work.mkdir()
    for name, expected in common.TOOLS.items():
        common.require_hash(Path(getattr(args, name)), expected)
    qualification, pilot, qbytes = common.producer_metadata(Path(args.producer))
    source = common.source_frames(Path(args.archive), qualification)
    yuv = work / 'source.yuv'
    conversion = k5d_encoder.convert_rgb(args.ffmpeg, source, yuv, Fraction(30, 1))
    del source
    if yuv.stat().st_size != common.N * common.FRAME_BYTES:
        raise ValueError('full source length mismatch')
    common.require_hash(yuv, common.SOURCE_YUV)
    common.expected_native_decode(pilot)
    control_dir = work / 'gold-control'
    control_dir.mkdir()
    old = svtav1.code(yuv, control_dir, width=common.W, height=common.H, fps=Fraction(30, 1),
                      frames=48, crf=62, preset=4, threads=1, full_range=False,
                      encoder=args.encoder, decoder=args.decoder, timeout=600)
    common.require_hash(Path(old['stream']), common.NATIVE62)
    common.require_hash(Path(old['decoded']), common.NATIVE62_DECODE)
    publication = work / 'publication'
    publication.mkdir()
    (publication / 'source-qualification.json').write_bytes(qbytes)
    records = []
    for crf in RATES:
        point = next(p for p in pilot['points'] if p['codec'] == 'svt' and p['point'] == crf)
        packet = common.tar_member(Path(args.producer), f'publish/svt-{crf}/stream.psbl', point['wire_sha256'])
        stream = ball_stream.unpack(packet)
        if (stream.frames != 48 or stream.width != common.W or stream.height != common.H
                or stream.fps != Fraction(30, 1) or stream.codec != 'svt-av1-ivf'
                or common.digest(stream.native) != point['native_sha256']):
            raise ValueError('accepted native packet mismatch')
        track, _ = ball.decode_frames(stream.ball_payload, 48, common.W, common.H)
        patches, positions = patch_frames(yuv, track, args.frames)
        directory = work / f'patch-{crf}'
        directory.mkdir()
        source_patch = directory / 'input.yuv'
        source_patch.write_bytes(patches.tobytes())
        coded = svtav1.code(source_patch, directory, width=PAD, height=PAD, fps=Fraction(30, 1),
                            frames=args.frames, crf=20, preset=4, threads=1, full_range=False,
                            encoder=args.encoder, decoder=args.decoder, timeout=600)
        base = publication / f'base-{crf}.ivf'
        patch = publication / f'patch-{crf}.ivf'
        base.write_bytes(stream.native)
        shutil.copyfile(coded['stream'], patch)
        row = {'crf': crf, 'positions': positions, 'resource_id': stream.resource_id,
               'ball_payload_hex': stream.ball_payload.hex(), 'ball_payload_sha256': common.digest(stream.ball_payload),
               'wire_sha256': point['wire_sha256'], 'base_sha256': point['native_sha256'],
               'base_bytes': base.stat().st_size, 'patch_sha256': common.k5d_source_oracle.sha(patch),
               'patch_bytes': patch.stat().st_size,
               'patch_payload_bytes': sum(svtav1.ivf_frames(patch.read_bytes())), 'patch_input_sha256': common.k5d_source_oracle.sha(source_patch),
               'base_yuv_sha256': point['encoder_decoded_sha256'],
               'patch_yuv_sha256': common.k5d_source_oracle.sha(Path(coded['decoded'])), 'patch_encode': coded}
        row['replay'] = decode_pair(base, patch, positions, args.decoder, directory / 'replay',
                                    row['base_yuv_sha256'], row['patch_yuv_sha256'])
        records.append(row)
    receipt = {'schema': 1, 'scope': 'finite 32px coded patch, transmitted BALL positions and free placement/framing',
               'frames': args.frames, 'producer_sha256': common.PRODUCER, 'source_archive_sha256': common.ARCHIVE,
               'source_yuv_sha256': common.SOURCE_YUV, 'qualification_sha256': common.QUALIFICATION,
               'native62_sha256': common.k5d_source_oracle.sha(Path(old['stream'])),
               'native62_decode_sha256': common.k5d_source_oracle.sha(Path(old['decoded'])),
               'native62_control': old, 'conversion_command': conversion, 'tools': common.TOOLS,
               'points': records, 'promotable': False}
    with (publication / 'receipt.json').open('x') as out:
        json.dump(receipt, out, indent=2, allow_nan=False)
        out.write('\n')
    return receipt


def members() -> list[str]:
    return ['receipt.json', 'source-qualification.json', *[f'{kind}-{crf}.ivf' for crf in RATES for kind in ('base', 'patch')]]


def publish(root: Path, archive: Path) -> dict[str, Any]:
    names = members()
    if {p.name for p in root.iterdir()} != set(names):
        raise ValueError('pixel publication inventory mismatch')
    hashes = {}
    with tarfile.open(archive, 'x') as tar:
        for name in names:
            path = root / name
            if not path.is_file() or path.is_symlink():
                raise ValueError('nonregular publication member')
            hashes[name] = common.k5d_source_oracle.sha(path)
            tar.add(path, arcname='publish/' + name, recursive=False)
    return {'sha256': common.k5d_source_oracle.sha(archive), 'bytes': archive.stat().st_size, 'members': hashes}


def unpack(archive: Path, expected: dict[str, Any], work: Path) -> Path:
    common.require_hash(archive, expected['sha256'])
    if archive.stat().st_size != expected['bytes'] or set(expected['members']) != set(members()):
        raise ValueError('pixel archive inventory mismatch')
    work.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as tar:
        entries = tar.getmembers()
        if (len(entries) != len(members()) or {e.name for e in entries} != {'publish/' + n for n in members()}
                or any(not entry.isfile() for entry in entries)):
            raise ValueError('invalid pixel archive inventory')
        tar.extractall(work, filter='data')
    root = work / 'publish'
    for name, digest in expected['members'].items():
        common.require_hash(root / name, digest)
    return root


def validate_artifact(root: Path, decoder: str, work: Path) -> dict[str, bool]:
    common.require_hash(Path(decoder), common.TOOLS['decoder'])
    common.require_hash(root / 'source-qualification.json', common.QUALIFICATION)
    result = json.loads((root / 'receipt.json').read_text())
    frames = result['frames']
    if frames not in (16, 48) or [p['crf'] for p in result['points']] != list(RATES):
        raise ValueError('invalid point span')
    checks = {'source': result['source_yuv_sha256'] == common.SOURCE_YUV,
              'producer': result['producer_sha256'] == common.PRODUCER,
              'source_archive': result['source_archive_sha256'] == common.ARCHIVE,
              'qualification': result['qualification_sha256'] == common.QUALIFICATION,
              'native62': result['native62_sha256'] == common.NATIVE62,
              'native62_decode': result['native62_decode_sha256'] == common.NATIVE62_DECODE,
              'tools': result['tools'] == common.TOOLS, 'not_promoted': result['promotable'] is False}
    cmd = result['native62_control']['encode_command']
    checks['native62_command'] = cmd == svtav1.encode_command(cmd[0], Path(cmd[2]), Path(cmd[4]),
        width=common.W, height=common.H, fps=Fraction(30, 1), frames=48, crf=62, preset=4, full_range=False)
    work.mkdir(parents=True, exist_ok=False)
    for row in result['points']:
        crf = row['crf']
        base, patch = root / f'base-{crf}.ivf', root / f'patch-{crf}.ivf'
        base_data, patch_data = base.read_bytes(), patch.read_bytes()
        expected_base, expected_wire, expected_bytes = EXPECTED_PACKETS[crf]
        checks[f'{crf}:accepted_packet'] = (row['base_sha256'] == expected_base
            and row['wire_sha256'] == expected_wire and row['base_bytes'] == expected_bytes)
        stream = reconstruct_packet(row, base_data)
        track, _ = ball.decode_frames(stream.ball_payload, 48, common.W, common.H)
        positions = row['positions']
        checks[f'{crf}:positions'] = (len(positions) == frames and all(
            {k: r[k] for k in ('present', 'position', 'left', 'top')} == placement(p)
            for r, p in zip(positions, track)))
        checks[f'{crf}:streams'] = (common.digest(base_data) == row['base_sha256']
            and common.digest(patch_data) == row['patch_sha256'] and len(base_data) == row['base_bytes']
            and len(patch_data) == row['patch_bytes']
            and sum(svtav1.ivf_frames(patch_data)) == row['patch_payload_bytes']
            and len(svtav1.ivf_frames(base_data)) == 48
            and len(svtav1.ivf_frames(patch_data)) == frames
            and common.digest(stream.ball_payload) == row['ball_payload_sha256'])
        command = row['patch_encode']['encode_command']
        checks[f'{crf}:patch_command'] = command == svtav1.encode_command(command[0], Path(command[2]),
            Path(command[4]), width=64, height=64, fps=Fraction(30, 1), frames=frames,
            crf=20, preset=4, full_range=False)
        replay = decode_pair(base, patch, positions, decoder, work / str(crf),
                             row['base_yuv_sha256'], row['patch_yuv_sha256'])
        checks[f'{crf}:fresh_decoders_and_composite'] = replay == row['replay']
    return checks


def run_fleet(args: argparse.Namespace) -> dict[str, Any]:
    if os.sched_getaffinity(0) != {0} or os.environ.get('PS_CPU_ALLOWANCE') != '1':
        raise ValueError('requires CPU 0 with allowance 1')
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    started, cpu = time.monotonic(), common.cpu_seconds()
    prepare(args)
    work = scratch / 'pixel-private'
    archive = publish(work / 'publication', stage / 'published.tar')
    root = unpack(stage / 'published.tar', archive, work / 'published-copy')
    checks = validate_artifact(root, args.decoder, work / 'published-replay')
    if not all(checks.values()):
        raise ValueError('internal replay failed')
    cost = common.cost_record(time.monotonic() - started, common.cpu_seconds() - cpu, args.frames)
    record = {'schema': 1, 'frames': args.frames, 'archive': archive, 'cost': cost, 'internal_checks': checks,
              'scope': 'physical finite pixel oracle, no source-denial claim'}
    with (stage / 'pixel-oracle-stage.json').open('x') as out:
        json.dump(record, out, indent=2, allow_nan=False)
        out.write('\n')
    return record


def validate_fleet(stage: Path, decoder: str, temp: Path) -> dict[str, Any]:
    record = json.loads((stage / 'pixel-oracle-stage.json').read_text())
    with tempfile.TemporaryDirectory(prefix='pixel-oracle-', dir=temp) as folder:
        root = unpack(stage / 'published.tar', record['archive'], Path(folder) / 'archive')
        checks = validate_artifact(root, decoder, Path(folder) / 'replay')
    cost = record['cost']
    checks['body_cost'] = cost == common.cost_record(cost['wall_seconds'], cost['cpu_seconds'], record['frames']) and cost['passed']
    checks['internal'] = bool(record['internal_checks']) and all(record['internal_checks'].values())
    if (stage / 'execution.json').exists():
        execution = json.loads((stage / 'execution.json').read_text())
        checks['whole_command_cost'] = execution['exit_code'] == 0 and common.cost_record(execution['seconds'], 0, record['frames'])['passed']
    return {'passed': all(checks.values()), 'checks': checks, 'scope': 'finite physical pixel oracle replay and cost'}


def historical_scores(archive: Path, pilot: dict[str, Any]) -> list[dict[str, Any]]:
    common.require_hash(archive, common.SCORE_ARCHIVE)
    points = []
    sizes = {48: (90949, 91272), 55: (63879, 64177), 62: (34146, 34421)}
    with tarfile.open(archive) as tar:
        for crf in RATES:
            name = f'publish/point-svt-{crf}.json'
            selected = [e for e in tar.getmembers() if e.name == name and e.isfile()]
            if len(selected) != 1:
                raise ValueError('historical point absent or ambiguous')
            handle = tar.extractfile(selected[0])
            if handle is None:
                raise ValueError('historical point unreadable')
            item = json.load(handle)
            old = item['point']
            expected = next(p for p in pilot['points'] if p['codec'] == 'svt' and p['point'] == crf)
            if (item['checkpoint_sha256'] != common.CHECKPOINT or old['native_sha256'] != expected['native_sha256']
                    or old['wire_sha256'] != expected['wire_sha256']
                    or (old['native_bytes'], old['wire_bytes']) != sizes[crf]):
                raise ValueError('historical rate point binding mismatch')
            points.append({'crf': crf, 'native_bytes': sizes[crf][0], 'wire_bytes': sizes[crf][1],
                           'baseline': old['baseline']['metrics'], 'streak': old['streak']['metrics']})
    return points


def score(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from experiments.visor.h3b import dists_crops, lpips_reduce
    from src.codecs import quality

    if torch.cuda.device_count() != 1 or tuple(torch.cuda.get_device_capability(0)) != (8, 9):
        raise ValueError('one Ada CUDA device required')
    torch.cuda.reset_peak_memory_stats()
    work = Path(os.environ['PS_SCRATCH_DIR']) / 'pixel-score-private'
    work.mkdir(exist_ok=False)
    started = time.monotonic()
    common.require_hash(Path(args.decoder), common.TOOLS['decoder'])
    qualification, pilot, _ = common.producer_metadata(Path(args.producer))
    for key in ('lpips_backbone', 'dists_backbone'):
        common.require_hash(Path(getattr(args, key)), pilot['identities'][key]['sha256'])
    stage_record = json.loads(Path(args.cpu_record).read_text())
    if stage_record['archive']['sha256'] != args.cpu_archive_sha256 or not stage_record['cost']['passed']:
        raise ValueError('CPU archive or cost binding mismatch')
    root = unpack(Path(args.cpu_archive), stage_record['archive'], work / 'archive')
    physical = validate_artifact(root, args.decoder, work / 'physical')
    if not all(physical.values()):
        raise ValueError('pixel oracle physical replay failed')
    receipt = json.loads((root / 'receipt.json').read_text())
    if receipt['frames'] != 48 or args.frames not in (16, 48):
        raise ValueError('scoring requires complete 48-frame packets')
    source = common.source_frames(Path(args.archive), qualification)
    labels = [(int(r['label_kind']), float(r['label_x']), float(r['label_y'])) for r in qualification['rows']]
    lpips = quality.load_lpips(Path(args.lpips_backbone), 'cuda')
    dists = quality.load_dists(Path(args.dists_backbone), 'cuda')
    nets = (lpips, dists)
    if any(parameter.device.type != 'cuda' for net in nets for parameter in net.parameters()):
        raise ValueError('metric network device mismatch')
    mask = np.ones((common.H, common.W), dtype=bool)
    self_lp = lpips_reduce(lpips, [source[0]], source[0], [[mask]], 'cuda')[0][0]
    self_di = dists_crops(dists, [source[0]], source[0], mask, 'cuda')[0]
    self_standard = common.standard_lpips(source[:1], source[:1], lpips)[0]
    if any(not np.isfinite(value) or abs(value) > 1e-5 for value in (self_lp, self_di, self_standard)):
        raise ValueError('metric self-control failed')
    old_points = historical_scores(Path(args.score_archive), pilot)
    native_rgb = np.load(work / 'physical' / '62' / 'base.npy', mmap_mode='r')
    old = next(item for item in old_points if item['crf'] == 62)
    native_metrics = k5d_encoder.metrics(source, native_rgb, labels, nets)
    parity = common.compare_metric_control(native_metrics, old['baseline'])
    rows = []
    for row in receipt['points']:
        crf = row['crf']
        base = np.load(work / 'physical' / str(crf) / 'base.npy', mmap_mode='r')
        patch = np.load(work / 'physical' / str(crf) / 'patch.npy', mmap_mode='r')
        rendered = compose(base, patch, row['positions'])
        selected = rendered[:args.frames]
        values = k5d_encoder.metrics(source[:args.frames], selected, labels[:args.frames], nets)
        standard = common.standard_lpips(source[:args.frames], selected, lpips)
        rows.append({'crf': crf, 'base_bytes': row['base_bytes'], 'patch_bytes': row['patch_bytes'],
                     'coded_bytes': row['base_bytes'] + row['patch_bytes'],
                     'free_patch_ivf_framing_bytes': row['base_bytes'] + row['patch_payload_bytes'], 'base_sha256': row['base_sha256'],
                     'patch_sha256': row['patch_sha256'], 'composite_sha256': row['replay']['composite_rgb_sha256'],
                     'frames': values, 'full_lpips_standard': standard})
    import socket
    result = {'schema': 1, 'scope': 'finite coded 32px patch, transmitted positions; free placement and framing',
              'frames': args.frames, 'available_receiver_frames': 48, 'packet_duration_seconds': 48 / 30,
              'promotable': False, 'source_archive_sha256': common.ARCHIVE, 'producer_sha256': common.PRODUCER,
              'historical_score_archive_sha256': common.SCORE_ARCHIVE, 'cpu_archive_sha256': args.cpu_archive_sha256,
              'metric_control': parity, 'self_control': {'lpips': self_lp, 'dists': self_di, 'standard_lpips': self_standard},
              'metric_definition': {'full_lpips': 'historical full spatial-map mean',
                                    'full_lpips_standard': 'standard Alex feature-layer mean sum',
                                    'crop_lpips': 'historical separately cropped 64x64 GT window'},
              'weights': {'lpips': quality.lpips_record(Path(args.lpips_backbone)),
                          'dists': quality.dists_record(Path(args.dists_backbone))},
              'gpu': {'host': socket.gethostname(), 'name': torch.cuda.get_device_name(0),
                      'capability': list(torch.cuda.get_device_capability(0)),
                      'uuid': str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')),
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'torch': torch.__version__,
                      'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                      'device_count': torch.cuda.device_count(), 'network_parameters_cuda': True},
              'code_sha256': common.k5d_source_oracle.sha(Path(__file__)), 'physical_replay': physical,
              'historical_points': old_points, 'rows': rows, 'wall_seconds': time.monotonic() - started}
    return result


def analyse(record: dict[str, Any]) -> dict[str, Any]:
    from src.pointstream.evaluation import anchor_sensitivity

    if (record.get('frames') != 48 or record.get('available_receiver_frames') != 48
            or record.get('metric_control', {}).get('passed') is not True):
        raise ValueError('full coverage and historical metric control required')
    if (record.get('producer_sha256') != common.PRODUCER or record.get('source_archive_sha256') != common.ARCHIVE
            or record.get('historical_score_archive_sha256') != common.SCORE_ARCHIVE):
        raise ValueError('analysis source identity mismatch')
    if [r['crf'] for r in record['rows']] != list(RATES) or [r['crf'] for r in record['historical_points']] != list(RATES):
        raise ValueError('invalid rate matrix')
    curves: dict[str, list[dict[str, Any]]] = {'coded_pixels': [], 'free_patch_ivf_framing': [], 'native': [], 'streak': []}
    for row in record['rows']:
        if row['coded_bytes'] != row['base_bytes'] + row['patch_bytes'] or row['coded_bytes'] <= 0:
            raise ValueError('incorrect coded byte accounting')
        if not row['base_bytes'] < row['free_patch_ivf_framing_bytes'] <= row['coded_bytes']:
            raise ValueError('invalid relaxed patch framing price')
        means = common.metric_means(row['frames'])
        curves['coded_pixels'].append({'crf': row['crf'], 'bytes': row['coded_bytes'], 'means': means})
        curves['free_patch_ivf_framing'].append({'crf': row['crf'], 'bytes': row['free_patch_ivf_framing_bytes'], 'means': means})
    for row in record['historical_points']:
        for kind, n, metric in (('native', 'native_bytes', 'baseline'), ('streak', 'wire_bytes', 'streak')):
            curves[kind].append({'crf': row['crf'], 'bytes': row[n], 'means': common.metric_means(row[metric])})
    comparisons: dict[str, Any] = {}
    for anchor in ('native', 'streak'):
        comparisons[anchor] = {}
        for metric in ('full_lpips', 'full_dists', 'crop_lpips', 'crop_dists', 'rgb_psnr'):
            sign = 1 if metric == 'rgb_psnr' else -1
            comparisons[anchor][metric] = anchor_sensitivity(
                [(row['bytes'], sign * row['means'][metric]) for row in curves['coded_pixels']],
                [(row['bytes'], sign * row['means'][metric]) for row in curves[anchor]])
    relaxed: dict[str, Any] = {}
    for anchor in ('native', 'streak'):
        relaxed[anchor] = {}
        for metric in ('full_lpips', 'full_dists', 'crop_lpips', 'crop_dists', 'rgb_psnr'):
            sign = 1 if metric == 'rgb_psnr' else -1
            relaxed[anchor][metric] = anchor_sensitivity(
                [(row['bytes'], sign * row['means'][metric]) for row in curves['free_patch_ivf_framing']],
                [(row['bytes'], sign * row['means'][metric]) for row in curves[anchor]])
    native = comparisons['native']
    dists = native['crop_dists']['bd_rate']['hull_pchip']
    lpips = native['crop_lpips']['bd_rate']['hull_pchip']
    gate = None if dists is None or lpips is None else dists <= -.10 and lpips < 0
    return {'schema': 1, 'scope': 'finite fixed 32px, CRF20 pixel family; free placement/framing; no universal bound',
            'physical_curves': curves, 'comparisons': comparisons, 'free_patch_ivf_framing_sensitivity': relaxed,
            'native_gate_passed': gate,
            'decision_rule': 'at least 10% crop DISTS saving and negative crop LPIPS BD-rate vs native admits possible paid patch test; compare against streak',
            'billing': 'complete base IVF plus complete patch IVF; free placement and framing; no paid transport claim',
            'interpolation': 'PCHIP log-rate over common quality, anchor upper concave hull; no extrapolation',
            'standard_lpips': 'oracle scalar retained separately; not compared with unauthenticated historical scalar'}


def validate_score(record: dict[str, Any], frames: int) -> dict[str, bool]:
    import math

    checks = {'frames': record['frames'] == frames and frames in (16, 48),
              'full_receiver': record['available_receiver_frames'] == 48 and record['packet_duration_seconds'] == 1.6,
              'not_promoted': record['promotable'] is False,
              'source': record['source_archive_sha256'] == common.ARCHIVE and record['producer_sha256'] == common.PRODUCER,
              'old_scores': record['historical_score_archive_sha256'] == common.SCORE_ARCHIVE,
              'control': record['metric_control']['passed'] is True
                 and record['metric_control']['tolerance_absolute'] == 1e-6
                 and all(math.isfinite(x) and x <= 1e-6 for x in record['metric_control']['max_absolute_difference'].values()),
              'self': all(math.isfinite(record['self_control'][x]) and abs(record['self_control'][x]) <= 1e-5
                          for x in ('lpips', 'dists', 'standard_lpips')),
              'cuda': record['gpu']['capability'] == [8, 9] and record['gpu']['device_count'] == 1
                  and record['gpu']['network_parameters_cuda'] is True
                  and 0 < record['gpu']['peak_reserved_bytes'] <= 8000 * 1024**2,
              'physical': bool(record['physical_replay']) and all(record['physical_replay'].values()),
              'rates': [x['crf'] for x in record['rows']] == list(RATES),
              'old_rates': [x['crf'] for x in record['historical_points']] == list(RATES)}
    for row in record['rows']:
        actual = row['frames']
        checks[f"{row['crf']}:frame_count"] = (len(actual) == frames and [x['frame'] for x in actual] == list(range(frames))
            and len(row['full_lpips_standard']) == frames and row['coded_bytes'] == row['base_bytes'] + row['patch_bytes']
            and row['base_bytes'] > 0 and row['patch_bytes'] > 0
            and row['base_bytes'] < row['free_patch_ivf_framing_bytes'] <= row['coded_bytes'])
        checks[f"{row['crf']}:finite"] = all(type(item.get(key)) in (int, float) and math.isfinite(item[key])
            for item in actual for key in ('frame_psnr', 'full_lpips', 'full_dists', 'crop_lpips', 'crop_dists')) and all(
                type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in row['full_lpips_standard'])
    return checks


def score_spec(deadline: str, archive: str, archive_sha: str, record: str, record_sha: str) -> dict[str, Any]:
    spec = common.gpu_spec(deadline, archive, archive_sha, record, record_sha)
    spec['entrypoint'] = ['-m', 'experiments.tennis.ball_pixel_oracle']
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.ball_pixel_oracle', '--validate-score']
    spec['gpu_memory_mib'] = 8000
    spec['smoke']['representative_basis'] = 'Full 48-frame native metric control and packet replay; first16 composite scores for each coded pixel point.'
    spec['gpu_sharing'] = {'mode': 'shared', 'margin_mib': 8192,
                           'basis': 'Perceptual values and exact pixels do not depend on concurrent GPU timing.'}
    spec['contention'] = {'policy': 'continue', 'basis': 'Shared GPU scoring; time is feasibility only.', 'resume_attempts': 0}
    return spec


def score_fleet(args: argparse.Namespace) -> dict[str, Any]:
    if os.sched_getaffinity(0) != {1} or os.environ.get('PS_CPU_ALLOWANCE') != '1':
        raise ValueError('requires CPU 1 with allowance 1')
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    started, cpu = time.monotonic(), common.cpu_seconds()
    result = score(args)
    checks = validate_score(result, args.frames)
    if not all(checks.values()):
        raise ValueError('score integrity check failed')
    path = scratch / 'pixel-scores.json'
    with path.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    archive = stage / 'published.tar'
    with tarfile.open(archive, 'x') as tar:
        tar.add(path, arcname='publish/scores.json', recursive=False)
    cost = common.gpu_cost(time.monotonic() - started, common.cpu_seconds() - cpu, args.frames)
    stage_record = {'schema': 1, 'frames': args.frames, 'archive': {'sha256': common.k5d_source_oracle.sha(archive),
        'bytes': archive.stat().st_size, 'scores_sha256': common.k5d_source_oracle.sha(path)},
        'cost': cost, 'internal_checks': checks}
    with (stage / 'pixel-score-stage.json').open('x') as out:
        json.dump(stage_record, out, indent=2, allow_nan=False)
        out.write('\n')
    return stage_record


def validate_score_fleet(stage: Path) -> dict[str, Any]:
    record = json.loads((stage / 'pixel-score-stage.json').read_text())
    archive = stage / 'published.tar'
    common.require_hash(archive, record['archive']['sha256'])
    if archive.stat().st_size != record['archive']['bytes']:
        raise ValueError('score archive size mismatch')
    payload = common.tar_member(archive, 'publish/scores.json', record['archive']['scores_sha256'])
    with tarfile.open(archive) as tar:
        if len(tar.getmembers()) != 1:
            raise ValueError('unexpected score archive members')
    checks = validate_score(json.loads(payload), record['frames'])
    cost = record['cost']
    checks['body_cost'] = cost == common.gpu_cost(cost['wall_seconds'], cost['cpu_seconds'], record['frames']) and cost['passed']
    if (stage / 'execution.json').exists():
        execution = json.loads((stage / 'execution.json').read_text())
        checks['whole_command_cost'] = execution['exit_code'] == 0 and common.gpu_cost(execution['seconds'], 0, record['frames'])['passed']
    return {'passed': all(checks.values()), 'checks': checks, 'scope': 'pixel score identity and cost'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fleet', action='store_true')
    parser.add_argument('--mode', choices=('run', 'score'), default='run')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--validate-score', action='store_true')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--score-spec', type=Path)
    parser.add_argument('--deadline')
    parser.add_argument('--analyse', type=Path)
    parser.add_argument('--analysis-output', type=Path)
    for name in ('producer', 'archive', 'score-archive', 'cpu-archive', 'cpu-archive-sha256',
                 'cpu-record', 'cpu-record-sha256', 'lpips-backbone', 'dists-backbone'):
        parser.add_argument('--' + name)
    parser.add_argument('--encoder', default='SvtAv1EncApp')
    parser.add_argument('--decoder', default='dav1d')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--frames', type=int, choices=(16, 48))
    args = parser.parse_args()
    if args.spec or args.score_spec:
        if not args.deadline:
            parser.error('spec needs a deadline')
        if args.score_spec:
            if any(getattr(args, x) is None for x in ('cpu_archive', 'cpu_archive_sha256', 'cpu_record', 'cpu_record_sha256')):
                parser.error('score spec requires CPU archive and record paths and hashes')
            config = score_spec(args.deadline, args.cpu_archive, args.cpu_archive_sha256, args.cpu_record, args.cpu_record_sha256)
            target = args.score_spec
        else:
            config, target = make_spec(args.deadline), args.spec
        with target.open('x') as out:
            json.dump(config, out, indent=2)
            out.write('\n')
        return
    if args.analyse:
        if args.analysis_output is None:
            parser.error('--analyse requires --analysis-output')
        data = args.analyse.read_bytes()
        result = analyse(json.loads(data))
        result['score_sha256'] = common.digest(data)
        with args.analysis_output.open('x') as out:
            json.dump(result, out, indent=2, allow_nan=False)
            out.write('\n')
        return
    if args.validate or args.validate_score:
        if args.validate_score:
            result = validate_score_fleet(Path(os.environ['PS_STAGE_DIR']))
        else:
            decoder = shutil.which(args.decoder)
            if decoder is None:
                raise FileNotFoundError('dav1d missing')
            result = validate_fleet(Path(os.environ['PS_STAGE_DIR']), decoder, Path(os.environ['TMPDIR']))
        with Path(os.environ['PS_VALIDATION_PATH']).open('x') as out:
            json.dump(result, out, allow_nan=False)
            out.write('\n')
        print(json.dumps(result))
        if not result['passed']:
            raise SystemExit(1)
        return
    if not args.fleet or not args.producer or not args.archive or args.frames is None:
        parser.error('fleet operation requires producer, archive, and frames')
    keys = ('decoder',) if args.mode == 'score' else ('ffmpeg', 'encoder', 'decoder')
    for key in keys:
        path = shutil.which(getattr(args, key))
        if path is None:
            raise FileNotFoundError(key + ' missing')
        setattr(args, key, path)
    if args.mode == 'score':
        required = ('score_archive', 'cpu_archive', 'cpu_archive_sha256', 'cpu_record', 'lpips_backbone', 'dists_backbone')
        if any(getattr(args, key) is None for key in required):
            parser.error('score mode requires staged producer, source, historical scores, CPU archive/record, and weights')
        stage = score_fleet(args)
    else:
        stage = run_fleet(args)
    print(json.dumps({'frames': args.frames, 'cost': stage['cost']}))
    if not stage['cost']['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

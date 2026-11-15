"""Frozen privileged GT ball ROI native-preparation stage; no jobs are submitted.

This module intentionally does not claim a metric comparison.  Its output is a
thin, independently decodable input for a separately admitted metric stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import socket
import tempfile
import tarfile
import time
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k5d_encoder, k5d_source_oracle, w1_roi
from src.codecs import svtav1

PRODUCER = '9074362c3f62ec18fba75acb3393c6a0803d907ad235a57bce98eb050c30b128'
CHECKPOINT = '1536eb02a9d2bce08c83aa064d0e184cf348a997aa999d8f980a918998aa9d03'
QUALIFICATION = '1bd32c3fb0c57255e334173d9be8adda99d424fed2e64ba4c29c541ae5ac14dc'
ARCHIVE = 'bb45372d011ef4092d78173169c8ad6280d7dbdb9998efaf2b1c72a63ca1dd18'
SOURCE_YUV = 'a40b172b7a3e72c0fcf75dff15584564e3d220ffa6f1653439d045387f2d9963'
NATIVE62_DECODE = '652302beac4aeee7b7c749a45cac1a6f421c5e780e6d21721a75a636430e753d'
NATIVE62 = 'cd79e3979e00124a8ae201c1b22f69651ae353a90b96da1acd1fbaaff757224b'
TOOLS = {'ffmpeg': '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2',
         'encoder': '4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4',
         'decoder': 'e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6'}
W, H, N = 1280, 720, 48
FRAME_BYTES = W * H * 3 // 2


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if k5d_source_oracle.sha(path) != expected:
        raise ValueError(f'SHA256 mismatch: {path}')


def producer_metadata(path: Path) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    require_hash(path, PRODUCER)
    with tarfile.open(path, 'r:*') as tar:
        def read(name: str) -> bytes:
            matches = [m for m in tar.getmembers() if m.name == name and m.isfile()]
            if len(matches) != 1:
                raise ValueError(f'missing or duplicate producer member: {name}')
            stream = tar.extractfile(matches[0])
            if stream is None:
                raise ValueError('unreadable producer member')
            return stream.read()
        checkpoint = read('publish/checkpoint.json')
        if digest(checkpoint) != CHECKPOINT:
            raise ValueError('checkpoint identity mismatch')
        manifest = json.loads(checkpoint)
        q = read('publish/source-qualification.json')
        pilot = read('publish/pilot.json')
        if (digest(q) != QUALIFICATION or manifest['files']['source-qualification.json'] != digest(q)
                or manifest['files']['pilot.json'] != digest(pilot)):
            raise ValueError('producer metadata identity mismatch')
    return json.loads(q), json.loads(pilot), q


def source_frames(archive: Path, qualification: dict[str, Any]) -> np.ndarray:
    import cv2

    require_hash(archive, ARCHIVE)
    rows = qualification['rows']
    if (qualification['clip'] != 'tracknet/game10/Clip3' or len(rows) != N
            or [r['frame'] for r in rows] != list(range(N))):
        raise ValueError('qualification clip/frame mismatch')
    images = []
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        for row in rows:
            name = row['member']
            if names.count(name) != 1 or not name.startswith('Dataset/game10/Clip3/'):
                raise ValueError('missing or duplicate qualified JPEG')
            data = z.read(name)
            if digest(data) != row['source_sha256']:
                raise ValueError(f'JPEG identity mismatch: {name}')
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape != (H, W, 3):
                raise ValueError('JPEG geometry mismatch')
            images.append(bgr[:, :, ::-1])
    return np.stack(images)


def regions(rows: list[dict[str, Any]], frames: int) -> np.ndarray:
    if frames not in (16, 48) or len(rows) != N:
        raise ValueError('invalid frame count')
    masks = np.zeros((frames, H, W), dtype=bool)
    for i, row in enumerate(rows[:frames]):
        x, y = float(row['label_x']), float(row['label_y'])
        if row['label_kind'] != 1 or not (0 <= x < W and 0 <= y < H):
            raise ValueError('expected visible valid GT centre')
        left = min(max(int(round(x)) - 32, 0), W - 64)
        top = min(max(int(round(y)) - 32, 0), H - 64)
        masks[i, top:top + 64, left:left + 64] = True
    return masks


def expected_native_decode(pilot: dict[str, Any]) -> str:
    points = [p for p in pilot['points'] if p['codec'] == 'svt' and p['point'] == 62]
    if len(points) != 1 or points[0]['native_sha256'] != NATIVE62:
        raise ValueError('producer native point mismatch')
    result = points[0]['encoder_decoded_sha256']
    if result != NATIVE62_DECODE:
        raise ValueError('missing native decode hash')
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.frames not in (16, 48):
        raise ValueError('frames must be 16 or 48')
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve()
    work = Path(args.work).resolve()
    if work == scratch or not work.is_relative_to(scratch):
        raise ValueError('work must be a new private directory under PS_SCRATCH_DIR')
    work.mkdir(parents=True, exist_ok=False)
    for key, expected in TOOLS.items():
        require_hash(Path(getattr(args, key)), expected)
    qualification, pilot, qualification_bytes = producer_metadata(Path(args.producer))
    expected_decode = expected_native_decode(pilot)
    source = source_frames(Path(args.archive), qualification)
    yuv = work / 'source.yuv'
    conversion = k5d_encoder.convert_rgb(args.ffmpeg, source, yuv, Fraction(30, 1))
    del source
    if yuv.stat().st_size != N * FRAME_BYTES:
        raise ValueError('source YUV size mismatch')
    require_hash(yuv, SOURCE_YUV)
    control = work / 'control'
    control.mkdir()
    common = dict(width=W, height=H, fps=Fraction(30, 1), preset=4, threads=1,
                  full_range=False, encoder=args.encoder, decoder=args.decoder, timeout=600)
    native = svtav1.code(yuv, control, frames=N, crf=62, **common)
    require_hash(Path(native['stream']), NATIVE62)
    require_hash(Path(native['decoded']), expected_decode)
    if native['file_bytes'] != 34146 or native['decoded_frames'] != N:
        raise ValueError('native control size mismatch')
    pub = work / 'publication'
    pub.mkdir()
    (pub / 'source-qualification.json').write_bytes(qualification_bytes)
    roi = pub / 'roi.txt'
    roi.write_text(w1_roi.roi_text(regions(qualification['rows'], args.frames), W, H, -40, outside=True))
    points = []
    for crf in (35, 48, 62):
        run_dir = work / f'crf-{crf}'
        run_dir.mkdir()
        result = svtav1.code(yuv, run_dir, frames=args.frames, crf=crf,
                             extra=['--roi-map-file', str(roi)], **common)
        data = Path(result['stream']).read_bytes()
        if len(svtav1.ivf_frames(data)) != args.frames or result['decoded_frames'] != args.frames:
            raise ValueError('encoded/decode frame count mismatch')
        result['decoded_sha256'] = k5d_source_oracle.sha(Path(result['decoded']))
        result['stream'] = f'crf-{crf}.ivf'
        result.pop('decoded')
        (pub / result['stream']).write_bytes(data)
        result['crf'] = crf
        points.append(result)
    report = {'schema': 1, 'scope': 'privileged GT 64x64, outside +40; development clip; no task metric',
              'frames': args.frames, 'promotable': False, 'complete_span': args.frames == 48, 'producer_sha256': PRODUCER,
              'source_archive_sha256': ARCHIVE, 'source_qualification_sha256': QUALIFICATION,
              'source_yuv_sha256': SOURCE_YUV, 'qualification_rows': qualification['rows'], 'conversion_command': conversion,
              'tools_sha256': TOOLS, 'native62_control': native,
              'native62_sha256': k5d_source_oracle.sha(Path(native['stream'])),
              'native62_decode_sha256': k5d_source_oracle.sha(Path(native['decoded'])), 'roi_sha256': k5d_source_oracle.sha(roi),
              'points': points, 'wall_seconds': time.monotonic() - started}
    with (pub / 'receipt.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    return report



def cpu_seconds() -> float:
    usage = [resource.getrusage(kind) for kind in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    return sum(item.ru_utime + item.ru_stime for item in usage)


def cost_record(wall: float, cpu: float, frames: int) -> dict[str, Any]:
    if wall <= 0 or cpu < 0 or frames not in (16, 48):
        raise ValueError('invalid cost measurement')
    forecast = 1.5 * wall * 48 / frames + 30
    return {'frames': frames, 'target_frames': 48, 'wall_seconds': wall, 'cpu_seconds': cpu,
            'forecast_seconds': forecast, 'full_cap_seconds': 900, 'fixed_reserve_seconds': 30,
            'passed': forecast <= 900 if frames == 16 else wall <= 900,
            'basis': 'entrypoint body wall after imports, including archive and internal replay; smoke 1.5 * wall * 48 / 16 + 30; full uses body wall'}


MEMBERS = ('receipt.json', 'source-qualification.json', 'roi.txt', 'crf-35.ivf', 'crf-48.ivf', 'crf-62.ivf')


def publish(root: Path, destination: Path) -> dict[str, Any]:
    if {p.name for p in root.iterdir()} != set(MEMBERS):
        raise ValueError('unexpected publication inventory')
    hashes = {}
    with tarfile.open(destination, 'x') as tar:
        for name in MEMBERS:
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('publication contains nonregular member')
            hashes[name] = k5d_source_oracle.sha(path)
            tar.add(path, arcname='publish/' + name, recursive=False)
    return {'sha256': k5d_source_oracle.sha(destination), 'bytes': destination.stat().st_size,
            'members': hashes}


def extract_thin(archive: Path, expected: dict[str, Any], destination: Path) -> Path:
    require_hash(archive, expected['sha256'])
    if archive.stat().st_size != expected['bytes'] or set(expected['members']) != set(MEMBERS):
        raise ValueError('publication inventory mismatch')
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        if (len(members) != len(MEMBERS) or {x.name for x in members} != {'publish/' + n for n in MEMBERS}
                or any(not x.isfile() for x in members)):
            raise ValueError('unexpected archive contents')
        tar.extractall(destination, filter='data')
    root = destination / 'publish'
    for name, sha in expected['members'].items():
        require_hash(root / name, sha)
    return root


def validate_publication(root: Path, decoder: str, work: Path) -> dict[str, bool]:
    require_hash(Path(decoder), TOOLS['decoder'])
    receipt = json.loads((root / 'receipt.json').read_text())
    count = receipt['frames']
    if count not in (16, 48) or receipt['promotable'] is not False:
        raise ValueError('invalid span or promotion marker')
    checks = {'producer': receipt['producer_sha256'] == PRODUCER,
              'source_archive': receipt['source_archive_sha256'] == ARCHIVE,
              'qualification': receipt['source_qualification_sha256'] == QUALIFICATION,
              'source_yuv': receipt['source_yuv_sha256'] == SOURCE_YUV,
              'native_stream': receipt['native62_sha256'] == NATIVE62,
              'native_decode': receipt['native62_decode_sha256'] == NATIVE62_DECODE,
              'complete_native_control': receipt['native62_control']['temporal_units'] == 48,
              'tool_hashes': receipt['tools_sha256'] == TOOLS}
    qualification_path = root / 'source-qualification.json'
    require_hash(qualification_path, QUALIFICATION)
    qualification = json.loads(qualification_path.read_text())
    checks['qualified_rows'] = qualification['rows'] == receipt['qualification_rows']
    expected_roi = w1_roi.roi_text(regions(qualification['rows'], count), W, H, -40, outside=True)
    checks['gt_map'] = ((root / 'roi.txt').read_text() == expected_roi
                        and k5d_source_oracle.sha(root / 'roi.txt') == receipt['roi_sha256'])
    native_command = receipt['native62_control']['encode_command']
    checks['native_command'] = native_command == svtav1.encode_command(
        native_command[0], Path(native_command[2]), Path(native_command[4]), width=W, height=H,
        fps=Fraction(30, 1), frames=48, crf=62, preset=4, full_range=False)
    points = receipt['points']
    if [p['crf'] for p in points] != [35, 48, 62]:
        raise ValueError('ROI rate matrix differs')
    work.mkdir(parents=True, exist_ok=False)
    for point in points:
        crf = point['crf']
        stream = root / f'crf-{crf}.ivf'
        data = stream.read_bytes()
        decoded = work / f'{crf}.yuv'
        cmd = point['encode_command']
        checks[f'{crf}:command'] = cmd == svtav1.encode_command(
            cmd[0], Path(cmd[2]), Path(cmd[4]), width=W, height=H, fps=Fraction(30, 1),
            frames=count, crf=crf, preset=4, full_range=False, extra=['--roi-map-file', cmd[-1]])
        checks[f'{crf}:stream'] = (point['stream'] == stream.name and point['stream_sha256'] == digest(data)
                                   and point['file_bytes'] == len(data)
                                   and len(svtav1.ivf_frames(data)) == count)
        command = svtav1.decode_command(decoder, stream, decoded, threads=1)
        svtav1.run(command, timeout=120)
        checks[f'{crf}:fresh_decode'] = (decoded.stat().st_size == count * FRAME_BYTES
                                         and k5d_source_oracle.sha(decoded) == point['decoded_sha256'])
    if count == 48:
        roi62 = next(point for point in points if point['crf'] == 62)
        checks['roi62_effective_pixels'] = roi62['decoded_sha256'] != NATIVE62_DECODE
    return checks


def fleet_run(args: argparse.Namespace) -> dict[str, Any]:
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    if os.sched_getaffinity(0) != {0} or os.environ.get('PS_CPU_ALLOWANCE') != '1':
        raise ValueError('one CPU affinity, index 0, and allowance 1 required')
    started, cpu_start = time.monotonic(), cpu_seconds()
    args.work = str(scratch / 'roi-private')
    result = run(args)
    archive = publish(Path(args.work) / 'publication', stage / 'published.tar')
    replay_root = extract_thin(stage / 'published.tar', archive, Path(args.work) / 'internal-archive')
    internal_checks = validate_publication(replay_root, args.decoder, Path(args.work) / 'internal-replay')
    if not all(internal_checks.values()):
        raise ValueError('internal published archive replay failed')
    cost = cost_record(time.monotonic() - started, cpu_seconds() - cpu_start, args.frames)
    record = {'schema': 1, 'archive': archive, 'cost': cost, 'frames': args.frames,
              'actual_native62_sha256': result['native62_sha256'],
              'actual_native62_decode_sha256': result['native62_decode_sha256'],
              'internal_fresh_replay': internal_checks,
              'scope': 'physical preparation, not a scored or promoted comparison'}
    with (stage / 'ball-roi-stage.json').open('x') as out:
        json.dump(record, out, indent=2, allow_nan=False)
        out.write('\n')
    return record


def fleet_validate(stage: Path, temporary: Path, decoder: str) -> dict[str, Any]:
    record = json.loads((stage / 'ball-roi-stage.json').read_text())
    cost = record['cost']
    expected_cost = cost_record(cost['wall_seconds'], cost['cpu_seconds'], record['frames'])
    with tempfile.TemporaryDirectory(prefix='ball-roi-validate-', dir=temporary) as directory:
        base = Path(directory)
        root = extract_thin(stage / 'published.tar', record['archive'], base / 'archive')
        checks = validate_publication(root, decoder, base / 'replay')
    execution_path = stage / 'execution.json'
    if execution_path.exists():
        execution = json.loads(execution_path.read_text())
        checks['whole_command_cost'] = (execution['exit_code'] == 0
                                         and cost_record(execution['seconds'], 0, record['frames'])['passed'])
    checks.update({'internal_fresh_replay': bool(record['internal_fresh_replay']) and all(record['internal_fresh_replay'].values()),
                   'cost': cost == expected_cost and cost['passed'],
                   'frame_binding': json.loads((stage / 'ball-roi-stage.json').read_text())['frames'] == cost['frames'],
                   'native_actual': record['actual_native62_sha256'] == NATIVE62,
                   'decode_actual': record['actual_native62_decode_sha256'] == NATIVE62_DECODE})
    return {'passed': all(checks.values()), 'checks': checks,
            'scope': 'fresh physical replay and cost only; no scored comparison'}


def cpu_spec(deadline: str) -> dict[str, Any]:
    data = '/home/itec/emanuele/Datasets/pointstream-data'
    inputs = [
        {'name': 'producer', 'path': data + '/jobs/fleet/inbox/20261113T210157Z-41ee26e3/full/published.tar',
         'sha256': PRODUCER, 'extract': False},
        {'name': 'archive', 'path': data + '/tennis/k5b-2026-10-20/inputs/tracknet-Dataset.zip',
         'sha256': ARCHIVE, 'extract': False},
    ]
    return {'schema': 1, 'entrypoint': ['-m', 'experiments.tennis.ball_roi_control'],
            'arguments': ['--fleet', '--producer', '{staged:producer}', '--archive', '{staged:archive}',
                          '--frames', '{frames}'],
            'staged_inputs': inputs, 'inputs': [{'path': p['path'], 'sha256': p['sha256']} for p in inputs],
            'environment': {'path': data + '/environments/pointstream-20261009T214002Z.tar.gz',
                            'sha256': '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'},
            'verify_staged_inputs': True, 'device': 'cpu', 'hosts': ['gpu6'],
            'gpu_models': [], 'gpu_memory_mib': 0, 'cpu_threads': 1, 'single_cpu_campaign': True, 'cpu_affinity_index': 0, 'local_storage_gib': 20,
            'required_commands': ['ffmpeg'],
            'scale': {'frames': {'smoke': 16, 'full': 48}},
            'smoke': {'seconds': 450, 'representative_basis': 'Full 48-frame authenticated source and native62 control; three ROI encodes of first16 and fresh decodes.'},
            'full': {'seconds': 900}, 'validator': ['{python}', '-m', 'experiments.tennis.ball_roi_control', '--validate'],
            'validator_seconds': 60, 'budget_seconds': 1530, 'deadline': deadline, 'stall_seconds': 450,
            'contention': {'policy': 'stop', 'resume_attempts': 0}}



SCORE_ARCHIVE = 'fd9c161132525e3d9d143412122f0281a9a5714278643208fd04368dd081d876'
SCORE_MEMBER = '3d4f29e6446a74d308e19858bead7c65a0ac5a1b784fede76fdd24c91eddda7b'


def tar_member(path: Path, name: str, expected: str) -> bytes:
    with tarfile.open(path, 'r:*') as tar:
        members = [item for item in tar.getmembers() if item.name == name and item.isfile()]
        if len(members) != 1:
            raise ValueError(f'missing or duplicate member {name}')
        stream = tar.extractfile(members[0])
        if stream is None:
            raise ValueError('unreadable member')
        data = stream.read()
    if digest(data) != expected:
        raise ValueError(f'member SHA mismatch {name}')
    return data


def compare_metric_control(actual: list[dict[str, Any]], saved: list[dict[str, Any]]) -> dict[str, Any]:
    if len(actual) != 48 or len(saved) != 48:
        raise ValueError('historical control must cover 48 frames')
    exact = ('frame', 'sse', 'pixels')
    floating = ('frame_psnr', 'full_lpips', 'full_dists', 'crop_lpips', 'crop_dists')
    maxima: dict[str, float] = {key: 0.0 for key in floating}
    for measured, reference in zip(actual, saved):
        if any(measured.get(key) != reference.get(key) for key in exact):
            raise ValueError('historical control integer mismatch')
        for key in floating:
            a, b = measured.get(key), reference.get(key)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise ValueError(f'missing historical metric {key}')
            difference = abs(a - b)
            if not math.isfinite(difference) or difference > 1e-6:
                raise ValueError(f'historical metric parity failure: {key} = {difference}')
            maxima[key] = max(maxima[key], difference)
    return {'passed': True, 'tolerance_absolute': 1e-6, 'max_absolute_difference': maxima,
            'scope': '48-frame native62 historical full-map/crop metrics and exact RGB SSE'}


def standard_lpips(source: np.ndarray, images: np.ndarray, net: Any) -> list[float]:
    import torch

    original = net.spatial
    values = []
    try:
        net.spatial = False
        for a, b in zip(source, images):
            x = torch.from_numpy(np.ascontiguousarray(a)).to('cuda').permute(2, 0, 1)[None].float() / 127.5 - 1.0
            y = torch.from_numpy(np.ascontiguousarray(b)).to('cuda').permute(2, 0, 1)[None].float() / 127.5 - 1.0
            with torch.inference_mode():
                value = float(net(y, x).item())
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid standard LPIPS')
            values.append(value)
    finally:
        net.spatial = original
    return values


def score(args: argparse.Namespace) -> dict[str, Any]:
    """Score authenticated ROI packets only after original native metric parity.

    This helper does not dispatch a GPU job or make a BD-rate verdict.
    """
    import torch

    from experiments.tennis import windows
    from src.codecs import quality

    started = time.monotonic()
    if torch.cuda.device_count() != 1 or tuple(torch.cuda.get_device_capability(0)) != (8, 9):
        raise ValueError('historical metric control requires one Ada CUDA device')
    torch.cuda.reset_peak_memory_stats()
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve()
    work = Path(args.work).resolve()
    if work == scratch or not work.is_relative_to(scratch):
        raise ValueError('scoring work must be private scratch')
    work.mkdir(parents=True, exist_ok=False)
    require_hash(Path(args.decoder), TOOLS['decoder'])
    require_hash(Path(args.score_archive), SCORE_ARCHIVE)
    qualification, pilot, _ = producer_metadata(Path(args.producer))
    for arg, key in (('lpips_backbone', 'lpips_backbone'), ('dists_backbone', 'dists_backbone')):
        require_hash(Path(getattr(args, arg)), pilot['identities'][key]['sha256'])
    cpu_record = json.loads(Path(args.cpu_record).read_text())
    if cpu_record['archive']['sha256'] != args.cpu_archive_sha256:
        raise ValueError('CPU archive binding differs')
    root = extract_thin(Path(args.cpu_archive), cpu_record['archive'], work / 'cpu-archive')
    physical = validate_publication(root, args.decoder, work / 'roi-yuv')
    if not all(physical.values()):
        raise ValueError('ROI fresh decoder replay failed')
    receipt = json.loads((root / 'receipt.json').read_text())
    if receipt['frames'] != 48 or args.frames not in (16, 48) or not cpu_record['cost']['passed']:
        raise ValueError('CPU count or cost gate mismatch')
    source = source_frames(Path(args.archive), qualification)
    labels = [(int(r['label_kind']), float(r['label_x']), float(r['label_y'])) for r in qualification['rows']]
    native = work / 'native62.ivf'
    native.write_bytes(tar_member(Path(args.producer), 'publish/svt-62/native.bin', NATIVE62))
    native_yuv = work / 'native62.yuv'
    svtav1.run(svtav1.decode_command(args.decoder, native, native_yuv, threads=1), timeout=120)
    require_hash(native_yuv, NATIVE62_DECODE)
    native_rgb = windows.rgb_frames(native_yuv, W, H, N, work / 'native62-rgb.npy')
    networks = (quality.load_lpips(Path(args.lpips_backbone), 'cuda'),
                quality.load_dists(Path(args.dists_backbone), 'cuda'))
    if any(parameter.device.type != 'cuda' for network in networks for parameter in network.parameters()):
        raise ValueError('perceptual network device mismatch')
    gold = json.loads(tar_member(Path(args.score_archive), 'publish/point-svt-62.json', SCORE_MEMBER))
    if gold['checkpoint_sha256'] != CHECKPOINT or gold['point']['native_sha256'] != NATIVE62:
        raise ValueError('historical scored point identity mismatch')
    from experiments.visor.h3b import dists_crops, lpips_reduce
    mask = np.ones((H, W), dtype=bool)
    self_lpips = lpips_reduce(networks[0], [source[0]], source[0], [[mask]], 'cuda')[0][0]
    self_dists = dists_crops(networks[1], [source[0]], source[0], mask, 'cuda')[0]
    self_standard = standard_lpips(source[:1], source[:1], networks[0])[0]
    if any(not math.isfinite(x) or abs(x) > 1e-5 for x in (self_lpips, self_dists, self_standard)):
        raise ValueError('perceptual model self-control failed')
    native_metrics = k5d_encoder.metrics(source, native_rgb, labels, networks)
    parity = compare_metric_control(native_metrics, gold['point']['baseline']['metrics'])
    historical = []
    with tarfile.open(args.score_archive, 'r:*') as bundle:
        for crf in (48, 55, 62):
            name = f'publish/point-svt-{crf}.json'
            members = [item for item in bundle.getmembers() if item.name == name and item.isfile()]
            if len(members) != 1:
                raise ValueError('missing/duplicate historical scored point')
            handle = bundle.extractfile(members[0])
            if handle is None:
                raise ValueError('unreadable historical point')
            item = json.load(handle)
            point = item['point']
            producer_point = next(p for p in pilot['points'] if p['codec'] == 'svt' and p['point'] == crf)
            if (item['checkpoint_sha256'] != CHECKPOINT or point['native_sha256'] != producer_point['native_sha256']
                    or point['wire_sha256'] != producer_point['wire_sha256']):
                raise ValueError('historical point is not bound to producer')
            physical_ivf = {48: 90949, 55: 63879, 62: 34146}[crf]
            physical_wire = {48: 91272, 55: 64177, 62: 34421}[crf]
            if point['native_bytes'] != physical_ivf or point['wire_bytes'] != physical_wire:
                raise ValueError('historical physical packet size mismatch')
            historical.append({'crf': crf, 'native_bytes': physical_ivf, 'wire_bytes': physical_wire,
                               'baseline': point['baseline']['metrics'], 'streak': point['streak']['metrics']})
    rows = []
    for point in receipt['points']:
        crf = point['crf']
        yuv = work / 'roi-yuv' / f'{crf}.yuv'
        rgb = windows.rgb_frames(yuv, W, H, 48, work / f'{crf}-rgb.npy')[:args.frames]
        values = k5d_encoder.metrics(source[:args.frames], rgb, labels[:args.frames], networks)
        standard = standard_lpips(source[:args.frames], rgb, networks[0])
        rows.append({'crf': crf, 'file_bytes': point['file_bytes'], 'stream_sha256': point['stream_sha256'],
                     'decoded_yuv_sha256': point['decoded_sha256'], 'frames': values,
                     'full_lpips_standard': standard})
    record = {'schema': 1, 'scope': 'privileged GT ROI; functional packet decode; no task or source-denial claims',
              'frames': args.frames, 'available_receiver_frames': 48, 'packet_duration_seconds': 48 / 30,
              'promotable': False, 'metric_control': parity,
              'self_control': {'lpips': self_lpips, 'dists': self_dists, 'lpips_standard': self_standard, 'absolute_tolerance': 1e-5},
              'metric_definition': {'full_lpips': 'historical full-resolution spatial-map mean',
                                    'full_lpips_standard': 'standard LPIPS Alex scalar, feature-layer means',
                                    'crop_lpips': 'separate 64x64 GT crop, historical spatial-map mean'},
              'source_qualification_sha256': QUALIFICATION, 'source_archive_sha256': ARCHIVE,
              'producer_sha256': PRODUCER, 'historical_score_archive_sha256': SCORE_ARCHIVE,
              'historical_score_member_sha256': SCORE_MEMBER, 'cpu_archive_sha256': args.cpu_archive_sha256,
              'fresh_decoder_sha256': TOOLS['decoder'], 'fresh_native_yuv_sha256': NATIVE62_DECODE,
              'weights': {'lpips': quality.lpips_record(Path(args.lpips_backbone)),
                          'dists': quality.dists_record(Path(args.dists_backbone))},
              'gpu': {'host': socket.gethostname(), 'name': torch.cuda.get_device_name(0),
                      'uuid': str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')),  'capability': list(torch.cuda.get_device_capability(0)),
                      'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                      'device_count': torch.cuda.device_count(), 'network_parameters_cuda': True,
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved()},
              'source_code_sha256': k5d_source_oracle.sha(Path(__file__)),
              'physical_replay': physical, 'rows': rows, 'historical_points': historical, 'wall_seconds': time.monotonic() - started}
    with Path(args.score_output).open('x') as out:
        json.dump(record, out, indent=2, allow_nan=False)
        out.write('\n')
    return record


def metric_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    if len(rows) != 48 or [row['frame'] for row in rows] != list(range(48)):
        raise ValueError('analysis requires all 48 ordered metric frames')
    names = ('full_lpips', 'full_dists', 'crop_lpips', 'crop_dists')
    means: dict[str, float] = {}
    for name in names:
        values = [row[name] for row in rows]
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise ValueError(f'invalid metric values: {name}')
        means[name] = math.fsum(values) / 48
    sse = sum(row['sse'] for row in rows)
    pixels = sum(row['pixels'] for row in rows)
    if sse <= 0 or pixels <= 0:
        raise ValueError('invalid pooled RGB error')
    means['rgb_psnr'] = 10 * math.log10(255 * 255 * pixels / sse)
    return means


def analyse_score(record: dict[str, Any]) -> dict[str, Any]:
    from src.pointstream.evaluation import anchor_sensitivity

    if (record.get('frames') != 48 or record.get('available_receiver_frames') != 48
            or record.get('metric_control', {}).get('passed') is not True):
        raise ValueError('prefix or metric-control failure cannot support BD-rate')
    if (record.get('producer_sha256') != PRODUCER or record.get('source_archive_sha256') != ARCHIVE
            or record.get('historical_score_archive_sha256') != SCORE_ARCHIVE):
        raise ValueError('analysis source identity mismatch')
    roi = record['rows']
    old = record['historical_points']
    if [r['crf'] for r in roi] != [35, 48, 62] or [r['crf'] for r in old] != [48, 55, 62]:
        raise ValueError('analysis rate matrix mismatch')
    curves: dict[str, list[dict[str, Any]]] = {'roi': [], 'native': [], 'streak': []}
    for row in roi:
        if type(row['file_bytes']) is not int or row['file_bytes'] <= 0:
            raise ValueError('invalid full IVF price')
        curves['roi'].append({'crf': row['crf'], 'bytes': row['file_bytes'], 'means': metric_means(row['frames'])})
    for row in old:
        for arm, byte_key, metric_key in (('native', 'native_bytes', 'baseline'), ('streak', 'wire_bytes', 'streak')):
            if type(row[byte_key]) is not int or row[byte_key] <= 0:
                raise ValueError('invalid historical physical price')
            curves[arm].append({'crf': row['crf'], 'bytes': row[byte_key], 'means': metric_means(row[metric_key])})
    comparisons = {}
    for candidate, anchor in (('streak', 'roi'), ('native', 'roi'), ('roi', 'native')):
        result = {}
        for metric in ('full_lpips', 'full_dists', 'crop_lpips', 'crop_dists', 'rgb_psnr'):
            sign = 1 if metric == 'rgb_psnr' else -1
            test = [(row['bytes'], sign * row['means'][metric]) for row in curves[candidate]]
            baseline = [(row['bytes'], sign * row['means'][metric]) for row in curves[anchor]]
            result[metric] = anchor_sensitivity(test, baseline)
        comparisons[candidate + '_vs_' + anchor] = result
    return {'schema': 1, 'scope': '48-frame development footage, GT-privileged ROI, no task BD-rate',
            'metric_definitions': record['metric_definition'], 'physical_curves': curves,
            'comparisons': comparisons, 'rate_basis': 'complete native IVF or paid PSBL bytes',
            'bd_rate_basis': 'PCHIP log-rate, achievable anchor upper concave hull, common quality interval only; no extrapolation',
            'standard_lpips': 'ROI values retained per frame; historical curve lacks separately authenticated standard scalar values'}


def gpu_cost(wall: float, cpu: float, frames: int) -> dict[str, Any]:
    if wall <= 0 or cpu < 0 or frames not in (16, 48):
        raise ValueError('invalid score cost')
    forecast = 1.5 * wall * 48 / frames + 30
    return {'frames': frames, 'wall_seconds': wall, 'cpu_seconds': cpu, 'full_cap_seconds': 1800,
            'forecast_seconds': forecast, 'passed': forecast <= 1800 if frames == 16 else wall <= 1800,
            'basis': 'entrypoint body wall after imports, including scoring and publication; smoke 1.5 * wall * 48 / 16 + 30; full uses body wall'}


def validate_score_record(record: dict[str, Any], frames: int) -> dict[str, bool]:
    checks = {'frames': record['frames'] == frames and frames in (16, 48),
              'full_receiver': record['available_receiver_frames'] == 48 and record['packet_duration_seconds'] == 48 / 30,
              'not_promoted': record['promotable'] is False,
              'producer': record['producer_sha256'] == PRODUCER,
              'source': record['source_archive_sha256'] == ARCHIVE,
              'qualification': record['source_qualification_sha256'] == QUALIFICATION,
              'saved_control': record['historical_score_archive_sha256'] == SCORE_ARCHIVE
                  and record['historical_score_member_sha256'] == SCORE_MEMBER,
              'native_decode': record['fresh_native_yuv_sha256'] == NATIVE62_DECODE,
              'decoder': record['fresh_decoder_sha256'] == TOOLS['decoder'],
              'metric_parity': record['metric_control']['passed'] is True
                  and record['metric_control']['tolerance_absolute'] == 1e-6
                  and all(math.isfinite(v) and v <= 1e-6 for v in record['metric_control']['max_absolute_difference'].values()),
              'self_control': all(math.isfinite(record['self_control'][k]) and abs(record['self_control'][k]) <= 1e-5
                                  for k in ('lpips', 'dists', 'lpips_standard')),
              'cuda': record['gpu']['capability'] == [8, 9] and record['gpu']['device_count'] == 1
                  and record['gpu']['network_parameters_cuda'] is True
                  and 0 < record['gpu']['peak_reserved_bytes'] <= 8000 * 1024**2,
              'fresh_receivers': bool(record['physical_replay']) and all(record['physical_replay'].values())}
    rows = record['rows']
    checks['rates'] = [r['crf'] for r in rows] == [35, 48, 62]
    checks['historical_rates'] = [r['crf'] for r in record['historical_points']] == [48, 55, 62]
    for row in rows:
        frame_rows = row['frames']
        numeric = ('frame_psnr', 'full_lpips', 'full_dists', 'crop_lpips', 'crop_dists')
        checks[f"{row['crf']}:coverage"] = (len(frame_rows) == frames
            and [r['frame'] for r in frame_rows] == list(range(frames))
            and len(row['full_lpips_standard']) == frames and row['file_bytes'] > 0)
        checks[f"{row['crf']}:metrics"] = all(type(r.get(k)) in (int, float) and math.isfinite(r[k])
                                                   for r in frame_rows for k in numeric) and all(
            type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in row['full_lpips_standard'])
    return checks


def score_fleet_run(args: argparse.Namespace) -> dict[str, Any]:
    if os.sched_getaffinity(0) != {1} or os.environ.get('PS_CPU_ALLOWANCE') != '1':
        raise ValueError('one CPU affinity, index 1, and allowance 1 required')
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    args.work = str(scratch / 'score-private')
    args.score_output = str(scratch / 'scores.json')
    started, cpu = time.monotonic(), cpu_seconds()
    scores = score(args)
    checks = validate_score_record(scores, args.frames)
    if not all(checks.values()):
        raise ValueError('score integrity checks failed')
    archive = stage / 'published.tar'
    with tarfile.open(archive, 'x') as tar:
        tar.add(args.score_output, arcname='publish/scores.json', recursive=False)
    info = {'sha256': k5d_source_oracle.sha(archive), 'bytes': archive.stat().st_size,
            'scores_sha256': k5d_source_oracle.sha(Path(args.score_output))}
    cost = gpu_cost(time.monotonic() - started, cpu_seconds() - cpu, args.frames)
    result = {'schema': 1, 'frames': args.frames, 'archive': info, 'cost': cost, 'internal_checks': checks}
    with (stage / 'ball-roi-score-stage.json').open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    return result


def score_fleet_validate(stage: Path) -> dict[str, Any]:
    record = json.loads((stage / 'ball-roi-score-stage.json').read_text())
    archive = stage / 'published.tar'
    require_hash(archive, record['archive']['sha256'])
    if archive.stat().st_size != record['archive']['bytes']:
        raise ValueError('score archive size mismatch')
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        if len(members) != 1 or members[0].name != 'publish/scores.json' or not members[0].isfile():
            raise ValueError('unexpected score archive inventory')
        handle = tar.extractfile(members[0])
        if handle is None:
            raise ValueError('unreadable score archive')
        data = handle.read()
    if digest(data) != record['archive']['scores_sha256']:
        raise ValueError('score member mismatch')
    checks = validate_score_record(json.loads(data), record['frames'])
    cost = record['cost']
    checks['body_cost'] = cost == gpu_cost(cost['wall_seconds'], cost['cpu_seconds'], record['frames']) and cost['passed']
    execution_path = stage / 'execution.json'
    if execution_path.exists():
        execution = json.loads(execution_path.read_text())
        checks['whole_command_cost'] = (execution['exit_code'] == 0
                                         and gpu_cost(execution['seconds'], 0, record['frames'])['passed'])
    return {'passed': all(checks.values()), 'checks': checks,
            'scope': 'score identity and cost; 16-frame smoke cannot support BD-rate'}


def gpu_spec(deadline: str, cpu_archive: str, cpu_archive_sha: str,
             cpu_record: str, cpu_record_sha: str) -> dict[str, Any]:
    data = '/home/itec/emanuele/Datasets/pointstream-data'
    inputs = [
        {'name': 'producer', 'path': data + '/jobs/fleet/inbox/20261113T210157Z-41ee26e3/full/published.tar', 'sha256': PRODUCER, 'extract': False},
        {'name': 'archive', 'path': data + '/tennis/k5b-2026-10-20/inputs/tracknet-Dataset.zip', 'sha256': ARCHIVE, 'extract': False},
        {'name': 'old_score', 'path': data + '/jobs/fleet/inbox/20261113T220101Z-0cfa5558/full/published.tar', 'sha256': SCORE_ARCHIVE, 'extract': False},
        {'name': 'cpu_archive', 'path': cpu_archive, 'sha256': cpu_archive_sha, 'extract': False},
        {'name': 'cpu_record', 'path': cpu_record, 'sha256': cpu_record_sha, 'extract': False},
        {'name': 'lpips', 'path': data + '/visor/h3-2026-10-10/inputs/lpips/alexnet-owt-7be5be79.pth',
         'sha256': '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02', 'extract': False},
        {'name': 'dists', 'path': data + '/visor/h3b-2026-10-11/inputs/dists/vgg16-397923af.pth',
         'sha256': '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0', 'extract': False},
    ]
    return {'schema': 1, 'entrypoint': ['-m', 'experiments.tennis.ball_roi_control'],
            'arguments': ['--fleet', '--mode', 'score', '--producer', '{staged:producer}', '--archive', '{staged:archive}',
                          '--score-archive', '{staged:old_score}', '--cpu-archive', '{staged:cpu_archive}',
                          '--cpu-record', '{staged:cpu_record}', '--cpu-archive-sha256', cpu_archive_sha,
                          '--lpips-backbone', '{staged:lpips}', '--dists-backbone', '{staged:dists}', '--frames', '{frames}'],
            'staged_inputs': inputs, 'inputs': [{'path': item['path'], 'sha256': item['sha256']} for item in inputs],
            'environment': {'path': data + '/environments/pointstream-20261009T214002Z.tar.gz',
                            'sha256': '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'},
            'verify_staged_inputs': True, 'device': 'gpu', 'hosts': ['gpu6'], 'gpu_models': ['RTX 6000 Ada'],
            'gpu_memory_mib': 8000, 'cpu_threads': 1, 'single_cpu_campaign': True, 'cpu_affinity_index': 1,
            'local_storage_gib': 20, 'required_commands': ['ffmpeg'],
            'scale': {'frames': {'smoke': 16, 'full': 48}},
            'smoke': {'seconds': 450, 'representative_basis': '48-frame original native metric parity and full packet replay; score first16 of each ROI point.'},
            'full': {'seconds': 1800}, 'validator': ['{python}', '-m', 'experiments.tennis.ball_roi_control', '--validate-score'],
            'validator_seconds': 60, 'budget_seconds': 2430, 'deadline': deadline, 'stall_seconds': 450,
            'gpu_sharing': {'mode': 'shared', 'margin_mib': 8192,
                            'basis': 'Existing qualified Ada LPIPS/DISTS quality judging; exact native metric parity; no speed claim'},
            'contention': {'policy': 'continue', 'basis': 'Quality judging independent of shared walltime; foreign processes untouched'}}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fleet', action='store_true')
    parser.add_argument('--mode', choices=('run', 'score'), default='run')
    parser.add_argument('--validate-score', action='store_true')
    parser.add_argument('--score-spec', type=Path)
    parser.add_argument('--cpu-record-sha256')
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--analyse', type=Path)
    parser.add_argument('--analysis-output', type=Path)
    for name in ('score-archive', 'score-output', 'cpu-archive', 'cpu-archive-sha256', 'cpu-record', 'lpips-backbone', 'dists-backbone'):
        parser.add_argument('--' + name)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--deadline')
    for name in ('producer', 'archive', 'work'):
        parser.add_argument('--' + name)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--encoder', default='SvtAv1EncApp')
    parser.add_argument('--decoder', default='dav1d')
    parser.add_argument('--frames', type=int, choices=(16, 48))
    args = parser.parse_args()
    if args.score_spec:
        if not all((args.deadline, args.cpu_archive, args.cpu_archive_sha256, args.cpu_record, args.cpu_record_sha256)):
            parser.error('--score-spec requires deadline and CPU archive/record paths and hashes')
        with args.score_spec.open('x') as out:
            json.dump(gpu_spec(args.deadline, args.cpu_archive, args.cpu_archive_sha256, args.cpu_record, args.cpu_record_sha256), out, indent=2)
            out.write('\n')
        return
    if args.spec:
        if not args.deadline:
            parser.error('--spec requires --deadline')
        with args.spec.open('x') as out:
            json.dump(cpu_spec(args.deadline), out, indent=2)
            out.write('\n')
        return
    if args.validate_score:
        result = score_fleet_validate(Path(os.environ['PS_STAGE_DIR']))
        with Path(os.environ['PS_VALIDATION_PATH']).open('x') as out:
            json.dump(result, out, allow_nan=False)
            out.write('\n')
        print(json.dumps(result))
        if not result['passed']:
            raise SystemExit(1)
        return
    if args.fleet and args.mode == 'score':
        required = ('producer', 'archive', 'score_archive', 'cpu_archive', 'cpu_archive_sha256',
                    'cpu_record', 'lpips_backbone', 'dists_backbone', 'frames')
        if any(getattr(args, name) is None for name in required):
            parser.error('fleet score is missing a required staged input')
        binary = shutil.which(args.decoder)
        if binary is None:
            raise FileNotFoundError('dav1d missing')
        args.decoder = binary
        result = score_fleet_run(args)
        print(json.dumps({'frames': args.frames, 'cost': result['cost']}))
        if not result['cost']['passed']:
            raise SystemExit(1)
        return
    if args.analyse:
        if args.analysis_output is None:
            parser.error('--analyse requires --analysis-output')
        source_data = args.analyse.read_bytes()
        analysis = analyse_score(json.loads(source_data))
        analysis['score_sha256'] = digest(source_data)
        with args.analysis_output.open('x') as out:
            json.dump(analysis, out, indent=2, allow_nan=False)
            out.write('\n')
        return
    if args.score:
        standalone_required = ('producer', 'archive', 'work', 'score_archive', 'score_output', 'cpu_archive',
                    'cpu_archive_sha256', 'cpu_record', 'lpips_backbone', 'dists_backbone', 'frames')
        if any(getattr(args, name) is None for name in standalone_required):
            parser.error('score requires producer, archive, work, CPU archive/record/hash, historical score archive, weights, output and frames')
        binary = shutil.which(args.decoder)
        if binary is None:
            raise FileNotFoundError('dav1d missing')
        args.decoder = binary
        scored = score(args)
        print(json.dumps({'frames': scored['frames'], 'metric_control': scored['metric_control']}))
        return
    if args.validate:
        decoder = shutil.which(args.decoder)
        if decoder is None:
            raise FileNotFoundError('dav1d missing')
        value = fleet_validate(Path(os.environ['PS_STAGE_DIR']), Path(os.environ['TMPDIR']), decoder)
        with Path(os.environ['PS_VALIDATION_PATH']).open('x') as out:
            json.dump(value, out, allow_nan=False)
            out.write('\n')
        print(json.dumps(value))
        if not value['passed']:
            raise SystemExit(1)
        return
    if not args.producer or not args.archive or args.frames is None or (not args.fleet and not args.work):
        parser.error('producer, archive, frames and standalone work are required')
    for key in TOOLS:
        binary = shutil.which(getattr(args, key))
        if binary is None:
            raise FileNotFoundError(f'{key} missing')
        setattr(args, key, binary)
    result = fleet_run(args) if args.fleet else run(args)
    print(json.dumps({'frames': args.frames, 'fleet': args.fleet, 'result': result.get('cost', {})}))
    if args.fleet and not result['cost']['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

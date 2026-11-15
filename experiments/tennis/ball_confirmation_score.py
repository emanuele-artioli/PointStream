"""Frozen tennis confirmation scoring primitives; no training or encoder access.

The scoring inputs are authenticated full-length publications. The 16-frame
smoke is never scientific coverage and retains the full packet duration.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import ball_confirmation as confirmation, ball_roi_control as common, k5d_encoder

SOURCE_ARCHIVE = 'e5edc12c857658a119bf6d6cbe029ee64c09d6a860f1b975fe0708952895293d'
SOURCE_MANIFEST = 'df0dabf48b61ed3a096a9a6b4e54d823ca3b279c86d6a09e69ef0cd9bf2ceeb1'
LPIPS = '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02'
DISTS = '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'
LPIPS_HEAD = 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0'
DISTS_HEAD = 'f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218'
MATRIX = [('native_svt', p) for p in (48, 55, 62)] + [('native_dcvc', p) for p in (0, 21, 42)] + [
    ('roi_raw', p) for p in (35, 48, 62)] + [('roi_fitted', p) for p in (48, 55, 62)] + [
    ('ptp', p) for p in (48, 55, 62)] + [('streak', p) for p in (48, 55, 62)]


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha_text(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def labels_for_source(manifest: dict[str, Any], source_id: str) -> list[tuple[int, float, float]]:
    matches = [s for s in manifest['sources'] if s['id'] == source_id]
    if len(matches) != 1 or source_id not in confirmation.SOURCES:
        raise ValueError('unknown or duplicate frozen source')
    entry = matches[0]
    n, fps, _ = confirmation.SOURCES[source_id]
    if entry['frozen_window']['all_native_frame_count'] != n or entry['frozen_window']['fps_exact'] != [fps.numerator, fps.denominator]:
        raise ValueError('source window/cadence differs from freeze')
    labels = [(-1, 0.0, 0.0)] * n
    seen: set[int] = set()
    for row in entry['official_window_rows']:
        index, kind = row['native_frame_index'], row['visibility']
        if type(index) is not int or index < 0 or index >= n or index in seen or kind not in (0, 1):
            raise ValueError('invalid official frame or visibility')
        seen.add(index)
        if kind == 1:
            x, y = row['native_X'], row['native_Y']
            if row['GT720_X_exact'] != [2 * x, 3] or row['GT720_Y_exact'] != [2 * y, 3] or not row['crop_defined']:
                raise ValueError('invalid coordinate mapping')
            px, py = round(2 * x / 3), round(2 * y / 3)
            if not (0 <= px < confirmation.W and 0 <= py < confirmation.H):
                raise ValueError('crop centre outside 720 view')
            labels[index] = (1, float(px), float(py))
        else:
            if row['crop_defined'] or row['GT720_X_exact'] is not None or row['GT720_Y_exact'] is not None:
                raise ValueError('nonvisible frame must not define crop')
            labels[index] = (0, 0.0, 0.0)
    counts = entry['counts']
    if (sum(k == 1 for k, _, _ in labels) != counts['visible'] or
            sum(k == 0 for k, _, _ in labels) != counts['not_visible'] or
            sum(k == -1 for k, _, _ in labels) != counts['unknown']):
        raise ValueError('frozen label counts mismatch')
    return labels


def frame_hashes(frames: np.ndarray) -> list[str]:
    return [hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() for frame in frames]


def temporal_errors(source: np.ndarray, decoded: np.ndarray) -> list[dict[str, int]]:
    if source.shape != decoded.shape or source.dtype != np.uint8 or decoded.dtype != np.uint8:
        raise ValueError('temporal images differ in shape or dtype')
    result = []
    for i in range(1, len(source)):
        a = source[i].astype(np.int16) - source[i - 1].astype(np.int16)
        b = decoded[i].astype(np.int16) - decoded[i - 1].astype(np.int16)
        d = a.astype(np.int32) - b.astype(np.int32)
        result.append({'from_frame': i - 1, 'to_frame': i, 'sse': int(np.sum(d.astype(np.int64) ** 2)), 'pixels': int(d.size)})
    return result


def visibility_record(track: list[Any]) -> dict[str, Any]:
    visible = [p is not None for p in track]
    return {'visible': visible, 'transitions': [i for i in range(1, len(visible)) if visible[i] != visible[i - 1]]}


def score_one(source: np.ndarray, decoded: np.ndarray, labels: list[tuple[int, float, float]],
              nets: tuple[Any, Any]) -> dict[str, Any]:
    if source.shape != decoded.shape or len(source) != len(labels):
        raise ValueError('scoring span mismatch')
    rows = k5d_encoder.metrics(source, decoded, labels, nets)
    standard = common.standard_lpips(source, decoded, nets[0])
    for row, value, label in zip(rows, standard, labels):
        row['full_lpips_spatial_map'] = row.pop('full_lpips')
        row['full_lpips_standard'] = value
        row['label_kind'] = label[0]
    return {'frames': rows, 'temporal': temporal_errors(source, decoded),
            'decoded_rgb_frame_sha256': frame_hashes(decoded)}


def validate_scores(record: dict[str, Any], labels: list[tuple[int, float, float]], phase: str, full_n: int) -> dict[str, bool]:
    expected = 16 if phase == 'smoke' else full_n if phase == 'full' else 0
    n = record['frames']
    checks = {'coverage': n == expected and len(labels) == full_n and record['available_receiver_frames'] == full_n and record['full_frames'] == full_n,
              'promotion': record['promotable'] is (phase == 'full' and n == full_n),
              'matrix': [(r['family'], r['point']) for r in record['rows']] == MATRIX}
    for row in record['rows']:
        key = f"{row['family']}-{row['point']}"
        frames = row['frames']
        checks[key + '-shape'] = (len(frames) == n and len(row['decoded_rgb_frame_sha256']) == n
                                  and len(row['temporal']) == n - 1 and type(row['coded_bytes']) is int
                                  and row['coded_bytes'] > 0 and len(row['packet_sha256']) == 64
                                  and all(len(x) == 64 for x in row['decoded_rgb_frame_sha256']))
        good = len(frames) == n
        for i, frame in enumerate(frames):
            good &= (frame['frame'] == i and frame['label_kind'] == labels[i][0]
                     and type(frame['sse']) is int and frame['sse'] >= 0
                     and frame['pixels'] == confirmation.W * confirmation.H * 3)
            for field in ('frame_psnr', 'full_lpips_spatial_map', 'full_lpips_standard', 'full_dists'):
                v = frame[field]
                good &= type(v) in (int, float) and math.isfinite(v) and v >= 0
            for field in ('crop_lpips', 'crop_dists'):
                v = frame[field]
                good &= (type(v) in (int, float) and math.isfinite(v) and v >= 0) if labels[i][0] == 1 else v is None
        for i, pair in enumerate(row['temporal']):
            good &= (pair['from_frame'] == i and pair['to_frame'] == i + 1 and type(pair['sse']) is int
                     and pair['sse'] >= 0 and pair['pixels'] == confirmation.W * confirmation.H * 3)
        checks[key + '-metrics'] = good and row['full_lpips_standard'] == [x['full_lpips_standard'] for x in frames]
        total_pixels = sum(x['pixels'] for x in frames)
        total_sse = sum(x['sse'] for x in frames)
        from experiments.tennis import k5
        checks[key + '-pooled'] = bool(total_pixels and row['pooled_rgb_psnr'] == k5.mse_to_psnr(total_sse / total_pixels))
    return checks


def model_digest(net: Any) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(net.state_dict().items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode() + b'\0' + str(a.dtype).encode() + b'\0' + str(a.shape).encode() + b'\0' + a.tobytes())
    return h.hexdigest()


def authenticated_manifest(archive: Path) -> tuple[dict[str, Any], bytes]:
    import json
    import tarfile

    common.require_hash(archive, SOURCE_ARCHIVE)
    with tarfile.open(archive) as tar:
        found = [m for m in tar.getmembers() if m.name == 'manifest.json' and m.isfile()]
        if len(found) != 1:
            raise ValueError('source manifest missing or ambiguous')
        stream = tar.extractfile(found[0])
        if stream is None:
            raise ValueError('source manifest unreadable')
        data = stream.read()
    if common.digest(data) != SOURCE_MANIFEST:
        raise ValueError('source manifest hash mismatch')
    return json.loads(data), data


def read_full_inputs(source_id: str, root: Path, archive: Path, binding: Path, validation: Path,
                     execution: Path, archive_sha: str, binding_sha: str, validation_sha: str,
                     execution_sha: str, phase: str) -> dict[str, Any]:
    import json

    for path, sha in ((archive, archive_sha), (binding, binding_sha), (validation, validation_sha), (execution, execution_sha)):
        common.require_hash(path, sha)
    stage = json.loads(binding.read_bytes())
    gate = json.loads(validation.read_bytes())
    execution_record = json.loads(execution.read_bytes())
    n = confirmation.SOURCES[source_id][0]
    if (stage['phase'] != phase or stage['source_id'] != source_id or stage['frames'] != n
            or stage['archive']['sha256'] != archive_sha
            or gate.get('schema') != 'pointstream-ball-confirmation-accepted-full-v1'
            or gate.get('phase') != phase or gate.get('source_id') != source_id or gate.get('frames') != n
            or gate.get('archive_sha256') != archive_sha or gate.get('stage_sha256') != binding_sha
            or gate.get('execution_sha256') != execution_sha or gate.get('canonical_complete') is not True
            or gate.get('passed') is not True or not sha_text(gate.get('raw_full_validation_sha256'))
            or not sha_text(gate.get('dispatch_canonical_sha256'))
            or not isinstance(gate.get('producer_job_id'), str)
            or re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', gate['producer_job_id']) is None
            or not isinstance(gate.get('checks'), dict) or not gate['checks']
            or not all(v is True for v in gate['checks'].values())
            or execution_record['exit_code'] != 0 or not 0 < execution_record['seconds'] <= 1800):
        raise ValueError('full authenticated stage and accepted gate required')
    confirmation.extract_tree(archive, stage['archive'], root)
    receipt = json.loads((root / 'receipt.json').read_bytes())
    if canonical_sha(receipt['dispatch']) != gate['dispatch_canonical_sha256']:
        raise ValueError('accepted producer dispatch does not match publication')
    if not all(confirmation.validate_publication(root, stage).values()):
        raise ValueError('full physical publication failed validation')
    return stage


def rows_from_publications(base: Path, control: Path, n: int, work: Path) -> list[tuple[str, int, int, str, np.ndarray, list[str]]]:
    import json
    from experiments.tennis import windows

    b = json.loads((base / 'receipt.json').read_bytes())
    c = json.loads((control / 'receipt.json').read_bytes())
    rows = []
    for family, point in MATRIX:
        if family in ('native_svt', 'native_dcvc'):
            kind = 'svt' if family == 'native_svt' else 'dcvc'
            row = next(x for x in b['result']['bases'] if (x['kind'], x['point']) == (kind, point))
            directory = base / 'bases' / f'{kind}-{point}'
            packet = directory / ('stream.ivf' if kind == 'svt' else 'stream.psdc')
            rgb = np.load(directory / 'decoded.npy', mmap_mode='r')
            hashes = row['rgb_frame_sha256']
        elif family.startswith('roi_'):
            kind = family.removeprefix('roi_')
            row = next(x for x in c['result']['roi'] if (x['family'], x['crf']) == (kind, point))
            directory = control / 'controls' / f'roi-{kind}-{point}'
            packet = directory / 'stream.ivf'
            rgb = windows.rgb_frames(directory / 'decoded.yuv', confirmation.W, confirmation.H, n,
                                     work / f'{family}-{point}.npy')
            hashes = frame_hashes(rgb)
        else:
            row = next(x for x in c['result']['pixel'] if x['crf'] == point)
            directory = control / 'controls' / f'pixel-{point}'
            packet = directory / ('pixel.ptp' if family == 'ptp' else 'streak.psbl')
            path = directory / ('rendered.rgb' if family == 'ptp' else 'streak-replay/rendered.rgb')
            rgb = np.memmap(path, dtype=np.uint8, mode='r', shape=(n, confirmation.H, confirmation.W, 3))
            hashes = row['replay']['rgb_frame_sha256'] if family == 'ptp' else row['streak_frame_sha256']
        if len(hashes) != n or frame_hashes(rgb) != hashes:
            raise ValueError('decoded pixel frame identity mismatch')
        rows.append((family, point, packet.stat().st_size, confirmation.sha(packet), rgb, hashes))
    return rows


def run(args: Any) -> dict[str, Any]:
    import json
    import os
    import resource
    import socket
    import time
    import torch
    from experiments.visor.h3b import dists_crops, lpips_reduce
    from src.codecs import quality

    started = time.monotonic()
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    phase = os.environ['PS_STAGE']
    n, fps, _ = confirmation.SOURCES[args.source_id]
    frames = 16 if phase == 'smoke' else n if phase == 'full' else 0
    if args.frames != frames or len(os.sched_getaffinity(0)) != 1 or os.environ.get('PS_CPU_ALLOWANCE') != '1':
        raise ValueError('stage frame or CPU allocation mismatch')
    if (torch.cuda.device_count() != 1 or tuple(torch.cuda.get_device_capability(0)) not in ((8, 6), (8, 9))
            or torch.__version__ != '2.10.0+cu128' or torch.version.cuda != '12.8'):
        raise ValueError('unqualified CUDA runtime or class')
    torch.cuda.set_per_process_memory_fraction(min(1., 8000 * 1024**2 / torch.cuda.get_device_properties(0).total_memory), 0)
    stage = Path(os.environ['PS_STAGE_DIR'])
    dispatch = json.loads((stage / 'dispatch.json').read_bytes())
    if dispatch['staged_environment']['sha256'] != confirmation.RUNTIME_SHA:
        raise ValueError('runtime differs from freeze')
    work = Path(os.environ['PS_SCRATCH_DIR']) / 'confirmation-score-private'
    work.mkdir(exist_ok=False)
    manifest, manifest_bytes = authenticated_manifest(Path(args.source_archive))
    labels = labels_for_source(manifest, args.source_id)
    bases = read_full_inputs(args.source_id, work / 'bases', Path(args.bases_archive), Path(args.bases_stage),
                             Path(args.bases_validation), Path(args.bases_execution), args.bases_archive_sha256,
                             args.bases_stage_sha256, args.bases_validation_sha256, args.bases_execution_sha256, 'bases')
    read_full_inputs(args.source_id, work / 'controls', Path(args.controls_archive), Path(args.controls_stage),
                     Path(args.controls_validation), Path(args.controls_execution), args.controls_archive_sha256,
                     args.controls_stage_sha256, args.controls_validation_sha256, args.controls_execution_sha256, 'controls')
    base_receipt = json.loads((work / 'bases/receipt.json').read_bytes())
    control_receipt = json.loads((work / 'controls/receipt.json').read_bytes())
    sender = json.loads((work / 'controls/sender.json').read_bytes())
    if (base_receipt['source_archive_sha256'] != SOURCE_ARCHIVE
            or control_receipt['base_archive_sha256'] != bases['archive']['sha256']):
        raise ValueError('source/base/control lineage mismatch')
    source = np.memmap(work / 'bases/source.rgb', dtype=np.uint8, mode='r', shape=(n, confirmation.H, confirmation.W, 3))
    if frame_hashes(source) != base_receipt['source']['rgb_frame_sha256']:
        raise ValueError('source decoded frame hashes mismatch')
    common.require_hash(Path(args.lpips), LPIPS)
    common.require_hash(Path(args.dists), DISTS)
    lpips_record, dists_record = quality.lpips_record(Path(args.lpips)), quality.dists_record(Path(args.dists))
    if (lpips_record['heads_sha256'] != LPIPS_HEAD or dists_record['weights_sha256'] != DISTS_HEAD
            or lpips_record['torchmetrics'] != '1.9.0' or dists_record['torchmetrics'] != '1.9.0'):
        raise ValueError('metric heads or torchmetrics differ from frozen historical scorer')
    nets = (quality.load_lpips(Path(args.lpips), 'cuda'), quality.load_dists(Path(args.dists), 'cuda'))
    if any(p.device.type != 'cuda' for net in nets for p in net.parameters()):
        raise ValueError('network has non-CUDA parameters')
    sample = torch.from_numpy(np.ascontiguousarray(source[0, :64, :64])).to('cuda').permute(2, 0, 1)[None].float() / 255.
    with torch.inference_mode():
        alex_features = list(nets[0].net(sample * 2 - 1))
        dists_features = list(nets[1].forward_once(sample))
    if (not alex_features or not dists_features or any(t.device.type != 'cuda' or not bool(torch.isfinite(t).all())
                                                       for t in alex_features + dists_features)):
        raise ValueError('metric feature qualification failed')
    mask = np.ones((confirmation.H, confirmation.W), dtype=bool)
    self_control = {'lpips_spatial': lpips_reduce(nets[0], [source[0]], source[0], [[mask]], 'cuda')[0][0],
                    'dists': dists_crops(nets[1], [source[0]], source[0], mask, 'cuda')[0],
                    'lpips_standard': common.standard_lpips(source[:1], source[:1], nets[0])[0]}
    if any(not math.isfinite(x) or abs(x) > 1e-5 for x in self_control.values()):
        raise ValueError('model self-control failed')
    scored = []
    for family, point, packet_bytes, packet_sha, rgb, hashes in rows_from_publications(work / 'bases', work / 'controls', n, work):
        data = score_one(source[:frames], rgb[:frames], labels[:frames], nets)
        if data['decoded_rgb_frame_sha256'] != hashes[:frames]:
            raise ValueError('score frame pixels differ from authenticated receiver')
        from experiments.tennis import k5
        sse = sum(row['sse'] for row in data['frames'])
        pixels = sum(row['pixels'] for row in data['frames'])
        scored.append({'family': family, 'point': point, 'coded_bytes': packet_bytes, 'packet_sha256': packet_sha,
                       'full_lpips_standard': [row['full_lpips_standard'] for row in data['frames']],
                       'pooled_rgb_psnr': k5.mse_to_psnr(sse / pixels), 'rate_bps': packet_bytes * 8 * float(fps) / n, **data})
    cost = time.monotonic() - started
    cpu_end = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = cpu_end.ru_utime + cpu_end.ru_stime - cpu_started.ru_utime - cpu_started.ru_stime
    forecast = 1.25 * cost * n / 16 + 30 if phase == 'smoke' else cost
    result = {'schema': 1, 'source_id': args.source_id, 'frames': frames, 'available_receiver_frames': n,
              'fps': [fps.numerator, fps.denominator], 'packet_duration_seconds': n / float(fps),
              'full_frames': n, 'promotable': phase == 'full', 'source_archive_sha256': SOURCE_ARCHIVE, 'manifest_sha256': SOURCE_MANIFEST,
              'base_archive_sha256': args.bases_archive_sha256, 'controls_archive_sha256': args.controls_archive_sha256,
              'base_stage_sha256': args.bases_stage_sha256, 'controls_stage_sha256': args.controls_stage_sha256,
              'base_validation_sha256': args.bases_validation_sha256, 'controls_validation_sha256': args.controls_validation_sha256,
              'base_execution_sha256': args.bases_execution_sha256, 'controls_execution_sha256': args.controls_execution_sha256,
              'sender_visibility': {'raw': visibility_record(sender['initial_track']),
                  **{str(p): visibility_record(next(x['centres'] for x in sender['points']
                                                    if x['codec'] == 'svt' and x['point'] == p)) for p in (48, 55, 62)}},
              'execution_identity': {'source_sha256': dispatch['code']['source_sha256'], 'snapshot_sha256': dispatch['code']['snapshot_sha256'],
                                     'environment_sha256': dispatch['staged_environment']['sha256']},
              'scorer_code_sha256': confirmation.sha(Path(__file__)),
              'self_control': self_control, 'weights': {'lpips': lpips_record,
                  'dists': dists_record, 'lpips_parameters_sha256': model_digest(nets[0]),
                  'dists_parameters_sha256': model_digest(nets[1])},
              'gpu': {'name': torch.cuda.get_device_name(0), 'host': socket.gethostname(),
                      'capability': list(torch.cuda.get_device_capability(0)), 'torch': torch.__version__,
                      'cuda': torch.version.cuda, 'all_parameters_cuda': True,
                      'feature_qualification': {'alex_layers': len(alex_features), 'dists_layers': len(dists_features), 'cuda_finite': True},
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved()},
              'cost': {'entrypoint_body_wall_seconds': cost, 'entrypoint_body_cpu_seconds': cpu_seconds,
                       'forecast_seconds': forecast, 'cap_seconds': 1800,
                       'passed': forecast <= 1800, 'scope': 'body after module imports; shared GPU timing not throughput'},
              'metric_definitions': {'full_lpips_spatial_map': 'historical full-frame spatial map mean',
                  'full_lpips_standard': 'standard scalar LPIPS', 'crop': 'independent 64x64 official visible crop',
                  'temporal': 'RGB temporal-difference SSE against source'}, 'rows': scored}
    checks = validate_scores(result, labels, phase, n)
    if not all(checks.values()) or not result['cost']['passed']:
        raise ValueError('score coverage or cost gate failed')
    pub = work / 'publication'
    pub.mkdir()
    (pub / 'manifest.json').write_bytes(manifest_bytes)
    confirmation.write_receipt(pub / 'scores.json', result)
    info = {'source_id': args.source_id, 'frames': frames, 'phase': phase,
            'archive': confirmation.publish_tree(pub, stage / 'published.tar'), 'checks': checks}
    confirmation.write_receipt(stage / 'confirmation-score-stage.json', info)
    return info


def validate(stage: Path, phase: str) -> dict[str, Any]:
    import json
    import tempfile

    info = json.loads((stage / 'confirmation-score-stage.json').read_bytes())
    dispatch = json.loads((stage / 'dispatch.json').read_bytes())
    if info['source_id'] not in confirmation.SOURCES:
        raise ValueError('unknown source')
    n, fps, _ = confirmation.SOURCES[info['source_id']]
    with tempfile.TemporaryDirectory(prefix='confirmation-score-validation-', dir=stage.parent) as tmp:
        root = Path(tmp) / 'publication'
        confirmation.extract_tree(stage / 'published.tar', info['archive'], root)
        manifest_bytes = (root / 'manifest.json').read_bytes()
        if common.digest(manifest_bytes) != SOURCE_MANIFEST:
            raise ValueError('source freeze mismatch')
        labels = labels_for_source(json.loads(manifest_bytes), info['source_id'])
        result = json.loads((root / 'scores.json').read_bytes())
        checks = validate_scores(result, labels, phase, n)
    recorded_inputs = {x['sha256'] for x in dispatch['input_identity']}
    bound = [SOURCE_ARCHIVE, LPIPS, DISTS, result['base_archive_sha256'], result['base_stage_sha256'],
             result['base_validation_sha256'], result['base_execution_sha256'], result['controls_archive_sha256'],
             result['controls_stage_sha256'], result['controls_validation_sha256'], result['controls_execution_sha256']]
    identity = {'source_sha256': dispatch['code']['source_sha256'], 'snapshot_sha256': dispatch['code']['snapshot_sha256'],
                'environment_sha256': dispatch['staged_environment']['sha256']}
    body_wall = result['cost']['entrypoint_body_wall_seconds']
    expected_forecast = 1.25 * body_wall * n / 16 + 30 if phase == 'smoke' else body_wall
    visibility = result['sender_visibility']
    checks.update({'stage_binding': stage.name == phase == info['phase'] and info['frames'] == result['frames']
                   and info['source_id'] == result['source_id'],
                   'input_binding': all(x in recorded_inputs for x in bound),
                   'identity': result['execution_identity'] == identity and identity['environment_sha256'] == confirmation.RUNTIME_SHA,
                   'source_freeze': result['source_archive_sha256'] == SOURCE_ARCHIVE and result['manifest_sha256'] == SOURCE_MANIFEST,
                   'fps_and_duration': result['fps'] == [fps.numerator, fps.denominator] and result['packet_duration_seconds'] == n / float(fps),
                   'sender_visibility': set(visibility) == {'raw', '48', '55', '62'} and all(
                       len(v['visible']) == n and all(type(x) is bool for x in v['visible'])
                       and v['transitions'] == [i for i in range(1, n) if v['visible'][i] != v['visible'][i - 1]]
                       for v in visibility.values()),
                   'self_control': all(math.isfinite(v) and abs(v) <= 1e-5 for v in result['self_control'].values()),
                   'gpu': result['gpu']['capability'] in ([8, 6], [8, 9]) and result['gpu']['torch'] == '2.10.0+cu128'
                   and result['gpu']['cuda'] == '12.8' and result['gpu']['all_parameters_cuda'] is True
                   and result['gpu']['feature_qualification']['cuda_finite'] is True
                   and result['gpu']['feature_qualification']['alex_layers'] > 0
                   and result['gpu']['feature_qualification']['dists_layers'] > 0
                   and 0 < result['gpu']['peak_reserved_bytes'] <= 8000 * 1024**2,
                   'weights': (result['weights']['lpips']['backbone_sha256'] == LPIPS
                               and result['weights']['dists']['backbone_sha256'] == DISTS
                               and result['weights']['lpips']['heads_sha256'] == LPIPS_HEAD
                               and result['weights']['dists']['weights_sha256'] == DISTS_HEAD
                               and result['weights']['lpips']['torchmetrics'] == '1.9.0'
                               and result['weights']['dists']['torchmetrics'] == '1.9.0'
                               and len(result['weights']['lpips_parameters_sha256']) == 64
                               and len(result['weights']['dists_parameters_sha256']) == 64),
                   'cost': (body_wall > 0 and result['cost']['passed'] is True
                            and result['cost']['forecast_seconds'] == expected_forecast and expected_forecast <= 1800)})
    execution = stage / 'execution.json'
    if execution.exists():
        actual = json.loads(execution.read_bytes())
        forecast = 1.25 * actual['seconds'] * n / 16 + 30 if phase == 'smoke' else actual['seconds']
        checks['whole_command_cost'] = actual['exit_code'] == 0 and forecast <= 1800
    return {'passed': all(checks.values()), 'checks': checks, 'phase': phase, 'source_id': info['source_id'],
            'frames': result['frames'], 'archive_sha256': info['archive']['sha256'],
            'stage_sha256': confirmation.sha(stage / 'confirmation-score-stage.json'),
            'dispatch_canonical_sha256': canonical_sha(dispatch),
            'scope': 'perceptual coverage; no task or throughput claim'}


def spec(source_id: str, deadline: str, supplied: dict[str, tuple[str, str]]) -> dict[str, Any]:
    data = '/home/itec/emanuele/Datasets/pointstream-data'
    expected = {f'{phase}_{item}' for phase in ('bases', 'controls') for item in ('archive', 'stage', 'validation', 'execution')}
    if source_id not in confirmation.SOURCES or set(supplied) != expected:
        raise ValueError('source and eight full accepted input bindings required')
    paths = {'source_archive': (data + '/tennis/confirmation-source-20261115T062148Z-297a05e2/tennis-confirmation-sources.tar', SOURCE_ARCHIVE),
             'lpips': (data + '/visor/h3-2026-10-10/inputs/lpips/alexnet-owt-7be5be79.pth', LPIPS),
             'dists': (data + '/visor/h3b-2026-10-11/inputs/dists/vgg16-397923af.pth', DISTS), **supplied}
    items = [{'name': name, 'path': path, 'sha256': sha, 'extract': False} for name, (path, sha) in paths.items()]
    args = ['--fleet', '--source-id', source_id, '--frames', '{frames}']
    for name, (_, sha) in paths.items():
        args += ['--' + name.replace('_', '-'), '{staged:' + name + '}']
        if name in supplied:
            args += ['--' + name.replace('_', '-') + '-sha256', sha]
    return {'schema': 1, 'device': 'gpu', 'hosts': ['gpu2', 'gpu3', 'gpu5', 'gpu6'],
            'gpu_models': ['RTX A6000', 'RTX 6000 Ada'], 'gpu_memory_mib': 8000, 'cpu_threads': 1,
            'single_cpu_campaign': True, 'cpu_affinity_index': 0,
            'gpu_sharing': {'mode': 'shared', 'margin_mib': 8192, 'basis': 'Quality only; no throughput claim.'},
            'contention': {'policy': 'continue', 'basis': 'Quality only; no throughput claim.', 'resume_attempts': 0},
            'local_storage_gib': 30, 'entrypoint': ['-m', 'experiments.tennis.ball_confirmation_score'], 'arguments': args,
            'staged_inputs': items, 'inputs': [{'path': x['path'], 'sha256': x['sha256']} for x in items],
            'environment': {'path': data + '/environments/pointstream-20261009T214002Z.tar.gz', 'sha256': confirmation.RUNTIME_SHA},
            'verify_staged_inputs': True, 'required_commands': [],
            'scale': {'frames': {'smoke': 16, 'full': confirmation.SOURCES[source_id][0]}},
            'smoke': {'seconds': 450, 'representative_basis': 'All 18 complete packets, first 16 frames; full receiver cost remains charged.'},
            'full': {'seconds': 1800}, 'validator': ['{python}', '-m', 'experiments.tennis.ball_confirmation_score', '--validate'],
            'validator_seconds': 60, 'budget_seconds': 2430, 'stall_seconds': 1800, 'deadline': deadline}


def main() -> None:
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fleet', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--deadline')
    parser.add_argument('--source-id', choices=list(confirmation.SOURCES))
    parser.add_argument('--frames', type=int)
    parser.add_argument('--source-archive')
    parser.add_argument('--lpips')
    parser.add_argument('--dists')
    for phase in ('bases', 'controls'):
        for name in ('archive', 'stage', 'validation', 'execution'):
            parser.add_argument(f'--{phase}-{name}')
            parser.add_argument(f'--{phase}-{name}-sha256')
    args = parser.parse_args()
    if args.spec:
        if not args.source_id or not args.deadline:
            parser.error('--spec requires --source-id and --deadline')
        supplied = {}
        for part in ('bases', 'controls'):
            for name in ('archive', 'stage', 'validation', 'execution'):
                key = f'{part}_{name}'
                path, digest = getattr(args, key), getattr(args, key + '_sha256')
                if not path or not digest:
                    parser.error('--spec requires all eight accepted input paths and hashes')
                supplied[key] = (path, digest)
        confirmation.write_receipt(args.spec, spec(args.source_id, args.deadline, supplied))
        return
    stage = Path(os.environ['PS_STAGE_DIR'])
    phase = os.environ['PS_STAGE']
    if args.validate:
        result = validate(stage, phase)
        confirmation.write_receipt(Path(os.environ['PS_VALIDATION_PATH']), result)
        print(json.dumps(result))
        if not result['passed']:
            raise SystemExit(1)
        return
    required = ('source_id', 'frames', 'source_archive', 'lpips', 'dists') + tuple(
        f'{part}_{name}{suffix}' for part in ('bases', 'controls') for name in ('archive', 'stage', 'validation', 'execution')
        for suffix in ('', '_sha256'))
    if not args.fleet or any(getattr(args, key) is None for key in required):
        parser.error('fleet requires all frozen source, accepted stage and metric inputs')
    info = run(args)
    if phase == 'full':
        result = validate(stage, phase)
        confirmation.write_receipt(stage / 'full-validation.json', result)
        if not result['passed']:
            raise SystemExit(1)
    print(json.dumps(info))


if __name__ == '__main__':
    main()

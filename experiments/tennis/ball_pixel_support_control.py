"""Fixed spatial-support control for the accepted paid 32px tennis packets.

Centre and annulus are receiver-only, free mechanism ablations. Every arm is
charged the original complete PTP packet; no labels control the placements.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import ball_roi_control as common, k5d_encoder, paid_patch_profile as paid, windows
from src.pointstream import tracked_patch

PAID_ARCHIVE = '66df76d85fa37f4228a7b50abaaa75b0e71dcf28639fcdd2b7c584960db0e418'
PAID_STAGE = '40c13a6df8b352c4c8813ac31d50678fabc3d021b4e810ad95c3945481f10a0b'
SCORE_ARCHIVE = 'c9da533b7c3ca334b170636abcf82efd50f6dfa597b3b0aefcea6c6959a1352e'
LPIPS = '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02'
DISTS = '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'
LPIPS_HEAD = 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0'
DISTS_HEAD = 'f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218'
RATES = (48, 55, 62)


def compose(base: np.ndarray, tile: np.ndarray, positions: list[Any], arm: str) -> np.ndarray:
    if (arm not in ('whole32', 'centre16', 'annulus') or base.ndim != 4 or tile.ndim != 4
            or base.dtype != np.uint8 or tile.dtype != np.uint8 or base.shape[0] != len(positions)
            or tile.shape != (len(positions), 64, 64, 3) or base.shape[-1] != 3):
        raise ValueError('invalid patch composition')
    _, h, w, _ = base.shape
    output = np.array(base, copy=True)
    for i, position in enumerate(positions):
        if position is None:
            continue
        if (not isinstance(position, list) or len(position) != 2 or any(type(v) is not int for v in position)):
            raise ValueError('invalid serialized placement')
        x, y = position
        if x < 0 or y < 0 or x + 32 > w or y + 32 > h or x % 2 or y % 2:
            raise ValueError('out-of-bounds or odd placement')
        if arm in ('whole32', 'annulus'):
            output[i, y:y + 32, x:x + 32] = tile[i, 16:48, 16:48]
        if arm == 'centre16':
            output[i, y + 8:y + 24, x + 8:x + 24] = tile[i, 24:40, 24:40]
        elif arm == 'annulus':
            output[i, y + 8:y + 24, x + 8:x + 24] = base[i, y + 8:y + 24, x + 8:x + 24]
    return output


def compare_full(actual: list[dict[str, Any]], saved: list[dict[str, Any]], standard: list[float], expected: list[float]) -> dict[str, Any]:
    result = common.compare_metric_control(actual, saved)
    if len(standard) != 48 or len(expected) != 48:
        raise ValueError('complete standard LPIPS control required')
    differences = [abs(a - b) for a, b in zip(standard, expected)]
    if any(not math.isfinite(x) or x > 1e-6 for x in differences):
        raise ValueError('standard LPIPS parity failure')
    return {**result, 'scope': '48-frame whole32 paid pixels: full-map/crop metrics and exact RGB SSE',
            'standard_lpips_max_abs': max(differences)}


def ratio(native: float, arm: float, whole: float) -> float | None:
    denominator = native - whole
    return (native - arm) / denominator if denominator > 0 else None


def point_coverage(labels: list[tuple[int, float, float]], positions: list[Any]) -> dict[str, Any]:
    """Classify GT point locations for reporting only, never for compositing."""
    if len(labels) != len(positions):
        raise ValueError('coverage frame count mismatch')
    statuses = []
    for (kind, x, y), position in zip(labels, positions):
        if kind != 1 or not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('coverage requires visible finite source labels')
        if position is None:
            status = 'sender_none'
        else:
            left, top = position
            if left + 8 <= x < left + 24 and top + 8 <= y < top + 24:
                status = 'centre16'
            elif left <= x < left + 32 and top <= y < top + 32:
                status = 'annulus'
            else:
                status = 'outside32'
        statuses.append(status)
    return {'frame_status': statuses, 'counts': {key: statuses.count(key) for key in
            ('centre16', 'annulus', 'outside32', 'sender_none')}}


def analyse(rows: list[dict[str, Any]], historical: list[dict[str, Any]]) -> dict[str, Any]:
    if [r['crf'] for r in rows] != list(RATES) or any(len(r['annulus']['frames']) != 48 or len(r['centre16']['frames']) != 48 for r in rows):
        raise ValueError('complete 48-frame points required')
    out = []
    for row, old in zip(rows, historical):
        if old['crf'] != row['crf']:
            raise ValueError('native curve misaligned')
        values = {}
        for metric in ('crop_dists', 'crop_lpips'):
            base = common.metric_means(old['baseline'])[metric]
            whole = common.metric_means(row['whole32']['frames'])[metric]
            values[metric] = {arm: ratio(base, common.metric_means(row[arm]['frames'])[metric], whole)
                              for arm in ('annulus', 'centre16')}
        out.append({'crf': row['crf'], 'retained_reduction': values, 'point_coverage': row['point_coverage']})
    def meets(metric: str, arm: str, threshold: float) -> bool:
        return all((v := p['retained_reduction'][metric][arm]) is not None and v >= threshold for p in out)
    background = meets('crop_dists', 'annulus', .5) and all(
        (v := p['retained_reduction']['crop_lpips']['annulus']) is not None and v > 0 for p in out)
    central = meets('crop_dists', 'centre16', .8) and all(
        (v := p['retained_reduction']['crop_lpips']['centre16']) is not None and v > 0 for p in out)
    from src.pointstream.evaluation import anchor_sensitivity
    comparisons: dict[str, Any] = {}
    for arm in ('annulus', 'centre16'):
        comparisons[arm] = {}
        for metric in ('crop_dists', 'crop_lpips'):
            candidate = [(r['packet_bytes'], -common.metric_means(r[arm]['frames'])[metric]) for r in rows]
            native = [(old['native_bytes'], -common.metric_means(old['baseline'])[metric]) for old in historical]
            whole_curve = [(r['packet_bytes'], -common.metric_means(r['whole32']['frames'])[metric]) for r in rows]
            comparisons[arm][metric] = {name: {'bd_rate': anchor_sensitivity(candidate, curve),
                                                'ordinal': paid.ordinal_separation(candidate, curve)}
                                        for name, curve in (('native', native), ('whole32', whole_curve))}
    return {'points': out, 'annulus_context_support': background, 'central_support': central,
            'comparisons': comparisons,
            'outcome': 'mixed' if background == central else 'context_support' if background else 'central_support',
            'scope': 'descriptive nonlinear reductions; not additive causal shares or object boundaries'}


def decode_streams(packet: Path, root: Path, decoder: str) -> tuple[np.ndarray, np.ndarray, list[Any], dict[str, Any]]:
    meta, base, tile = tracked_patch.unpack(packet.read_bytes())
    native = meta['native']
    if (native['width'], native['height'], native['frames'], native['fps'], native['full_range']) != (common.W, common.H, 48, [30, 1], False):
        raise ValueError('packet native geometry differs')
    root.mkdir(exist_ok=False)
    base_receipt = tracked_patch.decode_native(base, root / 'base.yuv', native, decoder)
    base_rgb = windows.rgb_frames(root / 'base.yuv', common.W, common.H, 48, root / 'base.npy')
    if not tile:
        if any(p is not None for p in meta['positions']):
            raise ValueError('missing tile for present position')
        tile_rgb = np.zeros((48, 64, 64, 3), dtype=np.uint8)
        tile_receipt = None
    else:
        tile_receipt = tracked_patch.decode_native(tile, root / 'tile.yuv', {**native, 'width': 64, 'height': 64}, decoder)
        tile_rgb = windows.rgb_frames(root / 'tile.yuv', 64, 64, 48, root / 'tile.npy')
    return base_rgb, tile_rgb, meta['positions'], {'base': base_receipt, 'tile': tile_receipt,
                                                   'base_ivf_sha256': common.digest(base), 'tile_ivf_sha256': common.digest(tile)}


def gpu_cost(wall: float, cpu: float, frames: int) -> dict[str, Any]:
    if wall <= 0 or cpu < 0 or frames not in (16, 48):
        raise ValueError('invalid cost')
    forecast = 1.5 * wall * 3 + 30 if frames == 16 else wall
    return {'wall_seconds': wall, 'cpu_seconds': cpu, 'frames': frames, 'forecast_seconds': forecast,
            'full_cap_seconds': 1800, 'passed': forecast <= 1800,
            'basis': 'entrypoint body after imports; smoke includes fixed full48 control and 16-frame ablations; shared timing not throughput'}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import socket
    import torch
    from experiments.visor.h3b import dists_crops, lpips_reduce
    from src.codecs import quality

    if len(os.sched_getaffinity(0)) != 1 or os.environ.get('PS_CPU_ALLOWANCE') != '1':
        raise ValueError('requires one CPU and allowance')
    if (torch.cuda.device_count() != 1 or tuple(torch.cuda.get_device_capability(0)) not in ((8, 6), (8, 9))
            or torch.__version__ != '2.10.0+cu128' or torch.version.cuda != '12.8'):
        raise ValueError('qualified CUDA runtime required')
    torch.cuda.set_per_process_memory_fraction(min(1., 8000 * 1024**2 / torch.cuda.get_device_properties(0).total_memory), 0)
    stage_dir = Path(os.environ['PS_STAGE_DIR'])
    phase = os.environ['PS_STAGE']
    if args.frames != {'smoke': 16, 'full': 48}.get(phase):
        raise ValueError('stage frame count mismatch')
    dispatch = json.loads((stage_dir / 'dispatch.json').read_bytes())
    if dispatch['staged_environment']['sha256'] != '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661':
        raise ValueError('runtime mismatch')
    work = Path(os.environ['PS_SCRATCH_DIR']) / 'support-control-private'
    work.mkdir(exist_ok=False)
    started, cpu = time.monotonic(), common.cpu_seconds()
    common.require_hash(Path(args.decoder), common.TOOLS['decoder'])
    qualification, pilot, qualification_bytes = common.producer_metadata(Path(args.producer))
    source = common.source_frames(Path(args.archive), qualification)
    labels = [(int(r['label_kind']), float(r['label_x']), float(r['label_y'])) for r in qualification['rows']]
    common.require_hash(Path(args.paid_stage), PAID_STAGE)
    common.require_hash(Path(args.paid_archive), PAID_ARCHIVE)
    paid_stage = json.loads(Path(args.paid_stage).read_bytes())
    if (paid_stage['archive']['sha256'] != PAID_ARCHIVE or paid_stage['frames'] != 48 or not paid_stage['cost']['passed']
            or not paid_stage['internal_checks'] or not all(paid_stage['internal_checks'].values())):
        raise ValueError('accepted paid full stage required')
    root = paid.extract(Path(args.paid_archive), paid_stage['archive'], 'tennis', work / 'paid')
    physical = paid.validate_publication(root, args.decoder, work / 'receiver')
    if not all(physical.values()):
        raise ValueError('paid fresh receiver failed')
    receipt = json.loads((root / 'receipt.json').read_bytes())
    if receipt['frames'] != 48 or not receipt['score_reuse_eligible']:
        raise ValueError('paid full48 not eligible')
    common.require_hash(Path(args.saved_scores), SCORE_ARCHIVE)
    saved_bytes = common.tar_member(Path(args.saved_scores), 'publish/scores.json', paid.PIXEL_SCORE)
    saved = json.loads(saved_bytes)
    if saved['frames'] != 48 or saved['cpu_archive_sha256'] != paid.PIXEL_ARCHIVE or not saved['metric_control']['passed']:
        raise ValueError('saved pixel control mismatch')
    for name, sha in (('lpips', LPIPS), ('dists', DISTS)):
        common.require_hash(Path(getattr(args, name)), sha)
    lp_record, di_record = quality.lpips_record(Path(args.lpips)), quality.dists_record(Path(args.dists))
    if (lp_record['heads_sha256'] != LPIPS_HEAD or di_record['weights_sha256'] != DISTS_HEAD
            or lp_record['torchmetrics'] != '1.9.0' or di_record['torchmetrics'] != '1.9.0'):
        raise ValueError('frozen metric heads differ')
    nets = (quality.load_lpips(Path(args.lpips), 'cuda'), quality.load_dists(Path(args.dists), 'cuda'))
    if any(p.device.type != 'cuda' for net in nets for p in net.parameters()):
        raise ValueError('metric parameter not CUDA')
    mask = np.ones((common.H, common.W), dtype=bool)
    self_control = {'lpips': lpips_reduce(nets[0], [source[0]], source[0], [[mask]], 'cuda')[0][0],
                    'dists': dists_crops(nets[1], [source[0]], source[0], mask, 'cuda')[0],
                    'standard_lpips': common.standard_lpips(source[:1], source[:1], nets[0])[0]}
    if any(not math.isfinite(v) or abs(v) > 1e-5 for v in self_control.values()):
        raise ValueError('self-control failed')
    controls: list[dict[str, Any]] = []
    for crf, old in zip(RATES, saved['rows']):
        if old['crf'] != crf:
            raise ValueError('saved score point mismatch')
        packet = root / f'{crf}.ptp'
        base, tile, positions, decoded = decode_streams(packet, work / f'decoded-{crf}', args.decoder)
        whole = compose(base, tile, positions, 'whole32')
        actual_sha = [common.digest(frame.tobytes()) for frame in whole]
        info = json.loads((root / f'{crf}.json').read_bytes())
        if (actual_sha != info['decode']['rgb_frame_sha256'] or common.digest(whole.tobytes()) != info['decode']['decoded_sha256']
                or info['decode']['decoded_sha256'] != old['composite_sha256']
                or decoded['base_ivf_sha256'] != old['base_sha256'] or decoded['tile_ivf_sha256'] != old['patch_sha256']):
            raise ValueError('all-frame whole32 paid receiver parity failed')
        full_metrics = k5d_encoder.metrics(source, whole, labels, nets)
        standard = common.standard_lpips(source, whole, nets[0])
        check = compare_full(full_metrics, old['frames'], standard, old['full_lpips_standard'])
        controls.append({'crf': crf, 'packet_bytes': packet.stat().st_size, 'packet_sha256': common.k5d_source_oracle.sha(packet),
                         'frame_sha256': actual_sha, 'decode': decoded, 'parity': check,
                         'point_coverage': point_coverage(labels, positions),
                         'whole32': {'frames': full_metrics, 'full_lpips_standard': standard}})
    # No ablation is scored until every full48 real receiver and metric control has passed.
    rows = []
    for control in controls:
        crf = control['crf']
        base, tile, positions, _ = decode_streams(root / f'{crf}.ptp', work / f'ablation-decode-{crf}', args.decoder)
        arms = {}
        for arm in ('centre16', 'annulus'):
            image = compose(base[:args.frames], tile[:args.frames], positions[:args.frames], arm)
            arms[arm] = {'frames': k5d_encoder.metrics(source[:args.frames], image, labels[:args.frames], nets),
                         'full_lpips_standard': common.standard_lpips(source[:args.frames], image, nets[0]),
                         'frame_sha256': [common.digest(x.tobytes()) for x in image]}
        rows.append({**control, **arms})
    cpu_used = common.cpu_seconds() - cpu
    cost = gpu_cost(time.monotonic() - started, cpu_used, args.frames)
    result = {'schema': 1, 'phase': phase, 'frames': args.frames, 'promotable': args.frames == 48,
              'scope': 'fixed receiver-only spatial support control; full original PTP packet price; not object segmentation or causal attribution',
              'paid_archive_sha256': PAID_ARCHIVE, 'paid_stage_sha256': PAID_STAGE, 'score_sha256': paid.PIXEL_SCORE,
              'source_archive_sha256': common.ARCHIVE, 'producer_sha256': common.PRODUCER,
              'physical_checks': physical, 'full48_controls_passed': all(r['parity']['passed'] for r in controls),
              'historical_points': saved['historical_points'], 'self_control': self_control,
              'weights': {'lpips': lp_record, 'dists': di_record},
              'gpu': {'name': torch.cuda.get_device_name(0), 'host': socket.gethostname(), 'capability': list(torch.cuda.get_device_capability(0)),
                      'torch': torch.__version__, 'cuda': torch.version.cuda, 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                      'all_parameters_cuda': True},
              'execution_identity': {'source_sha256': dispatch['code']['source_sha256'], 'snapshot_sha256': dispatch['code']['snapshot_sha256'],
                                     'environment_sha256': dispatch['staged_environment']['sha256']},
              'cost': cost, 'rows': rows}
    if args.frames == 48:
        result['analysis'] = analyse(rows, saved['historical_points'])
    checks = validate_result(result, phase, saved, root, labels)
    if not all(checks.values()) or not cost['passed']:
        raise ValueError('support control integrity or cost gate failed')
    publication = work / 'publication'
    publication.mkdir()
    paid.write_json(publication / 'scores.json', result)
    (publication / 'reference.json').write_bytes(saved_bytes)
    (publication / 'paid.tar').write_bytes(Path(args.paid_archive).read_bytes())
    (publication / 'paid-stage.json').write_bytes(Path(args.paid_stage).read_bytes())
    (publication / 'qualification.json').write_bytes(qualification_bytes)
    archive = stage_dir / 'published.tar'
    with tarfile.open(archive, 'x') as tar:
        for name in ('scores.json', 'reference.json', 'paid.tar', 'paid-stage.json', 'qualification.json'):
            tar.add(publication / name, arcname='publish/' + name, recursive=False)
    info = {'frames': args.frames, 'phase': phase, 'archive_sha256': common.k5d_source_oracle.sha(archive),
            'archive_bytes': archive.stat().st_size, 'scores_sha256': common.k5d_source_oracle.sha(publication / 'scores.json'),
            'checks': checks, 'cost': cost}
    paid.write_json(stage_dir / 'support-control-stage.json', info)
    return info


def validate_result(result: dict[str, Any], phase: str, saved: dict[str, Any], paid_root: Path,
                    labels: list[tuple[int, float, float]]) -> dict[str, bool]:
    frames = {'smoke': 16, 'full': 48}.get(phase)
    checks = {'coverage': frames is not None and result['frames'] == frames and result['phase'] == phase
              and result['promotable'] is (phase == 'full'),
              'source': result['source_archive_sha256'] == common.ARCHIVE and result['producer_sha256'] == common.PRODUCER,
              'paid': result['paid_archive_sha256'] == PAID_ARCHIVE and result['paid_stage_sha256'] == PAID_STAGE,
              'reference': (result['score_sha256'] == paid.PIXEL_SCORE and saved['frames'] == 48
                            and saved['cpu_archive_sha256'] == paid.PIXEL_ARCHIVE
                            and saved['source_archive_sha256'] == common.ARCHIVE and saved['producer_sha256'] == common.PRODUCER
                            and saved['metric_control']['passed'] is True),
              'physical': (result['full48_controls_passed'] is True and bool(result['physical_checks'])
                           and all(v is True for v in result['physical_checks'].values())),
              'points': [r['crf'] for r in result['rows']] == list(RATES),
              'self': all(math.isfinite(v) and abs(v) <= 1e-5 for v in result['self_control'].values()),
              'weights': result['weights']['lpips']['backbone_sha256'] == LPIPS and result['weights']['lpips']['heads_sha256'] == LPIPS_HEAD
              and result['weights']['dists']['backbone_sha256'] == DISTS and result['weights']['dists']['weights_sha256'] == DISTS_HEAD,
              'gpu': result['gpu']['capability'] in ([8, 6], [8, 9]) and result['gpu']['torch'] == '2.10.0+cu128'
              and result['gpu']['cuda'] == '12.8' and result['gpu']['all_parameters_cuda'] is True
              and 0 < result['gpu']['peak_reserved_bytes'] <= 8000 * 1024**2,
              'historical': result['historical_points'] == saved['historical_points']}
    for row, old in zip(result['rows'], saved['rows']):
        crf = row['crf']
        packet = paid_root / f'{crf}.ptp'
        metadata, _, _ = tracked_patch.unpack(packet.read_bytes())
        checks[f'{crf}:point_coverage'] = row['point_coverage'] == point_coverage(labels, metadata['positions'])
        info = json.loads((paid_root / f'{crf}.json').read_bytes())
        checks[f'{crf}:packet'] = (row['packet_bytes'] == packet.stat().st_size and row['packet_sha256'] == common.k5d_source_oracle.sha(packet)
             and row['frame_sha256'] == info['decode']['rgb_frame_sha256'] and len(row['frame_sha256']) == 48
             and info['decode']['decoded_sha256'] == old['composite_sha256'] and old['crf'] == crf
             and row['decode']['base_ivf_sha256'] == old['base_sha256'] and row['decode']['tile_ivf_sha256'] == old['patch_sha256'])
        try:
            parity = compare_full(row['whole32']['frames'], old['frames'], row['whole32']['full_lpips_standard'], old['full_lpips_standard'])
            checks[f'{crf}:metric_parity'] = parity == row['parity'] and parity['passed'] is True
        except (ValueError, KeyError, TypeError):
            checks[f'{crf}:metric_parity'] = False
        for name in ('centre16', 'annulus'):
            arm = row[name]
            good = (len(arm['frames']) == frames and len(arm['full_lpips_standard']) == frames and len(arm['frame_sha256']) == frames)
            for i, value in enumerate(arm['frames']):
                good &= (value['frame'] == i and type(value['sse']) is int and value['sse'] >= 0
                         and value['pixels'] == common.W * common.H * 3)
                for key in ('frame_psnr', 'full_lpips', 'full_dists', 'crop_lpips', 'crop_dists'):
                    x = value[key]
                    good &= type(x) in (int, float) and math.isfinite(x) and x >= 0
            good &= all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in arm['full_lpips_standard'])
            checks[f'{crf}:{name}'] = bool(good)
    checks['analysis'] = result.get('analysis') == analyse(result['rows'], saved['historical_points']) if phase == 'full' else 'analysis' not in result
    return checks


def validate(stage: Path, phase: str, decoder: str, tmp: Path) -> dict[str, Any]:
    info = json.loads((stage / 'support-control-stage.json').read_bytes())
    archive = stage / 'published.tar'
    common.require_hash(archive, info['archive_sha256'])
    if archive.stat().st_size != info['archive_bytes']:
        raise ValueError('archive size mismatch')
    with tempfile.TemporaryDirectory(prefix='support-control-validation-', dir=tmp) as directory:
        work = Path(directory)
        with tarfile.open(archive) as tar:
            names = {'publish/scores.json', 'publish/reference.json', 'publish/paid.tar', 'publish/paid-stage.json', 'publish/qualification.json'}
            if len(tar.getmembers()) != len(names) or {m.name for m in tar.getmembers()} != names or any(not m.isfile() for m in tar.getmembers()):
                raise ValueError('unexpected publication inventory')
            tar.extractall(work, filter='data')
        pub = work / 'publish'
        common.require_hash(pub / 'scores.json', info['scores_sha256'])
        common.require_hash(pub / 'reference.json', paid.PIXEL_SCORE)
        common.require_hash(pub / 'paid-stage.json', PAID_STAGE)
        common.require_hash(pub / 'paid.tar', PAID_ARCHIVE)
        common.require_hash(pub / 'qualification.json', common.QUALIFICATION)
        qualification = json.loads((pub / 'qualification.json').read_bytes())
        labels = [(int(r['label_kind']), float(r['label_x']), float(r['label_y'])) for r in qualification['rows']]
        if len(labels) != 48:
            raise ValueError('expected full48 source labels')
        bound = json.loads((pub / 'paid-stage.json').read_bytes())
        root = paid.extract(pub / 'paid.tar', bound['archive'], 'tennis', work / 'paid')
        physical = paid.validate_publication(root, decoder, work / 'receiver')
        saved = json.loads((pub / 'reference.json').read_bytes())
        result = json.loads((pub / 'scores.json').read_bytes())
        checks = validate_result(result, phase, saved, root, labels)
        checks['fresh_paid_receiver'] = all(physical.values())
        for row in result['rows']:
            crf = row['crf']
            base, tile, positions, _ = decode_streams(root / f'{crf}.ptp', work / f'arm-replay-{crf}', decoder)
            for arm in ('centre16', 'annulus'):
                n = result['frames']
                pixels = compose(base[:n], tile[:n], positions[:n], arm)
                checks[f'{crf}:{arm}:actual_pixel_hashes'] = (
                    [common.digest(x.tobytes()) for x in pixels] == row[arm]['frame_sha256'])
    dispatch = json.loads((stage / 'dispatch.json').read_bytes())
    identity = {'source_sha256': dispatch['code']['source_sha256'], 'snapshot_sha256': dispatch['code']['snapshot_sha256'],
                'environment_sha256': dispatch['staged_environment']['sha256']}
    checks['stage'] = stage.name == phase == info['phase'] and info['frames'] == result['frames']
    checks['dispatch'] = result['execution_identity'] == identity and identity['environment_sha256'] == '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'
    input_hashes = {item['sha256'] for item in dispatch['input_identity']}
    checks['inputs'] = {common.PRODUCER, common.ARCHIVE, PAID_ARCHIVE, PAID_STAGE, SCORE_ARCHIVE, LPIPS, DISTS} <= input_hashes
    cost = result['cost']
    checks['cost'] = cost == gpu_cost(cost['wall_seconds'], cost['cpu_seconds'], result['frames']) and cost['passed']
    execution = stage / 'execution.json'
    if execution.exists():
        e = json.loads(execution.read_bytes())
        checks['whole_command_cost'] = e['exit_code'] == 0 and gpu_cost(e['seconds'], 0, result['frames'])['passed']
    return {'passed': all(checks.values()), 'checks': checks, 'phase': phase, 'frames': result['frames'],
            'archive_sha256': info['archive_sha256'], 'scope': 'paid packet and metric control, not task or transport evidence'}


def spec(deadline: str) -> dict[str, Any]:
    data = '/home/itec/emanuele/Datasets/pointstream-data'
    inputs = [('producer', data + '/jobs/fleet/inbox/20261113T210157Z-41ee26e3/full/published.tar', common.PRODUCER),
              ('archive', data + '/tennis/k5b-2026-10-20/inputs/tracknet-Dataset.zip', common.ARCHIVE),
              ('paid_archive', data + '/jobs/fleet/inbox/20261115T032830Z-e14124b6/full/published.tar', PAID_ARCHIVE),
              ('paid_stage', data + '/jobs/fleet/inbox/20261115T032830Z-e14124b6/full/paid-profile-stage.json', PAID_STAGE),
              ('saved_scores', data + '/jobs/fleet/inbox/20261115T020807Z-26806765/full/published.tar', SCORE_ARCHIVE),
              ('lpips', data + '/visor/h3-2026-10-10/inputs/lpips/alexnet-owt-7be5be79.pth', LPIPS),
              ('dists', data + '/visor/h3b-2026-10-11/inputs/dists/vgg16-397923af.pth', DISTS)]
    items = [{'name': name, 'path': path, 'sha256': sha, 'extract': False} for name, path, sha in inputs]
    args = ['--fleet', '--frames', '{frames}']
    for name, _, _ in inputs:
        args += ['--' + name.replace('_', '-'), '{staged:' + name + '}']
    return {'schema': 1, 'device': 'gpu', 'hosts': ['gpu5', 'gpu6'],
            'gpu_models': ['RTX 6000 Ada'], 'gpu_memory_mib': 8000, 'cpu_threads': 1,
            'single_cpu_campaign': True, 'cpu_affinity_index': 0,
            'gpu_sharing': {'mode': 'shared', 'margin_mib': 8192, 'basis': 'Perceptual quality only; no throughput claim.'},
            'contention': {'policy': 'continue', 'basis': 'Quality only; no throughput claim.', 'resume_attempts': 0},
            'local_storage_gib': 20, 'entrypoint': ['-m', 'experiments.tennis.ball_pixel_support_control'],
            'arguments': args, 'staged_inputs': items,
            'inputs': [{'path': x['path'], 'sha256': x['sha256']} for x in items],
            'environment': {'path': data + '/environments/pointstream-20261009T214002Z.tar.gz',
                            'sha256': '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'},
            'verify_staged_inputs': True, 'required_commands': [], 'scale': {'frames': {'smoke': 16, 'full': 48}},
            'smoke': {'seconds': 450, 'representative_basis': 'All full48 paid receivers and metric parity first; all three 16-frame ablations.'},
            'full': {'seconds': 1800}, 'validator': ['{python}', '-m', 'experiments.tennis.ball_pixel_support_control', '--validate'],
            'validator_seconds': 60, 'budget_seconds': 2430, 'stall_seconds': 1800, 'deadline': deadline}


def main() -> None:
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fleet', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--deadline')
    parser.add_argument('--frames', type=int, choices=(16, 48))
    for name in ('producer', 'archive', 'paid-archive', 'paid-stage', 'saved-scores', 'lpips', 'dists'):
        parser.add_argument('--' + name)
    parser.add_argument('--decoder', default=str(Path(sys.prefix) / 'bin/dav1d'))
    args = parser.parse_args()
    if args.spec:
        if not args.deadline:
            parser.error('--spec requires --deadline')
        paid.write_json(args.spec, spec(args.deadline))
        return
    stage, phase = Path(os.environ['PS_STAGE_DIR']), os.environ['PS_STAGE']
    if args.validate:
        checked = validate(stage, phase, args.decoder, Path(os.environ['TMPDIR']))
        paid.write_json(Path(os.environ['PS_VALIDATION_PATH']), checked)
        print(json.dumps(checked))
        if not checked['passed']:
            raise SystemExit(1)
        return
    if not args.fleet or args.frames is None or any(getattr(args, name) is None for name in
            ('producer', 'archive', 'paid_archive', 'paid_stage', 'saved_scores', 'lpips', 'dists')):
        parser.error('fleet requires frozen paid/source/metric inputs and --frames')
    result = run(args)
    if phase == 'full':
        checked = validate(stage, phase, args.decoder, Path(os.environ['TMPDIR']))
        paid.write_json(stage / 'full-validation.json', checked)
        if not checked['passed']:
            raise SystemExit(1)
    print(json.dumps({'frames': result['frames'], 'cost': result['cost']}))


if __name__ == '__main__':
    main()

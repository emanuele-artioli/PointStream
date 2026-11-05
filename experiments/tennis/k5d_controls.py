"""Frozen sensitivity controls on retained calibration streams; never encodes video.

The prospective spec is inadmissible until independently accepted calibration
smoke and full archives are explicitly bound by path and SHA256.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
import sys
import time
from typing import Any
import zipfile

import numpy as np

from experiments.tennis import k5d_encoder as enc, k5d_fleet as fleet, k5d_source_oracle as oracle
from experiments.tennis import k5b_trackers as trk, windows as tw
from experiments.tennis.k5d_analysis import score_frames
from src.pointstream import ball, ball_stream

ARMS = ('native', 'streak', 'disk', 'wrong_position', 'brightness')
DECODER_SHA = 'e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6'
CONTRACT_SHA = '18c88bf2a70ede9a11b793d278924490ff5aa621feb7a703347a8d417e133ac5'
ENV_SHA = '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'
CALIBRATION_SHA = {'calibration_smoke': 'a39643080425a1b05da7bade7a74c0a96bbf14848434c121ae7f5ab8353e6873',
                   'calibration_full': 'f0dd118a5d64ca148be675a097606aa6dacb8ae49ffcac639ad30911ae74a727'}


def variants(stream: ball_stream.Stream, source: np.ndarray, native: np.ndarray) -> tuple[dict[str, ball_stream.Stream], bool]:
    if stream.codec != 'svt-av1-ivf' or stream.resource_id != 'svt':
        raise ValueError('only retained SVT calibration stream is permitted')
    track, apps = ball.decode_frames(stream.ball_payload, stream.frames, stream.width, stream.height)
    disk = [None if a is None else replace(a, exposure=0) for a in apps]
    bright = [None if a is None else replace(a, colour=(255, 255, 255)) for a in apps]
    shifted = [None if p is None else ((p[0] + 64) % stream.width, p[1]) for p in track]
    def make(t: ball.Track, a: list[ball.Appearance | None]) -> ball_stream.Stream:
        return replace(stream, ball_payload=ball.encode_frames(t, a, stream.width, stream.height))
    off = make([None] * stream.frames, [None] * stream.frames)
    disk_stream = make(track, disk)
    disk_images = ball_stream.render_from_native_decoded_frames(disk_stream, native)
    guard_off = enc.pooled_psnr(source, disk_images) < enc.pooled_psnr(source, native)
    if guard_off:
        disk_stream = off
    return {'native': off, 'streak': stream, 'disk': disk_stream,
            'wrong_position': make(shifted, apps), 'brightness': make(track, bright)}, guard_off


def members(path: Path) -> dict[str, bytes]:
    names = ('pilot.json', 'source-qualification.json', 'svt-62/native.bin', 'svt-62/stream.psbl',
             'svt-62/baseline.psbl', 'svt-62/point.json')
    result = {}
    with tarfile.open(path) as tar:
        for name in names:
            matches = [m for m in tar.getmembers() if m.name == 'publish/' + name]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 100_000_000:
                raise ValueError('calibration member missing, ambiguous or oversized: ' + name)
            f = tar.extractfile(matches[0])
            if f is None:
                raise ValueError('unreadable calibration member')
            result[name] = f.read()
    return result


def bound_inputs(stage: Path, *, full_frames: int = 48, calibration_sha: dict[str, str] | None = None,
                 extra_sha: dict[str, str] | None = None) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    if full_frames not in (48, 188):
        raise ValueError('unregistered controls scope')
    calibration_sha = CALIBRATION_SHA if calibration_sha is None else calibration_sha
    extra_sha = {} if extra_sha is None else extra_sha
    spec = json.loads((stage.parent / 'spec.json').read_text())
    stage_data = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if (spec.get('single_cpu_campaign') is not True or spec.get('cpu_threads') != 1
            or spec.get('gpu_sharing') != {'mode': 'exclusive'} or dispatch.get('gpu_sharing') != {'mode': 'exclusive'}
            or spec.get('gpu_models') != ['RTX A6000']
            or spec.get('device') != 'gpu' or not 0 < spec.get('budget_seconds', 0) <= 1200
            or spec.get('environment', {}).get('sha256') != ENV_SHA
            or stage_data['environment']['sha256'] != ENV_SHA or len(dispatch.get('cpu_affinity', [])) != 1
            or spec.get('budget_seconds', 0) > 550
            or spec.get('scale', {}).get('frames') != {'smoke': 16, 'full': full_frames}
            or spec.get('smoke', {}).get('seconds', 0) > (100 if full_frames == 188 else 130) or spec.get('full', {}).get('seconds', 0) > 400):
        raise ValueError('wrong environment or bounded single-CPU campaign')
    declared = {x['name']: x for x in spec['staged_inputs']}
    actual = {x['name']: x for x in stage_data['inputs']}
    names = {'archive', 'balltrack_code', 'balltrack_weight', 'wasb_code', 'wasb_weights', 'wasb_config',
             'lpips_backbone', 'dists_backbone', 'calibration_smoke', 'calibration_full'} | set(extra_sha)
    if set(declared) != names or set(actual) != names or len(actual) != len(stage_data['inputs']):
        raise ValueError('wrong input set')
    root = Path(stage_data['local_root']).resolve(strict=True)
    paths = {}
    for name in names:
        item, staged = declared[name], actual[name]
        digest = item['sha256']
        if (len(digest) != 64 or set(digest) == {'0'} or staged['sha256'] != digest
                or staged.get('extract', False) != item.get('extract', False)):
            raise ValueError('unbound staged identity: ' + name)
        if name in calibration_sha and digest != calibration_sha[name]:
            raise ValueError('calibration publication changed: ' + name)
        if name in extra_sha and digest != extra_sha[name]:
            raise ValueError('pilot receipt identity changed: ' + name)
        if name in fleet.EXPECTED and name not in ('wasb_code', 'wasb_weights', 'balltrack_code') and digest != fleet.EXPECTED[name]:
            raise ValueError('changed immutable input: ' + name)
        path = Path(staged['path']).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('staged input is not host local')
        paths[name] = path
    for name, digest in calibration_sha.items():
        if oracle.sha(paths[name]) != digest:
            raise ValueError('calibration bytes differ: ' + name)
    for name, digest in extra_sha.items():
        if oracle.sha(paths[name]) != digest:
            raise ValueError('pilot bytes differ: ' + name)
    for name in ('wasb_code', 'balltrack_code'):
        entry = actual[name]
        archive = paths[name].parent / Path(entry['source']).name
        if oracle.sha(archive) != entry['sha256'] or oracle.tree_sha(paths[name]) != fleet.archive_tree_sha(archive):
            raise ValueError('code tree differs from verified archive')
    if oracle.sha(paths['wasb_weights'] / 'wasb_tennis_best.pth.tar') != fleet.EXPECTED['wasb_weights']:
        raise ValueError('WASB checkpoint changed')
    oracle.bind_config(paths['wasb_code'], fleet.EXPECTED['wasb_config'])
    return paths, spec, dispatch


def check_calibration(data: dict[str, bytes], frames: int) -> tuple[dict[str, Any], dict[str, Any], ball_stream.Stream]:
    pilot, source, point = (json.loads(data[n]) for n in ('pilot.json', 'source-qualification.json', 'svt-62/point.json'))
    oracle.validate_report(source)
    matching = [p for p in pilot['points'] if p['codec'] == 'svt' and p['point'] == 62]
    wire = data['svt-62/stream.psbl']
    stream = ball_stream.unpack(wire)
    baseline = ball_stream.unpack(data['svt-62/baseline.psbl'])
    if (len(matching) != 1 or matching[0] != point or pilot['clip'] != 'tracknet/game10/Clip3'
            or source['clip'] != pilot['clip'] or pilot['frames'] != frames or source['frames'] != frames or stream.frames != frames
            or stream.codec != 'svt-av1-ivf' or stream.native != data['svt-62/native.bin']
            or baseline.native != stream.native or any(ball.decode_frames(baseline.ball_payload, frames, stream.width, stream.height)[0])
            or hashlib.sha256(wire).hexdigest() != point['wire_sha256']
            or hashlib.sha256(stream.native).hexdigest() != point['native_sha256']
            or point['receiver']['wire_sha256'] != point['wire_sha256']
            or point['receiver']['native_sha256'] != point['native_sha256']
            or point['receiver']['frames'] != frames
            or point['receiver']['decoded_yuv_sha256'] != point['encoder_decoded_sha256']
            or point['receiver']['rendered_rgb_sha256'] != point['encoder_rendered_sha256']
            or not point['decode_parity'] or not point['render_parity']):
        raise ValueError('retained calibration identities or parity fail')
    return source, point, stream


def run(stage: Path, scratch: Path, frames: int, *, full_frames: int = 48,
        calibration_sha: dict[str, str] | None = None, extra_sha: dict[str, str] | None = None) -> None:
    import torch
    from src.codecs import quality
    from experiments.tennis import k5d_worker
    started = time.monotonic()
    paths, _, dispatch = bound_inputs(stage, full_frames=full_frames, calibration_sha=calibration_sha, extra_sha=extra_sha)
    if frames != {'smoke': 16, 'full': full_frames}.get(stage.name):
        raise ValueError('stage frame mismatch')
    if len(os.sched_getaffinity(0)) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('single CPU and one CUDA GPU required')
    contract = fleet.ROOT / 'docs/experiments/k5d-inputs/calibration-controls-contract-20261104.json'
    if oracle.sha(contract) != CONTRACT_SHA:
        raise ValueError('registered controls contract changed')
    if oracle.sha(fleet.SOURCE) != fleet.SOURCE_SHA or oracle.sha(fleet.EXPOSURE) != fleet.EXPOSURE_SHA:
        raise ValueError('frozen source manifest/exposure changed')
    cal = paths['calibration_smoke' if frames == 16 else 'calibration_full']
    cal_data = members(cal)
    source_report, point, stream = check_calibration(cal_data, frames)
    if list(torch.cuda.get_device_capability(0)) != source_report['device']['capability']:
        raise ValueError('GPU class differs from retained calibration')
    identities = source_report['identities']
    for name in ('archive', 'balltrack_weight', 'wasb_config'):
        if identities[name]['sha256'] != fleet.EXPECTED[name]:
            raise ValueError('calibration judge/source identity changed: ' + name)
    for name in ('balltrack_code', 'wasb_code'):
        if identities[name]['tree_sha256'] != oracle.tree_sha(paths[name]):
            raise ValueError('calibration judge code changed: ' + name)
    if identities['wasb_weights']['sha256'] != fleet.EXPECTED['wasb_weights']:
        raise ValueError('calibration WASB weights changed')
    selected = oracle.select(json.loads(fleet.SOURCE.read_text()), 'tracknet/game10/Clip3', 0, frames)
    found = [c for c in trk.tracknet_clips(paths['archive'], [10]) if c['id'] == 'tracknet/game10/Clip3']
    if len(found) != 1:
        raise ValueError('source clip not unique')
    clip = found[0]
    oracle.validate_rows(clip['rows'], selected)
    with zipfile.ZipFile(paths['archive']) as z:
        if hashlib.sha256(z.read(selected['label'])).hexdigest() != selected['label_sha256']:
            raise ValueError('label archive changed')
        for i, row in enumerate(clip['rows'][:frames]):
            member = f"{clip['dir']}/{row[0]}"
            if source_report['rows'][i]['member'] != member or source_report['rows'][i]['source_sha256'] != hashlib.sha256(z.read(member)).hexdigest():
                raise ValueError('source frame differs from calibration')
    clip = {**clip, 'rows': clip['rows'][:frames]}
    labels = trk.tracknet_labels(clip)
    if labels != [(r['label_kind'], r['label_x'], r['label_y']) for r in source_report['rows']]:
        raise ValueError('label rows differ from calibration')
    source = trk.tracknet_frames(paths['archive'], clip)
    work = scratch / 'controls-work'
    work.mkdir()
    decoder = Path(sys.prefix) / 'bin/dav1d'
    if oracle.sha(decoder) != DECODER_SHA:
        raise ValueError('decoder binary differs from calibration environment')
    wire = work / 'retained.psbl'
    wire.write_bytes(cal_data['svt-62/stream.psbl'])
    resource = work / 'resources.json'
    resource.write_text(json.dumps({'resources': {'svt': {'codec': 'svt-av1-ivf', 'decoder': str(decoder), 'decoder_sha256': DECODER_SHA}}}))
    receipt = k5d_worker.decode(wire, resource, work / 'receive')
    if (receipt['decoded_yuv_sha256'] != point['encoder_decoded_sha256']
            or receipt['rendered_rgb_sha256'] != point['encoder_rendered_sha256']):
        raise ValueError('fresh native/streak parity against calibration failed')
    yuv = np.memmap(work / 'receive/native.yuv', np.uint8, 'r', shape=(frames, stream.height * 3 // 2, stream.width))
    native = np.stack([tw.yuv_to_rgb(x, stream.width, stream.height) for x in yuv])
    streams, guard_off = variants(stream, source, native)
    tracker = trk.BallTrack(paths['balltrack_code'], paths['balltrack_weight'])
    zoo = trk.Zoo(paths['wasb_code'], paths['wasb_weights'], names=('wasb-step1',))
    if next(tracker.net.parameters()).device.type != 'cuda' or next(zoo.settings['wasb-step1']['net'].parameters()).device.type != 'cuda':
        raise ValueError('judges not on CUDA')
    nets = (quality.load_lpips(paths['lpips_backbone'], 'cuda'), quality.load_dists(paths['dists_backbone'], 'cuda'))
    publish = scratch / 'publish'
    publish.mkdir()
    results = {}
    streak_images = ball_stream.render_from_native_decoded_frames(stream, native)
    for name in ARMS:
        arm = streams[name]
        packed = ball_stream.pack(arm)
        if ball_stream.unpack(packed) != arm:
            raise ValueError('PSBL roundtrip mismatch')
        (publish / (name + '.psbl')).write_bytes(packed)
        images = ball_stream.render_from_native_decoded_frames(arm, native)
        if name == 'streak' and hashlib.sha256(images.tobytes()).hexdigest() != receipt['rendered_rgb_sha256']:
            raise ValueError('streak render parity failed')
        reference = streak_images if name in ('wrong_position', 'brightness') else native
        changed = int(np.count_nonzero(np.any(images != reference, axis=(1, 2, 3))))
        metric_rows = enc.metrics(source, images, labels, nets)
        for i, row in enumerate(metric_rows):
            row['crop_psnr'] = (enc.old.psnr(enc.crop64(source[i], labels[i][1], labels[i][2]),
                                             enc.crop64(images[i], labels[i][1], labels[i][2]))
                                if labels[i][0] == 1 else None)
        judges = {'wasb': enc.prediction_rows(labels, zoo.run('wasb-step1', images)),
                  'balltrack': enc.prediction_rows(labels, tracker.run(images))}
        if name in ('native', 'streak'):
            original = point['baseline' if name == 'native' else 'streak']['judges']
            if judges != original:
                raise ValueError('retained calibration judge outputs differ: ' + name)
        results[name] = {'changed_frames': changed, 'status': 'informative' if changed else 'uninformative',
                         'native_bytes': len(arm.native), 'ball_bytes': len(arm.ball_payload),
                         'wire_bytes': len(packed), 'wire_sha256': hashlib.sha256(packed).hexdigest(),
                         'coding_claim': name in ('native', 'streak', 'disk'),
                         'billed_bytes': (len(arm.native) if name == 'native' else len(packed))
                         if name in ('native', 'streak', 'disk') else None,
                         'metrics': metric_rows, 'judges': judges}
    result = {'scope': 'diagnostic controls; no judge validation or realism claim',
              'coding_rate_basis': {'native': 'physical IVF bytes', 'streak': 'complete PSBL bytes',
                                    'disk': 'complete PSBL bytes',
                                    'native_zero_ball_psbl': 'diagnostic ledger only; excluded from baseline charge',
                                    'wrong_position_and_brightness': 'diagnostic only; no coding claim'},
              'frames': frames,
              'clip': 'tracknet/game10/Clip3', 'labels': labels, 'results': results,
              'disk_guard_off': guard_off, 'receiver': receipt, 'calibration_archive_sha256': oracle.sha(cal),
              'calibration_wire_sha256': point['wire_sha256'], 'code_identity': dispatch['code']['source_sha256'],
              'gpu': torch.cuda.get_device_name(0), 'cuda_capability': list(torch.cuda.get_device_capability(0)),
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'decoder_sha256': DECODER_SHA,
              'source_archive_sha256': fleet.EXPECTED['archive'], 'contract_sha256': CONTRACT_SHA,
              'metric_models': {'lpips': quality.lpips_record(paths['lpips_backbone']),
                                'dists': quality.dists_record(paths['dists_backbone'])},
              'judge_identities': identities, 'seconds': time.monotonic() - started}
    (publish / 'controls.json').write_text(json.dumps(result, allow_nan=False) + '\n')
    validate_result(result, frames)


def validate_result(report: dict[str, Any], frames: int) -> None:
    if report.get('frames') != frames or report.get('clip') != 'tracknet/game10/Clip3' or set(report.get('results', {})) != set(ARMS):
        raise ValueError('wrong control coverage')
    labels = [tuple(x) for x in report['labels']]
    if len(labels) != frames:
        raise ValueError('label coverage mismatch')
    for name, result in report['results'].items():
        if len(result['metrics']) != frames or [x['frame'] for x in result['metrics']] != list(range(frames)):
            raise ValueError('metric coverage mismatch')
        billed = (result['native_bytes'] if name == 'native' else result['wire_bytes']) if name in ('native', 'streak', 'disk') else None
        if result['billed_bytes'] != billed or result['coding_claim'] != (billed is not None):
            raise ValueError('coding-arm billing basis mismatch')
        if result['status'] != ('informative' if result['changed_frames'] else 'uninformative'):
            raise ValueError('inactive control mislabeled')
        for i, row in enumerate(result['metrics']):
            for key in ('frame_psnr', 'full_lpips', 'full_dists', 'crop_psnr', 'crop_lpips', 'crop_dists'):
                value = row[key]
                required = not key.startswith('crop') or labels[i][0] == 1
                if required and (not isinstance(value, (float, int)) or not math.isfinite(value)):
                    raise ValueError('missing or nonfinite metric')
                if not required and value is not None:
                    raise ValueError('unexpected crop for unlabelled frame')
        for judge in ('wasb', 'balltrack'):
            got = result['judges'][judge]
            preds = [(p['detected'], p['x'], p['y']) for p in got['rows']]
            if len(preds) != frames or got['scores'] != score_frames(labels, preds):
                raise ValueError('judge scoring/coverage mismatch')
    disk = report['results']['disk']
    native = report['results']['native']
    if report['disk_guard_off'] and disk['changed_frames'] != 0:
        raise ValueError('disk guard failed to turn off')
    if sum(x['sse'] for x in disk['metrics']) > sum(x['sse'] for x in native['metrics']):
        raise ValueError('disk violates pooled RGB PSNR nondecrease guard')



def cost_forecast(execution: dict[str, Any], clock: dict[str, Any], now: float, full_limit: float) -> float:
    elapsed = execution['seconds']
    if (execution.get('exit_code') != 0 or execution.get('timing_contaminated') is not False
            or isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed) or elapsed <= 0
            or not 0 < clock['started'] <= clock['phase_started'] <= now < clock['budget_end']):
        raise ValueError('invalid smoke execution or campaign clock')
    projected = elapsed * 3 * 1.5 + 20
    if projected > full_limit or projected + 20 > clock['budget_end'] - now:
        raise ValueError('full control cost forecast exceeds budget')
    return projected


def full188_forecast(seconds16: float, seconds48: float, current16: float, *, reserve: float) -> float:
    """Prospective bound; admission still requires SHA-bound pilot receipts and review."""
    values = (seconds16, seconds48, current16, reserve)
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
            or not 0 < seconds16 <= seconds48 or current16 <= 0 or reserve <= 0):
        raise ValueError('invalid pilot cost evidence')
    marginal = (seconds48 - seconds16) / 32
    fixed = seconds16 - 16 * marginal
    if fixed < 0:
        raise ValueError('negative fixed pilot cost')
    return 1.5 * (fixed + 188 * marginal) * max(1., current16 / seconds16) + reserve


def validate_smoke(stage: Path, output: Path, *, full_frames: int = 48,
                   calibration_sha: dict[str, str] | None = None, extra_sha: dict[str, str] | None = None) -> None:
    try:
        paths, spec, dispatch = bound_inputs(stage, full_frames=full_frames, calibration_sha=calibration_sha, extra_sha=extra_sha)
        with tarfile.open(stage / 'published.tar') as tar:
            f = tar.extractfile('publish/controls.json')
            if f is None:
                raise ValueError('missing controls report')
            report = json.load(f)
            wires = {}
            for name in ARMS:
                item = tar.extractfile('publish/' + name + '.psbl')
                if item is None:
                    raise ValueError('missing persisted PSBL')
                wires[name] = item.read()
        validate_result(report, 16)
        original_source, point, original_stream = check_calibration(members(paths['calibration_smoke']), 16)
        expected_labels = [(r['label_kind'], r['label_x'], r['label_y']) for r in original_source['rows']]
        if ([tuple(row) for row in report['labels']] != expected_labels
                or report['calibration_archive_sha256'] != oracle.sha(paths['calibration_smoke'])
                or report['code_identity'] != dispatch['code']['source_sha256']
                or report['contract_sha256'] != CONTRACT_SHA
                or report['decoder_sha256'] != DECODER_SHA
                or oracle.sha(Path(sys.prefix) / 'bin/dav1d') != DECODER_SHA):
            raise ValueError('calibration labels, runtime or provenance mismatch')
        for name, original_name in (('native', 'baseline'), ('streak', 'streak')):
            if report['results'][name]['judges'] != point[original_name]['judges']:
                raise ValueError('calibration judge output mismatch')
        for name, wire in wires.items():
            stream = ball_stream.unpack(wire)
            row = report['results'][name]
            if (stream.native != original_stream.native or len(wire) != row['wire_bytes']
                    or len(stream.ball_payload) != row['ball_bytes']
                    or hashlib.sha256(wire).hexdigest() != row['wire_sha256']):
                raise ValueError('persisted wire identity/accounting failed')
        if wires['streak'] != ball_stream.pack(original_stream):
            raise ValueError('streak is not the retained calibration wire')
        track, apps = ball.decode_frames(original_stream.ball_payload, 16, original_stream.width, original_stream.height)
        empty = ball.encode_frames([None] * 16, [None] * 16, original_stream.width, original_stream.height)
        expected_disk = (empty if report['disk_guard_off'] else ball.encode_frames(
            track, [None if a is None else replace(a, exposure=0) for a in apps],
            original_stream.width, original_stream.height))
        if (ball_stream.unpack(wires['native']).ball_payload != empty
                or ball_stream.unpack(wires['disk']).ball_payload != expected_disk):
            raise ValueError('native/disk control symbols do not follow the registered transformations')
        shifted = [None if p is None else ((p[0] + 64) % original_stream.width, p[1]) for p in track]
        bright = [None if a is None else replace(a, colour=(255, 255, 255)) for a in apps]
        expected_shift = ball.encode_frames(shifted, apps, original_stream.width, original_stream.height)
        expected_bright = ball.encode_frames(track, bright, original_stream.width, original_stream.height)
        if (ball_stream.unpack(wires['wrong_position']).ball_payload != expected_shift
                or ball_stream.unpack(wires['brightness']).ball_payload != expected_bright):
            raise ValueError('diagnostic symbols differ from the registered transformations')
        with zipfile.ZipFile(paths['archive']) as z:
            for row in original_source['rows']:
                if hashlib.sha256(z.read(row['member'])).hexdigest() != row['source_sha256']:
                    raise ValueError('original source JPEG identity mismatch')
        if report['calibration_wire_sha256'] != point['wire_sha256'] or report['receiver']['rendered_rgb_sha256'] != point['encoder_rendered_sha256']:
            raise ValueError('retained pixel parity failed')
        execution = json.loads((stage / 'execution.json').read_text())
        clock = dispatch['campaign_clock']
        now = time.monotonic()
        elapsed = execution['seconds']
        if full_frames == 188:
            if extra_sha is None:
                raise ValueError('pilot identities required')
            if (execution.get('exit_code') != 0 or execution.get('timing_contaminated') is not False
                    or isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed)
                    or elapsed <= 0 or execution.get('gpu_sharing') != {'mode': 'exclusive'}
                    or not 0 < clock['started'] <= clock['phase_started'] <= now < clock['budget_end']
                    or clock['budget_end'] - clock['started'] > spec['budget_seconds'] + 1):
                raise ValueError('invalid smoke execution or campaign clock')
            audit = json.loads(paths['cost_audit'].read_text())
            if (audit['job_id'] != '20261105T021434Z-3ec159d8'
                    or audit['stages']['smoke']['archive_sha256'] != extra_sha['pilot_smoke']
                    or audit['stages']['full']['archive_sha256'] != extra_sha['pilot_full']):
                raise ValueError('pilot provenance differs')
            with tarfile.open(paths['pilot_smoke']) as pilot:
                pilot_file = pilot.extractfile('publish/controls.json')
                if pilot_file is None:
                    raise ValueError('missing pilot smoke controls')
                prior = json.load(pilot_file)
            validate_result(prior, 16)
            for key in ('labels', 'results', 'disk_guard_off', 'calibration_archive_sha256', 'calibration_wire_sha256', 'metric_models'):
                if report[key] != prior[key]:
                    raise ValueError('16-frame control parity differs: ' + key)
            with tarfile.open(paths['pilot_full']) as pilot:
                pilot_file = pilot.extractfile('publish/controls.json')
                if pilot_file is None:
                    raise ValueError('missing pilot full controls')
                prior_full = json.load(pilot_file)
            validate_result(prior_full, 48)
            projected = full188_forecast(audit['stages']['smoke']['command_seconds'], audit['stages']['full']['command_seconds'], elapsed, reserve=40)
            if projected > spec['full']['seconds'] or projected + 10 > clock['budget_end'] - now:
                raise ValueError('188-frame control cost exceeds stage or campaign')
        else:
            projected = cost_forecast(execution, clock, now, spec['full']['seconds'])
        remaining = clock['budget_end'] - now
        result = {'passed': True, 'checks': {'five_arms': True, 'native_streak_parity': True,
                  'metrics_judges_coverage': True, 'cost_projection': True},
                  'projection': {'smoke_seconds': elapsed, 'full_seconds': projected, 'remaining_seconds': remaining}}
    except Exception as exc:
        output.write_text(json.dumps({'passed': False, 'checks': {'controls': False}, 'error': f'{type(exc).__name__}: {exc}'}) + '\n')
        raise
    output.write_text(json.dumps(result) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--validate-smoke', action='store_true')
    args = parser.parse_args()
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.validate_smoke:
        validate_smoke(stage, Path(os.environ['PS_VALIDATION_PATH']))
    elif args.frames in (16, 48):
        run(stage, Path(os.environ['PS_SCRATCH_DIR']), args.frames)
    else:
        parser.error('16 or 48 frames required for cost pilot')


if __name__ == '__main__':
    main()

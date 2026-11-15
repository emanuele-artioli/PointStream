"""Fresh automatic BALL sender and historic 48-frame parity job.

The generic sender consumes source RGB and six ordered decoded native bases.
It uses no ground-truth coordinates. The fleet adapter compares a full replay
with immutable historical packets; a prefix smoke never proves parity.
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import ball_roi_control as common, k5b1b, k5c_independent_ball as old
from experiments.tennis import k5b_trackers as trackers, k5d_encoder, k5d_source_oracle, paid_patch_profile as paid, windows
from src.pointstream import ball, ball_stream, tracked_patch

MODEL_CODE = '5bec7ca535d2562ce0252a45f3dded0cfd5897571a6e29a10247a87884d961ff'
MODEL_TREE = 'f7a5c1c6284365f84c324d62169186ef1312c4d921bc0df5dbacd16ef5361964'
MODEL_WEIGHT = 'b052479e6229d840a1a7f42dbc83a6b752a53ef8bc5841a4b957acd519563fec'
POINTS = old.POINTS


def derive(source: np.ndarray, bases: list[tuple[tuple[str, int], np.ndarray]], fps: Fraction,
           tracker: Any) -> dict[str, Any]:
    """Run historical tracker, six-base fit, global guard and BALL round trip.

    Caller owns device/resource validation. The concrete BallTrack wrapper uses
    1280x720; generic geometry here is for explicitly compatible tracker input.
    """
    if not isinstance(fps, Fraction) or fps <= 0:
        raise ValueError('positive exact frame rate required')
    if (source.ndim != 4 or source.dtype != np.uint8 or source.shape[-1] != 3 or len(source) < 1
            or [key for key, _ in bases] != list(POINTS)
            or any(base.shape != source.shape or base.dtype != np.uint8 for _, base in bases)):
        raise ValueError('source and six ordered decoded RGB bases must match')
    n, h, w, _ = source.shape
    started = time.monotonic()
    predicted = tracker.run(source)
    if len(predicted) != n:
        raise ValueError('tracker frame count mismatch')
    start = k5b1b.track_of(predicted, (h, w))
    tracked_seconds = time.monotonic() - started
    fit_start = time.monotonic()
    fits = old.fit_streak(start, source, [frames for _, frames in bases], (h, w))
    fit_seconds = time.monotonic() - fit_start
    if len(fits) != len(POINTS):
        raise ValueError('fit did not produce every base point')
    rows = []
    for (key, decoded), fit in zip(bases, fits):
        track, appearances = fit['track'], fit['appearances']
        if len(track) != n or len(appearances) != n:
            raise ValueError('fit length mismatch')
        guard = k5d_encoder.pooled_psnr(source, old.render_frames(decoded, track, appearances)) < k5d_encoder.pooled_psnr(source, decoded)
        if guard:
            track, appearances = [None] * n, [None] * n
        payload = ball.encode_frames(track, appearances, w, h)
        quantized, decoded_appearances = ball.decode_frames(payload, n, w, h)
        if quantized != track or decoded_appearances != appearances:
            raise ValueError('BALL quantization round trip mismatch')
        centres = [list(p) if p is not None else None for p in quantized]
        origins = [tracked_patch.origin(p, w, h) for p in centres]
        rows.append({'codec': key[0], 'point': key[1], 'centres': centres, 'origins': origins,
                     'ball_payload_sha256': common.digest(payload), 'ball_payload_hex': payload.hex(),
                     'guard_disabled': guard, 'fit_delta_se': fit['delta_se']})
    return {'frames': n, 'size': [w, h], 'fps': [fps.numerator, fps.denominator], 'points': rows,
            'initial_track': [list(p) if p is not None else None for p in start],
            'timing': {'source_track_wall_seconds': tracked_seconds, 'six_base_fit_wall_seconds': fit_seconds}}


def extract_inputs(producer: Path, work: Path, frames: int) -> tuple[dict[str, Any], dict[str, Any], list[tuple[tuple[str, int], np.ndarray]], dict[tuple[str, int], bytes]]:
    """Copy only authenticated original YUV bases and three historical PSBLs."""
    qualification, pilot, _ = common.producer_metadata(producer)
    expected_points = {(p['codec'], p['point']): p for p in pilot['points']}
    if set(expected_points) != set(POINTS) or frames not in (16, 48):
        raise ValueError('historical point matrix mismatch')
    work.mkdir(parents=True, exist_ok=False)
    bases = []
    packets: dict[tuple[str, int], bytes] = {}
    with tarfile.open(producer) as tar:
        checkpoint_file = tar.extractfile('publish/checkpoint.json')
        if checkpoint_file is None:
            raise ValueError('checkpoint missing')
        checkpoint_bytes = checkpoint_file.read()
        if common.digest(checkpoint_bytes) != common.CHECKPOINT:
            raise ValueError('checkpoint identity mismatch')
        checkpoint = json.loads(checkpoint_bytes)
        members = tar.getmembers()
        for key in POINTS:
            kind, point = key
            rel = f'{kind}-{point}/receiver-evidence/native.yuv'
            name = 'publish/' + rel
            matches = [m for m in members if m.name == name and m.isfile()]
            if len(matches) != 1 or matches[0].size != 48 * common.FRAME_BYTES:
                raise ValueError('missing/ambiguous historical native YUV')
            stream = tar.extractfile(matches[0])
            if stream is None:
                raise ValueError('historical native YUV unreadable')
            dst = work / f'{kind}-{point}.yuv'
            import hashlib
            digest = hashlib.sha256()
            with dst.open('xb') as output:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
                    if frames == 48:
                        output.write(chunk)
                    elif output.tell() < frames * common.FRAME_BYTES:
                        output.write(chunk[:frames * common.FRAME_BYTES - output.tell()])
            measured = digest.hexdigest()
            if (measured != checkpoint['files'][rel] or measured != expected_points[key]['encoder_decoded_sha256']
                    or dst.stat().st_size != frames * common.FRAME_BYTES):
                raise ValueError('historical native decoded YUV identity mismatch')
            rgb = windows.rgb_frames(dst, common.W, common.H, frames, work / f'{kind}-{point}.npy')
            bases.append((key, rgb))
            if kind == 'svt':
                psbl_rel = f'{kind}-{point}/stream.psbl'
                matches = [m for m in members if m.name == 'publish/' + psbl_rel and m.isfile()]
                if len(matches) != 1:
                    raise ValueError('missing/ambiguous historical PSBL')
                handle = tar.extractfile(matches[0])
                if handle is None:
                    raise ValueError('historical PSBL unreadable')
                data = handle.read()
                if (common.digest(data) != checkpoint['files'][psbl_rel]
                        or common.digest(data) != expected_points[key]['wire_sha256']):
                    raise ValueError('historical PSBL identity mismatch')
                packets[key] = data
    return qualification, pilot, bases, packets


def parity(actual: dict[str, Any], packets: dict[tuple[str, int], bytes], oracle_receipt: dict[str, Any]) -> dict[str, Any]:
    if actual['frames'] == 16:
        return {'applicable': False, 'passed': None, 'reason': 'prefix tracker median and temporal context differ'}
    if actual['frames'] != 48 or [r['crf'] for r in oracle_receipt['points']] != [48, 55, 62]:
        raise ValueError('full48 and accepted oracle required')
    checks = {}
    for row in actual['points']:
        if row['codec'] != 'svt':
            continue
        key = (row['codec'], row['point'])
        stream = ball_stream.unpack(packets[key])
        track, _ = ball.decode_frames(stream.ball_payload, 48, common.W, common.H)
        centres = [list(p) if p is not None else None for p in track]
        saved = next(p for p in oracle_receipt['points'] if p['crf'] == row['point'])
        pixel_centres = [p['position'] for p in saved['positions']]
        pixel_origins = [[p['left'], p['top']] if p['present'] else None for p in saved['positions']]
        checks[str(row['point'])] = {'historical_centres_visibility': row['centres'] == centres,
                                    'historical_payload': row['ball_payload_hex'] == stream.ball_payload.hex(),
                                    'oracle_centres': row['centres'] == pixel_centres,
                                    'oracle_quantized_origins': row['origins'] == pixel_origins}
    if set(checks) != {'48', '55', '62'}:
        raise ValueError('missing SVT parity point')
    return {'applicable': True, 'checks': checks, 'passed': all(all(c.values()) for c in checks.values())}


def validate_result(result: dict[str, Any], packets: dict[tuple[str, int], bytes], oracle_receipt: dict[str, Any]) -> dict[str, bool]:
    frames = result['frames']
    if frames not in (16, 48) or [tuple((p['codec'], p['point'])) for p in result['points']] != list(POINTS):
        raise ValueError('invalid sender result span/matrix')
    checks = {'source_identity': result['source_archive_sha256'] == common.ARCHIVE,
              'producer_identity': result['producer_sha256'] == common.PRODUCER,
              'checkpoint_identity': result['checkpoint_sha256'] == common.CHECKPOINT,
              'model_code': result['model_code_archive_sha256'] == MODEL_CODE and result['model_code_tree_sha256'] == MODEL_TREE,
              'model_weight': result['model_weight_sha256'] == MODEL_WEIGHT,
              'gpu': (result['gpu']['device_count'] == 1 and result['gpu']['capability'] == [8, 9]
                      and result['gpu']['model_on_cuda'] is True and result['gpu']['torch'] == '2.10.0+cu128'
                      and 0 < result['gpu']['peak_reserved_bytes'] <= 8000 * 1024**2),
              'fps': result['fps'] == [30, 1], 'size': result['size'] == [common.W, common.H],
              'track_count': len(result['initial_track']) == frames}
    for row in result['points']:
        key = row['codec'] + '-' + str(row['point'])
        payload = bytes.fromhex(row['ball_payload_hex'])
        track, _ = ball.decode_frames(payload, frames, common.W, common.H)
        centres = [list(p) if p is not None else None for p in track]
        checks[key + ':quantized'] = (common.digest(payload) == row['ball_payload_sha256']
            and centres == row['centres'] and row['origins'] == [tracked_patch.origin(p, common.W, common.H) for p in centres]
            and len(centres) == frames)
    recalculated = parity(result, packets, oracle_receipt)
    checks['parity'] = result['parity'] == recalculated and (not recalculated['applicable'] or recalculated['passed'])
    checks['prefix_not_promoted'] = result['promotable'] is (frames == 48 and recalculated['passed'] is True)
    return checks


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if len(os.sched_getaffinity(0)) != 1 or os.environ.get('PS_CPU_ALLOWANCE') != '1':
        raise ValueError('one CPU and allowance required')
    if (torch.cuda.device_count() != 1 or tuple(torch.cuda.get_device_capability(0)) != (8, 9)
            or torch.__version__ != '2.10.0+cu128' or torch.version.cuda != '12.8'):
        raise ValueError('requires historical Ada and frozen Torch CUDA runtime')
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8000 * 1024**2 / total_memory), 0)
    stage = Path(os.environ['PS_STAGE_DIR'])
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if dispatch['staged_environment']['sha256'] != isolated_runtime():
        raise ValueError('packed runtime identity mismatch')
    if k5d_source_oracle.tree_sha(Path(args.model_code)) != MODEL_TREE:
        raise ValueError('BallTrack code tree differs')
    common.require_hash(Path(args.model_weight), MODEL_WEIGHT)
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    work = scratch / 'fresh-sender-private'
    work.mkdir(exist_ok=False)
    start, cpu = time.monotonic(), common.cpu_seconds()
    oracle_receipt, oracle_bytes, oracle_stage = paid.authenticated_oracle(Path(args.pixel_archive), Path(args.pixel_stage), work / 'oracle')
    qualification, pilot, bases, packets = extract_inputs(Path(args.producer), work / 'historical-bases', args.frames)
    if qualification['frames'] != 48 or pilot['frames'] != 48 or pilot['clip'] != 'tracknet/game10/Clip3':
        raise ValueError('incorrect historical source scope')
    source = common.source_frames(Path(args.archive), qualification)[:args.frames]
    tracker = trackers.BallTrack(Path(args.model_code), Path(args.model_weight))
    model_on_cuda = next(tracker.net.parameters()).device.type == 'cuda'
    if not model_on_cuda:
        raise ValueError('BallTrack did not load on CUDA')
    result = derive(source, bases, Fraction(30, 1), tracker)
    result.update({'schema': 1, 'scope': 'fresh historical sender parity; no new quality or deployment claim',
                   'source_archive_sha256': common.ARCHIVE, 'producer_sha256': common.PRODUCER,
                   'checkpoint_sha256': common.CHECKPOINT, 'source_qualification_sha256': common.QUALIFICATION,
                   'model_code_archive_sha256': MODEL_CODE, 'model_code_tree_sha256': MODEL_TREE,
                   'model_weight_sha256': MODEL_WEIGHT, 'pixel_archive_sha256': paid.PIXEL_ARCHIVE,
                   'pixel_stage_sha256': paid.PIXEL_STAGE,
                   'execution_identity': {'source_sha256': dispatch['code']['source_sha256'],
                       'snapshot_sha256': dispatch['code']['snapshot_sha256'], 'environment_sha256': isolated_runtime()},
                   'gpu': {'name': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
                           'device_count': torch.cuda.device_count(), 'model_on_cuda': model_on_cuda,
                           'torch': torch.__version__, 'cuda': torch.version.cuda,
                           'peak_reserved_bytes': torch.cuda.max_memory_reserved()},
                   'sender_source_sha256': common.k5d_source_oracle.sha(Path(__file__))})
    comparison = parity(result, packets, oracle_receipt)
    result['parity'] = comparison
    result['promotable'] = args.frames == 48 and comparison['passed'] is True
    checks = validate_result(result, packets, oracle_receipt)
    if not all(checks.values()):
        raise ValueError('sender output or full historic parity failed')
    pub = work / 'publication'
    pub.mkdir()
    (pub / 'oracle-receipt.json').write_bytes(oracle_bytes)
    (pub / 'oracle-stage.json').write_bytes(oracle_stage)
    for key, data in packets.items():
        (pub / f'{key[0]}-{key[1]}.psbl').write_bytes(data)
    paid.write_json(pub / 'sender.json', result)
    archive = publish(pub, stage / 'published.tar')
    cost = common.cost_record(time.monotonic() - start, common.cpu_seconds() - cpu, args.frames)
    record = {'schema': 1, 'frames': args.frames, 'archive': archive, 'cost': cost, 'checks': checks,
              'scope': 'historical source parity; shared GPU timing not throughput evidence'}
    paid.write_json(stage / 'fresh-sender-stage.json', record)
    return record


def isolated_runtime() -> str:
    return '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'


def members() -> list[str]:
    return ['sender.json', 'oracle-receipt.json', 'oracle-stage.json', *[f'svt-{p}.psbl' for p in (48, 55, 62)]]


def publish(root: Path, archive: Path) -> dict[str, Any]:
    if {p.name for p in root.iterdir()} != set(members()):
        raise ValueError('sender publication inventory mismatch')
    hashes = {}
    with tarfile.open(archive, 'x') as tar:
        for name in members():
            path = root / name
            if not path.is_file() or path.is_symlink():
                raise ValueError('invalid sender publication member')
            hashes[name] = common.k5d_source_oracle.sha(path)
            tar.add(path, arcname='publish/' + name, recursive=False)
    return {'sha256': common.k5d_source_oracle.sha(archive), 'bytes': archive.stat().st_size, 'members': hashes}


def binding_checks(record: dict[str, Any], sender: dict[str, Any], dispatch: dict[str, Any], phase: str) -> dict[str, bool]:
    expected = {'smoke': 16, 'full': 48}.get(phase)
    identity = {'source_sha256': dispatch['code']['source_sha256'],
                'snapshot_sha256': dispatch['code']['snapshot_sha256'],
                'environment_sha256': dispatch['staged_environment']['sha256']}
    return {'stage_frame_binding': expected is not None and record['frames'] == sender['frames'] == expected,
            'full_promotable': phase != 'full' or sender['promotable'] is True,
            'dispatch_identity': sender['execution_identity'] == identity and identity['environment_sha256'] == isolated_runtime()}


def validate(stage: Path, temp: Path) -> dict[str, Any]:
    record = json.loads((stage / 'fresh-sender-stage.json').read_text())
    archive = stage / 'published.tar'
    common.require_hash(archive, record['archive']['sha256'])
    if archive.stat().st_size != record['archive']['bytes'] or set(record['archive']['members']) != set(members()):
        raise ValueError('sender archive binding mismatch')
    with tempfile.TemporaryDirectory(prefix='fresh-sender-validation-', dir=temp) as folder:
        with tarfile.open(archive) as tar:
            entries = tar.getmembers()
            if (len(entries) != len(members()) or {m.name for m in entries} != {'publish/' + n for n in members()}
                    or any(not m.isfile() for m in entries)):
                raise ValueError('invalid sender publication')
            tar.extractall(folder, filter='data')
        root = Path(folder) / 'publish'
        for name, sha in record['archive']['members'].items():
            common.require_hash(root / name, sha)
        common.require_hash(root / 'oracle-stage.json', paid.PIXEL_STAGE)
        oracle_stage = json.loads((root / 'oracle-stage.json').read_text())
        if oracle_stage['archive']['sha256'] != paid.PIXEL_ARCHIVE:
            raise ValueError('oracle source archive identity mismatch')
        common.require_hash(root / 'oracle-receipt.json', oracle_stage['archive']['members']['receipt.json'])
        oracle_receipt = json.loads((root / 'oracle-receipt.json').read_text())
        packets = {('svt', p): (root / f'svt-{p}.psbl').read_bytes() for p in (48, 55, 62)}
        for key, data in packets.items():
            expected = next(p for p in oracle_receipt['points'] if p['crf'] == key[1])['wire_sha256']
            if common.digest(data) != expected:
                raise ValueError('original PSBL identity mismatch')
        sender = json.loads((root / 'sender.json').read_text())
        checks = validate_result(sender, packets, oracle_receipt)
        phase = os.environ['PS_STAGE']
        checks.update(binding_checks(record, sender, json.loads((stage / 'dispatch.json').read_text()), phase))
        checks['stage_directory'] = stage.name == phase
    cost = record['cost']
    checks['body_cost'] = cost == common.cost_record(cost['wall_seconds'], cost['cpu_seconds'], record['frames']) and cost['passed']
    if (stage / 'execution.json').exists():
        execution = json.loads((stage / 'execution.json').read_text())
        checks['whole_command_cost'] = execution['exit_code'] == 0 and common.cost_record(execution['seconds'], 0, record['frames'])['passed']
    return {'passed': all(checks.values()), 'checks': checks,
            'scope': 'historic sender parity; 16-frame smoke is cost/correctness only'}


def spec(deadline: str) -> dict[str, Any]:
    data = '/home/itec/emanuele/Datasets/pointstream-data'
    paths = [
        ('producer', data + '/jobs/fleet/inbox/20261113T210157Z-41ee26e3/full/published.tar', common.PRODUCER, False),
        ('archive', data + '/tennis/k5b-2026-10-20/inputs/tracknet-Dataset.zip', common.ARCHIVE, False),
        ('pixel_archive', data + '/jobs/fleet/inbox/20261115T020136Z-f5dd76c2/full/published.tar', paid.PIXEL_ARCHIVE, False),
        ('pixel_stage', data + '/jobs/fleet/inbox/20261115T020136Z-f5dd76c2/full/pixel-oracle-stage.json', paid.PIXEL_STAGE, False),
        ('model_code', data + '/tennis/k5b-2026-10-20/inputs/RacketVision-c44af2a.tar', MODEL_CODE, True),
        ('model_weight', data + '/tennis/k5b-2026-10-20/inputs/balltrack_best.state_dict.pt', MODEL_WEIGHT, False),
    ]
    items = [{'name': name, 'path': path, 'sha256': sha, 'extract': extract} for name, path, sha, extract in paths]
    arguments = ['--fleet', '--frames', '{frames}']
    for name, _, _, _ in paths:
        arguments += ['--' + name.replace('_', '-'), '{staged:' + name + '}']
    return {'schema': 1, 'device': 'gpu', 'hosts': ['gpu5', 'gpu6'], 'gpu_models': ['RTX 6000 Ada'],
            'gpu_memory_mib': 8000, 'cpu_threads': 1, 'single_cpu_campaign': True, 'cpu_affinity_index': 0,
            'gpu_sharing': {'mode': 'shared', 'margin_mib': 8192,
                            'basis': 'Exact sender parity depends on authenticated pixels and outputs; shared timing is not throughput.'},
            'contention': {'policy': 'continue', 'basis': 'Shared Ada parity; no timing claim.', 'resume_attempts': 0},
            'local_storage_gib': 20, 'entrypoint': ['-m', 'experiments.tennis.fresh_ball_sender'],
            'arguments': arguments, 'staged_inputs': items,
            'inputs': [{'path': item['path'], 'sha256': item['sha256']} for item in items],
            'environment': {'path': data + '/environments/pointstream-20261009T214002Z.tar.gz', 'sha256': isolated_runtime()},
            'verify_staged_inputs': True, 'required_commands': [], 'scale': {'frames': {'smoke': 16, 'full': 48}},
            'smoke': {'seconds': 450, 'representative_basis': 'Actual historical decoded backgrounds and sender; prefix16 is cost/correctness only because full-window median differs.'},
            'full': {'seconds': 900}, 'validator': ['{python}', '-m', 'experiments.tennis.fresh_ball_sender', '--validate'],
            'validator_seconds': 60, 'budget_seconds': 1530, 'stall_seconds': 900, 'deadline': deadline}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fleet', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--deadline')
    parser.add_argument('--frames', type=int, choices=(16, 48))
    for name in ('producer', 'archive', 'pixel-archive', 'pixel-stage', 'model-code', 'model-weight'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.spec:
        if not args.deadline:
            parser.error('--spec requires --deadline')
        paid.write_json(args.spec, spec(args.deadline))
        return
    if args.validate:
        result = validate(Path(os.environ['PS_STAGE_DIR']), Path(os.environ['TMPDIR']))
        paid.write_json(Path(os.environ['PS_VALIDATION_PATH']), result)
        print(json.dumps(result))
        if not result['passed']:
            raise SystemExit(1)
        return
    if not args.fleet or args.frames is None or any(getattr(args, k) is None for k in
            ('producer', 'archive', 'pixel_archive', 'pixel_stage', 'model_code', 'model_weight')):
        parser.error('fleet requires all staged inputs and --frames')
    result = run(args)
    print(json.dumps({'frames': args.frames, 'cost': result['cost']}))
    if not result['cost']['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

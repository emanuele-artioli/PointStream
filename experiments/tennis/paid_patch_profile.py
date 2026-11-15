"""CPU adapters for the paid PTP1 tracked-pixel profile; no job dispatch.

Tennis binds prior transmitted BALL positions and compares all full-clip pixels
against the accepted coded-pixel oracle. Egocentric mode is a full-range native
fallback correctness check; it makes no perceptual-quality claim.
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

from experiments.tennis import ball_pixel_oracle as oracle, ball_roi_control as common, k5d_encoder
from src.codecs import svtav1
from src.pointstream import codec, tracked_patch

PIXEL_ARCHIVE = 'a06f2fca3a2bcad45123d5e06a233a6f2dadbcc91c96060658f759df81452739'
PIXEL_STAGE = '5b90815b56bcef79bbf9f77e955ac22ae15aa236f0aa2195b79600d825357259'
EGO_ARCHIVE = '766265d350d67f6ae25d7fe7670f28b2e671d843110306f651037dc92a288b1e'
EGO_SOURCE = '15cc7ab12ce80c6a8775bc645e5b00d30f42767ec72ec7462931512e8189d94f'
DATA = '/home/itec/emanuele/Datasets/pointstream-data'


def authenticated_oracle(archive: Path, record: Path, work: Path) -> tuple[dict[str, Any], bytes, bytes]:
    common.require_hash(record, PIXEL_STAGE)
    stage = json.loads(record.read_text())
    if stage['archive']['sha256'] != PIXEL_ARCHIVE or stage['frames'] != 48 or not stage['cost']['passed']:
        raise ValueError('accepted oracle stage mismatch')
    root = oracle.unpack(archive, stage['archive'], work)
    data = (root / 'receipt.json').read_bytes()
    receipt = json.loads(data)
    if (receipt['frames'] != 48 or receipt['producer_sha256'] != common.PRODUCER
            or receipt['source_yuv_sha256'] != common.SOURCE_YUV
            or [p['crf'] for p in receipt['points']] != list(oracle.RATES)):
        raise ValueError('accepted oracle receipt mismatch')
    for point in receipt['points']:
        expected = oracle.EXPECTED_PACKETS[point['crf']]
        if (point['base_sha256'], point['wire_sha256'], point['base_bytes']) != expected:
            raise ValueError('oracle packet binding mismatch')
        if len(point['positions']) != 48 or len(point['replay']['composite_frame_sha256']) != 48:
            raise ValueError('oracle position or pixel coverage mismatch')
    return receipt, data, record.read_bytes()


def ego_source(archive: Path, work: Path) -> tuple[Path, bytes]:
    common.require_hash(archive, EGO_ARCHIVE)
    with tarfile.open(archive) as tar:
        entries = tar.getmembers()
        expected = {'source.yuv', 'masks.npz', 'preparation.json'}
        if len(entries) != 3 or {m.name for m in entries} != expected or any(not m.isfile() for m in entries):
            raise ValueError('unexpected prepared input members')
        reader = tar.extractfile('source.yuv')
        metadata = tar.extractfile('preparation.json')
        if reader is None or metadata is None:
            raise ValueError('missing prepared source')
        source = work / 'ego-source.yuv'
        with source.open('xb') as out:
            shutil.copyfileobj(reader, out, length=1024 * 1024)
        preparation = metadata.read()
    common.require_hash(source, EGO_SOURCE)
    if source.stat().st_size != 1920 * 1080 * 3 // 2 * 64:
        raise ValueError('prepared source size mismatch')
    info = json.loads(preparation)
    if (info['outputs']['source.yuv']['sha256'] != EGO_SOURCE or info['width'] != 1920
            or info['height'] != 1080 or info['frames'] != 64 or Fraction(info['fps']) != 50
            or info['source_record']['full_range'] is not True):
        raise ValueError('prepared source metadata mismatch')
    return source, preparation


def tennis_gold(args: argparse.Namespace, work: Path) -> tuple[Path, dict[str, Any]]:
    qualification, pilot, _ = common.producer_metadata(Path(args.producer))
    source_rgb = common.source_frames(Path(args.archive), qualification)
    source = work / 'tennis-source.yuv'
    conversion = k5d_encoder.convert_rgb(args.ffmpeg, source_rgb, source, Fraction(30, 1))
    del source_rgb
    common.require_hash(source, common.SOURCE_YUV)
    if source.stat().st_size != 48 * common.FRAME_BYTES:
        raise ValueError('gold source length mismatch')
    common.expected_native_decode(pilot)
    control = work / 'control'
    control.mkdir()
    result = svtav1.code(source, control, width=1280, height=720, fps=Fraction(30, 1), frames=48,
                         crf=62, preset=4, threads=1, full_range=False,
                         encoder=args.encoder, decoder=args.decoder, timeout=600)
    common.require_hash(Path(result['stream']), common.NATIVE62)
    common.require_hash(Path(result['decoded']), common.NATIVE62_DECODE)
    return source, {'source_sha256': common.SOURCE_YUV, 'conversion_command': conversion,
                    'native62_sha256': common.NATIVE62, 'native62_yuv_sha256': common.NATIVE62_DECODE,
                    'native62_control': result}


def prefix(source: Path, output: Path, frame_bytes: int, frames: int, full: int) -> Path:
    if frames == full:
        return source
    if frames != 16:
        raise ValueError('unsupported smoke span')
    remaining = frame_bytes * frames
    with source.open('rb') as reader, output.open('xb') as writer:
        while remaining:
            block = reader.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError('truncated prefix source')
            writer.write(block)
            remaining -= len(block)
    return output


def config_for(domain: str, frames: int, point: dict[str, Any] | None) -> dict[str, Any]:
    if domain == 'tennis':
        if point is None or len(point['positions']) != 48 or frames not in (16, 48):
            raise ValueError('invalid tennis oracle positions')
        return {'profile': 'tracked-pixel-v1', 'width': 1280, 'height': 720, 'fps': [30, 1], 'frames': frames,
                'full_range': False, 'crf': point['crf'], 'preset': 4, 'input_format': 'yuv420p',
                'positions': [p['position'] for p in point['positions'][:frames]],
                'position_provenance': {'kind': 'retained-automatic-PSBL', 'wire_sha256': point['wire_sha256'],
                                        'source_archive_sha256': common.ARCHIVE, 'oracle_archive_sha256': PIXEL_ARCHIVE}}
    if domain == 'ego' and point is None and frames in (16, 64):
        return {'profile': 'tracked-pixel-v1', 'width': 1920, 'height': 1080, 'fps': [50, 1], 'frames': frames,
                'full_range': True, 'crf': 50, 'preset': 8, 'input_format': 'yuv420p',
                'positions': [None] * frames,
                'position_provenance': {'kind': 'all-absent-native-fallback', 'source_archive_sha256': EGO_ARCHIVE}}
    raise ValueError('unsupported domain or stage')


def write_json(path: Path, value: Any) -> None:
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def parity(point: dict[str, Any], base: bytes, patch: bytes, decoded: dict[str, Any], frames: int) -> dict[str, Any]:
    if frames == 16:
        return {'applicable': False, 'passed': None, 'reason': 'prefix encoding has different temporal context'}
    if frames != 48:
        raise ValueError('invalid tennis parity count')
    expected = point['replay']
    checks = {'native_stream': common.digest(base) == point['base_sha256'],
              'patch_stream': common.digest(patch) == point['patch_sha256'],
              'all_rgb_frames': decoded['rgb_frame_sha256'] == expected['composite_frame_sha256'],
              'aggregate_rgb': decoded['decoded_sha256'] == expected['composite_rgb_sha256']}
    return {'applicable': True, 'passed': all(checks.values()), 'checks': checks}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if os.sched_getaffinity(0) != {0} or os.environ.get('PS_CPU_ALLOWANCE') != '1':
        raise ValueError('requires one CPU at affinity index 0')
    for key, sha in common.TOOLS.items():
        common.require_hash(Path(getattr(args, key)), sha)
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    work = scratch / 'paid-private'
    work.mkdir(exist_ok=False)
    pub = work / 'publication'
    pub.mkdir()
    if args.domain == 'tennis':
        if args.frames not in (16, 48):
            raise ValueError('invalid tennis span')
        oracle_receipt, receipt_bytes, stage_bytes = authenticated_oracle(Path(args.pixel_archive), Path(args.pixel_stage), work / 'oracle')
        (pub / 'oracle-receipt.json').write_bytes(receipt_bytes)
        (pub / 'oracle-stage.json').write_bytes(stage_bytes)
        source, source_info = tennis_gold(args, work)
        source = prefix(source, work / 'prefix.yuv', common.FRAME_BYTES, args.frames, 48)
        points: list[dict[str, Any] | None] = list(oracle_receipt['points'])
    elif args.domain == 'ego':
        if args.frames not in (16, 64):
            raise ValueError('invalid ego span')
        source, metadata = ego_source(Path(args.prepared), work)
        (pub / 'preparation.json').write_bytes(metadata)
        source_info = {'source_sha256': EGO_SOURCE, 'prepared_archive_sha256': EGO_ARCHIVE}
        source = prefix(source, work / 'prefix.yuv', 1920 * 1080 * 3 // 2, args.frames, 64)
        points = [None]
    else:
        raise ValueError('invalid domain')
    records = []
    for point in points:
        config = config_for(args.domain, args.frames, point)
        label = str(point['crf']) if point is not None else 'fallback'
        packet = pub / f'{label}.ptp'
        encoding = tracked_patch.encode(source, packet, config, scratch=work, encoder=args.encoder, decoder=args.decoder)
        rendered = work / f'{label}.rgb'
        decoding = tracked_patch.decode(packet, rendered, scratch=work, decoder=args.decoder)
        inspection = tracked_patch.inspect(codec.read_packet(packet))
        if (decoding['decoded_sha256'] != encoding['decoded_sha256']
                or decoding['rgb_frame_sha256'] != encoding['rgb_frame_sha256']
                or len(decoding['rgb_frame_sha256']) != args.frames):
            raise ValueError('public receiver replay differs from encoder receipt')
        _, base, patch = tracked_patch.unpack(packet.read_bytes())
        result = parity(point, base, patch, decoding, args.frames) if point else {
            'applicable': False, 'passed': None, 'reason': 'egocentric native fallback has no quality reuse'}
        if result['applicable'] and not result['passed']:
            raise ValueError('full paid output differs from accepted pixel oracle')
        if args.domain == 'ego' and (patch or inspection['metadata']['native']['full_range'] is not True):
            raise ValueError('egocentric fallback or color range mismatch')
        write_json(pub / f'{label}.json', {'config': config, 'encode': encoding, 'decode': decoding,
                                          'inspect': inspection, 'oracle_parity': result})
        records.append({'label': label, 'packet_sha256': common.k5d_source_oracle.sha(packet),
                        'packet_bytes': packet.stat().st_size, 'oracle_parity': result})
    result = {'schema': 1, 'domain': args.domain, 'frames': args.frames, 'source': source_info,
              'points': records, 'frozen_runtime_tools': common.TOOLS,
              'score_reuse_eligible': args.domain == 'tennis' and args.frames == 48,
              'scope': 'functional PTP1; source denial and tracker generation cost not established'}
    write_json(pub / 'receipt.json', result)
    return result


def publication_members(domain: str) -> list[str]:
    labels = ('48', '55', '62') if domain == 'tennis' else ('fallback',)
    provenance = ['oracle-receipt.json', 'oracle-stage.json'] if domain == 'tennis' else ['preparation.json']
    return ['receipt.json', *provenance, *[f'{label}.{ext}' for label in labels for ext in ('ptp', 'json')]]


def publish(root: Path, archive: Path, domain: str) -> dict[str, Any]:
    names = publication_members(domain)
    if {p.name for p in root.iterdir()} != set(names):
        raise ValueError('publication inventory mismatch')
    hashes = {}
    with tarfile.open(archive, 'x') as tar:
        for name in names:
            path = root / name
            if not path.is_file() or path.is_symlink():
                raise ValueError('publication member is not a regular file')
            hashes[name] = common.k5d_source_oracle.sha(path)
            tar.add(path, arcname='publish/' + name, recursive=False)
    return {'sha256': common.k5d_source_oracle.sha(archive), 'bytes': archive.stat().st_size, 'members': hashes}


def extract(archive: Path, binding: dict[str, Any], domain: str, work: Path) -> Path:
    common.require_hash(archive, binding['sha256'])
    names = publication_members(domain)
    if archive.stat().st_size != binding['bytes'] or set(binding['members']) != set(names):
        raise ValueError('archive binding mismatch')
    work.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as tar:
        entries = tar.getmembers()
        if (len(entries) != len(names) or {e.name for e in entries} != {'publish/' + name for name in names}
                or any(not e.isfile() for e in entries)):
            raise ValueError('archive has unexpected members')
        tar.extractall(work, filter='data')
    root = work / 'publish'
    for name, expected in binding['members'].items():
        common.require_hash(root / name, expected)
    return root


def validate_publication(root: Path, decoder: str, work: Path) -> dict[str, bool]:
    common.require_hash(Path(decoder), common.TOOLS['decoder'])
    receipt = json.loads((root / 'receipt.json').read_text())
    domain, frames = receipt['domain'], receipt['frames']
    if domain not in ('tennis', 'ego') or frames not in ((16, 48) if domain == 'tennis' else (16, 64)):
        raise ValueError('unregistered domain/frame span')
    checks: dict[str, bool] = {'tools': receipt['frozen_runtime_tools'] == common.TOOLS,
        'promotion': receipt['score_reuse_eligible'] == (domain == 'tennis' and frames == 48)}
    if domain == 'tennis':
        stage_path = root / 'oracle-stage.json'
        common.require_hash(stage_path, PIXEL_STAGE)
        stage = json.loads(stage_path.read_text())
        checks['oracle_archive'] = stage['archive']['sha256'] == PIXEL_ARCHIVE and stage['frames'] == 48 and stage['cost']['passed']
        oracle_path = root / 'oracle-receipt.json'
        common.require_hash(oracle_path, stage['archive']['members']['receipt.json'])
        oracle_data = json.loads(oracle_path.read_text())
        checks['gold_source'] = (receipt['source']['source_sha256'] == common.SOURCE_YUV
                                 and receipt['source']['native62_sha256'] == common.NATIVE62
                                 and receipt['source']['native62_yuv_sha256'] == common.NATIVE62_DECODE)
        expected_points: list[dict[str, Any] | None] = list(oracle_data['points'])
        if [p['crf'] for p in oracle_data['points']] != list(oracle.RATES):
            raise ValueError('oracle point matrix mismatch')
    else:
        checks['prepared_source'] = (receipt['source']['source_sha256'] == EGO_SOURCE
                                     and receipt['source']['prepared_archive_sha256'] == EGO_ARCHIVE)
        expected_points = [None]
    if len(receipt['points']) != len(expected_points):
        raise ValueError('paid point count mismatch')
    work.mkdir(parents=True, exist_ok=False)
    for saved, point in zip(receipt['points'], expected_points):
        label = str(point['crf']) if point else 'fallback'
        data = (root / f'{label}.ptp').read_bytes()
        meta, base, patch = tracked_patch.unpack(data)
        info = json.loads((root / f'{label}.json').read_text())
        config = config_for(domain, frames, point)
        positions = [tracked_patch.origin(pos, config['width'], config['height']) for pos in config['positions']]
        checks[f'{label}:configuration'] = (saved['label'] == label and info['config'] == config
            and meta['positions'] == positions and meta['native']['frames'] == frames
            and meta['native']['fps'] == config['fps'] and meta['native']['full_range'] == config['full_range']
            and meta['native']['width'] == config['width'] and meta['native']['height'] == config['height'])
        checks[f'{label}:packet'] = (saved['packet_sha256'] == common.digest(data) and saved['packet_bytes'] == len(data)
            and info['inspect'] == tracked_patch.inspect(data))
        decoded = tracked_patch.decode(root / f'{label}.ptp', work / f'{label}.rgb', scratch=work, decoder=decoder)
        checks[f'{label}:fresh_receiver'] = (decoded['decoded_sha256'] == info['decode']['decoded_sha256']
            and decoded['rgb_frame_sha256'] == info['decode']['rgb_frame_sha256']
            and len(decoded['rgb_frame_sha256']) == frames and decoded['output_format'] == 'rgb24')
        replay = parity(point, base, patch, decoded, frames) if point else {'applicable': False, 'passed': None,
                                                                            'reason': 'egocentric native fallback has no quality reuse'}
        checks[f'{label}:parity'] = replay == saved['oracle_parity'] == info['oracle_parity'] and (not replay['applicable'] or replay['passed'])
        if domain == 'ego':
            checks['ego_native_only'] = not patch and all(p is None for p in meta['positions']) and meta['native']['full_range'] is True
    return checks


def cost_record(wall: float, cpu: float, frames: int, domain: str) -> dict[str, Any]:
    target = 48 if domain == 'tennis' else 64 if domain == 'ego' else 0
    if wall <= 0 or cpu < 0 or frames not in (16, target) or not target:
        raise ValueError('invalid cost input')
    forecast = 1.5 * wall * target / frames + 30
    return {'frames': frames, 'target_frames': target, 'wall_seconds': wall, 'cpu_seconds': cpu,
            'forecast_seconds': forecast, 'full_cap_seconds': 900,
            'passed': forecast <= 900 if frames == 16 else wall <= 900,
            'basis': 'entrypoint body after imports includes packing and internal replay; smoke forecast 1.5 * body * target/16 + 30'}


def run_fleet(args: argparse.Namespace) -> dict[str, Any]:
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    start, cpu = time.monotonic(), common.cpu_seconds()
    prepare(args)
    work = scratch / 'paid-private'
    binding = publish(work / 'publication', stage / 'published.tar', args.domain)
    root = extract(stage / 'published.tar', binding, args.domain, work / 'internal-archive')
    checks = validate_publication(root, args.decoder, work / 'internal-replay')
    if not all(checks.values()):
        raise ValueError('internal archive or public receiver replay failed')
    cost = cost_record(time.monotonic() - start, common.cpu_seconds() - cpu, args.frames, args.domain)
    result = {'schema': 1, 'domain': args.domain, 'frames': args.frames, 'archive': binding,
              'cost': cost, 'internal_checks': checks}
    write_json(stage / 'paid-profile-stage.json', result)
    return result


def validate_fleet(stage: Path, decoder: str, temp: Path) -> dict[str, Any]:
    record = json.loads((stage / 'paid-profile-stage.json').read_text())
    with tempfile.TemporaryDirectory(prefix='ptp-validate-', dir=temp) as folder:
        root = extract(stage / 'published.tar', record['archive'], record['domain'], Path(folder) / 'archive')
        checks = validate_publication(root, decoder, Path(folder) / 'replay')
    cost = record['cost']
    checks['cost'] = cost == cost_record(cost['wall_seconds'], cost['cpu_seconds'], record['frames'], record['domain']) and cost['passed']
    checks['internal_replay'] = bool(record['internal_checks']) and all(record['internal_checks'].values())
    if (stage / 'execution.json').exists():
        execution = json.loads((stage / 'execution.json').read_text())
        checks['whole_command_cost'] = execution['exit_code'] == 0 and cost_record(execution['seconds'], 0, record['frames'], record['domain'])['passed']
    return {'passed': all(checks.values()), 'checks': checks,
            'scope': 'public packet, fresh native decoder and bounded cost; no OS source-denial claim'}


def spec(domain: str, deadline: str) -> dict[str, Any]:
    if domain == 'tennis':
        paths = [('producer', DATA + '/jobs/fleet/inbox/20261113T210157Z-41ee26e3/full/published.tar', common.PRODUCER),
                 ('archive', DATA + '/tennis/k5b-2026-10-20/inputs/tracknet-Dataset.zip', common.ARCHIVE),
                 ('pixel_archive', DATA + '/jobs/fleet/inbox/20261115T020136Z-f5dd76c2/full/published.tar', PIXEL_ARCHIVE),
                 ('pixel_stage', DATA + '/jobs/fleet/inbox/20261115T020136Z-f5dd76c2/full/pixel-oracle-stage.json', PIXEL_STAGE)]
        arguments = ['--producer', '{staged:producer}', '--archive', '{staged:archive}',
                     '--pixel-archive', '{staged:pixel_archive}', '--pixel-stage', '{staged:pixel_stage}']
        full = 48
        basis = 'Full48 source and native62 gold; three paid packets and fresh decodes for first16 with exact positions.'
    elif domain == 'ego':
        paths = [('prepared', DATA + '/jobs/fleet/inbox/20261114T233516Z-4223e78c/full/published.tar', EGO_ARCHIVE)]
        arguments = ['--prepared', '{staged:prepared}']
        full = 64
        basis = 'Accepted64 full-range source; first16 full pipeline, native fallback, public decode and frame verification.'
    else:
        raise ValueError('unknown domain')
    items = [{'name': name, 'path': path, 'sha256': sha, 'extract': False} for name, path, sha in paths]
    return {'schema': 1, 'device': 'cpu', 'hosts': ['gpu6'], 'gpu_models': [], 'gpu_memory_mib': 0,
            'cpu_threads': 1, 'single_cpu_campaign': True, 'cpu_affinity_index': 0, 'local_storage_gib': 20,
            'entrypoint': ['-m', 'experiments.tennis.paid_patch_profile'],
            'arguments': ['--fleet', '--domain', domain, *arguments, '--frames', '{frames}'],
            'staged_inputs': items, 'inputs': [{'path': item['path'], 'sha256': item['sha256']} for item in items],
            'environment': {'path': DATA + '/environments/pointstream-20261009T214002Z.tar.gz',
                            'sha256': '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'},
            'verify_staged_inputs': True, 'required_commands': ['ffmpeg'],
            'scale': {'frames': {'smoke': 16, 'full': full}}, 'smoke': {'seconds': 450, 'representative_basis': basis},
            'full': {'seconds': 900}, 'validator': ['{python}', '-m', 'experiments.tennis.paid_patch_profile', '--validate'],
            'validator_seconds': 60, 'budget_seconds': 1530, 'stall_seconds': 450, 'deadline': deadline,
            'contention': {'policy': 'stop', 'resume_attempts': 0}}



PIXEL_SCORE = '00e2885b59bed2a2a497c77a778abec29e0045280b0110e3b2bc23dba492ac16'


def ordinal_separation(candidate: list[tuple[float, float]], anchor: list[tuple[float, float]]) -> dict[str, Any]:
    """Strict endpoint separation on common rate support; quality is higher-is-better."""
    low = max(min(r for r, _ in candidate), min(r for r, _ in anchor))
    high = min(max(r for r, _ in candidate), max(r for r, _ in anchor))
    if high <= low:
        return {'status': 'no_common_rate_interval', 'common_rate_interval': None}
    c_low, c_high = min(q for _, q in candidate), max(q for _, q in candidate)
    a_low, a_high = min(q for _, q in anchor), max(q for _, q in anchor)
    relation = ('candidate_strictly_better' if c_low > a_high else
                'candidate_strictly_worse' if c_high < a_low else 'not_strictly_separated')
    return {'status': relation, 'common_rate_interval': [low, high],
            'candidate_quality_range': [c_low, c_high], 'anchor_quality_range': [a_low, a_high],
            'scope': 'strict observed quality-range separation on common sampled rate support; no BD-rate inferred'}


def reprice(score_path: Path, stage_path: Path, expected_stage_sha: str, archive: Path,
            decoder: str, temporary: Path) -> dict[str, Any]:
    """Reuse accepted pixel scores only after replaying the paid full48 archive."""
    from src.pointstream.evaluation import anchor_sensitivity

    common.require_hash(score_path, PIXEL_SCORE)
    common.require_hash(stage_path, expected_stage_sha)
    score = json.loads(score_path.read_text())
    stage = json.loads(stage_path.read_text())
    if (stage['domain'] != 'tennis' or stage['frames'] != 48 or not stage['cost']['passed']
            or not stage['internal_checks'] or not all(stage['internal_checks'].values())):
        raise ValueError('paid source requires accepted full48 physical stage')
    if (score['frames'] != 48 or score['available_receiver_frames'] != 48 or not score['metric_control']['passed']
            or score['cpu_archive_sha256'] != PIXEL_ARCHIVE or score['source_archive_sha256'] != common.ARCHIVE
            or score['producer_sha256'] != common.PRODUCER):
        raise ValueError('saved pixel score identity mismatch')
    with tempfile.TemporaryDirectory(prefix='paid-reprice-', dir=temporary) as folder:
        base = Path(folder)
        root = extract(archive, stage['archive'], 'tennis', base / 'archive')
        checks = validate_publication(root, decoder, base / 'receiver')
        if not all(checks.values()):
            raise ValueError('paid packet validation failed')
        paid_receipt = json.loads((root / 'receipt.json').read_text())
        if not paid_receipt['score_reuse_eligible']:
            raise ValueError('paid score reuse not admitted')
        paid_rows = []
        for old, saved in zip(score['rows'], paid_receipt['points']):
            if int(saved['label']) != old['crf'] or not saved['oracle_parity']['passed']:
                raise ValueError('paid point and pixel scores differ')
            info = json.loads((root / f"{saved['label']}.json").read_text())
            if info['decode']['decoded_sha256'] != old['composite_sha256']:
                raise ValueError('accepted score pixels differ from paid decode')
            packet = (root / f"{saved['label']}.ptp").read_bytes()
            _, native, patch = tracked_patch.unpack(packet)
            if common.digest(native) != old['base_sha256'] or common.digest(patch) != old['patch_sha256']:
                raise ValueError('accepted score stream identities differ')
            paid_rows.append({'crf': old['crf'], 'bytes': len(packet), 'means': common.metric_means(old['frames']),
                              'packet_sha256': common.digest(packet)})
    if [p['crf'] for p in paid_rows] != list(oracle.RATES) or [p['crf'] for p in score['historical_points']] != list(oracle.RATES):
        raise ValueError('invalid scored matrix')
    curves: dict[str, list[dict[str, Any]]] = {'paid': paid_rows, 'native': [], 'streak': []}
    for row in score['historical_points']:
        for name, price, metric in (('native', 'native_bytes', 'baseline'), ('streak', 'wire_bytes', 'streak')):
            curves[name].append({'crf': row['crf'], 'bytes': row[price], 'means': common.metric_means(row[metric])})
    comparisons: dict[str, Any] = {}
    for anchor in ('native', 'streak'):
        comparisons[anchor] = {}
        for metric in ('full_lpips', 'full_dists', 'crop_lpips', 'crop_dists', 'rgb_psnr'):
            sign = 1 if metric == 'rgb_psnr' else -1
            candidate = [(p['bytes'], sign * p['means'][metric]) for p in curves['paid']]
            baseline = [(p['bytes'], sign * p['means'][metric]) for p in curves[anchor]]
            comparisons[anchor][metric] = {'bd_rate': anchor_sensitivity(candidate, baseline),
                                           'ordinal': ordinal_separation(candidate, baseline)}
    return {'schema': 1, 'scope': 'paid PTP1, exact 48-frame pixel parity; no source-denial or tracker throughput claim',
            'score_sha256': PIXEL_SCORE, 'paid_stage_sha256': expected_stage_sha,
            'paid_archive_sha256': stage['archive']['sha256'], 'physical_curves': curves,
            'comparisons': comparisons, 'basis': 'complete paid PTP1 bytes; common-quality PCHIP and anchor hull only',
            'standard_lpips': 'not compared without historical standard scalar scores'}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fleet', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--reprice-score', type=Path)
    parser.add_argument('--paid-stage', type=Path)
    parser.add_argument('--paid-stage-sha256')
    parser.add_argument('--paid-archive', type=Path)
    parser.add_argument('--reprice-output', type=Path)
    parser.add_argument('--deadline')
    parser.add_argument('--domain', choices=('tennis', 'ego'))
    for name in ('producer', 'archive', 'pixel-archive', 'pixel-stage', 'prepared'):
        parser.add_argument('--' + name)
    parser.add_argument('--encoder', default='SvtAv1EncApp')
    parser.add_argument('--decoder', default='dav1d')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--frames', type=int)
    args = parser.parse_args()
    if args.reprice_score:
        if not all((args.paid_stage, args.paid_stage_sha256, args.paid_archive, args.reprice_output)):
            parser.error('reprice requires paid stage path/hash, archive and output')
        decoder = shutil.which(args.decoder)
        if decoder is None:
            raise FileNotFoundError('dav1d missing')
        result = reprice(args.reprice_score, args.paid_stage, args.paid_stage_sha256, args.paid_archive,
                         decoder, Path(os.environ['TMPDIR']))
        write_json(args.reprice_output, result)
        return
    if args.spec:
        if not args.domain or not args.deadline:
            parser.error('spec needs domain and deadline')
        write_json(args.spec, spec(args.domain, args.deadline))
        return
    if args.validate:
        decoder = shutil.which(args.decoder)
        if decoder is None:
            raise FileNotFoundError('dav1d missing')
        result = validate_fleet(Path(os.environ['PS_STAGE_DIR']), decoder, Path(os.environ['TMPDIR']))
        write_json(Path(os.environ['PS_VALIDATION_PATH']), result)
        print(json.dumps(result))
        if not result['passed']:
            raise SystemExit(1)
        return
    if not args.fleet or not args.domain or args.frames is None:
        parser.error('fleet requires domain and frames')
    required = ('producer', 'archive', 'pixel_archive', 'pixel_stage') if args.domain == 'tennis' else ('prepared',)
    if any(getattr(args, name) is None for name in required):
        parser.error('missing staged inputs')
    for name in common.TOOLS:
        binary = shutil.which(getattr(args, name))
        if binary is None:
            raise FileNotFoundError(name + ' missing')
        setattr(args, name, binary)
    result = run_fleet(args)
    print(json.dumps({'domain': args.domain, 'frames': args.frames, 'cost': result['cost']}))
    if not result['cost']['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

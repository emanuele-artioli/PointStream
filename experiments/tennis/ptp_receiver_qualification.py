"""Bounded fleet bridge for fresh-exec PTP1 source-denial qualification.

The worker receives packet bytes and reviewed generic runtime resources, not
source pixels, labels, hints or expected output. This claims neither network
nor MCP isolation, nor independent observation of native process births.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import time
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any

from experiments.tennis import ball_roi_control as common, k5d_isolated_receiver as kernel
from experiments.tennis import paid_patch_profile as paid, ptp_isolated_receiver as isolated
from src.pointstream import tracked_patch

DATA = paid.DATA
POLICY = '4fa67547b1288e46f979040f0de5315288bd8bfc5d6065e3b6abaeeee32bea1c'
MODEL = 'b052479e6229d840a1a7f42dbc83a6b752a53ef8bc5841a4b957acd519563fec'
FIRST_JPEG = '23c699909b21a42efb00704f25b4563809f7d1afc9bb43d20cfdf54b436fcd10'
EGO_MASKS = '2cd5d3879f53d61b68f54154a110dcfa12ce126ddc37cb564a4a9e97ef598916'
VIDEO = 'c94c6cf88542ef08057224f5834b911737fb5e25515af3a705bd8b92a7bb4338'
ORIGINAL = {
    'archive': DATA + '/tennis/k5b-2026-10-20/inputs/tracknet-Dataset.zip',
    'producer': DATA + '/jobs/fleet/inbox/20261113T210157Z-41ee26e3/full/published.tar',
    'prepared': DATA + '/jobs/fleet/inbox/20261114T233516Z-4223e78c/full/published.tar',
    'model': DATA + '/tennis/k5b-2026-10-20/inputs/balltrack_best.state_dict.pt',
    'policy': DATA + '/jobs/fleet/checks/k5d-SVT-source-cache-71ef32b0c16fdc74e3030075de428061/config.json',
    'ego_video': DATA + '/ladder-autonomy-20261114/intake/p09-103-20261114T233342Z-ed1fc777/P09_103.MP4',
}
FIXED = {'archive': common.ARCHIVE, 'producer': common.PRODUCER, 'prepared': paid.EGO_ARCHIVE,
         'model': MODEL, 'policy': POLICY}


def row(path: Path, expected: str | None = None) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f'protected input must exist: {path}')
    sha = common.k5d_source_oracle.sha(path)
    if expected is not None and sha != expected:
        raise ValueError(f'protected file hash differs: {path}')
    return {'path': str(path), 'sha256': sha, 'size': path.stat().st_size}


def bind_input(path: Path, stage_path: Path, archive_sha: str, stage_sha: str, domain: str, root: Path) -> tuple[Path, dict[str, Any]]:
    common.require_hash(stage_path, stage_sha)
    stage = json.loads(stage_path.read_text())
    frames = 48 if domain == 'tennis' else 64
    if (stage['domain'] != domain or stage['frames'] != frames or stage['archive']['sha256'] != archive_sha
            or not stage['cost']['passed'] or not stage['internal_checks'] or not all(stage['internal_checks'].values())):
        raise ValueError('paid stage is not a full accepted physical output')
    unpacked = paid.extract(path, stage['archive'], domain, root)
    receipt = json.loads((unpacked / 'receipt.json').read_text())
    if receipt['domain'] != domain or receipt['frames'] != frames:
        raise ValueError('paid artifact scope mismatch')
    if domain == 'tennis' and receipt['score_reuse_eligible'] is not True:
        raise ValueError('tennis paid pixels not accepted for reuse')
    return unpacked, receipt


def context(stage: Path) -> dict[str, Any]:
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    identity = {'source_sha256': dispatch['code']['source_sha256'],
                'snapshot_sha256': dispatch['code']['snapshot_sha256'],
                'environment_sha256': dispatch['staged_environment']['sha256']}
    if identity['environment_sha256'] != isolated.RUNTIME_SHA:
        raise ValueError('packed runtime mismatch')
    return {'job_id': stage.parent.name, 'stage': stage.name, 'execution_identity': identity, 'gpu': {'kind': 'cpu'}}


def source_roles(args: argparse.Namespace, work: Path, tennis_root: Path, ego_root: Path) -> dict[str, list[dict[str, Any]]]:
    protected: dict[str, list[dict[str, Any]]] = {role: [] for role in isolated.ROLES}
    for name in ('archive', 'producer', 'prepared'):
        protected['source_archive'].append(row(Path(ORIGINAL[name]), FIXED[name]))
        protected['staged_source_archive'].append(row(Path(getattr(args, name)), FIXED[name]))
    ego_video = Path(ORIGINAL['ego_video'])
    if ego_video.is_file():
        protected['source_archive'].append(row(ego_video, VIDEO))
    for path in (Path(ORIGINAL['model']), Path(args.model)):
        protected['private_detector'].append(row(path, MODEL))
    qualification, _, qualification_bytes = common.producer_metadata(Path(args.producer))
    qpath = work / 'actual-source-qualification.json'
    qpath.write_bytes(qualification_bytes)
    protected['ground_truth'].append(row(qpath, common.QUALIFICATION))
    first = qualification['rows'][0]
    jpeg = work / 'actual-source-frame.jpg'
    with zipfile.ZipFile(args.archive) as source:
        names = source.namelist()
        if names.count(first['member']) != 1:
            raise ValueError('qualified source JPEG is missing or ambiguous')
        jpeg.write_bytes(source.read(first['member']))
    protected['source_frame'].append(row(jpeg, first['source_sha256']))
    source_rgb = common.source_frames(Path(args.archive), qualification)
    tennis_yuv = work / 'actual-tennis-source.yuv'
    from experiments.tennis import k5d_encoder
    k5d_encoder.convert_rgb(args.ffmpeg, source_rgb, tennis_yuv, Fraction(30, 1))
    del source_rgb
    protected['source_yuv'].append(row(tennis_yuv, common.SOURCE_YUV))
    ego_yuv, _ = paid.ego_source(Path(args.prepared), work)
    protected['source_yuv'].append(row(ego_yuv, paid.EGO_SOURCE))
    ego_frame = work / 'actual-ego-frame.yuv'
    with ego_yuv.open('rb') as source, ego_frame.open('xb') as output:
        data = source.read(1920 * 1080 * 3 // 2)
        if len(data) != 1920 * 1080 * 3 // 2:
            raise ValueError('missing actual ego source frame')
        output.write(data)
    protected['source_frame'].append(row(ego_frame))
    mask = work / 'actual-ego-masks.npz'
    with tarfile.open(args.prepared) as tar:
        entry = tar.extractfile('masks.npz')
        if entry is None:
            raise ValueError('actual ego labels missing')
        with mask.open('xb') as output:
            shutil.copyfileobj(entry, output, length=1024 * 1024)
    protected['ground_truth'].append(row(mask, EGO_MASKS))
    for root, labels in ((tennis_root, ('48', '55', '62')), (ego_root, ('fallback',))):
        for label in labels:
            info = json.loads((root / f'{label}.json').read_text())
            path = work / f'actual-config-{label}.json'
            paid.write_json(path, info['config'])
            protected['encoder_state'].append(row(path))
    isolated.protected_evidence(protected)
    return protected


def expected_case(root: Path, label: str, domain: str) -> dict[str, Any]:
    info = json.loads((root / f'{label}.json').read_text())
    packet = root / f'{label}.ptp'
    data = packet.read_bytes()
    meta, _, patch = tracked_patch.unpack(data)
    inspected = tracked_patch.inspect(data)
    count = 48 if domain == 'tennis' else 64
    if (info['inspect'] != inspected or meta['native']['frames'] != count
            or len(info['decode']['rgb_frame_sha256']) != count
            or info['encode']['rgb_frame_sha256'] != info['decode']['rgb_frame_sha256']
            or info['encode']['decoded_sha256'] != info['decode']['decoded_sha256']):
        raise ValueError('accepted packet/expected RGB binding differs')
    if domain == 'tennis' and (not patch or not info['oracle_parity']['passed']):
        raise ValueError('tennis point lacks oracle parity')
    if domain == 'ego' and (patch or any(p is not None for p in meta['positions'])):
        raise ValueError('ego point is not native fallback')
    return {'domain': domain, 'label': label, 'packet': str(packet), 'packet_sha256': common.digest(data),
            'packet_bytes': len(data), 'frames': count,
            'expected': {'decoded_sha256': info['decode']['decoded_sha256'],
                         'rgb_frame_sha256': info['decode']['rgb_frame_sha256']}}



def verify_policy(policy: dict[str, Any], base_bytes: bytes, policy_sha: str, host: str) -> None:
    if common.digest(base_bytes) != POLICY:
        raise ValueError('reviewed base policy identity mismatch')
    if policy_sha == POLICY:
        if policy != json.loads(base_bytes) or host not in policy.get('profiles', {}):
            raise ValueError('legacy policy differs from authenticated base or host')
        return
    from experiments.tennis import ptp_cpu_policy_capture as capture

    capture.validate_derivation(policy, base_bytes)
    capture.verify_links(policy, host)

def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.sched_getaffinity(0) != {0} or os.environ.get('PS_CPU_ALLOWANCE') != '1':
        raise ValueError('requires one CPU at affinity index 0')
    if args.cases not in (1, 4):
        raise ValueError('only one or four complete packet cases are registered')
    for name, sha in FIXED.items():
        if name != 'policy':
            common.require_hash(Path(getattr(args, name)), sha)
    common.require_hash(Path(args.base_policy), POLICY)
    common.require_hash(Path(args.policy), args.policy_sha256)
    common.require_hash(Path(args.decoder), common.TOOLS['decoder'])
    common.require_hash(Path(args.ffmpeg), common.TOOLS['ffmpeg'])
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    start, cpu = time.monotonic(), common.cpu_seconds()
    work = scratch / 'ptp-qualification-private'
    work.mkdir(exist_ok=False)
    tennis, _ = bind_input(Path(args.tennis_archive), Path(args.tennis_stage), args.tennis_archive_sha256,
                           args.tennis_stage_sha256, 'tennis', work / 'tennis')
    ego, _ = bind_input(Path(args.ego_archive), Path(args.ego_stage), args.ego_archive_sha256,
                        args.ego_stage_sha256, 'ego', work / 'ego')
    run_context = context(stage)
    base_bytes = Path(args.base_policy).read_bytes()
    old_policy = json.loads(Path(args.policy).read_text())
    host = os.uname().nodename.split('.')[0]
    verify_policy(old_policy, base_bytes, args.policy_sha256, host)
    source_root = Path(__file__).resolve().parents[2]
    amended = isolated.amend_policy(old_policy, source_root, run_context['execution_identity'])
    # Verify every existing reviewed generic-runtime and system entry before forking.
    kernel.inventory(amended)
    policy_file = work / 'amended-policy.json'
    paid.write_json(policy_file, amended)
    policy_sha = common.k5d_source_oracle.sha(policy_file)
    protected = source_roles(args, work, tennis, ego)
    case_list = [expected_case(tennis, label, 'tennis') for label in ('48', '55', '62')]
    case_list.append(expected_case(ego, 'fallback', 'ego'))
    selected = case_list[:args.cases]
    publication = work / 'publication'
    publication.mkdir()
    shutil.copyfile(policy_file, publication / 'policy.json')
    shutil.copyfile(args.policy, publication / 'old-policy.json')
    shutil.copyfile(args.base_policy, publication / 'base-policy.json')
    results = []
    for index, case in enumerate(selected):
        result = isolated.replay(Path(case['packet']), work / f'receiver-{index}', policy_path=policy_file,
            policy_sha=policy_sha, packet_sha=case['packet_sha256'], protected=protected, context=run_context,
            expected=case['expected'], decoder=Path(args.decoder), timeout=180)
        result_path = publication / f'proof-{index}.json'
        paid.write_json(result_path, result)
        results.append({k: v for k, v in case.items() if k != 'packet'})
    body = {'schema': 1, 'host': os.uname().nodename.split('.')[0], 'cases': args.cases, 'context': run_context, 'policy_sha256': policy_sha,
            'original_policy_sha256': args.policy_sha256, 'input_policy_sha256': args.policy_sha256,
            'base_policy_sha256': POLICY, 'runtime_sha256': isolated.RUNTIME_SHA,
            'source_inputs': {name: FIXED[name] for name in FIXED if name != 'policy'},
            'paid_tennis_archive_sha256': args.tennis_archive_sha256, 'paid_tennis_stage_sha256': args.tennis_stage_sha256,
            'paid_ego_archive_sha256': args.ego_archive_sha256, 'paid_ego_stage_sha256': args.ego_stage_sha256,
            'ego_original_video_present': Path(ORIGINAL['ego_video']).is_file(), 'records': results,
            'scope': 'actual source, labels, encoder state and private model file reads denied after fresh exec; no network or MCP isolation'}
    paid.write_json(publication / 'manifest.json', body)
    binding = publish(publication, stage / 'published.tar', args.cases)
    replay_root = extract(stage / 'published.tar', binding, args.cases, work / 'internal-archive')
    internal_checks = validate_proofs(replay_root, args.cases, args.policy_sha256)
    if not all(internal_checks.values()):
        raise ValueError('internal qualification proof validation failed')
    cost = cost_record(time.monotonic() - start, common.cpu_seconds() - cpu, args.cases)
    result = {'schema': 1, 'cases': args.cases, 'archive': binding, 'cost': cost, 'internal_checks': internal_checks,
              'scope': 'PTP filesystem denial under inherited native restrictions'}
    paid.write_json(stage / 'ptp-qualification-stage.json', result)
    return result


def names(cases: int) -> list[str]:
    if cases not in (1, 4):
        raise ValueError('invalid case count')
    return ['manifest.json', 'policy.json', 'old-policy.json', 'base-policy.json', *[f'proof-{i}.json' for i in range(cases)]]


def publish(root: Path, archive: Path, count: int) -> dict[str, Any]:
    expected = names(count)
    if {p.name for p in root.iterdir()} != set(expected):
        raise ValueError('unexpected proof inventory')
    files = {}
    with tarfile.open(archive, 'x') as tar:
        for name in expected:
            path = root / name
            if not path.is_file() or path.is_symlink():
                raise ValueError('proof file must be regular')
            files[name] = common.k5d_source_oracle.sha(path)
            tar.add(path, arcname='publish/' + name, recursive=False)
    return {'sha256': common.k5d_source_oracle.sha(archive), 'bytes': archive.stat().st_size, 'members': files}


def extract(archive: Path, binding: dict[str, Any], count: int, work: Path) -> Path:
    common.require_hash(archive, binding['sha256'])
    expected = names(count)
    if archive.stat().st_size != binding['bytes'] or set(binding['members']) != set(expected):
        raise ValueError('proof archive binding mismatch')
    work.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as tar:
        entries = tar.getmembers()
        if (len(entries) != len(expected) or {e.name for e in entries} != {'publish/' + n for n in expected}
                or any(not e.isfile() for e in entries)):
            raise ValueError('unexpected proof archive members')
        tar.extractall(work, filter='data')
    root = work / 'publish'
    for name, sha in binding['members'].items():
        common.require_hash(root / name, sha)
    return root


def proof_bindings(proof: dict[str, Any], manifest: dict[str, Any], policy: dict[str, Any]) -> dict[str, bool]:
    """Bind archived witness identities to independently frozen generic/data inputs.

    This uses archived metadata, never reopening ephemeral protected files. The
    PTP verifier separately links every role row to an actual EACCES/EPERM probe.
    Arbitrary encoder-config and ego-frame hashes remain observed evidence;
    their source contents are not inferred from a filename or a claimed role.
    """
    filesystem = proof['filesystem_proof']
    host = manifest.get('host')
    if not isinstance(host, str) or host not in policy['profiles']:
        raise ValueError('qualification host absent from archived policy')
    profile = policy['profiles'][host]
    checks = {
        'profile_inventory': filesystem['profile_sha256'] == kernel._object_sha(profile),
        'runtime_inventory': filesystem['runtime_inventory_sha256'] == kernel._object_sha(profile['runtime_files']),
        'source_inventory': filesystem['source_inventory_sha256'] == kernel._object_sha(profile['source_files']),
        'profile_runtime': profile.get('runtime_archive_sha256') == isolated.RUNTIME_SHA,
    }
    roles = filesystem['protected_roles']
    # Paths can disappear after fleet cleanup: compare frozen path/hash records.
    checks['protected_records'] = set(roles) == set(isolated.ROLES) and all(
        isinstance(rows, list) and rows and all(
            isinstance(r, dict) and set(r) == {'path', 'sha256', 'size'}
            and isinstance(r['path'], str) and Path(r['path']).is_absolute()
            and type(r['size']) is int and r['size'] > 0
            for r in rows) for rows in roles.values())
    if not checks['protected_records']:
        return checks
    names = ('archive', 'producer', 'prepared')
    original = [(ORIGINAL[name], FIXED[name]) for name in names]
    video_present = manifest.get('ego_original_video_present')
    if video_present is True:
        original.append((ORIGINAL['ego_video'], VIDEO))
    checks['original_sources'] = (type(video_present) is bool and
        [(r['path'], r['sha256']) for r in roles['source_archive']] == original)
    checks['staged_sources'] = [r['sha256'] for r in roles['staged_source_archive']] == [FIXED[name] for name in names]
    models = roles['private_detector']
    checks['private_models'] = (len(models) == 2 and models[0]['path'] == ORIGINAL['model']
                                and all(r['sha256'] == MODEL for r in models))
    checks['source_yuv'] = [(r['sha256'], r['size']) for r in roles['source_yuv']] == [
        (common.SOURCE_YUV, 48 * common.FRAME_BYTES), (paid.EGO_SOURCE, 64 * 1920 * 1080 * 3 // 2)]
    checks['source_labels'] = [r['sha256'] for r in roles['ground_truth']] == [common.QUALIFICATION, EGO_MASKS]
    frames = roles['source_frame']
    checks['actual_frames'] = (len(frames) == 2 and frames[0]['sha256'] == FIRST_JPEG
                               and frames[1]['size'] == 1920 * 1080 * 3 // 2)
    # Full stage prepares all four real config files even in one-case smoke.
    checks['encoder_config_coverage'] = [Path(r['path']).name for r in roles['encoder_state']] == [
        'actual-config-48.json', 'actual-config-55.json', 'actual-config-62.json', 'actual-config-fallback.json']
    return checks


def validate_proofs(root: Path, cases: int, expected_policy_sha: str) -> dict[str, bool]:
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['cases'] != cases or len(manifest['records']) != cases:
        raise ValueError('case count mismatch')
    common.require_hash(root / 'policy.json', manifest['policy_sha256'])
    common.require_hash(root / 'base-policy.json', POLICY)
    common.require_hash(root / 'old-policy.json', expected_policy_sha)
    old_policy = json.loads((root / 'old-policy.json').read_text())
    verify_policy(old_policy, (root / 'base-policy.json').read_bytes(), expected_policy_sha, manifest['host'])
    actual_policy = json.loads((root / 'policy.json').read_text())
    reconstructed = isolated.amend_policy(old_policy, Path(__file__).resolve().parents[2],
                                         manifest['context']['execution_identity'])
    checks = {'actual_host': manifest['host'] == os.uname().nodename.split('.')[0],
              'policy_amendment': reconstructed == actual_policy,
              'original_policy': (manifest['original_policy_sha256'] == expected_policy_sha
                                  and manifest['input_policy_sha256'] == expected_policy_sha
                                  and manifest['base_policy_sha256'] == POLICY),
              'runtime': manifest['runtime_sha256'] == isolated.RUNTIME_SHA,
              'fixed_inputs': manifest['source_inputs'] == {name: FIXED[name] for name in FIXED if name != 'policy'}}
    expected_cases = [('tennis', '48'), ('tennis', '55'), ('tennis', '62'), ('ego', 'fallback')]
    for i, record in enumerate(manifest['records']):
        proof = json.loads((root / f'proof-{i}.json').read_text())
        if (record['domain'], record['label']) != expected_cases[i]:
            raise ValueError('case ordering or scope mismatch')
        observer = proof['filesystem_proof']['worker_observer']
        isolated.verify_receipt(proof, context=manifest['context'], policy_sha=manifest['policy_sha256'],
                                packet_sha=record['packet_sha256'], observer=observer)
        checks.update({f'{i}:binding:{key}': value for key, value in proof_bindings(proof, manifest, actual_policy).items()})
        decode = proof['decode']
        checks[f'{i}:exact_reconstruction'] = (proof['parent_validation']['exact_rgb_and_frame_hashes'] is True
            and decode['decoded_sha256'] == record['expected']['decoded_sha256']
            and decode['rgb_frame_sha256'] == record['expected']['rgb_frame_sha256']
            and decode['decoded_frames'] == record['frames'] == (48 if record['domain'] == 'tennis' else 64)
            and decode['packet_bytes'] == record['packet_bytes'])
    return checks


def cost_record(wall: float, cpu: float, cases: int) -> dict[str, Any]:
    if wall <= 0 or cpu < 0 or cases not in (1, 4):
        raise ValueError('invalid qualification cost')
    forecast = 1.5 * wall * 4 / cases + 30
    return {'cases': cases, 'target_cases': 4, 'wall_seconds': wall, 'cpu_seconds': cpu,
            'forecast_seconds': forecast, 'full_cap_seconds': 900,
            'passed': forecast <= 900 if cases == 1 else wall <= 900,
            'basis': 'body after imports including full actual source preparation and fresh-exec proof; smoke 1.5 * wall * 4 + 30'}


def validate_fleet(stage: Path, temp: Path) -> dict[str, Any]:
    record = json.loads((stage / 'ptp-qualification-stage.json').read_text())
    spec = json.loads((stage.parent / 'spec.json').read_text())
    policy_items = [item for item in spec['staged_inputs'] if item['name'] == 'policy']
    base_items = [item for item in spec['staged_inputs'] if item['name'] == 'base_policy']
    if len(policy_items) != 1 or len(base_items) != 1 or base_items[0]['sha256'] != POLICY:
        raise ValueError('spec policy bindings differ')
    expected_policy_sha = policy_items[0]['sha256']
    with tempfile.TemporaryDirectory(prefix='ptp-proof-', dir=temp) as folder:
        root = extract(stage / 'published.tar', record['archive'], record['cases'], Path(folder) / 'archive')
        checks = validate_proofs(root, record['cases'], expected_policy_sha)
        manifest = json.loads((root / 'manifest.json').read_text())
        checks['host_in_frozen_spec'] = manifest['host'] in spec['hosts']
    cost = record['cost']
    checks['body_cost'] = cost == cost_record(cost['wall_seconds'], cost['cpu_seconds'], record['cases']) and cost['passed']
    checks['internal_validation'] = bool(record['internal_checks']) and all(record['internal_checks'].values())
    if (stage / 'execution.json').exists():
        execution = json.loads((stage / 'execution.json').read_text())
        checks['whole_command_cost'] = (execution['exit_code'] == 0
                                         and cost_record(execution['seconds'], 0, record['cases'])['passed'])
    return {'passed': all(checks.values()), 'checks': checks,
            'scope': 'fresh-exec source-denial and exact RGB; no network/MCP isolation or independently observed native births'}


def make_spec(deadline: str, paid_inputs: dict[str, dict[str, str]],
              policy_descriptor: dict[str, Any] | None = None) -> dict[str, Any]:
    if set(paid_inputs) != {'tennis_archive', 'tennis_stage', 'ego_archive', 'ego_stage'}:
        raise ValueError('four exact paid full-stage paths and hashes required')
    if policy_descriptor is None:
        selected_policy = {'path': ORIGINAL['policy'], 'sha256': POLICY, 'hosts': ['gpu6']}
    else:
        selected_policy = policy_descriptor
        if (set(selected_policy) != {'path', 'sha256', 'hosts'}
                or not isinstance(selected_policy['path'], str) or not Path(selected_policy['path']).is_absolute()
                or not isinstance(selected_policy['sha256'], str) or len(selected_policy['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in selected_policy['sha256'])
                or selected_policy['sha256'] == POLICY or not isinstance(selected_policy['hosts'], list)
                or not selected_policy['hosts']
                or any(not isinstance(host, str) or host not in {'gpu1', 'gpu3', 'gpu5'} for host in selected_policy['hosts'])
                or len(set(selected_policy['hosts'])) != len(selected_policy['hosts'])):
            raise ValueError('invalid frozen CPU policy descriptor')
    items: list[dict[str, Any]] = [{'name': name, 'path': ORIGINAL[name], 'sha256': sha, 'extract': False}
                                   for name, sha in FIXED.items() if name != 'policy']
    items.extend([{'name': 'policy', 'path': selected_policy['path'], 'sha256': selected_policy['sha256'], 'extract': False},
                  {'name': 'base_policy', 'path': ORIGINAL['policy'], 'sha256': POLICY, 'extract': False}])
    for name, info in paid_inputs.items():
        if (set(info) != {'path', 'sha256'} or not Path(info['path']).is_absolute() or len(info['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in info['sha256'])):
            raise ValueError('paid input requires an absolute path and lowercase SHA256')
        items.append({'name': name, **info, 'extract': False})
    arguments = ['--fleet', '--cases', '{cases}']
    for item in items:
        arguments += ['--' + item['name'].replace('_', '-'), '{staged:' + item['name'] + '}']
        if item['name'] in paid_inputs or item['name'] == 'policy':
            arguments += ['--' + item['name'].replace('_', '-') + '-sha256', item['sha256']]
    return {'schema': 1, 'device': 'cpu', 'hosts': selected_policy['hosts'], 'gpu_models': [], 'gpu_memory_mib': 0,
            'cpu_threads': 1, 'single_cpu_campaign': True, 'cpu_affinity_index': 0, 'local_storage_gib': 20,
            'entrypoint': ['-m', 'experiments.tennis.ptp_receiver_qualification'], 'arguments': arguments,
            'staged_inputs': items, 'inputs': [{'path': p['path'], 'sha256': p['sha256']} for p in items],
            'environment': {'path': DATA + '/environments/pointstream-20261009T214002Z.tar.gz',
                            'sha256': isolated.RUNTIME_SHA}, 'verify_staged_inputs': True,
            'required_commands': ['ffmpeg'], 'scale': {'cases': {'smoke': 1, 'full': 4}},
            'smoke': {'seconds': 450, 'representative_basis': 'All actual protected inputs/materialization and one full48 tennis PTP receiver with fresh restricted exec; full repeats for three tennis and one full64 ego packet.'},
            'full': {'seconds': 900}, 'validator': ['{python}', '-m', 'experiments.tennis.ptp_receiver_qualification', '--validate'],
            'validator_seconds': 60, 'budget_seconds': 1530, 'stall_seconds': 450, 'deadline': deadline,
            'contention': {'policy': 'stop', 'resume_attempts': 0}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fleet', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--paid-inputs', type=Path)
    parser.add_argument('--policy-descriptor', type=Path)
    parser.add_argument('--base-policy')
    parser.add_argument('--policy-sha256')
    parser.add_argument('--deadline')
    parser.add_argument('--cases', type=int, choices=(1, 4))
    for name in (*FIXED, 'tennis_archive', 'tennis_stage', 'ego_archive', 'ego_stage'):
        parser.add_argument('--' + name.replace('_', '-'))
    for name in ('tennis_archive', 'tennis_stage', 'ego_archive', 'ego_stage'):
        parser.add_argument('--' + name.replace('_', '-') + '-sha256')
    parser.add_argument('--decoder', default='dav1d')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    args = parser.parse_args()
    if args.spec:
        if args.paid_inputs is None or args.deadline is None:
            parser.error('spec requires --paid-inputs JSON and --deadline')
        descriptor = json.loads(args.policy_descriptor.read_text()) if args.policy_descriptor else None
        paid.write_json(args.spec, make_spec(args.deadline, json.loads(args.paid_inputs.read_text()), descriptor))
        return
    if args.validate:
        result = validate_fleet(Path(os.environ['PS_STAGE_DIR']), Path(os.environ['TMPDIR']))
        paid.write_json(Path(os.environ['PS_VALIDATION_PATH']), result)
        print(json.dumps(result))
        if not result['passed']:
            raise SystemExit(1)
        return
    if not args.fleet or args.cases is None:
        parser.error('fleet requires --cases')
    required = (*FIXED, 'base_policy', 'policy_sha256', 'tennis_archive', 'tennis_stage', 'ego_archive', 'ego_stage',
                'tennis_archive_sha256', 'tennis_stage_sha256', 'ego_archive_sha256', 'ego_stage_sha256')
    if any(getattr(args, name) is None for name in required):
        parser.error('fleet requires all authenticated staged inputs and paid hashes')
    for name in ('decoder', 'ffmpeg'):
        binary = shutil.which(getattr(args, name))
        if binary is None:
            raise FileNotFoundError(name + ' missing')
        setattr(args, name, str(Path(binary).resolve()))
    result = run(args)
    print(json.dumps({'cases': args.cases, 'cost': result['cost']}))
    if not result['cost']['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

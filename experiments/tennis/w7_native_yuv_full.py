"""Future W7 75-role native-YUV byte-proof worker; requires reviewed pilot.

No specimen spec is emitted without actual complete SHA-bound pilot evidence
and all archived inputs. No job is submitted by this module.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

from experiments.tennis import w7_byteproof as bp, w7_mkv_profile as profile, w7_remux_job as old


def independent_check(row: dict[str, Any], original: Path, work: Path, temp: Path,
                      ffmpeg: Path, ffprobe: Path, dav1d: Path) -> dict[str, Any]:
    record = bp.validate_proof(row, original, work, native_yuv=True)
    returned = temp / 'validator-returned.ivf'
    subprocess.run([str(ffmpeg), '-nostdin', '-v', 'error', '-threads', '1', '-i', str(work / 'delivered.mkv'),
                    '-map', '0:v:0', '-an', '-c:v', 'copy', '-fps_mode', 'passthrough', '-f', 'ivf', '-n', str(returned)],
                   capture_output=True, check=True, timeout=30)
    checked = bp.compare_ivf(original, returned)
    if (not checked['timestamps_equal'] or not checked['ordered_payloads_equal']
            or bp.digest(returned) != bp.digest(work / 'decoder-scratch.ivf')):
        raise ValueError('validator extraction differs from ordered original payloads')
    yuv = temp / 'validator-native.yuv'
    subprocess.run([str(dav1d), '-i', str(returned), '-o', str(yuv), '--threads', '1', '--quiet'],
                   capture_output=True, check=True, timeout=90)
    if (yuv.stat().st_size != row['width'] * row['height'] * 3 // 2 * 240
            or bp.digest(yuv) != row['historical_native_yuv_sha256']):
        raise ValueError('independent native YUV mismatch')
    commands = profile.commands(str(ffmpeg), str(ffprobe), 'unused', original, work)
    source = json.loads(subprocess.run(commands['probe_original'], capture_output=True, check=True, timeout=30).stdout)
    delivered = json.loads(subprocess.run(commands['probe_mkv'], capture_output=True, check=True, timeout=30).stdout)
    before, after = bp.probe(source, row['width'], row['height']), bp.probe(delivered, row['width'], row['height'])
    if (before['timestamps'] != after['timestamps'] or before['colors'] != record['original_color_metadata']
            or after['colors'] != record['delivered_color_metadata']):
        raise ValueError('independent timestamps or colors differ')
    return record


def rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    bp.representative_three(inventory)
    result = inventory['points'] + inventory['controls'] + inventory['new_points']
    if len(result) != 75 or len({(x['window'], x['name']) for x in result}) != 75:
        raise ValueError('logical role coverage differs')
    if any(x['frames'] != 240 or not isinstance(x['historical_native_yuv_sha256'], str)
           or len(x['historical_native_yuv_sha256']) != 64 for x in result):
        raise ValueError('historical native YUV identity missing')
    return result


def forecast(max_point: float, prior_charge: float, pilot_charge: float, reserve: float, *, count: int = 78) -> float:
    values = (max_point, prior_charge, pilot_charge, reserve)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('invalid measured full forecast inputs')
    if prior_charge != 138 or count != 78:
        raise ValueError('original charge or repeated smoke scope differs')
    estimated = 1.5 * count * max_point + reserve
    if estimated > 600 - prior_charge - pilot_charge:
        raise ValueError('full 75-role workload exceeds remaining phase budget')
    return estimated


def verify_archives(inventory: dict[str, Any], archive_map: dict[str, str], local: dict[str, str]) -> None:
    """Verify every referenced archive and every role's exact native member."""
    import tarfile
    all_rows = rows(inventory)
    expected = {x['encode_archive_sha256'] for x in all_rows}
    if set(archive_map) != expected or set(local) != expected:
        raise ValueError('complete 18-archive mapping required')
    for sha in sorted(expected):
        path = Path(local[sha])
        if bp.digest(path) != sha or not Path(archive_map[sha]).is_absolute():
            raise ValueError('actual archive identity differs')
        with tarfile.open(path) as tar:
            for row in (x for x in all_rows if x['encode_archive_sha256'] == sha):
                matches = [m for m in tar.getmembers() if m.name == row['ivf_member']]
                if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 100_000_000:
                    raise ValueError('missing, ambiguous or oversized native member')
                stream = tar.extractfile(matches[0])
                if stream is None:
                    raise ValueError('native member unavailable')
                import hashlib
                if hashlib.sha256(stream.read()).hexdigest() != row['original_ivf_sha256']:
                    raise ValueError('native member SHA256 differs')


def pilot_evidence(status: dict[str, Any], stage: Path, charged: float) -> dict[str, Any]:
    """Bind the *actual* completed YUV pilot, never a proposed job or estimate."""
    if status.get('status') != 'complete' or status.get('remote_status', {}).get('exit_code') != 0:
        raise ValueError('completed pilot status required')
    gate = status.get('gate') or {}
    validation = gate.get('validation') or {}
    required = {'three_frozen_roles', 'independent_mkv_extract', 'native_yuv', 'timestamps', 'cost_forecast'}
    if (gate.get('passed') is not True or validation.get('passed') is not True
            or any(validation.get('checks', {}).get(key) is not True for key in required)):
        raise ValueError('native YUV pilot gate incomplete')
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    execution = json.loads((stage / 'execution.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    if (dispatch['code']['git_head'] != '83842fb3' and not dispatch['code']['git_head'].startswith('83842fb3')):
        raise ValueError('pilot source revision differs')
    if (gate['source_sha256'] != dispatch['code']['source_sha256']
            or dispatch['staged_environment']['sha256'] != old.ENV_SHA
            or execution.get('exit_code') != 0 or execution.get('timing_contaminated') is not False
            or bp.digest(stage / 'published.tar') != staging['published']['sha256']):
        raise ValueError('pilot execution, publication or source identity differs')
    if isinstance(charged, bool) or not isinstance(charged, (float, int)) or not math.isfinite(charged) or charged <= 30:
        raise ValueError('actual inclusive pilot charge required')
    max_point = validation['max_worker_and_validator_point_seconds']
    estimate = forecast(max_point, 138, charged, 60)
    return {'job_id': status['job_id'], 'pilot_source_sha256': gate['source_sha256'],
            'pilot_snapshot_sha256': dispatch['code']['snapshot_sha256'],
            'pilot_publication_sha256': staging['published']['sha256'],
            'pilot_execution_sha256': bp.digest(stage / 'execution.json'),
            'pilot_dispatch_sha256': bp.digest(stage / 'dispatch.json'),
            'pilot_staging_sha256': bp.digest(stage / 'staging.json'),
            'pilot_gate_sha256': bp.digest(stage.parent / 'gate.json'),
            'pilot_charge_seconds': charged, 'original_charge_seconds': 138,
            'max_point_seconds': max_point, 'full_forecast_seconds': estimate,
            'scope': '75 independent roles plus repeated three-role smoke; 60-second reserve'}


def context(stage: Path) -> tuple[list[dict[str, Any]], dict[str, Path], dict[str, Any], dict[str, Any]]:
    import os
    from experiments.tennis import w7_native_yuv_pilot as pilot
    audit_path = Path(__file__).resolve().parents[2] / 'docs/experiments/w7-inputs/native-yuv-9b0e94ee-independent-audit-20261105.json'
    if bp.digest(audit_path) != '7adeee91d8d1d29d102319c918d57c9a614c2dcb1bb4b9102559a7014ae6e5f8':
        raise ValueError('audited native-YUV pilot evidence changed')
    audit = json.loads(audit_path.read_text())
    if (audit.get('job_id') != '20261105T024114Z-9b0e94ee' or audit.get('status') != 'complete'
            or audit.get('inclusive_charge_seconds') != 54 or audit.get('gate', {}).get('passed') is not True):
        raise ValueError('pilot receipt or inclusive cost mismatch')
    inventory = json.loads(pilot.INVENTORY.read_text())
    if bp.digest(pilot.INVENTORY) != pilot.INVENTORY_SHA:
        raise ValueError('frozen role inventory changed')
    selected = rows(inventory)
    stage_data = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    spec = json.loads((stage.parent / 'spec.json').read_text())
    if (spec.get('device') != 'cpu' or spec.get('single_cpu_campaign') is not True or spec.get('cpu_threads') != 1
            or spec.get('budget_seconds') != 348 or spec.get('validator_seconds') != 45
            or spec.get('prior_cpu_charge_seconds') != 192 or spec.get('remaining_phase_cpu_seconds') != 408
            or spec.get('external_audit_reserve_seconds') != 60
            or spec.get('entrypoint') != ['-m', 'experiments.tennis.w7_native_yuv_full']
            or spec.get('arguments') != ['--stage', '{stage}']
            or spec.get('scale') != {'stage': {'smoke': 'smoke', 'full': 'full'}}
            or spec.get('validator') != ['{python}', '-m', 'experiments.tennis.w7_native_yuv_full', '--validate-smoke']
            or spec.get('smoke', {}).get('seconds') != 40 or spec.get('full', {}).get('seconds') != 240
            or spec.get('environment', {}).get('sha256') != old.ENV_SHA
            or stage_data['environment']['sha256'] != old.ENV_SHA or len(dispatch.get('cpu_affinity', [])) != 1
            or len(os.sched_getaffinity(0)) != 1):
        raise ValueError('full 75-role campaign scope or CPU bounds differ')
    entries = {x['name']: x for x in stage_data['inputs']}
    declarations = {x['name']: x for x in spec['staged_inputs']}
    hashes = {x['encode_archive_sha256'] for x in selected}
    asset_names = {'arc_' + h[:16] for h in hashes}
    if len(asset_names) != len(hashes) or set(entries) != asset_names or set(declarations) != asset_names:
        raise ValueError('staged archive coverage differs')
    root = Path(stage_data['local_root']).resolve(strict=True)
    paths = {}
    for sha in hashes:
        name = 'arc_' + sha[:16]
        item, declared = entries[name], declarations[name]
        tree = Path(item['path']).resolve(strict=True)
        archive = tree.parent / Path(item['source']).name
        if (item['sha256'] != sha or declared['sha256'] != sha or item.get('extract') is not True
                or declared.get('extract') is not True or not tree.is_relative_to(root) or bp.digest(archive) != sha):
            raise ValueError('archive source, bytes or local root differ')
        paths[sha] = tree
    bins = Path(stage_data['environment']['bin']).resolve(strict=True)
    if (not bins.is_relative_to(root) or bp.digest(bins / 'ffmpeg') != old.FFMPEG_SHA
            or bp.digest(bins / 'ffprobe') != old.FFPROBE_SHA or bp.digest(bins / 'dav1d') != profile.DAV1D_SHA):
        raise ValueError('packed native tools differ')
    paths.update(ffmpeg=bins / 'ffmpeg', ffprobe=bins / 'ffprobe', dav1d=bins / 'dav1d')
    return selected, paths, spec, dispatch


def source(row: dict[str, Any], trees: dict[str, Path]) -> Path:
    root = trees[row['encode_archive_sha256']]
    member = Path(row['ivf_member'])
    if not member.parts or member.parts[0] != 'publish':
        raise ValueError('invalid IVF archive member')
    path = (root / member).resolve(strict=True)
    if not path.is_relative_to(root) or bp.digest(path) != row['original_ivf_sha256']:
        raise ValueError('source IVF differs from scored historical stream')
    return path


def run(stage: Path, scratch: Path) -> None:
    import tempfile
    import time
    from experiments.tennis import w7_native_yuv_pilot as pilot
    all_rows, paths, _, _ = context(stage)
    chosen = bp.representative_three(json.loads(pilot.INVENTORY.read_text())) if stage.name == 'smoke' else all_rows
    if stage.name not in ('smoke', 'full'):
        raise ValueError('invalid stage')
    out = scratch / 'publish'
    out.mkdir(exist_ok=False)
    result = []
    for index, row in enumerate(chosen):
        native = source(row, paths)
        work = out / f'role-{index:03d}'
        started = time.monotonic()
        bp.prove(row, native, work, paths['ffmpeg'], paths['ffprobe'], dav1d=paths['dav1d'], native_yuv=True)
        with tempfile.TemporaryDirectory(prefix='w7-native-proof-', dir=os.environ['TMPDIR']) as temp:
            checked = independent_check(row, native, work, Path(temp), paths['ffmpeg'], paths['ffprobe'], paths['dav1d'])
        independent = {'source_ivf_sha256': row['original_ivf_sha256'],
                       'delivered_mkv_sha256': checked['delivered_mkv_sha256'],
                       'returned_ivf_sha256': checked['reconstructed_ivf_sha256'],
                       'native_yuv_sha256': row['historical_native_yuv_sha256'],
                       'native_decoder_sha256': profile.DAV1D_SHA,
                       'checks': {'reextract': True, 'payloads': True, 'timestamps': True, 'native_yuv': True}}
        (work / 'independent-validation.json').write_text(json.dumps(independent) + '\n')
        result.append({'index': index, 'window': row['window'], 'name': row['name'],
                       'proof_sha256': bp.digest(work / 'proof.json'),
                       'independent_validation_sha256': bp.digest(work / 'independent-validation.json'),
                       'seconds': time.monotonic() - started})
        # Preserve the completed prefix even if a later point fails.
        (out / 'coverage.json').write_text(json.dumps({'completed': result}, indent=2) + '\n')
    if len(result) != (3 if stage.name == 'smoke' else 75):
        raise ValueError('logical role coverage incomplete')


def validate(stage: Path, target: Path) -> None:
    import tarfile
    import tempfile
    import time
    from experiments.tennis import w7_native_yuv_pilot as pilot
    try:
        _, paths, spec, dispatch = context(stage)
        chosen = bp.representative_three(json.loads(pilot.INVENTORY.read_text()))
        staging = json.loads((stage / 'staging.json').read_text())
        with tempfile.TemporaryDirectory(prefix='w7-full-smoke-', dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            with tarfile.open(stage / 'published.tar') as tar:
                if bp.digest(stage / 'published.tar') != staging['published']['sha256']:
                    raise ValueError('smoke archive changed')
                members = tar.getmembers()
                if len(members) > 45 or any(not (x.isfile() or x.isdir()) for x in members):
                    raise ValueError('unexpected smoke members')
                tar.extractall(root, filter='data')
            coverage = json.loads((root / 'publish/coverage.json').read_text())['completed']
            if len(coverage) != 3:
                raise ValueError('three full-path smoke roles required')
            for i, row in enumerate(chosen):
                work = root / 'publish' / f'role-{i:03d}'
                if (coverage[i]['index'] != i or coverage[i]['window'] != row['window']
                        or coverage[i]['name'] != row['name'] or coverage[i]['proof_sha256'] != bp.digest(work / 'proof.json')
                        or coverage[i]['independent_validation_sha256'] != bp.digest(work / 'independent-validation.json')):
                    raise ValueError('smoke role proof mismatch')
                temp = root / f'validate-{i}'
                temp.mkdir()
                independent_check(row, source(row, paths), work, temp, paths['ffmpeg'], paths['ffprobe'], paths['dav1d'])
        execution = json.loads((stage / 'execution.json').read_text())
        clock = dispatch['campaign_clock']
        now = time.monotonic()
        if (execution.get('exit_code') != 0 or execution.get('timing_contaminated') is not False
                or not 0 < clock['started'] <= clock['phase_started'] <= now < clock['budget_end']):
            raise ValueError('invalid actual smoke execution or clock')
        live_max = max(x['seconds'] for x in coverage)
        audit_path = Path(__file__).resolve().parents[2] / 'docs/experiments/w7-inputs/native-yuv-9b0e94ee-independent-audit-20261105.json'
        reference = json.loads(audit_path.read_text())['gate']['validation']['max_worker_and_validator_point_seconds']
        estimate = 1.5 * 75 * max(live_max, reference) + 30
        created = json.loads((stage.parent / 'state.json').read_text())['created']
        elapsed = max(now - clock['started'], time.time() - created)
        if estimate > spec['full']['seconds'] or estimate + 60 > 408 - elapsed or estimate + 15 > clock['budget_end'] - now:
            raise ValueError('full phase cannot fit actual observed cost or inclusive budget')
        target.write_text(json.dumps({'passed': True, 'checks': {'three_frozen_roles': True,
            'independent_native_yuv': True, 'all_frame_timestamps': True, 'cost_forecast': True},
            'full_forecast_seconds': estimate, 'phase_elapsed_seconds': elapsed}) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'full_native_yuv': False}, 'error': str(exc)}) + '\n')
        raise



def audit_full(archive: Path, expected_sha: str, inventory: dict[str, Any]) -> dict[str, Any]:
    """Validate every saved logical role and billed output without model inference."""
    import hashlib
    import tarfile
    if bp.digest(archive) != expected_sha:
        raise ValueError('full publication SHA256 mismatch')
    selected = rows(inventory)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        if len(members) > 600 or any(not (x.isfile() or x.isdir()) for x in members):
            raise ValueError('unsupported full publication members')
        def read(name: str) -> bytes:
            matches = [m for m in members if m.name == name]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 100_000_000:
                raise ValueError('missing, ambiguous or oversized saved proof')
            f = tar.extractfile(matches[0])
            if f is None:
                raise ValueError('saved proof unavailable')
            return f.read()
        coverage = json.loads(read('publish/coverage.json'))['completed']
        if len(coverage) != 75:
            raise ValueError('full logical coverage incomplete')
        for i, row in enumerate(selected):
            entry = coverage[i]
            folder = f'publish/role-{i:03d}/'
            raw = read(folder + 'proof.json')
            proof = json.loads(raw)
            validation_raw = read(folder + 'independent-validation.json')
            validation = json.loads(validation_raw)
            mkv, returned = read(folder + 'delivered.mkv'), read(folder + 'decoder-scratch.ivf')
            if (entry['index'] != i or entry['window'] != row['window'] or entry['name'] != row['name']
                    or entry['proof_sha256'] != hashlib.sha256(raw).hexdigest()
                    or entry['independent_validation_sha256'] != hashlib.sha256(validation_raw).hexdigest()
                    or any(proof.get(key) != value for key, value in row.items())
                    or proof.get('profile') != profile.PROFILE or proof.get('native_yuv_proven') is not True
                    or proof.get('native_decoder_sha256') != profile.DAV1D_SHA
                    or proof.get('native_yuv_sha256') != row['historical_native_yuv_sha256']
                    or proof.get('ordered_payloads_equal') is not True or proof.get('timestamps_equal') is not True
                    or proof.get('delivered_mkv_bytes') != len(mkv)
                    or proof.get('delivered_mkv_sha256') != hashlib.sha256(mkv).hexdigest()
                    or proof.get('reconstructed_ivf_sha256') != hashlib.sha256(returned).hexdigest()
                    or validation.get('checks') != {'reextract': True, 'payloads': True, 'timestamps': True, 'native_yuv': True}
                    or validation.get('source_ivf_sha256') != row['original_ivf_sha256']
                    or validation.get('delivered_mkv_sha256') != proof['delivered_mkv_sha256']
                    or validation.get('returned_ivf_sha256') != proof['reconstructed_ivf_sha256']
                    or validation.get('native_yuv_sha256') != row['historical_native_yuv_sha256']
                    or validation.get('native_decoder_sha256') != profile.DAV1D_SHA):
                raise ValueError('full role proof, native validation or billed MKV mismatch')
    return {'passed': True, 'logical_roles': 75, 'archive_sha256': expected_sha,
            'scope': 'saved-artifact audit; independent re-extraction/redecode ran in worker for each role'}

def main() -> None:
    import sys
    stage = Path(os.environ['PS_STAGE_DIR'])
    if sys.argv[1:] == ['--validate-smoke']:
        validate(stage, Path(os.environ['PS_VALIDATION_PATH']))
    elif sys.argv[1:] == ['--stage', stage.name] and stage.name in ('smoke', 'full'):
        run(stage, Path(os.environ['PS_SCRATCH_DIR']))
    else:
        raise ValueError('invalid full-study arguments')


if __name__ == '__main__':
    main()

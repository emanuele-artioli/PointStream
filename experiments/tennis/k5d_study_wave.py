"""Read-only wave proposal for the frozen 30-clip K5d study.

This never dispatches, changes a ledger, creates a cost gate, or authorizes GPU
work. An operator must inspect the returned proposal and build each real spec.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from experiments.jobs import inbox
from experiments.tennis import k5d_checkpoint_fleet as cf, k5d_full_study as study
from experiments.tennis import k5d_source_oracle as oracle
from experiments.tennis import k5d_checkpoint as cp, k5d_encoder as enc
from experiments.tennis import k5d_cpu_accounting as accounting

CONFIRMATIONS = ('tracknet/game9/Clip5', 'tracknet/game10/Clip12')
STATES = {'pending', 'running', 'complete', 'failed', 'unknown'}



@contextlib.contextmanager
def unpack_publication(archive: Path, sha: str) -> Iterator[Path]:
    """Extract an uncompressed fleet tar within a bounded local-space reserve."""
    if oracle.sha(archive) != sha:
        raise ValueError('archive SHA256 mismatch')
    archive_bytes = archive.stat().st_size
    # Native and wire archives can exceed the calibration helper's 2 GB limit.
    if not 0 < archive_bytes <= 64 * 1024**3:
        raise ValueError('publication exceeds audit archive bound')
    with tempfile.TemporaryDirectory(prefix='k5d-wave-audit-') as folder:
        if shutil.disk_usage(folder).free < 3 * archive_bytes + 1024**3:
            raise ValueError('insufficient audit scratch space')
        with tarfile.open(archive, 'r:') as bundle:
            members = bundle.getmembers()
            names = [member.name for member in members]
            if (len(members) > 10000 or len(names) != len(set(names))
                    or any(not (m.isfile() or m.isdir()) for m in members)
                    or sum(m.size for m in members) > archive_bytes):
                raise ValueError('invalid publication members or size')
            bundle.extractall(folder, filter='data')
        yield Path(folder) / 'publish'

def read(path: Path, sha: str) -> dict[str, Any]:
    return study.checked_json(path, sha)


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timezone required')
    return parsed.astimezone(timezone.utc)


def verify_completed(clip: str, phase: str, entry: dict[str, Any], frames: int,
                     identity: dict[str, str], preparation: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Verify fleet provenance and independently inspect the actual full artifact."""
    job = entry.get('job_id', '')
    if not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', job):
        raise ValueError('invalid completed job ID')
    receipt = read(Path(entry['receipt_path']), entry['receipt_sha256'])
    if (receipt.get('job_id') != job or receipt.get('state') != 'succeeded'
            or receipt.get('clip') != clip or receipt.get('phase') != phase
            or receipt.get('frames') != frames or receipt.get('execution_identity') != identity
            or receipt.get('gpu_model') != 'RTX A6000'):
        raise ValueError('completed receipt provenance differs')
    names = ('smoke', 'full')
    publications = receipt.get('publications', {})
    if set(publications) != set(names) or entry.get('publications') != publications:
        raise ValueError('complete smoke and full publications required')
    job_dir = Path(receipt['job_directory'])
    if job_dir.name != job:
        raise ValueError('completed job directory differs')
    records_top = receipt['records']
    spec = read(job_dir / 'spec.json', records_top['spec_sha256'])
    ready = read(job_dir / 'ready.json', records_top['ready_sha256'])
    state = read(job_dir / 'state.json', records_top['state_sha256'])
    terminal = read(job_dir / 'run/status.json', records_top['terminal_sha256'])
    gate = read(job_dir / 'gate.json', records_top['gate_sha256'])
    if (spec.get('entrypoint') != ['-m', 'experiments.tennis.k5d_scoped_namespace', 'outer']
            or spec.get('gpu_sharing') != {'mode': 'exclusive'}
            or ready.get('source_sha256') != identity['source_sha256']
            or ready.get('snapshot_sha256') != identity['snapshot_sha256']
            or ready.get('spec_sha256') != inbox.digest(spec)
            or state.get('job_id') != job or state.get('status') != 'complete' or terminal.get('status') != 'complete'
            or terminal.get('exit_code') != 0 or terminal.get('timing_contaminated', False) is not False
            or gate.get('passed') is not True or gate.get('source_sha256') != identity['source_sha256']
            or gate.get('spec_sha256') != inbox.digest(spec) or gate.get('inputs') != spec.get('inputs')
            or gate.get('validation', {}).get('passed') is not True):
        raise ValueError('canonical terminal, source, exclusive namespace or gate differs')
    for stage in names:
        records = receipt['records'][stage]
        dispatch = read(job_dir / stage / 'dispatch.json', records['dispatch_sha256'])
        staging = read(job_dir / stage / 'staging.json', records['staging_sha256'])
        execution = read(job_dir / stage / 'execution.json', records['execution_sha256'])
        artifact = publications[stage]
        control = spec.get('entrypoint') in (['-m', 'experiments.tennis.k5d_control_study'],
                                                ['-m', 'experiments.tennis.k5d_scoped_namespace', 'outer'])
        expected_frames = frames if control or stage == 'full' else 16
        code = dispatch['code']
        gpu = dispatch['environment']['gpu']['name']
        if (execution.get('exit_code') != 0 or execution.get('timing_contaminated') is not False
                or execution.get('contention') != [] or dispatch.get('gpu_sharing') != {'mode': 'exclusive'}
                or code['spec_sha256'] != inbox.digest(spec)
                or code['source_sha256'] != identity['source_sha256']
                or code['snapshot_sha256'] != identity['snapshot_sha256']
                or dispatch['staged_environment']['sha256'] != identity['environment_sha256']
                or dispatch['staged_inputs'] != staging['inputs']
                or dispatch['input_identity'] != spec['inputs']
                or staging['published']['sha256'] != artifact['sha256']
                or Path(staging['published']['path']).parts[-3:] != (job, stage, 'published.tar')
                or 'RTX A6000' not in gpu or 'Ada' in gpu
                or spec['gpu_models'] != ['RTX A6000']
                or cf.option(dispatch['command'], '--clip') != clip
                or cf.option(dispatch['command'], '--frames') != str(expected_frames)
                or spec['entrypoint'][1] not in dispatch['command']
                or '--accounted-child' in dispatch['command']
                or phase not in dispatch['command']):
            raise ValueError('completed fleet execution provenance differs')
        if not cf.digest_string(artifact['sha256']) or oracle.sha(Path(artifact['local_path'])) != artifact['sha256']:
            raise ValueError('completed publication SHA mismatch')
        if phase == 'prepare' and stage == 'full' and not cf.digest_string(artifact.get('checkpoint_sha256', '')):
            raise ValueError('prepared checkpoint digest missing')
        if control:
            if spec.get('entrypoint') != ['-m', 'experiments.tennis.k5d_scoped_namespace', 'outer']:
                raise ValueError('completed control stage requires qualified namespace')
            log_path = job_dir / stage / 'command.log'
            if oracle.sha(log_path) != records['command_log_sha256']:
                raise ValueError('CPU command log SHA mismatch')
            log_text = log_path.read_text()
            records_cpu = [json.loads(line.removeprefix('K5D_CPU_ACCOUNTING ')) for line in log_text.splitlines()
                           if line.startswith('K5D_CPU_ACCOUNTING ')]
            if len(records_cpu) != 1 or records_cpu[0].get('method') != 'linux-pidns-init-waited-rusage-v1':
                raise ValueError('qualified namespace accounting method required')
            accounting.read_receipt(log_text, phase=phase, stage=stage, clip=clip,
                                    frames=frames, identity=identity)
        if control and stage == 'smoke':
            with unpack_publication(Path(artifact['local_path']), artifact['sha256']) as output:
                readiness = json.loads((output / 'readiness.json').read_text())
                if (readiness.get('identity') != identity or readiness.get('source', {}).get('frames') != frames
                        or readiness['source'].get('labels') != frames
                        or readiness.get('scope') != 'CPU asset, source header and checkpoint readiness; no GPU scoring'):
                    raise ValueError('readiness artifact differs')
    return validate_full(clip, phase, frames, identity, publications['full'], preparation, dispatch)



def validate_full(clip: str, phase: str, frames: int, identity: dict[str, str],
                  artifact: dict[str, Any], preparation: dict[str, Any] | None,
                  dispatch: dict[str, Any]) -> dict[str, Any] | None:
    """Reconstruct score from the exact producer and validate native data and metrics."""
    with unpack_publication(Path(artifact['local_path']), artifact['sha256']) as output:
        if phase == 'prepare':
            record = cp.verify(output, artifact['checkpoint_sha256'], identity)
            enc.validate(output, prepared=True)
            if record['clip'] != clip or record['frames'] != frames:
                raise ValueError('prepared full timeline differs')
            prepared = json.loads((output / 'pilot.json').read_text())
            return {'manifest_files': record['files'],
                    'point_metadata_sha256': [inbox.digest(p) for p in prepared['points']]}
        if preparation is None:
            raise ValueError('score requires exact prepared publication')
        prepared = preparation['publications']['full']
        checkpoint_sha = prepared['checkpoint_sha256']
        option = '--checkpoint-sha256' if '--checkpoint-sha256' in dispatch['command'] else '--checkpoint-full-sha256'
        if (cf.option(dispatch['command'], option) != checkpoint_sha
                or not any(x.get('sha256') == prepared['sha256'] for x in dispatch.get('staged_inputs', []))):
            raise ValueError('score dispatch is not bound to prepared publication')
        trusted = preparation.get('_prepared_evidence')
        if trusted is not None:
            assembled = output / 'assembled'
            actual_files = cp.manifest_files(assembled)
            expected_files = trusted['manifest_files']
            changed = {'pilot.json', *(f'{codec}-{point}/point.json' for codec, point in enc.POINTS)}
            if ({k: v for k, v in actual_files.items() if k not in changed}
                    != {k: v for k, v in expected_files.items() if k not in changed}):
                raise ValueError('scored output differs from root-pinned producer manifest')
            summary = json.loads((output / 'coalesced-validation.json').read_text())
            if summary['checkpoint_sha256'] != checkpoint_sha or summary['execution_identity'] != identity:
                raise ValueError('score checkpoint lineage differs')
            report = json.loads((assembled / 'pilot.json').read_text())
            entries = summary['points']
            if len(entries) != 6 or len(report['points']) != 6:
                raise ValueError('six scored points required')
            for index, item in enumerate(entries):
                name = item['name']
                if Path(name).name != name or oracle.sha(output / name) != item['sha256']:
                    raise ValueError('score point hash differs')
                point_record = json.loads((output / name).read_text())
                point = point_record['point']
                baseline = {**point['baseline'], 'metrics': [], 'judges': {}}
                streak = {**point['streak'], 'metrics': [], 'judges': {}, 'reuse': None}
                original = {**point, 'baseline': baseline, 'streak': streak}
                if (point_record['checkpoint_sha256'] != checkpoint_sha or point_record['execution_identity'] != identity
                        or inbox.digest(original) != trusted['point_metadata_sha256'][index]
                        or report['points'][index] != point
                        or point_record['receiver_replay']['decoded_yuv_sha256'] != point['encoder_decoded_sha256']
                        or point_record['receiver_replay']['rendered_rgb_sha256'] != point['encoder_rendered_sha256']):
                    raise ValueError('scored point differs from root-pinned producer')
            enc.validate(assembled)
            return None
        with unpack_publication(Path(prepared['local_path']), prepared['sha256']) as root:
            record = cp.verify(root, checkpoint_sha, identity)
            if record['clip'] != clip or record['frames'] != frames:
                raise ValueError('score producer timeline differs')
            summary = json.loads((output / 'coalesced-validation.json').read_text())
            if summary['checkpoint_sha256'] != checkpoint_sha or summary['execution_identity'] != identity:
                raise ValueError('score checkpoint lineage differs')
            points = []
            for item in summary['points']:
                name = item['name']
                if Path(name).name != name:
                    raise ValueError('unsafe point name')
                points.append({'path': str(output / name), 'sha256': item['sha256']})
            with tempfile.TemporaryDirectory(prefix='k5d-wave-') as folder:
                rebuilt = Path(folder) / 'assembled'
                cp.assemble(root, checkpoint_sha, points, rebuilt, identity)
                actual = output / 'assembled'
                enc.validate(actual)
                if json.loads((rebuilt / 'pilot.json').read_text()) != json.loads((actual / 'pilot.json').read_text()):
                    raise ValueError('score assembly differs from exact producer and point receipts')
        return None




def verify_independent_confirmation(clip: str, phase: str, entry: dict[str, Any],
                                    identity: dict[str, str]) -> None:
    """Require a separately recorded, hash-bound audit of each real confirmation."""
    audit = read(Path(entry['independent_audit_path']), entry['independent_audit_sha256'])
    receipt = read(Path(entry['receipt_path']), entry['receipt_sha256'])
    checks = audit.get('checks', {})
    required = {'fleet_provenance', 'source_frame_identity', 'published_content',
                'namespace_cpu_accounting', 'teardown', 'native_fresh_decode'}
    if phase == 'score':
        required |= {'all_six_metrics_judges', 'producer_checkpoint_join'}
    if (audit.get('schema') != 1 or audit.get('scope') != 'independent full confirmation audit'
            or audit.get('audited') is not True or audit.get('passed') is not True
            or audit.get('job_id') != entry['job_id'] or audit.get('clip') != clip or audit.get('phase') != phase
            or audit.get('execution_identity') != identity or audit.get('receipt_sha256') != entry['receipt_sha256']
            or audit.get('command_log_sha256') != receipt['records']['full']['command_log_sha256']
            or not isinstance(checks, dict) or any(checks.get(k) is not True for k in required)):
        raise ValueError('independent confirmation audit differs')

def verified_cpu_calibration(samples: dict[str, Any], identity: dict[str, str],
                             verified_inventory: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, float]]:
    """Fit CPU only from the two independently validated full confirmation jobs per phase."""
    if set(samples) != {'prepare', 'score'}:
        raise ValueError('both CPU calibration phases required')
    fitted = {}
    selection = study.selection()
    for phase in ('prepare', 'score'):
        entries = samples[phase]
        if not isinstance(entries, list) or len(entries) != 2 or {x.get('clip') for x in entries} != set(CONFIRMATIONS):
            raise ValueError('both registered CPU confirmation clips required')
        values = []
        jobs = set()
        for sample in entries:
            clip = sample['clip']
            frames = selection[clip]['label_rows']
            entry = sample['entry']
            jobs.add(entry['job_id'])
            if verified_inventory is not None:
                saved = verified_inventory[clip][phase]
                if saved.get('state') != 'complete' or any(entry.get(key) != saved.get(key) or key not in entry or key not in saved
                        for key in ('job_id', 'receipt_sha256', 'publications', 'independent_audit_sha256')):
                    raise ValueError('CPU calibration differs from verified inventory')
            if phase == 'score':
                if not sample.get('preparation'):
                    raise ValueError('score calibration needs its actual producer job')
                if verified_inventory is None:
                    verify_completed(clip, 'prepare', sample['preparation'], frames, identity)
                elif sample['preparation']['job_id'] != verified_inventory[clip]['prepare']['job_id']:
                    raise ValueError('CPU score producer differs from verified inventory')
            if verified_inventory is None:
                verify_completed(clip, phase, entry, frames, identity, sample.get('preparation'))
            verify_independent_confirmation(clip, phase, entry, identity)
            receipt = read(Path(entry['receipt_path']), entry['receipt_sha256'])
            directory = Path(receipt['job_directory'])
            spec = read(directory / 'spec.json', receipt['records']['spec_sha256'])
            if spec.get('entrypoint') != ['-m', 'experiments.tennis.k5d_scoped_namespace', 'outer']:
                raise ValueError('CPU calibration requires qualified namespace execution')
            log = directory / 'full' / 'command.log'
            if oracle.sha(log) != receipt['records']['full']['command_log_sha256']:
                raise ValueError('CPU calibration command log changed')
            log_text = log.read_text()
            records_cpu = [json.loads(line.removeprefix('K5D_CPU_ACCOUNTING ')) for line in log_text.splitlines()
                           if line.startswith('K5D_CPU_ACCOUNTING ')]
            if len(records_cpu) != 1 or records_cpu[0].get('method') != 'linux-pidns-init-waited-rusage-v1':
                raise ValueError('CPU calibration lacks namespace accounting')
            seconds = accounting.read_receipt(log_text, phase=phase, stage='full', clip=clip,
                                              frames=frames, identity=identity)
            values.append((frames, seconds))
        if len(jobs) != 2:
            raise ValueError('distinct measured confirmation jobs required')
        values.sort()
        (n1, t1), (n2, t2) = values
        slope = max(0.0, (t2 - t1) / (n2 - n1))
        fixed = max(0.0, t1 - slope * n1, t2 - slope * n2)
        fitted[phase] = {'fixed_seconds': fixed, 'marginal_seconds_per_frame': slope, 'margin': 1.5}
    return fitted


OVERHEAD_KINDS = ('readiness', 'staging', 'validation', 'publication', 'audit', 'isolation')


def verified_noncommand(bundle: dict[str, Any], identity: dict[str, str],
                        rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Verify disjoint measured groups and command-covered isolation."""
    if (bundle.get('schema') != 1 or bundle.get('failure_reserve') != 'one-bounded-campaign'
            or bundle.get('command_covered') not in ([], ['isolation'])
            or not isinstance(bundle.get('groups'), dict) or len(bundle['groups']) != 1):
        raise ValueError('audited noncommand groups and prospective failure reserve required')
    covered = set(bundle['command_covered'])
    for item in bundle['groups'].values():
        scopes = item.get('covers')
        if (not isinstance(scopes, list) or not scopes or len(scopes) != len(set(scopes))
                or not set(scopes).issubset(set(OVERHEAD_KINDS)) or covered.intersection(scopes)):
            raise ValueError('invalid or overlapping noncommand coverage')
        covered.update(scopes)
    if covered != set(OVERHEAD_KINDS):
        raise ValueError('all noncommand categories must be covered')
    jobs = {rows[clip][phase]['job_id']: (clip, phase, rows[clip][phase])
            for clip in CONFIRMATIONS for phase in ('prepare', 'score')}
    if len(jobs) != 4:
        raise ValueError('four distinct confirmation jobs required')
    if bundle['command_covered']:
        for clip in CONFIRMATIONS:
            for phase in ('prepare', 'score'):
                entry = rows[clip][phase]
                receipt = read(Path(entry['receipt_path']), entry['receipt_sha256'])
                path = Path(receipt['job_directory']) / 'full/command.log'
                if oracle.sha(path) != receipt['records']['full']['command_log_sha256']:
                    raise ValueError('command-covered isolation log differs')
                log = path.read_text()
                accounting.read_receipt(log, phase=phase, stage='full', clip=clip,
                                        frames=study.selection()[clip]['label_rows'], identity=identity)
                namespace = [json.loads(line.removeprefix('K5D_NAMESPACE_CANDIDATE '))
                             for line in log.splitlines() if line.startswith('K5D_NAMESPACE_CANDIDATE ')]
                if len(namespace) != 1 or namespace[0].get('init_cpu_seconds') is None:
                    raise ValueError('command-covered isolation CPU absent')
    result = {}
    measured_intervals: dict[str, list[tuple[float, float]]] = {}
    now = datetime.now(timezone.utc)
    for kind, item in bundle['groups'].items():
        if not isinstance(kind, str) or not kind or item.get('frequency') != 'per_campaign':
            raise ValueError('noncommand group identity or frequency differs')
        raw = read(Path(item['raw_path']), item['raw_sha256'])
        audit = read(Path(item['audit_path']), item['audit_sha256'])
        job = raw.get('job_id')
        if (not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', str(job))
                or job not in jobs or (raw.get('clip'), raw.get('phase')) != jobs[job][:2]
                or raw.get('schema') != 1 or raw.get('kind') != kind or raw.get('covers') != item['covers']
                or raw.get('execution_identity') != identity
                or raw.get('cpu_method') not in ('waited-rusage-delta-v1', 'single-cpu-interval-upper-bound-v1')
                or raw.get('complete_descendant_wait') is not True
                or raw.get('observed_at') is None or timestamp(raw['observed_at']) > now
                or audit.get('schema') != 1 or audit.get('scope') != 'independent noncommand measurement audit'
                or audit.get('audited') is not True or audit.get('passed') is not True
                or audit.get('job_id') != job or audit.get('kind') != kind or audit.get('covers') != item['covers']
                or audit.get('raw_sha256') != item['raw_sha256'] or audit.get('execution_identity') != identity
                or audit.get('checks', {}).get('raw_job_and_scope_join') is not True
                or audit.get('checks', {}).get('complete_process_tree') is not True
                or audit.get('checks', {}).get('gpu_claim_coverage') is not True):
            raise ValueError('noncommand evidence identity or independent audit differs')
        # The checked fleet records bind every group to one actual confirmation.
        directory = Path(raw['job_directory'])
        if directory.name != job:
            raise ValueError('noncommand job directory differs')
        job_entry = jobs[job][2]
        job_receipt = read(Path(job_entry['receipt_path']), job_entry['receipt_sha256'])
        if Path(job_receipt['job_directory']) != directory:
            raise ValueError('noncommand measurement uses another job directory')
        status = read(directory / 'run/status.json', raw['status_sha256'])
        spec = read(directory / 'spec.json', raw['spec_sha256'])
        if status.get('status') != 'complete' or not isinstance(spec.get('entrypoint'), list):
            raise ValueError('noncommand source job status differs')
        stage = raw['stage']
        if stage not in ('smoke', 'full'):
            raise ValueError('invalid noncommand stage')
        dispatch = read(directory / stage / 'dispatch.json', raw['dispatch_sha256'])
        code = dispatch['code']
        if (code.get('source_sha256') != identity['source_sha256']
                or code.get('snapshot_sha256') != identity['snapshot_sha256']
                or dispatch['staged_environment']['sha256'] != identity['environment_sha256']
                or code.get('spec_sha256') != inbox.digest(spec)
                or not raw.get('gpu_uuid') or raw['gpu_uuid'] != dispatch['environment']['gpu']['uuid']):
            raise ValueError('noncommand fleet dispatch identity differs')
        canonical = read(directory / 'state.json', raw['state_sha256'])
        supervisor = read(directory / 'run/supervisor.json', raw['supervisor_sha256'])
        claims = read(directory / 'run/claim.json', raw['claim_sha256'])
        claim = claims['device_claim']
        if (canonical.get('job_id') != job or canonical.get('status') != 'complete'
                or claim.get('device_uuid') != raw['gpu_uuid'] or claim.get('sharing') != 'exclusive'
                or claim.get('pid') != supervisor.get('pid')
                or claim.get('proc_start_time') != supervisor.get('proc_start_time')
                or claim.get('job_dir') != str(directory / 'run')
                or not (canonical['created'] - 1.1 <= supervisor['proc_start_time'] <= claim['created_at']
                        <= status['ended'] <= canonical['updated'])
                or raw.get('gpu_claim_start') != supervisor['proc_start_time']
                or raw.get('gpu_claim_end') != canonical['updated']):
            raise ValueError('GPU claim interval differs from canonical supervisor and claim')
        before, after = raw['cpu_before'], raw['cpu_after']
        counter_records = []
        for label in ('before', 'after'):
            counter_path = Path(raw[f'counter_{label}_path']).resolve(strict=True)
            if not counter_path.is_relative_to(directory.resolve(strict=True)):
                raise ValueError('CPU counter source is outside its recorded job')
            counter_records.append(read(counter_path, raw[f'counter_{label}_sha256']))
        for declared, measured_record in zip((before, after), counter_records):
            child = measured_record.get('children_resource_usage')
            if (not isinstance(child, list) or len(child) < 2
                    or declared.get('children_user') != child[0] or declared.get('children_system') != child[1]):
                raise ValueError('CPU child counters differ from hashed raw execution')
            if raw['cpu_method'] == 'waited-rusage-delta-v1':
                own = measured_record.get('self_resource_usage')
                if (not isinstance(own, list) or len(own) < 2
                        or declared.get('self_user') != own[0] or declared.get('self_system') != own[1]):
                    raise ValueError('measured supervisor counters unavailable or mismatched')
            elif (raw.get('self_cpu_observed') is not False or declared.get('self_user') != 0
                  or declared.get('self_system') != 0):
                raise ValueError('upper bound must disclose unmeasured supervisor CPU')
        cpu_values = [before.get(k) for k in ('self_user', 'self_system', 'children_user', 'children_system')]
        end_values = [after.get(k) for k in ('self_user', 'self_system', 'children_user', 'children_system')]
        values = [*cpu_values, *end_values, raw.get('gpu_claim_start'), raw.get('gpu_claim_end'),
                  raw.get('interval_start'), raw.get('interval_end')]
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values)
                or any(b < a for a, b in zip(cpu_values, end_values))
                or raw['gpu_claim_end'] < raw['gpu_claim_start'] or raw['interval_end'] <= raw['interval_start']):
            raise ValueError('invalid noncommand measurement counters')
        interval = (float(raw['interval_start']), float(raw['interval_end']))
        previous = measured_intervals.setdefault(job, [])
        if any(interval[0] < end and start < interval[1] for start, end in previous):
            raise ValueError('overlapping noncommand intervals must be one group')
        previous.append(interval)
        measured = sum(b - a for a, b in zip(cpu_values, end_values))
        if measured <= 0:
            raise ValueError('noncommand raw CPU counters missing')
        if raw['cpu_method'] == 'single-cpu-interval-upper-bound-v1':
            affinity = dispatch.get('cpu_affinity')
            if (spec.get('single_cpu_campaign') is not True or spec.get('cpu_threads') != 1
                    or not isinstance(affinity, list) or len(affinity) != 1
                    or audit['checks'].get('one_cpu_affinity_and_tree') is not True
                    or audit['checks'].get('external_costs_covered') is not True):
                raise ValueError('single-CPU interval lacks audited affinity, tree or external scope')
            cpu = raw['interval_end'] - raw['interval_start']
            if measured > cpu + 0.05:
                raise ValueError('CPU counters exceed single-CPU interval')
        else:
            cpu = measured
        full_dispatch = read(directory / 'full/dispatch.json', raw['full_dispatch_sha256'])
        full_execution = read(directory / 'full/execution.json', raw['full_execution_sha256'])
        full_elapsed = full_execution.get('seconds')
        if (full_execution.get('exit_code') != 0 or full_execution.get('timing_contaminated') is not False
                or full_execution.get('contention') != [] or full_dispatch.get('gpu_sharing') != {'mode': 'exclusive'}
                or full_dispatch.get('code', {}).get('source_sha256') != identity['source_sha256']
                or full_dispatch.get('code', {}).get('snapshot_sha256') != identity['snapshot_sha256']
                or isinstance(full_elapsed, bool) or not isinstance(full_elapsed, (float, int))
                or not math.isfinite(full_elapsed) or full_elapsed <= 0
                or not supervisor['proc_start_time'] <= full_dispatch['started']
                or full_dispatch['started'] + full_elapsed > status['ended']):
            raise ValueError('full command is not contained in authoritative claim interval')
        gpu = raw['gpu_claim_end'] - raw['gpu_claim_start'] - full_elapsed
        if gpu <= 0:
            raise ValueError('noncommand GPU claim interval invalid')
        if raw['cpu_method'] == 'single-cpu-interval-upper-bound-v1':
            if (raw['interval_start'] != supervisor['proc_start_time']
                    or raw['interval_end'] != canonical['updated']
                    or audit['checks'].get('full_command_contained') is not True):
                raise ValueError('single-CPU interval differs from canonical whole campaign')
            full_log = directory / 'full/command.log'
            if oracle.sha(full_log) != job_receipt['records']['full']['command_log_sha256']:
                raise ValueError('noncommand full CPU log differs')
            cpu_command = accounting.read_receipt(full_log.read_text(), phase=raw['phase'], stage='full',
                         clip=raw['clip'], frames=study.selection()[raw['clip']]['label_rows'], identity=identity)
            cpu -= cpu_command
        elif audit['checks'].get('full_command_excluded') is not True:
            raise ValueError('measured noncommand CPU overlaps full command')
        if cpu <= 0:
            raise ValueError('noncommand CPU bound invalid')
        # External transport, authoring and audit are outside the one-core
        # campaign interval. Their independently audited caps are charged once.
        external = raw.get('external_cpu')
        if (not isinstance(external, dict) or not isinstance(external.get('sources'), list)
                or not external['sources'] or audit.get('external_cpu') != external
                or audit['checks'].get('external_caps_source_join') is not True):
            raise ValueError('hash-bound external CPU cap evidence required')
        external_cpu = external.get('upper_seconds')
        if (isinstance(external_cpu, bool) or not isinstance(external_cpu, (int, float))
                or not math.isfinite(external_cpu) or external_cpu <= 0):
            raise ValueError('invalid external CPU cap')
        for source in external['sources']:
            if not isinstance(source, dict) or set(source) != {'path', 'sha256'}:
                raise ValueError('invalid external CPU source')
            if oracle.sha(Path(source['path'])) != source['sha256']:
                raise ValueError('external CPU source differs')
        cpu += external_cpu
        result[kind] = {'cpu': 1.5 * cpu, 'gpu': 1.5 * gpu, 'cpu_method': raw['cpu_method'],
                        'covers': item['covers'], 'frequency': 'per_campaign'}
    return {'groups': result, 'command_covered': bundle['command_covered'],
            'failure_reserve': 'one-bounded-campaign'}


def completed_summary(entry: dict[str, Any]) -> dict[str, Any]:
    """Small immutable stage identity carried forward by a root-pinned certificate."""
    return {'job_id': entry['job_id'], 'receipt_sha256': entry['receipt_sha256'],
            'independent_audit_sha256': entry.get('independent_audit_sha256'),
            'publications': {stage: {'sha256': item['sha256'],
                **({'checkpoint_sha256': item['checkpoint_sha256']} if 'checkpoint_sha256' in item else {})}
                for stage, item in entry['publications'].items()},
            **({'prepared_evidence': entry['_prepared_evidence']} if entry.get('_prepared_evidence') is not None else {})}


def inventory(ledger: dict[str, Any], identity: dict[str, str],
              trusted_completed: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    selected = study.selection()
    if (ledger.get('schema') != 1 or ledger.get('selection_sha256') != study.fleet.SOURCE_SHA
            or ledger.get('execution_identity') != identity or ledger.get('gpu_model') != 'RTX A6000'):
        raise ValueError('ledger study identity differs')
    rows = ledger.get('clips')
    if not isinstance(rows, list) or len(rows) != 30:
        raise ValueError('complete 30-clip ledger required')
    by_clip = {row['clip']: row for row in rows}
    if len(by_clip) != 30 or set(by_clip) != set(selected):
        raise ValueError('ledger clip set differs from frozen inventory')
    for clip, row in by_clip.items():
        frames = selected[clip]['label_rows']
        if row.get('frames') != frames:
            raise ValueError('ledger changes a registered timeline')
        for phase in ('prepare', 'score'):
            entry = row[phase]
            if entry.get('state') not in STATES:
                raise ValueError('invalid ledger stage state')
            if entry['state'] == 'complete':
                previous = (trusted_completed or {}).get(clip, {}).get(phase)
                if previous is None:
                    proof = verify_completed(clip, phase, entry, frames, identity, row['prepare'] if phase == 'score' else None)
                    if phase == 'prepare':
                        entry['_prepared_evidence'] = proof
                else:
                    expected = {k: v for k, v in previous.items() if k != 'prepared_evidence'}
                    if completed_summary(entry) != expected:
                        raise ValueError('completed stage changed since root-pinned certificate')
                    if phase == 'prepare':
                        if not isinstance(previous.get('prepared_evidence'), dict):
                            raise ValueError('root-pinned producer evidence missing')
                        entry['_prepared_evidence'] = previous['prepared_evidence']
            elif entry['state'] == 'pending' and set(entry) != {'state'}:
                raise ValueError('pending work has unresolved prior submission')
            elif entry['state'] != 'pending' and not entry.get('job_id'):
                raise ValueError('unfinished stage lacks job identity')
        if row['score']['state'] != 'pending' and row['prepare']['state'] != 'complete':
            raise ValueError('score lacks completed preparation')
    return by_clip


def inspect_hosts(snapshot: dict[str, Any], now: datetime) -> None:
    if (snapshot.get('schema') != 1 or snapshot.get('gpu_models') != ['RTX A6000']
            or set(snapshot.get('inspected_hosts', [])) != {f'gpu{i}' for i in range(1, 7)}
            or timestamp(snapshot['observed_at']) > now or timestamp(snapshot['valid_until']) <= now):
        raise ValueError('fresh all-host inspection required')
    if snapshot.get('active_or_uncertain_jobs') != []:
        raise ValueError('active or uncertain jobs require inspection before any proposal')


def propose(*, ledger_path: Path, ledger_sha: str, gate_path: Path, gate_sha: str,
            evidence_path: Path, evidence_sha: str, inspection_path: Path, inspection_sha: str,
            identity: dict[str, str], gpu_budget_seconds: float, cpu_budget_seconds: float,
            now: datetime, control_assets: dict[str, str] | None = None) -> dict[str, Any]:
    """Return at most one bounded prepare/score job, never submit it."""
    if now.tzinfo is None:
        raise ValueError('timezone required')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
           for x in (gpu_budget_seconds, cpu_budget_seconds)):
        raise ValueError('positive explicitly remaining budgets required')
    cert = None
    if control_assets is not None:
        from experiments.tennis import k5d_full_admission as admission
        gate = read(gate_path, gate_sha)
        cert = admission.read_certificate(Path(control_assets['certificate']), gate['certificate_sha256'])
    rows = inventory(read(ledger_path, ledger_sha), identity,
                     trusted_completed=None if cert is None else cert['completed'])
    inspect_hosts(read(inspection_path, inspection_sha), now.astimezone(timezone.utc))
    uncertain = [(clip, phase, row[phase]['state']) for clip, row in rows.items() for phase in ('prepare', 'score')
                 if row[phase]['state'] in ('running', 'failed', 'unknown')]
    if uncertain:
        raise ValueError('existing or uncertain job must be inspected; no replay')
    confirmations_done = all(rows[clip]['score']['state'] == 'complete' for clip in CONFIRMATIONS)
    candidates = list(CONFIRMATIONS) if not confirmations_done else sorted(rows)
    for clip in candidates:
        row = rows[clip]
        if row['prepare']['state'] == 'pending':
            phase = 'prepare'
        elif row['score']['state'] == 'pending':
            phase = 'score'
        else:
            continue
        if control_assets is None:
            _, evidence, _ = study.admission(gate_path, gate_sha, evidence_path, evidence_sha, clip, identity)
            cost = evidence['cost'][phase]
            if isinstance(cost.get('margin'), bool) or not isinstance(cost.get('margin'), (int, float)) or not math.isfinite(cost['margin']) or cost['margin'] < 1.5:
                raise ValueError('cost margin must be at least 1.5')
            raise ValueError('process-tree CPU bound is not established; no wave proposal')
        from experiments.tennis import k5d_control_study as control
        required = {'bridge_path', 'bridge_sha', 'calibration_source', 'consumer_source', 'budget_ledger',
                    'budget_ledger_sha', 'budget_attestation', 'budget_attestation_sha', 'native_cpu_receipt',
                    'gpu_parity_receipt', 'certificate', 'microbenchmark', 'microbenchmark_audit'}
        if set(control_assets) != required:
            raise ValueError('complete full-study control evidence required')
        a = control_assets
        evidence, frames = control.approved(gate_path, gate_sha, evidence_path, evidence_sha,
            Path(a['bridge_path']), a['bridge_sha'], Path(a['calibration_source']), Path(a['consumer_source']),
            Path(a['budget_ledger']), a['budget_ledger_sha'], Path(a['budget_attestation']), a['budget_attestation_sha'],
            identity, clip, phase, {'native_cpu': Path(a['native_cpu_receipt']), 'gpu_parity': Path(a['gpu_parity_receipt'])},
            {'certificate': Path(a['certificate']), 'microbenchmark': Path(a['microbenchmark']),
             'microbenchmark_audit': Path(a['microbenchmark_audit'])})
        assert cert is not None
        cpu_model = cert['cpu_models'][phase]
        gpu_full = study.forecast(evidence, phase, frames, evidence['cost'][phase]['seconds16'])
        cpu_full = cpu_model['margin'] * (cpu_model['fixed_seconds'] + cpu_model['marginal_seconds_per_frame'] * frames)
        caps = {}
        for resource, command in (('gpu', gpu_full), ('cpu', cpu_full)):
            d = evidence['programme'][resource]
            caps[resource] = command + sum(d['noncommand_groups'].values()) + d['failed_start_reserve_seconds']
        if gpu_full > 2670 or caps['gpu'] > gpu_budget_seconds or caps['cpu'] > cpu_budget_seconds:
            raise ValueError('next stage exceeds recovery or remaining budgets')
        return {'schema': 1, 'action': phase, 'clip': clip, 'frames': frames, 'execution_identity': identity,
                'ledger_sha256': ledger_sha, 'gpu_seconds_upper': caps['gpu'], 'cpu_seconds_upper': caps['cpu'],
                'scope': 'review-only qualified remaining-stage proposal; no dispatch or authority'}
    return {'schema': 1, 'action': 'complete', 'selection_sha256': study.fleet.SOURCE_SHA,
            'ledger_sha256': ledger_sha, 'scope': 'all registered clips have completed prepare and score receipts'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.config.read_text())
    for name in ('ledger_path', 'gate_path', 'evidence_path', 'inspection_path'):
        data[name] = Path(data[name])
    result = propose(**data, now=datetime.now(timezone.utc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

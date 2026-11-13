"""Author and verify a compact, root-pinned full-study admission certificate.

Authoring is a one-time server-side raw audit. A worker trusts only the exact
certificate SHA independently reviewed and pinned in its gate; it never opens
prior publication or noncommand paths named by that certificate. This trust
boundary is intentional: the certificate is not an independent audit itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_checkpoint_fleet as cf, k5d_fleet as fleet
from experiments.tennis import k5d_full_study as study, k5d_source_oracle as oracle

MAX_CERT_BYTES = 256 * 1024
DATA_ROOT = Path('/home/itec/emanuele/Datasets/pointstream-data')


def read_certificate(path: Path, expected: Any) -> dict[str, Any]:
    if not isinstance(expected, str) or not cf.digest_string(expected):
        raise ValueError('root-pinned certificate SHA required')
    if not path.is_file() or path.stat().st_size > MAX_CERT_BYTES:
        raise ValueError('certificate missing or exceeds size limit')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('certificate SHA differs from root-pinned gate')
    result = json.loads(raw)
    if not isinstance(result, dict) or result.get('schema') != 1:
        raise ValueError('certificate schema differs')
    return result



def check_small_paths(paths: list[Path]) -> None:
    for path in paths:
        if not path.is_file() or path.stat().st_size > MAX_CERT_BYTES:
            raise ValueError('compact worker input missing or exceeds size limit')


def check_microbenchmark(path: Path, audit_path: Path, gate: dict[str, Any], identity: dict[str, str], cert_sha: str) -> None:
    check_small_paths([path, audit_path])
    expected, audit_sha = gate.get('certificate_microbench_sha256'), gate.get('certificate_microbench_audit_sha256')
    if (not isinstance(expected, str) or not cf.digest_string(expected) or oracle.sha(path) != expected
            or not isinstance(audit_sha, str) or not cf.digest_string(audit_sha) or oracle.sha(audit_path) != audit_sha):
        raise ValueError('root-pinned microbenchmark and independent audit SHAs required')
    result, audit = json.loads(path.read_text()), json.loads(audit_path.read_text())
    cap = gate.get('certificate_validation_reserve_seconds')
    if (isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0
            or result.get('scope') != 'actual compact certificate verification microbenchmark'
            or result.get('certificate_sha256') != cert_sha or result.get('execution_identity') != identity
            or not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', str(result.get('job_id')))
            or type(result.get('iterations')) is not int or result['iterations'] <= 0
            or any(isinstance(result.get(k), bool) or not isinstance(result.get(k), (int, float))
                   or not math.isfinite(result[k]) or result[k] < 0 for k in ('wall_seconds', 'cpu_seconds'))
            or 1.5 * result['wall_seconds'] > cap or audit.get('scope') != 'independent certificate microbenchmark audit'
            or audit.get('passed') is not True or audit.get('raw_sha256') != expected
            or audit.get('job_id') != result['job_id'] or audit.get('execution_identity') != identity):
        raise ValueError('certificate microbenchmark is unqualified or over reserve')


def _deadline(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('certificate deadline needs timezone')
    return parsed.astimezone(timezone.utc)


def check_certificate(cert: dict[str, Any], gate: dict[str, Any], identity: dict[str, str],
                      clip: str, phase: str, cost_sha: str, ledger_sha: str, attestation_sha: str,
                      qualification_paths: dict[str, Path] | None, receipt: dict[str, Any]) -> None:
    """Cheap verifier: root's pinned hash attests the expensive primary audit."""
    from experiments.tennis import k5d_ada_study as ada
    gpu_profile = ada.profile(gate)
    model, capability = ada.hardware(gate)
    if ada.profile(cert) != gpu_profile or ada.profile(receipt) != gpu_profile:
        raise ValueError('certificate, cost and gate GPU profiles differ')
    if (gate.get('schema') != 1 or gate.get('approved') is not True or gate.get('release') != 'full-study'
            or gate.get('purpose') != 'full-retrospective-benchmark-cpu-readiness'
            or gate.get('selection_sha256') != fleet.SOURCE_SHA or gate.get('consumer_execution_identity') != identity
            or gate.get('gpu_models') != [model] or gate.get('gpu_capability') != capability
            or gate.get('fidelity_only_for_unqualified') is not True
            or gate.get('cost_receipt_sha256') != cost_sha
            or cert.get('scope') != 'server-side full raw admission audit; root must review exact SHA'
            or cert.get('selection_sha256') != fleet.SOURCE_SHA or cert.get('execution_identity') != identity
            or cert.get('cost_sha256') != cost_sha or cert.get('cost_core_sha256') != cost_core_sha256(receipt)
            or cert.get('ledger_sha256') != ledger_sha
            or cert.get('attestation_sha256') != attestation_sha
            or cert.get('qualification') != gate.get('confirmation_qualification')
            or not isinstance(cert.get('targets'), dict) or len(cert['targets']) > 30
            or _deadline(cert['valid_until']) <= datetime.now(timezone.utc)):
        raise ValueError('certificate, gate, budget, source or validity differs')
    if gpu_profile == ada.ADA and cert.get('receiver_isolation_config_sha256') != gate.get('receiver_isolation_config_sha256'):
        raise ValueError('Ada certificate isolation config differs')
    target = cert['targets'].get(clip, {}).get(phase)
    if (phase not in ('prepare', 'score') or clip not in (gate.get('allowed_clips') or [])
            or not isinstance(target, dict) or target.get('state') != 'pending'
            or type(target.get('frames')) is not int or target['frames'] <= 0):
        raise ValueError('target is not certified pending')
    if phase == 'score' and not isinstance(target.get('producer'), dict):
        raise ValueError('score target lacks certified producer')
    authored, expires = _deadline(cert['authored_at']), _deadline(cert['valid_until'])
    now = datetime.now(timezone.utc)
    if authored > now or not 0 < (expires - authored).total_seconds() <= 28800:
        raise ValueError('certificate timestamp range differs')
    from experiments.tennis import k5d_control_study as control
    selected = study.selection()
    pending = cert.get('pending')
    if (not isinstance(pending, list) or len(pending) > 60
            or any(not isinstance(x, list) or len(x) != 2 or x[0] not in selected
                   or x[1] not in ('prepare', 'score') for x in pending)
            or len({tuple(x) for x in pending}) != len(pending)):
        raise ValueError('certificate pending inventory differs')
    pending_set = {tuple(x) for x in pending}
    completed = cert.get('completed')
    if not isinstance(completed, dict) or len(completed) > 30 or any(
            c not in selected or not isinstance(phases, dict) or not set(phases).issubset({'prepare', 'score'})
            for c, phases in completed.items()):
        raise ValueError('certificate completed inventory differs')
    completed_set = {(c, p) for c, phases in completed.items() for p in phases}
    if (pending_set & completed_set or pending_set | completed_set !=
            {(c, p) for c in selected for p in ('prepare', 'score')}):
        raise ValueError('certificate does not partition the complete inventory')
    for c, p in completed_set:
        item = completed[c][p]
        if (not isinstance(item, dict) or not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', str(item.get('job_id')))
                or not isinstance(item.get('receipt_sha256'), str) or not cf.digest_string(item['receipt_sha256'])
                or not isinstance(item.get('publications'), dict) or set(item['publications']) != {'smoke', 'full'}
                or any(not isinstance(x.get('sha256'), str) or not cf.digest_string(x['sha256'])
                       for x in item['publications'].values())):
            raise ValueError('certificate completed-stage proof differs')
        if c in ('tracknet/game9/Clip5', 'tracknet/game10/Clip12') and (
                not isinstance(item.get('independent_audit_sha256'), str)
                or not cf.digest_string(item['independent_audit_sha256'])):
            raise ValueError('certificate confirmation audit identity missing')
    inventory = {c: {p: {'state': 'pending' if (c, p) in pending_set else 'complete'}
                     for p in ('prepare', 'score')} for c in selected}
    if not all((c, p) in completed_set for c in ('tracknet/game9/Clip5', 'tracknet/game10/Clip12')
               for p in ('prepare', 'score')):
        raise ValueError('certificate lacks all four confirmation stages')
    blocked = cert.get('blocked_targets')
    if (not isinstance(blocked, list) or any(not isinstance(x, list) or len(x) != 2 for x in blocked)
            or len({tuple(x) for x in blocked}) != len(blocked)
            or not {tuple(x) for x in blocked}.issubset(pending_set)):
        raise ValueError('certificate blocked targets differ')
    expected = {(c, p) for c, p in pending_set
                if (p == 'prepare' or inventory[c]['prepare']['state'] == 'complete') and [c, p] not in blocked}
    actual = {(c, p) for c, phases in cert['targets'].items()
              for p in phases if isinstance(phases, dict)}
    if actual != expected or any(not isinstance(phases, dict) for phases in cert['targets'].values()):
        raise ValueError('certificate targets differ from pending inventory')
    for c, p in actual:
        entry = cert['targets'][c][p]
        if entry.get('frames') != selected[c]['label_rows'] or entry.get('state') != 'pending':
            raise ValueError('certificate target timeline differs')
        if p == 'score':
            producer = entry.get('producer', {})
            completed = cert.get('completed', {}).get(c, {}).get('prepare', {})
            publication = completed.get('publications', {}).get('full', {})
            job = completed.get('job_id')
            remote = DATA_ROOT / 'jobs/fleet/inbox' / str(job) / 'full/published.tar'
            if gpu_profile == ada.ADA:
                ada.score_producer(gate, c, cert)
            if (producer.get('job_id') != job or producer.get('remote') != str(remote)
                    or producer.get('archive_sha256') != publication.get('sha256')
                    or producer.get('checkpoint_sha256') != publication.get('checkpoint_sha256')
                    or not all(isinstance(producer.get(k), str) and cf.digest_string(producer[k])
                               for k in ('archive_sha256', 'checkpoint_sha256'))):
                raise ValueError('certified score producer differs from verified publication')
    actual_forecast = control.programme_projection(receipt, inventory=inventory,
                        noncommand=cert['noncommand'], certified_cpu=cert['cpu_models'])
    if actual_forecast != cert.get('forecast') or not all(actual_forecast[r]['fits'] is True for r in ('gpu', 'cpu')):
        raise ValueError('compact recomputed budget differs or does not fit')
    if qualification_paths is None or set(qualification_paths) != {'native_cpu', 'gpu_parity'}:
        raise ValueError('actual qualification receipt paths required')
    q = cert['qualification']
    parity = study.checked_json(qualification_paths['gpu_parity'], q['gpu_parity_receipt_sha256'])
    if gpu_profile == ada.ADA:
        ada.qualification(gate, parity, identity)
        ada.validate_calibration(receipt)
        if receipt['ada_calibration']['config_sha256'] != gate['receiver_isolation_config_sha256']:
            raise ValueError('Ada calibration and certified config differ')
    bootstrap = parity.get('namespace_bootstrap_seconds')
    reserve = gate.get('namespace_bootstrap_reserve_seconds')
    if (isinstance(bootstrap, bool) or not isinstance(bootstrap, (int, float)) or not math.isfinite(bootstrap)
            or bootstrap < 0 or isinstance(reserve, bool) or not isinstance(reserve, (int, float))
            or not math.isfinite(reserve) or reserve < max(5.0, 1.5 * bootstrap)
            or parity.get('namespace_bootstrap_audited') is not True):
        raise ValueError('namespace bootstrap reserve lacks measured audited parity evidence')
    for name, key in (('native_cpu', 'native_cpu_receipt_sha256'), ('gpu_parity', 'gpu_parity_receipt_sha256')):
        if oracle.sha(qualification_paths[name]) != q.get(key):
            raise ValueError('qualification receipt SHA differs')


def target_claim(cert: dict[str, Any], cert_sha: str, clip: str, phase: str, job: str) -> None:
    """Persistent one-winner claim; a prior different job requires manual audit."""
    if not job or '/' in job or job in ('.', '..'):
        raise ValueError('invalid job identity')
    key = hashlib.sha256(json.dumps([cert['execution_identity'], clip, phase], sort_keys=True).encode()).hexdigest()
    root = DATA_ROOT / 'jobs/fleet/k5d-full-target-claims'
    root.mkdir(parents=True, exist_ok=True)
    claim = root / key
    record = {'job_id': job, 'clip': clip, 'phase': phase, 'certificate_sha256': cert_sha}
    try:
        claim.mkdir()
    except FileExistsError:
        existing = json.loads((claim / 'identity.json').read_text())
        if existing != record:
            raise ValueError('target already claimed; inspect prior job without replay')
        return
    # Do not remove a partial claim on failure: that requires manual inspection.
    with (claim / 'identity.json').open('x') as out:
        json.dump(record, out, sort_keys=True)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())



COST_CORE_KEYS = ('schema', 'selection_sha256', 'calibration_execution_identity', 'consumer_execution_identity',
                  'bridge_sha256', 'gpu_model', 'gpu_capability', 'calibration_jobs', 'stage_caps_fit',
                  'calibration_archives', 'calibration_audit_sha256', 'score_timing_sha256',
                  'calibration_execution_sha256', 'cost', 'cpu_calibration')


def cost_core_sha256(receipt: dict[str, Any]) -> str:
    """Exclude only mutable budget/allocation fields from carried scientific evidence."""
    from experiments.jobs.inbox import digest
    if any(key not in receipt for key in COST_CORE_KEYS):
        raise ValueError('immutable scientific cost core incomplete')
    from experiments.tennis import k5d_ada_study as ada
    keys = COST_CORE_KEYS + (('registered_gpu_profile', 'ada_calibration') if ada.profile(receipt) == ada.ADA else ())
    if any(key not in receipt for key in keys):
        raise ValueError('Ada scientific cost core incomplete')
    return digest({key: receipt[key] for key in keys})

def author(config: dict[str, Any]) -> dict[str, Any]:
    """Run expensive immutable raw joins on the fleet server, before root review."""
    from experiments.tennis import k5d_control_study as control, k5d_study_wave as wave

    identity = config['identity']
    raw = config['raw']
    for item in raw.values():
        if not isinstance(item, dict) or 'path' not in item or 'sha256' not in item:
            raise ValueError('raw audit inputs require path and SHA')
    def path(name: str) -> Path:
        p = Path(raw[name]['path']).resolve(strict=True)
        if not p.is_relative_to(DATA_ROOT):
            raise ValueError('raw authoring input outside shared data root')
        if oracle.sha(p) != raw[name]['sha256']:
            raise ValueError('raw authoring input SHA differs')
        return p
    sources = {name: path(name) for name in raw}
    gate = study.checked_json(sources['gate'], raw['gate']['sha256'])
    incoming_cost = study.checked_json(sources['cost'], raw['cost']['sha256'])
    core_sha = cost_core_sha256(incoming_cost)
    from experiments.tennis import k5d_ada_study as ada
    gpu_profile = ada.profile(gate)
    if ada.profile(incoming_cost) != gpu_profile:
        raise ValueError('cost and authoring GPU profile differ')
    if gpu_profile == ada.ADA:
        ada.validate_calibration(incoming_cost)
        if incoming_cost['ada_calibration']['config_sha256'] != gate.get('receiver_isolation_config_sha256'):
            raise ValueError('Ada calibration and authoring config differ')
    if gate.get('release') != 'full-study':
        raise ValueError('full-study draft gate required')
    status = study.checked_json(sources['canonical_status'], raw['canonical_status']['sha256'])
    captured = _deadline(status['captured_at'])
    now = datetime.now(timezone.utc)
    if captured > now or (now - captured).total_seconds() > 120 or status.get('scope') != 'canonical all-job fleet status':
        raise ValueError('fresh canonical all-job status required')
    own_job = Path(os.environ['PS_STAGE_DIR']).parent.name
    inventory_doc = study.checked_json(sources['inventory'], raw['inventory']['sha256'])
    prior = None
    if 'prior_certificate' in sources:
        if 'prior_gate' not in sources:
            raise ValueError('prior root-pinned gate required for advancement')
        previous_gate = study.checked_json(sources['prior_gate'], raw['prior_gate']['sha256'])
        if (previous_gate.get('approved') is not True or previous_gate.get('release') != 'full-study'
                or previous_gate.get('certificate_sha256') != raw['prior_certificate']['sha256']):
            raise ValueError('prior certificate was not pinned by an approved gate')
        prior = read_certificate(sources['prior_certificate'], raw['prior_certificate']['sha256'])
        if (prior.get('execution_identity') != identity or prior.get('selection_sha256') != fleet.SOURCE_SHA
                or prior.get('cost_core_sha256') != core_sha
                or prior.get('qualification') != gate.get('confirmation_qualification')
                or ada.profile(prior) != gpu_profile or ada.profile(previous_gate) != gpu_profile
                or (gpu_profile == ada.ADA and prior.get('receiver_isolation_config_sha256') != gate.get('receiver_isolation_config_sha256'))):
            raise ValueError('prior audited source, costs or qualifications differ')
    inventory = wave.inventory(inventory_doc, identity, prior.get('completed') if prior else None, gpu_profile=gpu_profile) if gpu_profile == ada.ADA else wave.inventory(inventory_doc, identity, prior.get('completed') if prior else None)
    pending = [(clip, phase) for clip, row in inventory.items() for phase in ('prepare', 'score')
               if row[phase]['state'] == 'pending' and (phase == 'prepare' or row['prepare']['state'] == 'complete')]
    if not pending:
        raise ValueError('no eligible pending target')
    if not isinstance(status.get('jobs'), list):
        raise ValueError('canonical status jobs required')
    native_receipt = study.checked_json(sources['native_cpu'], raw['native_cpu']['sha256'])
    native_job = str(native_receipt.get('job_id', ''))
    if not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', native_job):
        raise ValueError('native qualification job identity missing')
    qualification_epoch = datetime.strptime(native_job[:16], '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).timestamp()
    ada_qualification = study.checked_json(sources['gpu_parity'], gate['confirmation_qualification']['gpu_parity_receipt_sha256']) if gpu_profile == ada.ADA else None
    inspected = []
    blocked: list[list[str]] = []
    for item in status['jobs']:
        job = item.get('job_id')
        state = item.get('status')
        if job == own_job:
            continue
        terminal = state in ('complete', 'failed', 'contended', 'budget_exhausted', 'expired', 'cancelled')
        source_hint = item.get('source_sha256') or (item.get('gate') or {}).get('source_sha256')
        if terminal and source_hint and source_hint != identity['source_sha256']:
            continue
        if terminal and not source_hint:
            created = item.get('created')
            if isinstance(created, bool) or not isinstance(created, (int, float)) or not math.isfinite(created):
                raise ValueError('terminal job without source requires creation time')
            if created < qualification_epoch:
                continue
        if not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', str(job)):
            raise ValueError('unidentified active or uncertain fleet job')
        directory = Path(str(item.get('directory', '')))
        if directory != DATA_ROOT / 'jobs/fleet/inbox' / job:
            raise ValueError('active job canonical directory differs')
        ready_path, spec_path = directory / 'ready.json', directory / 'spec.json'
        ready = json.loads(ready_path.read_text())
        live_spec = json.loads(spec_path.read_text())
        from experiments.jobs.inbox import digest
        if ready.get('spec_sha256') != digest(live_spec) or not cf.digest_string(str(ready.get('source_sha256', ''))):
            raise ValueError('active job identity is unknown')
        inspected.append({'job_id': job, 'status': state, 'source_sha256': ready['source_sha256'],
                          'ready_sha256': oracle.sha(ready_path), 'spec_sha256': oracle.sha(spec_path)})
        if ready['source_sha256'] == identity['source_sha256']:
            arguments = live_spec.get('arguments', [])
            if live_spec.get('entrypoint') == ['-m', 'experiments.tennis.k5d_native_pid_oracle']:
                continue
            if gpu_profile == ada.ADA:
                if ada.known_calibration_job(job, state, directory, incoming_cost):
                    continue
                assert ada_qualification is not None
                if ada.known_qualification_job(job, state, directory, ada_qualification, identity):
                    continue
            if (live_spec.get('entrypoint') != ['-m', 'experiments.tennis.k5d_scoped_namespace']
                    or arguments[:1] != ['outer'] or arguments.count('--') != 1):
                raise ValueError('same-consumer job target is unknown')
            worker = arguments[arguments.index('--') + 1:]
            if worker and worker[0] in ('parity16', 'parity-noop'):
                continue
            if not worker or worker[0] not in ('prepare', 'score'):
                raise ValueError('same-consumer job phase is unknown')
            active_clip, active_phase = cf.option(worker, '--clip'), worker[0]
            if terminal:
                entry = inventory.get(active_clip, {}).get(active_phase, {})
                if state != 'complete' or entry.get('state') != 'complete' or entry.get('job_id') != job:
                    raise ValueError('same-consumer terminal target is not reconciled in inventory')
            else:
                if (active_clip, active_phase) not in pending:
                    raise ValueError('same-consumer active job is not a pending target')
                blocked.append([active_clip, active_phase])
    if len({tuple(x) for x in blocked}) != len(blocked):
        raise ValueError('duplicate active target requires reconciliation')
    pending = [(c, p) for c, p in pending if [c, p] not in blocked]
    if not pending:
        raise ValueError('no unclaimed pending target')
    if prior:
        cost = incoming_cost
        ledger = study.checked_json(sources['ledger'], raw['ledger']['sha256'])
        attestation = study.checked_json(sources['attestation'], raw['attestation']['sha256'])
        control.check_budget(ledger, attestation, raw['ledger']['sha256'], raw['attestation']['sha256'], cost)
        cpu_models, noncommand = prior['cpu_models'], prior['noncommand']
        noncommand_refs = prior['noncommand_raw_sha256']
        forecast = control.programme_projection(cost, inventory=inventory, noncommand=noncommand,
                                                certified_cpu=cpu_models)
        control.check_forecast_allocation(attestation, forecast)
    else:
        qualification_paths = {'native_cpu': sources['native_cpu'], 'gpu_parity': sources['gpu_parity']}
        full_paths = {'inventory': sources['inventory'], 'noncommand': sources['noncommand']}
        clip, phase = pending[0]
        cost, _ = control.audit_raw_approval(sources['gate'], raw['gate']['sha256'], sources['cost'], raw['cost']['sha256'],
            sources['bridge'], raw['bridge']['sha256'], sources['old_source'], sources['new_source'],
            sources['ledger'], raw['ledger']['sha256'], sources['attestation'], raw['attestation']['sha256'],
            identity, clip, phase, qualification_paths, full_paths, authoring=True, verified_inventory=inventory)
        noncommand_doc = study.checked_json(sources['noncommand'], raw['noncommand']['sha256'])
        noncommand = wave.verified_noncommand(noncommand_doc, identity, inventory)
        cpu_models = wave.verified_cpu_calibration(cost['cpu_calibration'], identity, inventory, **({'gpu_profile': gpu_profile} if gpu_profile == ada.ADA else {}))
        forecast = control.programme_projection(cost, inventory=inventory, noncommand=noncommand)
        compact_forecast = control.programme_projection(cost, inventory=inventory, noncommand=noncommand,
                                                       certified_cpu=cpu_models)
        if forecast != compact_forecast:
            raise ValueError('compact and raw budget forecasts differ')
        noncommand_refs = {name: {'raw': group['raw_sha256'], 'audit': group['audit_sha256']}
                           for name, group in noncommand_doc['groups'].items()}
    if not all(forecast[r]['fits'] is True for r in ('gpu', 'cpu')):
        raise ValueError('audited remaining workload does not fit')
    targets: dict[str, Any] = {}
    selections = study.selection()
    for clip, phase in pending:
        target_entry: dict[str, Any] = {'state': 'pending', 'frames': selections[clip]['label_rows']}
        if phase == 'score':
            prep = inventory[clip]['prepare']
            receipt = wave.read(Path(prep['receipt_path']), prep['receipt_sha256'])
            published = prep['publications']['full']
            remote = Path(receipt['job_directory']) / 'full/published.tar'
            if not remote.is_absolute() or not remote.is_relative_to(DATA_ROOT):
                raise ValueError('producer archive is not in shared data root')
            target_entry['producer'] = {'job_id': prep['job_id'], 'archive_sha256': published['sha256'],
                                'checkpoint_sha256': published['checkpoint_sha256'], 'remote': str(remote)}
        targets.setdefault(clip, {})[phase] = target_entry
    if gpu_profile == ada.ADA:
        for phases in targets.values():
            if 'score' in phases:
                phases['score']['producer']['config_sha256'] = gate['receiver_isolation_config_sha256']
    valid_until = _deadline(config['valid_until'])
    if not now < valid_until or (valid_until - now).total_seconds() > 28800:
        raise ValueError('certificate validity must be within eight hours')
    result = {'schema': 1, 'scope': 'server-side full raw admission audit; root must review exact SHA',
              'selection_sha256': fleet.SOURCE_SHA, 'execution_identity': identity,
              'cost_sha256': raw['cost']['sha256'], 'cost_core_sha256': core_sha, 'ledger_sha256': raw['ledger']['sha256'],
              'attestation_sha256': raw['attestation']['sha256'], 'qualification': gate['confirmation_qualification'],
              'raw_sha256': {name: item['sha256'] for name, item in raw.items()}, 'forecast': forecast,
              'cpu_models': cpu_models, 'noncommand': noncommand,
              'pending': [[clip, phase] for clip, row in sorted(inventory.items())
                          for phase in ('prepare', 'score') if row[phase]['state'] == 'pending'],
              'completed': {clip: {phase: wave.completed_summary(row[phase])
                  for phase in ('prepare', 'score') if row[phase]['state'] == 'complete'}
                  for clip, row in sorted(inventory.items()) if any(row[p]['state'] == 'complete' for p in ('prepare', 'score'))},
              'noncommand_raw_sha256': noncommand_refs, 'inspected_active_jobs': inspected,
              'prior_certificate_sha256': raw['prior_certificate']['sha256'] if prior else None,
              'targets': targets, 'blocked_targets': blocked,
              'canonical_status_captured_at': status['captured_at'],
              'authored_at': now.isoformat(), 'valid_until': valid_until.isoformat()}
    if gpu_profile == ada.ADA:
        result['registered_gpu_profile'] = ada.ADA
        result['receiver_isolation_config_sha256'] = gate['receiver_isolation_config_sha256']
    if len(json.dumps(result, sort_keys=True, allow_nan=False).encode()) > MAX_CERT_BYTES:
        raise ValueError('certificate exceeds bounded size')
    return result



def author_spec(base: dict[str, Any], *, config_remote: str, config_local: Path, config_sha: str,
                deadline: str, smoke_seconds: int, budget_seconds: int) -> dict[str, Any]:
    """Create a CPU-only author job proposal; never dispatch or approve it."""
    if (type(smoke_seconds) is not int or not 1 <= smoke_seconds <= 600
            or type(budget_seconds) is not int or not smoke_seconds + 35 <= budget_seconds <= 2700):
        raise ValueError('invalid bounded author stage or campaign budget')
    item = cf.asset('admission_config', config_remote, config_sha, config_local, extract=False)
    result = json.loads(json.dumps(base))
    result.update(device='cpu', gpu_models=[], gpu_memory_mib=0, cpu_threads=1, single_cpu_campaign=True,
                  entrypoint=['-m', 'experiments.tennis.k5d_full_admission'],
                  arguments=['run', '--phase', '{phase}', '--config', '{staged:admission_config}', '--config-sha256', config_sha],
                  scale={'phase': {'smoke': 'smoke', 'full': 'full'}},
                  deadline=deadline, smoke={'seconds': smoke_seconds,
                    'representative_basis': 'One bounded server-side raw admission audit and actual compact-verifier benchmark.'},
                  full={'seconds': 5}, validator=['{python}', '-m', 'experiments.tennis.k5d_full_admission', 'validate'],
                  validator_seconds=30, budget_seconds=budget_seconds, verify_staged_inputs=True)
    result.pop('gpu_sharing', None)
    result.pop('contention', None)
    result['staged_inputs'] = [item]
    result['inputs'] = [{'path': item['path'], 'sha256': item['sha256']}]
    from experiments.jobs.inbox import validate_spec
    validate_spec(result)
    return result

def main() -> None:
    """Fleet CPU author; heavy raw audit runs only here, never in workers."""
    import time
    import resource

    from experiments.tennis import k5d_checkpoint as checkpoint, k5d_cost_pilot as cost

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run', 'validate'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--phase')
    parser.add_argument('--config-sha256')
    args = parser.parse_args()
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.mode == 'validate':
        target = Path(os.environ['PS_VALIDATION_PATH'])
        try:
            staging = json.loads((stage / 'staging.json').read_text())
            with cost.unpack(stage / 'published.tar', staging['published']['sha256']) as output:
                cert_path = output / 'certificate.json'
                certificate = read_certificate(cert_path, oracle.sha(cert_path))
                author_receipt = json.loads((output / 'author-receipt.json').read_text())
                bench = json.loads((output / 'microbenchmark.json').read_text())
                inputs = [item for item in staging['inputs'] if item['name'] == 'admission_config']
                if (len(inputs) != 1 or author_receipt.get('config_sha256') != inputs[0]['sha256']
                        or author_receipt.get('certificate_sha256') != oracle.sha(cert_path)
                        or author_receipt.get('microbenchmark_sha256') != oracle.sha(output / 'microbenchmark.json')
                        or bench.get('certificate_sha256') != oracle.sha(cert_path)
                        or certificate.get('execution_identity') != checkpoint.execution_identity()
                        or author_receipt.get('scope') != 'server raw audit; root review required'):
                    raise ValueError('author publication does not bind inputs and source')
            target.write_text(json.dumps({'passed': True, 'checks': {'compact_author_publication': True,
                              'exact_source_and_config': True}}) + '\n')
        except Exception as exc:
            target.write_text(json.dumps({'passed': False, 'checks': {'author_publication': False},
                                          'error': str(exc)}) + '\n')
            raise
        return
    if args.phase != os.environ.get('PS_STAGE'):
        raise ValueError('author phase differs from fleet stage')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('author requires GPU-hidden single-CPU fleet execution')
    output = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
    output.mkdir()
    if os.environ.get('PS_STAGE') == 'full':
        checkpoint.write_new(output / 'noop.json', {'scope': 'no second audit'})
        return
    if os.environ.get('PS_STAGE') != 'smoke' or args.config is None or not args.config_sha256:
        raise ValueError('author smoke requires immutable config')
    if oracle.sha(args.config) != args.config_sha256:
        raise ValueError('author configuration SHA differs')
    config = json.loads(args.config.read_text())
    if config.get('identity') != checkpoint.execution_identity():
        raise ValueError('author and consumer execution identities differ')
    start = time.monotonic()
    cert = author(config)
    checkpoint.write_new(output / 'certificate.json', cert)
    certificate_path = output / 'certificate.json'
    cert_sha = oracle.sha(certificate_path)
    raw = config['raw']
    gate = study.checked_json(Path(raw['gate']['path']), raw['gate']['sha256'])
    gate = {**gate, 'approved': True, 'certificate_sha256': cert_sha}
    receipt = study.checked_json(Path(raw['cost']['path']), raw['cost']['sha256'])
    ledger = study.checked_json(Path(raw['ledger']['path']), raw['ledger']['sha256'])
    attestation = study.checked_json(Path(raw['attestation']['path']), raw['attestation']['sha256'])
    qualification_paths = {'native_cpu': Path(raw['native_cpu']['path']), 'gpu_parity': Path(raw['gpu_parity']['path'])}
    target_clip = next(iter(cert['targets']))
    target_phase = next(iter(cert['targets'][target_clip]))
    from experiments.tennis import k5d_control_study as control
    wall_samples, cpu_samples = [], []
    for _ in range(3):
        cpu_start = resource.getrusage(resource.RUSAGE_SELF)
        wall_start = time.monotonic()
        checked = read_certificate(certificate_path, cert_sha)
        control.check_budget(ledger, attestation, raw['ledger']['sha256'], raw['attestation']['sha256'], receipt)
        check_certificate(checked, gate, config['identity'], target_clip, target_phase, raw['cost']['sha256'],
                          raw['ledger']['sha256'], raw['attestation']['sha256'], qualification_paths, receipt)
        wall_samples.append(time.monotonic() - wall_start)
        cpu_end = resource.getrusage(resource.RUSAGE_SELF)
        cpu_samples.append(cpu_end.ru_utime + cpu_end.ru_stime - cpu_start.ru_utime - cpu_start.ru_stime)
    checkpoint.write_new(output / 'microbenchmark.json', {'scope': 'actual compact certificate verification microbenchmark',
        'certificate_sha256': cert_sha, 'execution_identity': config['identity'], 'job_id': stage.parent.name,
        'iterations': len(wall_samples), 'wall_seconds': max(wall_samples), 'cpu_seconds': max(cpu_samples),
        'wall_samples_seconds': wall_samples, 'cpu_samples_seconds': cpu_samples,
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'verifier_sha256': oracle.sha(Path(__file__)),
        'control_sha256': oracle.sha(Path(control.__file__))})
    checkpoint.write_new(output / 'author-receipt.json', {'config_sha256': args.config_sha256,
        'certificate_sha256': cert_sha, 'microbenchmark_sha256': oracle.sha(output / 'microbenchmark.json'),
        'elapsed_seconds': time.monotonic() - start, 'scope': 'server raw audit; root review required'})


if __name__ == '__main__':
    main()

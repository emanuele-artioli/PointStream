"""Prospective CPU-readiness profile for the full-context K5d study.

This module grants no budget and requires separately approved immutable gate,
calibration proof and source bridge. Existing checkpoint defaults are unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_benchmark as benchmark, k5d_checkpoint as cp
from experiments.tennis import k5d_checkpoint_fleet as cf, k5d_coalesced as co
from experiments.tennis import k5d_control_bridge as bridge, k5d_control_readiness as readiness
from experiments.tennis import k5d_cost_pilot as cost, k5d_fleet as fleet, k5d_full_study as study
from experiments.tennis import k5d_study_evidence as evidence, k5d_source_oracle as oracle

# Saved fleet evidence independently audited against the d8b5 references.
CALIBRATION_EXECUTIONS = {
    'prepare16': ('94ebdb749a1ff6e4258a17a277ee6ce5dedb43ec88ee10a8ce112a47b1be1ff5', 168.21834516525269),
    'prepare48': ('d64c4a29d6fb594bec15fe2ad56837b6c76f953ddae4a4a7925d0f834e458c4e', 234.93895959854126),
    'score16': ('c95333b87b568acf805cc11c8ba21671521ef62ccad4d205294e535bae76ba7e', 110.028817653656),
    'score48': ('461048ad06202e58706ee5cc47d01c1e1a036461ba59f98da502fcc9f4ddf3a3', 201.41556024551392),
}
CALIBRATION_AUDIT_SHA = '09f7d8761c95c83b8c000cae9cc1f86d648e3ad415481b880b34240baee7421d'
CALIBRATION_TIMING = {'16': '6aa2841e2e72682b5e8a2e78dc4395cad0b23dfa2703310c12ee60848b584451',
                      '48': 'e30aff247dad65bbf3aaabb2ae5ee46d5bdcd9f1e45d00ea7b3a63a26e50e000'}
CALIBRATION_ARCHIVES = {
    'prepare16': '84575156c209d324400c1c0bbc710698c9c44aceb0e1eb5f764d08a239405b14',
    'prepare48': '4a5ba08205dee1b3a51ad4634dcf00b1f528534b91d772e30a316a4d860357a6',
    'score16': 'd6cb69e4c0f0cb7b9100283b8b5808d28146a2e0e31b269ab309249691ce20b7',
    'score48': '8dd50977e0fdd9ce15ec4d10529b7c1cd0f4cfaf9e140ee30a3de92d10c3d532',
}


def validate_calibration(receipt: dict[str, Any]) -> None:
    if receipt.get('budget_reconciled') is not True:
        raise ValueError('budget and active reservations require reconciliation')
    costs = receipt['cost']
    if (receipt.get('calibration_archives') != CALIBRATION_ARCHIVES
            or receipt.get('calibration_audit_sha256') != CALIBRATION_AUDIT_SHA
            or receipt.get('score_timing_sha256') != CALIBRATION_TIMING):
        raise ValueError('calibration publications, audit or timing differ from saved evidence')
    for key, (sha, measured) in CALIBRATION_EXECUTIONS.items():
        phase, frame = ('prepare', '16' if key.endswith('16') else '48') if key.startswith('prepare') else ('score', '16' if key.endswith('16') else '48')
        if receipt['calibration_execution_sha256'].get(key) != sha or costs[phase]['seconds' + frame] < measured:
            raise ValueError('calibration execution differs or understates measured cost')
    for phase in ('prepare', 'score'):
        if (any(isinstance(costs[phase][key], bool) or not isinstance(costs[phase][key], (int, float))
                or not math.isfinite(costs[phase][key])
                for key in ('margin', 'reserve_seconds', 'seconds16', 'seconds48'))
                or costs[phase]['margin'] < 1.5 or costs[phase]['reserve_seconds'] <= 0
                or costs[phase]['seconds16'] <= 0 or costs[phase]['seconds48'] < costs[phase]['seconds16']):
            raise ValueError('invalid calibration bounds')


def programme_projection(receipt: dict[str, Any], *, inventory: dict[str, dict[str, Any]] | None = None,
                         noncommand: dict[str, dict[str, Any]] | None = None,
                         certified_cpu: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    """Recompute all 60 GPU full stages and CPU-readiness campaigns, once each."""
    rows = study.selection()
    if len(rows) != 30 or sum(row['label_rows'] for row in rows.values()) != 5675:
        raise ValueError('frozen population differs')
    validate_calibration(receipt)
    costs = receipt['cost']
    if inventory is not None and (set(inventory) != set(rows) or any(
            inventory[clip][phase]['state'] not in ('pending', 'complete')
            for clip in rows for phase in ('prepare', 'score'))):
        raise ValueError('inventory contains unknown or unfinished work')
    pending = [(clip, phase) for clip in rows for phase in ('prepare', 'score')
               if inventory is None or inventory[clip][phase]['state'] == 'pending']
    full_compute = 0.0
    max_full = 0.0
    for clip, phase in pending:
        row = rows[clip]
        estimate = study.forecast(receipt, phase, row['label_rows'], costs[phase]['seconds16'])
        full_compute += estimate
        max_full = max(max_full, estimate)
    if max_full > 2670:
        raise ValueError('full stage exceeds recovery bound')
    cpu_costs = None
    if certified_cpu is not None or 'cpu_calibration' in receipt:
        if certified_cpu is None:
            from experiments.tennis import k5d_study_wave as wave
            cpu_costs = wave.verified_cpu_calibration(receipt['cpu_calibration'], receipt['consumer_execution_identity'], inventory)
        else:
            cpu_costs = certified_cpu
        if set(cpu_costs) != {'prepare', 'score'}:
            raise ValueError('both CPU calibration phases required')
        for model in cpu_costs.values():
            values = [model.get(k) for k in ('margin', 'fixed_seconds', 'marginal_seconds_per_frame')]
            if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                    for v in values) or model['margin'] < 1.5):
                raise ValueError('CPU calibration requires a finite 1.5 margin')
        if inventory is not None and certified_cpu is None:
            for phase in ('prepare', 'score'):
                for sample in receipt['cpu_calibration'][phase]:
                    entry, saved = sample['entry'], inventory[sample['clip']][phase]
                    if any(entry.get(key) != saved.get(key) or key not in entry or key not in saved
                           for key in ('job_id', 'receipt_sha256', 'publications', 'independent_audit_sha256')):
                        raise ValueError('CPU calibration differs from verified inventory')
    results = {}
    for resource in ('gpu', 'cpu'):
        d = receipt['programme'][resource]
        keys = ('readiness_seconds', 'stage_staging_seconds', 'validator_seconds', 'audit_seconds',
                'failed_start_reserve_seconds', 'ball_remaining_seconds', 'global_remaining_seconds')
        if any(isinstance(d[k], bool) or not isinstance(d[k], (float, int)) or not math.isfinite(d[k]) or d[k] < 0 for k in keys):
            raise ValueError('invalid programme budget assumptions')
        if not (20 <= d['readiness_seconds'] <= 60 and d['stage_staging_seconds'] > 0
                and d['validator_seconds'] > 0 and d['audit_seconds'] > 0 and d['failed_start_reserve_seconds'] > 0):
            raise ValueError('missing readiness, staging, validator or audit reserves')
        command_cost = full_compute
        max_command = max_full
        if resource == 'cpu' and cpu_costs is not None:
            estimates = [cpu_costs[phase]['margin'] * (cpu_costs[phase]['fixed_seconds']
                         + cpu_costs[phase]['marginal_seconds_per_frame'] * rows[clip]['label_rows'])
                         for clip, phase in pending]
            command_cost = sum(estimates)
            max_command = max(estimates, default=0.0)
        count = len(pending)
        if noncommand is not None:
            groups = noncommand['groups']
            declared = d.get('noncommand_groups')
            if not isinstance(declared, dict) or set(declared) != set(groups):
                raise ValueError('all audited noncommand groups require allowances')
            for name, sample in groups.items():
                value = declared[name]
                if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                        or value < sample[resource]):
                    raise ValueError('noncommand allowance understates audited measurement: ' + name)
            group_total = sum(declared.values())
            if count and d['failed_start_reserve_seconds'] < max_command + group_total:
                raise ValueError('failure reserve below one bounded campaign')
            if resource == 'gpu' and max_full + group_total > 2700:
                raise ValueError('longest campaign exceeds 45-minute recovery bound')
            incremental = command_cost + count * group_total + (d['failed_start_reserve_seconds'] if count else 0)
        else:
            campaign = max_full + d['readiness_seconds'] + 2 * d['stage_staging_seconds'] + d['validator_seconds']
            if campaign > 2700:
                raise ValueError('longest campaign exceeds 45-minute recovery bound')
            incremental = (command_cost + count * d['readiness_seconds'] + 2 * count * d['stage_staging_seconds']
                           + count * d['validator_seconds'] + count * d['audit_seconds'] + d['failed_start_reserve_seconds'])
        qualified = resource == 'gpu' or (inventory is not None and noncommand is not None and cpu_costs is not None)
        results[resource] = {'incremental_seconds': incremental if resource == 'gpu' or cpu_costs is not None else None,
                             'fits': qualified and incremental <= min(d['ball_remaining_seconds'], d['global_remaining_seconds'])}
    return {'campaigns': len(pending), 'full_stages': len(pending), 'gpu_smoke_inferences': 0,
            'full_frames_per_phase': 5675, 'max_full_seconds': max_full,
            'gpu': results['gpu'], 'cpu': results['cpu'],
            'cpu_method': ('verified command CPU and independently audited noncommand bounds'
                           if inventory is not None and noncommand is not None and cpu_costs is not None
                           else 'unreleased estimate; complete CPU bound not established')}


def check_budget(ledger: dict[str, Any], attestation: dict[str, Any], ledger_sha: str,
                 attestation_sha: str, receipt: dict[str, Any]) -> None:
    if (attestation.get('reviewed') is not True or attestation.get('ledger_sha256') != ledger_sha
            or attestation.get('active_reservations_reconciled') is not True
            or receipt.get('budget_ledger_sha256') != ledger_sha
            or receipt.get('budget_attestation_sha256') != attestation_sha):
        raise ValueError('budget snapshot and active reservations not bound')
    active = {}
    if 'reservations' in ledger:
        for reservation in ledger['reservations']:
            status = reservation.get('status')
            if not isinstance(status, str) or not status:
                raise ValueError('reservation status missing')
            if status.lower().startswith(('closed', 'released', 'superseded', 'cancelled')):
                continue
            phase = reservation['phase']
            if phase in active:
                raise ValueError('duplicate active reservation phase')
            remaining = {}
            for resource in ('gpu', 'cpu'):
                cap = reservation[resource + '_seconds']
                if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap < 0:
                    raise ValueError('invalid active reservation accounting')
                remaining[resource] = cap
            active[phase] = {'workstream': reservation['workstream'], **remaining}
        if active and ledger.get('reservation_amount_basis') != 'remaining_unspent':
            raise ValueError('active reservation amounts must explicitly be remaining unspent')
        if (active or 'active_reservations' in attestation) and attestation.get('active_reservations') != active:
            raise ValueError('active reservation attestation differs from ledger')
    covered = attestation.get('forecast_covered_reservation_phases', [])
    if (not isinstance(covered, list) or len(covered) != len(set(covered))
            or receipt.get('forecast_covered_reservation_phases', []) != covered
            or any(phase not in active or active[phase]['workstream'] != 'ball' for phase in covered)):
        raise ValueError('forecast-covered reservations do not match active ball allocation')
    for resource in ('gpu', 'cpu'):
        charged = sum(row.get(resource + '_seconds', 0) for row in ledger['charges'] if row['workstream'] == 'ball')
        global_charged = sum(row.get(resource + '_seconds', 0) for row in ledger['charges'])
        row = attestation[resource]
        programme = receipt['programme'][resource]
        if (row['ball_charged_seconds'] != charged or row['global_charged_seconds'] != global_charged
                or row['ball_limit_seconds'] != ledger['workstreams']['ball'][resource + '_limit_seconds']
                or row['spent_and_reserved_seconds'] < charged + sum(x[resource] for x in active.values() if x['workstream'] == 'ball')
                or row['global_remaining_seconds'] > ledger[resource + '_total_seconds'] - global_charged - sum(x[resource] for phase, x in active.items() if phase not in covered)
                or programme['ball_remaining_seconds'] > row['ball_limit_seconds'] - row['spent_and_reserved_seconds'] + sum(active[phase][resource] for phase in covered)
                or programme['global_remaining_seconds'] > row['global_remaining_seconds']):
            raise ValueError('programme budget exceeds reconciled ledger')



def check_forecast_allocation(attestation: dict[str, Any], forecast: dict[str, Any]) -> None:
    covered = attestation.get('forecast_covered_reservation_phases', [])
    if covered:
        active = attestation['active_reservations']
        for resource in ('gpu', 'cpu'):
            amount = forecast[resource]['incremental_seconds']
            if amount is None or amount > sum(active[phase][resource] for phase in covered):
                raise ValueError('pending forecast exceeds its reserved allocation')

def confirmation_budget(gate: dict[str, Any], receipt: dict[str, Any]) -> None:
    """Check only an independently reserved four-job scale confirmation.

    This does not certify CPU fit, containment, or full-study admission.
    """
    clips = ('tracknet/game9/Clip5', 'tracknet/game10/Clip12')
    if set(clips) != study.CONFIRMATION or gate.get('allowed_clips') != list(clips):
        raise ValueError('confirmation release must name both registered clips in order')
    caps = gate.get('confirmation_caps')
    if not isinstance(caps, dict) or set(caps) != {'gpu', 'cpu'}:
        raise ValueError('confirmation GPU and CPU caps required')
    rows = study.selection()
    for resource, maximum in (('gpu', 9000), ('cpu', 10800)):
        budget = caps[resource]
        if not isinstance(budget, dict) or set(budget) != {'jobs', 'external', 'total'}:
            raise ValueError('four job caps and external reserve required')
        jobs = budget['jobs']
        names = {f'{clip}:{phase}' for clip in clips for phase in ('prepare', 'score')}
        if not isinstance(jobs, dict) or set(jobs) != names:
            raise ValueError('exact confirmation jobs required')
        values = [*jobs.values(), budget['external'], budget['total']]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('invalid confirmation cap')
        if sum(jobs.values()) + budget['external'] > budget['total'] or budget['total'] > maximum:
            raise ValueError('confirmation cap exceeds tranche')
        programme = receipt['programme'][resource]
        if budget['total'] > min(programme['ball_remaining_seconds'], programme['global_remaining_seconds']):
            raise ValueError('confirmation cap exceeds reconciled budget')
        if resource == 'gpu':
            for clip in clips:
                for phase in ('prepare', 'score'):
                    predicted = study.forecast(receipt, phase, rows[clip]['label_rows'], receipt['cost'][phase]['seconds16'])
                    if jobs[f'{clip}:{phase}'] < predicted:
                        raise ValueError('confirmation GPU cap below numerical forecast')


PAIR_RELEASE = 'first-292-pair-v1'
PAIR_CLIP = 'tracknet/game9/Clip5'


def parent8_bridge_record(old_source: Path, calibration: dict[str, str], new_source: Path,
                          identity: dict[str, str]) -> dict[str, Any]:
    """Normalize the independently pinned resource-only archive comparison."""
    from experiments.tennis import k5d_same_job_source_bridge as scoped_bridge
    return scoped_bridge.bridge(old_source, calibration, new_source, identity)


def pair_producer(gate: dict[str, Any]) -> dict[str, Any]:
    producer = gate.get('pair_producer')
    if (not isinstance(producer, dict) or producer.get('audited') is not True
            or producer.get('clip') != PAIR_CLIP or producer.get('frames') != 292
            or producer.get('prepare_external_included') is not True
            or not isinstance(producer.get('job_id'), str) or not producer['job_id'].startswith('2026')
            or any(not isinstance(producer.get(key), str) or not cf.digest_string(producer[key])
                   for key in ('archive_sha256', 'checkpoint_sha256', 'audit_sha256'))
            or not isinstance(producer.get('remote'), str)
            or not Path(producer['remote']).is_absolute() or not Path(producer['remote']).is_relative_to(cf.DATA_ROOT)):
        raise ValueError('independently audited pair producer required')
    return producer


def validate_pair_checkpoint_asset(item: dict[str, str], producer: dict[str, Any]) -> None:
    """Bind both staged archive aliases and the checkpoint to the audited producer."""
    if (item.get('remote') != producer['remote']
            or item.get('sha256') != producer['archive_sha256']
            or item.get('archive_sha256') != producer['archive_sha256']
            or item.get('checkpoint_sha256') != producer['checkpoint_sha256']):
        raise ValueError('pair score checkpoint differs from audited producer')


def verify_pair_producer_audit(gate: dict[str, Any], path: Path, identity: dict[str, str]) -> None:
    producer = pair_producer(gate)
    audit = study.checked_json(path, producer['audit_sha256'])
    if (audit.get('schema') != 1 or audit.get('audited') is not True or audit.get('terminal_status') != 'COMPLETE'
            or audit.get('exclusive') is not True or audit.get('timing_contaminated') is not False
            or audit.get('consumer_execution_identity') != identity
            or any(audit.get(key) != producer.get(key) for key in
                   ('job_id', 'clip', 'frames', 'remote', 'archive_sha256', 'checkpoint_sha256',
                    'gpu_charge_seconds', 'cpu_charge_seconds', 'ledger_charge_phases', 'prepare_external_included'))):
        raise ValueError('independent pair producer audit differs')


def verify_pair_ledger_charge(gate: dict[str, Any], ledger: dict[str, Any]) -> None:
    """Join the audited whole prepare charge, including source prep and audit, to ledger rows."""
    producer = pair_producer(gate)
    phases = producer.get('ledger_charge_phases')
    if (not isinstance(phases, list) or not phases or len(phases) != len(set(phases))
            or any(not isinstance(phase, str) or not phase for phase in phases)):
        raise ValueError('pair producer charge phases required')
    rows = [row for row in ledger['charges'] if row.get('phase') in phases]
    if len(rows) != len(phases) or any(row.get('workstream') != 'ball' for row in rows):
        raise ValueError('pair producer charges missing from ledger')
    for resource in ('gpu', 'cpu'):
        amounts = [row.get(resource + '_seconds') for row in rows]
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                for v in amounts) or sum(amounts) != producer.get(resource + '_charge_seconds')):
            raise ValueError('pair producer charge differs from reconciled ledger')


def first_pair_budget(gate: dict[str, Any], receipt: dict[str, Any], attestation: dict[str, Any],
                      phase: str = 'prepare') -> None:
    """Reserve the complete requested pair; later study fit remains unproven."""
    if gate.get('allowed_clips') != [PAIR_CLIP] or gate.get('future_study_fit_proven') is not False:
        raise ValueError('pair release requires the registered clip and no full-study fit claim')
    caps = gate.get('confirmation_caps')
    if not isinstance(caps, dict) or set(caps) != {'gpu', 'cpu'}:
        raise ValueError('pair GPU and CPU caps required')
    forecast = {}
    for resource in ('gpu', 'cpu'):
        budget = caps[resource]
        if not isinstance(budget, dict) or set(budget) != {'jobs', 'external', 'total'}:
            raise ValueError('pair jobs and external reserve required')
        jobs = budget['jobs']
        if not isinstance(jobs, dict) or set(jobs) != {f'{PAIR_CLIP}:prepare', f'{PAIR_CLIP}:score'}:
            raise ValueError('exact pair jobs required')
        external = budget['external']
        if not isinstance(external, dict) or set(external) != {'prepare', 'score'}:
            raise ValueError('pair external caps must cover prepare and score separately')
        values = [*jobs.values(), *external.values(), budget['total']]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('invalid pair cap')
        if sum(jobs.values()) + sum(external.values()) > budget['total'] or budget['total'] > 3350:
            raise ValueError('pair cap exceeds inclusive allocation')
        remaining = budget['total']
        if phase == 'score':
            producer = pair_producer(gate)
            charged = producer.get(resource + '_charge_seconds')
            if (isinstance(charged, bool) or not isinstance(charged, (int, float))
                    or not math.isfinite(charged) or charged <= 0
                    or charged > jobs[f'{PAIR_CLIP}:prepare'] + external['prepare']):
                raise ValueError('invalid audited producer charge')
            remaining = jobs[f'{PAIR_CLIP}:score'] + external['score']
            if charged + remaining > budget['total']:
                raise ValueError('pair including producer charge exceeds inclusive allocation')
        elif phase != 'prepare':
            raise ValueError('invalid pair phase')
        if remaining > min(receipt['programme'][resource]['ball_remaining_seconds'],
                           receipt['programme'][resource]['global_remaining_seconds']):
            raise ValueError('pair cap exceeds reconciled budget')
        if resource == 'gpu':
            for kind in ('prepare', 'score'):
                predicted = study.forecast(receipt, kind, 292, receipt['cost'][kind]['seconds16'])
                if jobs[f'{PAIR_CLIP}:{kind}'] < predicted:
                    raise ValueError('pair GPU cap below numerical forecast')
        forecast[resource] = {'incremental_seconds': remaining}
    check_forecast_allocation(attestation, forecast)


def audit_raw_approval(gate_path: Path, gate_sha: str, cost_path: Path, cost_sha: str, bridge_path: Path, bridge_sha: str,
             old_source: Path, new_source: Path, ledger_path: Path, ledger_sha: str,
             budget_path: Path, budget_sha: str, identity: dict[str, str], clip: str, phase: str,
             qualification_paths: dict[str, Path] | None = None,
             full_evidence_paths: dict[str, Path] | None = None, *, authoring: bool = False,
             verified_inventory: dict[str, dict[str, Any]] | None = None) -> tuple[dict[str, Any], int]:
    gate = study.checked_json(gate_path, gate_sha)
    receipt = study.checked_json(cost_path, cost_sha)
    signed_bridge = study.checked_json(bridge_path, bridge_sha)
    if (gate.get('schema') != 1 or (gate.get('approved') is not True and not (authoring and gate.get('approved') is False))
            or gate.get('purpose') != 'full-retrospective-benchmark-cpu-readiness'
            or gate.get('selection_sha256') != fleet.SOURCE_SHA or gate.get('consumer_execution_identity') != identity
            or gate.get('cost_receipt_sha256') != cost_sha or gate.get('bridge_sha256') != bridge_sha
            or gate.get('gpu_models') != ['RTX A6000'] or gate.get('gpu_capability') != [8, 6]
            or gate.get('fidelity_only_for_unqualified') is not True):
        raise ValueError('CPU readiness release gate differs')
    evidence.historical(gate)
    clips = study.selection()
    allowed = gate.get('allowed_clips')
    if (not isinstance(allowed, list) or clip not in allowed or len(set(allowed)) != len(allowed)
            or not set(allowed).issubset(set(clips)) or gate.get('release') not in ('cost-confirmation', 'full-study', PAIR_RELEASE)
            or (gate['release'] == 'cost-confirmation' and not set(allowed).issubset(study.CONFIRMATION))):
        raise ValueError('clip is not admitted by this release')
    if gate['release'] == PAIR_RELEASE and clips[PAIR_CLIP]['label_rows'] != 292:
        raise ValueError('registered pair timeline differs')
    calibration = signed_bridge['calibration_execution_identity']
    if signed_bridge.get('approved') is not True or signed_bridge.get('consumer_execution_identity') != identity:
        raise ValueError('unapproved or wrong source bridge')
    if gate['release'] == PAIR_RELEASE:
        actual_bridge = parent8_bridge_record(old_source, calibration, new_source, identity)
        bridge_fields = tuple(actual_bridge)
    else:
        actual_bridge = bridge.bridge(old_source, calibration, new_source, identity)
        bridge_fields = ('calibration_execution_identity', 'consumer_execution_identity', 'calibration_snapshot_sha256',
                         'consumer_snapshot_sha256', 'changed_paths', 'unchanged_files')
    for name in bridge_fields:
        if signed_bridge.get(name) != actual_bridge[name]:
            raise ValueError('source bridge archived manifest differs: ' + name)
    if (receipt.get('schema') != 1 or receipt.get('passed') is not True
            or receipt.get('selection_sha256') != fleet.SOURCE_SHA
            or receipt.get('calibration_execution_identity') != calibration
            or receipt.get('consumer_execution_identity') != identity
            or receipt.get('bridge_sha256') != bridge_sha
            or receipt.get('gpu_model') != 'RTX A6000' or receipt.get('gpu_capability') != [8, 6]
            or receipt.get('calibration_jobs') != {'prepare': '20261105T025241Z-fe451f9e', 'score': '20261105T032433Z-1423f92a'}
            or receipt.get('stage_caps_fit') is not True
            or any(receipt['cost'][kind]['margin'] < 1.5 for kind in ('prepare', 'score'))):
        raise ValueError('full programme cost evidence does not authorize this profile')
    ledger = study.checked_json(ledger_path, ledger_sha)
    attestation = study.checked_json(budget_path, budget_sha)
    check_budget(ledger, attestation, ledger_sha, budget_sha, receipt)
    if gate['release'] == 'cost-confirmation':
        programme_projection(receipt)
    if gate['release'] in ('cost-confirmation', 'full-study', PAIR_RELEASE):
        qualification = gate.get('confirmation_qualification')
        if not isinstance(qualification, dict):
            raise ValueError('real-path confirmation qualification required')
        from experiments.tennis import k5d_scoped_namespace as namespace
        if (qualification.get('audited') is not True or qualification.get('cpu_native_pid_passed') is not True
                or qualification.get('gpu_16_parity_passed') is not True
                or qualification.get('consumer_execution_identity') != identity
                or qualification.get('gpu_model') != 'RTX A6000' or qualification.get('gpu_capability') != [8, 6]
                or qualification.get('namespace_wrapper_sha256') != oracle.sha(Path(namespace.__file__))
                or any(not isinstance(qualification.get(key), str) or not cf.digest_string(qualification[key])
                       for key in ('native_cpu_receipt_sha256', 'gpu_parity_receipt_sha256', 'unshare_sha256'))):
            raise ValueError('real-path qualification differs')
        if qualification_paths is None or set(qualification_paths) != {'native_cpu', 'gpu_parity'}:
            raise ValueError('both independently audited qualification receipts are required')
        native = study.checked_json(qualification_paths['native_cpu'], qualification['native_cpu_receipt_sha256'])
        parity = study.checked_json(qualification_paths['gpu_parity'], qualification['gpu_parity_receipt_sha256'])
        if (native.get('schema') != 1 or native.get('audited') is not True or native.get('passed') is not True
                or native.get('scope') != 'native CPU PID containment and accounting' or native.get('gpu_used') is not False
                or native.get('consumer_execution_identity') != identity
                or not isinstance(native.get('job_id'), str) or not native['job_id'].startswith('2026')
                or native.get('namespace_wrapper_sha256') != qualification['namespace_wrapper_sha256']
                or native.get('unshare_sha256') != qualification['unshare_sha256']
                or native.get('cases') != ['normal', 'timeout', 'sigterm', 'sigkill']
                or native.get('all_owned_pids_absent') is not True or native.get('independent_cpu_match') is not True
                or parity.get('schema') != 1 or parity.get('audited') is not True or parity.get('passed') is not True
                or parity.get('scope') != 'actual 16-frame GPU native and score parity'
                or not isinstance(parity.get('job_id'), str) or not parity['job_id'].startswith('2026')
                or parity.get('consumer_execution_identity') != identity or parity.get('gpu_model') != 'RTX A6000'
                or parity.get('gpu_capability') != [8, 6] or parity.get('all_six_exact') is not True
                or parity.get('namespace_wrapper_sha256') != qualification['namespace_wrapper_sha256']
                or parity.get('unshare_sha256') != qualification['unshare_sha256']):
            raise ValueError('actual native or GPU qualification receipt differs')
        if gate['release'] == PAIR_RELEASE and (parity.get('exclusive') is not True
                                                   or parity.get('timing_contaminated') is not False
                                                   or parity.get('source_bridge_sha256') != bridge_sha):
            raise ValueError('pair requires clean exclusive GPU qualification')
        if gate['release'] == 'cost-confirmation':
            confirmation_budget(gate, receipt)
        elif gate['release'] == PAIR_RELEASE:
            validate_calibration(receipt)
            if phase == 'score':
                verify_pair_ledger_charge(gate, ledger)
            first_pair_budget(gate, receipt, attestation, phase)
    if gate['release'] == 'full-study':
        qualification = gate.get('confirmation_qualification')
        if not isinstance(qualification, dict):
            raise ValueError('full-study qualification missing')
        if full_evidence_paths is None or set(full_evidence_paths) != {'inventory', 'noncommand'}:
            raise ValueError('full-study inventory and noncommand evidence required')
        from experiments.tennis import k5d_study_wave as wave
        inventory_doc = study.checked_json(full_evidence_paths['inventory'], gate['full_study_inventory_sha256'])
        noncommand_doc = study.checked_json(full_evidence_paths['noncommand'], gate['noncommand_evidence_sha256'])
        if verified_inventory is not None and not authoring:
            raise ValueError('preverified inventory is author-only')
        inventory = verified_inventory if verified_inventory is not None else wave.inventory(inventory_doc, identity)
        if any(inventory[clip][stage]['state'] != 'complete'
               for clip in wave.CONFIRMATIONS for stage in ('prepare', 'score')):
            raise ValueError('all four verified confirmation stages required')
        for confirmed_clip in wave.CONFIRMATIONS:
            for confirmed_phase in ('prepare', 'score'):
                entry = inventory[confirmed_clip][confirmed_phase]
                job_receipt = study.checked_json(Path(entry['receipt_path']), entry['receipt_sha256'])
                job_spec = study.checked_json(Path(job_receipt['job_directory']) / 'spec.json',
                                              job_receipt['records']['spec_sha256'])
                if cf.option(job_spec['arguments'], '--unshare-sha') != qualification['unshare_sha256']:
                    raise ValueError('confirmation namespace executable identity differs')
        if inventory[clip][phase]['state'] != 'pending':
            raise ValueError('requested target is not pending')
        if phase == 'score' and inventory[clip]['prepare']['state'] != 'complete':
            raise ValueError('score target has no verified producer')
        noncommand = wave.verified_noncommand(noncommand_doc, identity, inventory)
        projection = programme_projection(receipt, inventory=inventory, noncommand=noncommand)
        check_forecast_allocation(attestation, projection)
        if receipt.get('budget_fits') != {'gpu': True, 'cpu': True}:
            raise ValueError('full programme budget evidence differs')
        if not projection['gpu']['fits'] or not projection['cpu']['fits']:
            raise ValueError('recomputed full programme exceeds available budgets')
    if phase not in ('prepare', 'score'):
        raise ValueError('invalid stage')
    if gate['release'] == PAIR_RELEASE and phase == 'score':
        pair_producer(gate)
    return receipt, clips[clip]['label_rows']



def approved(gate_path: Path, gate_sha: str, cost_path: Path, cost_sha: str, bridge_path: Path, bridge_sha: str,
             old_source: Path, new_source: Path, ledger_path: Path, ledger_sha: str,
             budget_path: Path, budget_sha: str, identity: dict[str, str], clip: str, phase: str,
             qualification_paths: dict[str, Path] | None = None,
             full_evidence_paths: dict[str, Path] | None = None) -> tuple[dict[str, Any], int]:
    """Worker checks small signed inputs only; raw full-study audit is author-only."""
    gate = study.checked_json(gate_path, gate_sha)
    if gate.get('release') != 'full-study':
        return audit_raw_approval(gate_path, gate_sha, cost_path, cost_sha, bridge_path, bridge_sha,
                                  old_source, new_source, ledger_path, ledger_sha, budget_path, budget_sha,
                                  identity, clip, phase, qualification_paths, full_evidence_paths)
    if full_evidence_paths is None or set(full_evidence_paths) != {'certificate', 'microbenchmark', 'microbenchmark_audit'}:
        raise ValueError('full-study compact certificate required')
    from experiments.tennis import k5d_full_admission as admission
    cert = admission.read_certificate(full_evidence_paths['certificate'], gate.get('certificate_sha256'))
    admission.check_small_paths([gate_path, cost_path, ledger_path, budget_path, *([] if qualification_paths is None else qualification_paths.values())])
    admission.check_microbenchmark(full_evidence_paths['microbenchmark'], full_evidence_paths['microbenchmark_audit'],
                                   gate, identity, gate['certificate_sha256'])
    receipt = study.checked_json(cost_path, cost_sha)
    ledger = study.checked_json(ledger_path, ledger_sha)
    attestation = study.checked_json(budget_path, budget_sha)
    check_budget(ledger, attestation, ledger_sha, budget_sha, receipt)
    admission.check_certificate(cert, gate, identity, clip, phase, cost_sha, ledger_sha, budget_sha,
                                qualification_paths, receipt)
    check_forecast_allocation(attestation, cert['forecast'])
    return receipt, cert['targets'][clip][phase]['frames']

def spec(base: dict[str, Any], *, phase: str, clip: str, identity: dict[str, str], assets: dict[str, dict[str, str]],
         deadline: str, readiness_seconds: int, full_seconds: int, campaign_seconds: int) -> dict[str, Any]:
    if (base.get('device') != 'gpu' or base.get('cpu_threads') != 1
            or base.get('single_cpu_campaign') is not True
            or base.get('gpu_sharing') != {'mode': 'exclusive'}
            or base.get('contention') != {'policy': 'stop', 'resume_attempts': 0}):
        raise ValueError('exclusive single-core GPU campaign required')
    if (type(readiness_seconds) is not int or not 20 <= readiness_seconds <= 60
            or type(full_seconds) is not int or not 0 < full_seconds <= 2670
            or type(campaign_seconds) is not int or campaign_seconds > 2700
            or campaign_seconds < readiness_seconds + full_seconds + 30):
        raise ValueError('invalid readiness/full campaign bounds')
    if 'full_gate' not in assets:
        raise ValueError('missing approved evidence asset: full_gate')
    gate_data = study.checked_json(Path(assets['full_gate']['local']), assets['full_gate']['sha256'])
    full_release = gate_data.get('release') == 'full-study'
    for name, item in assets.items():
        if name == 'checkpoint' and gate_data.get('release') == PAIR_RELEASE and phase == 'score':
            validate_pair_checkpoint_asset(item, pair_producer(gate_data))
        if name == 'checkpoint' and item.get('archive_sha256') != item.get('sha256'):
            raise ValueError('checkpoint archive SHA aliases differ')
        if not (full_release and name == 'checkpoint'):
            cf.asset(name, item['remote'], item['sha256'], Path(item['local']), extract=False)
    required = {'full_gate', 'cost_receipt', 'fidelity_gate', 'budget_ledger', 'budget_attestation',
                'native_cpu_receipt', 'gpu_parity_receipt'}
    required |= {'certificate', 'certificate_microbench', 'certificate_microbench_audit'} if full_release else {'bridge', 'calibration_source', 'consumer_source'}
    if not required.issubset(assets):
        raise ValueError('missing approved evidence asset')
    if full_release and set(assets) != required | ({'checkpoint'} if phase == 'score' else set()):
        raise ValueError('full-study assets must be the exact compact set')
    if gate_data.get('release') == PAIR_RELEASE and phase == 'score':
        if 'pair_producer_audit' not in assets:
            raise ValueError('pair producer audit asset required')
        audit_asset = assets['pair_producer_audit']
        if audit_asset['sha256'] != pair_producer(gate_data)['audit_sha256']:
            raise ValueError('pair producer audit asset digest differs')
        verify_pair_producer_audit(gate_data, Path(audit_asset['local']), identity)
    if gate_data.get('release') == PAIR_RELEASE:
        fidelity = study.checked_json(Path(assets['fidelity_gate']['local']), assets['fidelity_gate']['sha256'])
        if fidelity.get('pair_cost_bound_sha256') != assets['cost_receipt']['sha256']:
            raise ValueError('pair fidelity cost evidence differs')
        benchmark.encoder.fidelity_gate(Path(assets['fidelity_gate']['local']), assets['fidelity_gate']['sha256'],
                                        fleet.EXPECTED['archive'], purpose=PAIR_RELEASE, clip=clip, frames=292)
    qualification_paths = {name: Path(assets[name + '_receipt']['local']) for name in ('native_cpu', 'gpu_parity')}
    full_evidence_paths = {'certificate': Path(assets['certificate']['local']),
                           'microbenchmark': Path(assets['certificate_microbench']['local']),
                           'microbenchmark_audit': Path(assets['certificate_microbench_audit']['local'])} if full_release else None
    receipt, frames = approved(Path(assets['full_gate']['local']), assets['full_gate']['sha256'],
                               Path(assets['cost_receipt']['local']), assets['cost_receipt']['sha256'],
                               Path(assets['bridge']['local']) if not full_release else Path('.'),
                               assets['bridge']['sha256'] if not full_release else '',
                               Path(assets['calibration_source']['local']) if not full_release else Path('.'),
                               Path(assets['consumer_source']['local']) if not full_release else Path('.'),
                               Path(assets['budget_ledger']['local']), assets['budget_ledger']['sha256'],
                               Path(assets['budget_attestation']['local']), assets['budget_attestation']['sha256'],
                               identity, clip, phase, qualification_paths, full_evidence_paths)
    if full_release:
        if full_evidence_paths is None:
            raise ValueError('compact certificate paths required')
        from experiments.tennis import k5d_full_admission as admission
        cert = admission.read_certificate(full_evidence_paths['certificate'], gate_data['certificate_sha256'])
        readiness_namespace = readiness_seconds - min(10, readiness_seconds / 4)
        if (gate_data['certificate_validation_reserve_seconds']
                + gate_data['namespace_bootstrap_reserve_seconds'] >= readiness_namespace):
            raise ValueError('compact verification leaves no readiness namespace time')
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        if not 0 < (admission._deadline(deadline) - now).total_seconds() <= 2700:
            raise ValueError('job deadline must be within 45 minutes')
        if admission._deadline(deadline) > admission._deadline(cert['valid_until']):
            raise ValueError('job deadline exceeds certificate validity')
    predicted = study.forecast(receipt, phase, frames, receipt['cost'][phase]['seconds16'])
    namespace_full = full_seconds - min(10, full_seconds / 4)
    verification_reserve = 0.0
    if full_release:
        verification_reserve = (gate_data['certificate_validation_reserve_seconds']
                                + gate_data['namespace_bootstrap_reserve_seconds'])
        if (isinstance(verification_reserve, bool) or not isinstance(verification_reserve, (int, float))
                or not math.isfinite(verification_reserve) or verification_reserve < 0):
            raise ValueError('invalid compact verification or namespace reserve')
    if predicted + verification_reserve > namespace_full:
        raise ValueError('forecast and verification exceed actual namespace timeout')
    extra = [cf.asset(name, item['remote'], item['sha256'], Path(item['local']), extract=False)
             for name, item in assets.items() if name != 'checkpoint']
    args = [phase, *base['arguments'], '--clip', clip, '--mode', '{mode}',
            '--full-gate', '{staged:full_gate}', '--full-gate-sha256', assets['full_gate']['sha256'],
            '--cost-receipt', '{staged:cost_receipt}', '--cost-receipt-sha256', assets['cost_receipt']['sha256'],
            '--budget-ledger', '{staged:budget_ledger}', '--budget-ledger-sha256', assets['budget_ledger']['sha256'],
            '--budget-attestation', '{staged:budget_attestation}', '--budget-attestation-sha256', assets['budget_attestation']['sha256'],
            '--fidelity-gate', '{staged:fidelity_gate}', '--fidelity-gate-sha256', assets['fidelity_gate']['sha256']]
    if not full_release:
        args += ['--bridge', '{staged:bridge}', '--bridge-sha256', assets['bridge']['sha256'],
                 '--calibration-source', '{staged:calibration_source}', '--consumer-source', '{staged:consumer_source}']
    if qualification_paths is not None:
        args += ['--native-cpu-receipt', '{staged:native_cpu_receipt}',
                 '--gpu-parity-receipt', '{staged:gpu_parity_receipt}']
    if full_release:
        args += ['--certificate', '{staged:certificate}', '--certificate-microbench', '{staged:certificate_microbench}',
                 '--certificate-microbench-audit', '{staged:certificate_microbench_audit}']
    if gate_data.get('release') == PAIR_RELEASE and phase == 'score':
        args += ['--pair-producer-audit', '{staged:pair_producer_audit}']
    if phase == 'score':
        if 'checkpoint' not in assets:
            raise ValueError('actual full-context checkpoint required')
        item = assets['checkpoint']
        if gate_data['release'] == PAIR_RELEASE:
            producer = pair_producer(gate_data)
            validate_pair_checkpoint_asset(item, producer)
        if full_release:
            producer = cert['targets'][clip]['score']['producer']
            if (item.get('remote') != producer['remote'] or item.get('sha256') != producer['archive_sha256']
                    or item.get('checkpoint_sha256') != producer['checkpoint_sha256']):
                raise ValueError('score checkpoint differs from certified producer')
            remote = Path(item['remote'])
            if not remote.is_absolute() or not remote.is_relative_to(cf.DATA_ROOT):
                raise ValueError('certified checkpoint outside shared data root')
            extra.append({'name': 'checkpoint', 'path': str(remote), 'sha256': item['sha256'], 'extract': True})
        else:
            extra.append(study.checkpoint(item, 'checkpoint', clip, frames, identity))
        args += ['--checkpoint', '{staged:checkpoint}', '--checkpoint-sha256', item['checkpoint_sha256']]
    result = cf.common_spec(base, extra, deadline)
    result.update(entrypoint=['-m', 'experiments.tennis.k5d_control_study'], arguments=args,
                  gpu_models=['RTX A6000'], budget_seconds=campaign_seconds, validator_seconds=30)
    result['scale']['frames'] = {'smoke': frames, 'full': frames}
    result['scale']['mode'] = {'smoke': 'readiness', 'full': 'full'}
    result['smoke'] = {'seconds': readiness_seconds, 'representative_basis': 'CPU validation of the complete source timeline, model identities, source bridge and prepared checkpoint.'}
    result['full'] = {'seconds': full_seconds}
    if gate_data['release'] in ('cost-confirmation', 'full-study', PAIR_RELEASE):
        if gate_data['release'] in ('cost-confirmation', PAIR_RELEASE):
            for resource in ('gpu', 'cpu'):
                cap = gate_data['confirmation_caps'][resource]['jobs'][f'{clip}:{phase}']
                if campaign_seconds > cap:
                    raise ValueError('campaign exceeds reserved confirmation job cap')
        else:
            for resource in ('gpu', 'cpu'):
                d = receipt['programme'][resource]
                if campaign_seconds + sum(d['noncommand_groups'].values()) > d['failed_start_reserve_seconds']:
                    raise ValueError('actual campaign cap exceeds prospective failure reserve')
        result['entrypoint'] = ['-m', 'experiments.tennis.k5d_scoped_namespace']
        result['arguments'] = ['outer', '--unshare-sha', gate_data['confirmation_qualification']['unshare_sha256'],
                               '--timeout', '{namespace_timeout}', '--', *args]
        result['scale']['namespace_timeout'] = {'smoke': str(readiness_seconds - min(10, readiness_seconds / 4)),
                                                 'full': str(full_seconds - min(10, full_seconds / 4))}
    result['validator'] = ['{python}', '-m', 'experiments.tennis.k5d_control_study', 'validate']
    from experiments.jobs.inbox import validate_spec
    validate_spec(result)
    return result


def verify_gate(args: argparse.Namespace, identity: dict[str, str]) -> int:
    _, frames = approved(Path(args.full_gate), args.full_gate_sha256, Path(args.cost_receipt), args.cost_receipt_sha256,
                         Path(args.bridge or '.'), args.bridge_sha256 or '', Path(args.calibration_source or '.'), Path(args.consumer_source or '.'),
                         Path(args.budget_ledger), args.budget_ledger_sha256, Path(args.budget_attestation),
                         args.budget_attestation_sha256, identity, args.clip, args.action,
                         {'native_cpu': Path(args.native_cpu_receipt), 'gpu_parity': Path(args.gpu_parity_receipt)}
                         if args.native_cpu_receipt and args.gpu_parity_receipt else None,
                         {'certificate': Path(args.certificate), 'microbenchmark': Path(args.certificate_microbench),
                          'microbenchmark_audit': Path(args.certificate_microbench_audit)}
                         if args.certificate and args.certificate_microbench and args.certificate_microbench_audit else None)
    if args.frames != frames:
        raise ValueError('full native timeline differs')
    gate = study.checked_json(Path(args.full_gate), args.full_gate_sha256)
    if gate.get('release') == PAIR_RELEASE and args.action == 'score':
        producer = pair_producer(gate)
        if not args.pair_producer_audit:
            raise ValueError('pair producer audit asset required')
        verify_pair_producer_audit(gate, Path(args.pair_producer_audit), identity)
        stage = Path(os.environ['PS_STAGE_DIR'])
        entries = [x for x in json.loads((stage / 'staging.json').read_text())['inputs'] if x['name'] == 'checkpoint']
        if (len(entries) != 1 or entries[0]['sha256'] != producer['archive_sha256']
                or entries[0]['source'] != producer['remote']
                or args.checkpoint_sha256 != producer['checkpoint_sha256']):
            raise ValueError('staged pair producer differs from audited producer')
    if args.certificate:
        from experiments.tennis import k5d_full_admission as admission
        gate = study.checked_json(Path(args.full_gate), args.full_gate_sha256)
        cert = admission.read_certificate(Path(args.certificate), gate['certificate_sha256'])
        stage = Path(os.environ['PS_STAGE_DIR'])
        spec_doc = json.loads((stage.parent / 'spec.json').read_text())
        if admission._deadline(spec_doc['deadline']) > admission._deadline(cert['valid_until']):
            raise ValueError('fleet deadline exceeds certificate validity')
        if args.action == 'score':
            producer = cert['targets'][args.clip]['score']['producer']
            entries = [x for x in json.loads((stage / 'staging.json').read_text())['inputs'] if x['name'] == 'checkpoint']
            if (len(entries) != 1 or entries[0]['sha256'] != producer['archive_sha256']
                    or entries[0]['source'] != producer['remote']
                    or args.checkpoint_sha256 != producer['checkpoint_sha256']):
                raise ValueError('staged score producer differs from certificate')
        admission.target_claim(cert, gate['certificate_sha256'], args.clip, args.action, stage.parent.name)
    return frames



def require_namespace_worker() -> None:
    """Bind the current PID-namespace child to its actual fleet dispatch."""
    from experiments.jobs.inbox import phase_command

    stage_name = os.environ.get('PS_STAGE', '')
    if stage_name not in ('smoke', 'full') or os.getppid() != 1:
        raise ValueError('actual PID-namespace init parent required')
    status = Path('/proc/self/status').read_text()
    fields = [line.split()[1:] for line in status.splitlines() if line.startswith('NSpid:')]
    if len(fields) != 1 or len(fields[0]) < 2 or int(fields[0][-1]) != os.getpid():
        raise ValueError('nested PID namespace identity required')
    stage = Path(os.environ['PS_STAGE_DIR'])
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if (spec.get('entrypoint') != ['-m', 'experiments.tennis.k5d_scoped_namespace']
            or spec.get('arguments', [])[:1] != ['outer']):
        raise ValueError('direct control-study fleet entrypoint prohibited')
    expected = phase_command(spec, stage_name, staging['inputs'], sys.executable)
    if expected != dispatch.get('command') or expected.count('--') != 1:
        raise ValueError('actual namespace dispatch differs from fleet specification')
    tail = expected[expected.index('--') + 1:]
    validate_namespace_child_argv(sys.argv[1:], tail)
    parent_fields = [line.split()[1:] for line in status.splitlines() if line.startswith('PPid:')]
    if len(parent_fields) != 1 or len(parent_fields[0]) != 1:
        raise ValueError('host parent PID missing')
    host_parent = int(parent_fields[0][0])
    parent_root = Path('/proc') / str(host_parent)
    parent_args = [x.decode() for x in (parent_root / 'cmdline').read_bytes().split(b'\0') if x]
    parent_status = (parent_root / 'status').read_text()
    nested = [line.split()[1:] for line in parent_status.splitlines() if line.startswith('NSpid:')]
    own_ns = os.readlink('/proc/self/ns/pid')
    parent_ns = os.readlink(parent_root / 'ns/pid')
    validate_namespace_parent(parent_args, nested, own_ns, parent_ns, expected, tail)



def validate_namespace_child_argv(actual: list[str], expected: list[str]) -> None:
    """The CLI removes --accounted-child before parsing the worker arguments."""
    if actual != expected:
        raise ValueError('namespace child arguments differ from fleet dispatch')

def validate_namespace_parent(parent_args: list[str], nested: list[list[str]], own_ns: str, parent_ns: str,
                              expected: list[str], tail: list[str]) -> None:
    """Join the host-visible parent to the frozen fleet namespace command."""
    from experiments.tennis import k5d_scoped_namespace as namespace

    if (len(nested) != 1 or len(nested[0]) < 2 or nested[0][-1] != '1' or own_ns != parent_ns
            or len(parent_args) < 10 or Path(parent_args[0]).resolve() != Path(sys.executable).resolve()
            or parent_args[1:4] != ['-m', 'experiments.tennis.k5d_scoped_namespace', 'inner']
            or parent_args[4] != '--outside' or parent_args[5] == own_ns
            or not parent_args[5].startswith('pid:[') or parent_args[6] != '--timeout'
            or parent_args[8] != '--' or parent_args[9:] != tail
            or float(parent_args[7]) != float(cf.option(expected, '--timeout'))
            or Path(namespace.__file__).resolve() != Path(__file__).resolve().with_name('k5d_scoped_namespace.py')):
        raise ValueError('actual namespace init identity or command differs')

def run(args: argparse.Namespace) -> None:
    require_namespace_worker()
    identity = cp.execution_identity()
    verify_gate(args, identity)
    gate = study.checked_json(Path(args.full_gate), args.full_gate_sha256)
    if gate.get('release') == PAIR_RELEASE:
        args.same_job_receiver = True
        if args.action == 'score':
            args.receiver_producer_job = pair_producer(gate)['job_id']
        fidelity = study.checked_json(Path(args.fidelity_gate), args.fidelity_gate_sha256)
        if fidelity.get('pair_cost_bound_sha256') != args.cost_receipt_sha256:
            raise ValueError('pair fidelity cost evidence differs')
        args.fidelity_purpose = PAIR_RELEASE
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    if {'smoke': 'readiness', 'full': 'full'}.get(os.environ.get('PS_STAGE', '')) != args.mode:
        raise ValueError('stage and control profile differ')
    if args.mode == 'readiness':
        result = readiness.inputs(args, stage, identity)
        output = scratch / 'publish'
        output.mkdir()
        cp.write_new(output / 'readiness.json', result)
    else:
        data = benchmark.build(args)
        if args.action == 'prepare':
            cp.prepare(data, identity)
        else:
            data.checkpoint = str(Path(args.checkpoint) / 'publish')
            co.score_all(data, args.checkpoint_sha256, identity)



def check_full_timeout(predicted: float, observed: float, timeout: float) -> None:
    values = (predicted, observed, timeout)
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values)
            or predicted + 1.5 * observed > timeout):
        raise ValueError('actual readiness and numerical forecast exceed full namespace timeout')

def validate() -> None:
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        stage = Path(os.environ['PS_STAGE_DIR'])
        staged = json.loads((stage / 'staging.json').read_text())
        with cost.unpack(stage / 'published.tar', staged['published']['sha256']) as output:
            report = json.loads((output / 'readiness.json').read_text())
            if (report.get('identity') != cp.execution_identity() or report['source']['frames'] <= 0
                    or report['source']['frames'] != report['source']['labels']
                    or report['scope'] != 'CPU asset, source header and checkpoint readiness; no GPU scoring'):
                raise ValueError('incomplete CPU readiness output')
        from experiments.tennis import k5d_cpu_accounting as accounting
        spec_data = json.loads((stage.parent / 'spec.json').read_text())
        args = spec_data['arguments']
        if (spec_data['entrypoint'] != ['-m', 'experiments.tennis.k5d_scoped_namespace']
                or spec_data.get('arguments', [])[:1] != ['outer']):
            raise ValueError('readiness validator requires namespace entrypoint')
        from experiments.jobs.inbox import phase_command
        dispatch = json.loads((stage / 'dispatch.json').read_text())
        if dispatch.get('command') != phase_command(spec_data, 'smoke', staged['inputs'], sys.executable):
            raise ValueError('readiness namespace dispatch differs from specification')
        if args.count('--') != 1:
            raise ValueError('namespace worker separator differs')
        args = args[args.index('--') + 1:]
        if '--accounted-child' in args:
            raise ValueError('fleet command may not bypass CPU wrapper')
        log_text = (stage / 'command.log').read_text()
        cpu_records = [json.loads(line.removeprefix('K5D_CPU_ACCOUNTING ')) for line in log_text.splitlines()
                       if line.startswith('K5D_CPU_ACCOUNTING ')]
        if len(cpu_records) != 1 or cpu_records[0].get('method') != 'linux-pidns-init-waited-rusage-v1':
            raise ValueError('namespace accounting method required')
        accounting.read_receipt(log_text, phase=args[0], stage='smoke',
                                clip=cf.option(args, '--clip'), frames=report['source']['frames'],
                                identity=cp.execution_identity())
        # The bounded CPU-only readiness smoke is the actual target-specific
        # payload measurement. Require its observed complete stage time, with
        # margin, plus the numerical forecast to fit the effective namespace
        # deadline before the GPU full stage can start.
        execution = json.loads((stage / 'execution.json').read_text())
        observed = execution.get('seconds')
        receipt_path = next(item['path'] for item in staged['inputs'] if item['name'] == 'cost_receipt')
        cost_receipt = study.checked_json(Path(receipt_path), cf.option(args, '--cost-receipt-sha256'))
        predicted = study.forecast(cost_receipt, args[0], report['source']['frames'],
                                   cost_receipt['cost'][args[0]]['seconds16'])
        full_timeout = float(spec_data['scale']['namespace_timeout']['full'])
        check_full_timeout(predicted, observed, full_timeout)
        target.write_text(json.dumps({'passed': True, 'checks': {'full_timeline_and_assets': True,
                                      'strict_frozen_identity': True}}) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'readiness': False}, 'error': str(exc)}) + '\n')
        raise


def main() -> None:
    if sys.argv[1:] == ['validate']:
        validate()
        return
    if sys.argv[1:2] == ['--make-spec']:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        options = parser.parse_args(sys.argv[2:])
        data = json.loads(options.config.read_text())
        result = spec(json.loads(cf.BASE.read_text()), **data)
        cp.write_new(options.output, result)
        return
    accounted_child = sys.argv[1:2] == ['--accounted-child']
    if accounted_child:
        del sys.argv[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'score'))
    for name in ('archive', 'balltrack-code', 'balltrack-weight', 'wasb-code', 'wasb-weights', 'wasb-config',
                 'image-ckpt', 'video-ckpt', 'lpips-backbone', 'dists-backbone', 'clip', 'full-gate', 'full-gate-sha256',
                 'cost-receipt', 'cost-receipt-sha256',
                 'fidelity-gate', 'fidelity-gate-sha256', 'budget-ledger', 'budget-ledger-sha256',
                 'budget-attestation', 'budget-attestation-sha256', 'mode'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--bridge')
    parser.add_argument('--bridge-sha256')
    parser.add_argument('--calibration-source')
    parser.add_argument('--consumer-source')
    parser.add_argument('--certificate')
    parser.add_argument('--certificate-microbench')
    parser.add_argument('--certificate-microbench-audit')
    parser.add_argument('--native-cpu-receipt')
    parser.add_argument('--gpu-parity-receipt')
    parser.add_argument('--study-inventory')
    parser.add_argument('--noncommand-evidence')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--pair-producer-audit')
    parser.add_argument('--frames', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'score' and (not args.checkpoint or not args.checkpoint_sha256):
        parser.error('score requires actual checkpoint and SHA')
    if accounted_child:
        run(args)
    else:
        from experiments.tennis import k5d_cpu_accounting as accounting
        command = [sys.executable, '-m', 'experiments.tennis.k5d_control_study',
                   '--accounted-child', *sys.argv[1:]]
        stage_name = os.environ.get('PS_STAGE', '')
        stage_dir = Path(os.environ['PS_STAGE_DIR'])
        spec_data = json.loads((stage_dir.parent / 'spec.json').read_text())
        seconds = spec_data[stage_name]['seconds']
        accounting.measure(command, phase=args.action, clip=args.clip, frames=args.frames,
                           stage=stage_name, identity=cp.execution_identity(), timeout_seconds=seconds - min(10, seconds / 4))


if __name__ == '__main__':
    main()

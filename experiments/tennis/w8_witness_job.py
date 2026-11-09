"""Proposed W8 context witness control. No compute release is implied.

The CPU gate is a root-pinned audit envelope, not an operator boolean. Its raw
proof hashes and exact source/runtime identities must be independently audited
before root supplies the envelope SHA to a prospective fleet spec.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.tennis import w8_strict_host_cost_job as host
from experiments.tennis import w8_input_job as fixed

CPU_SCOPE = 'root-reviewed W8 same-source CPU isolation and native lifecycle proof'
REQUIRED_PROOFS = ('namespace_lifecycle', 'native_normal', 'native_timeout_kill', 'strict_modes', 'assessment_policy', 'independent_audit')
REQUIRED_CHECKS = ('same_source_snapshot_environment', 'cpu_hidden_one_core', 'namespace_normal_timeout_term_kill',
                   'native_real_yuv_and_timeout_kill', 'owned_pid_starttick_teardown',
                   'strict_runtime_policy_deny_paths', 'strict_mode_dispatch_static_reviewed',
                   'strict_source_closure_and_clock_leaves', 'assessment_role_policy', 'counter_and_external_cost_coverage')


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _check_deadline(spec: dict[str, Any], *, reference_epoch: float | None = None) -> None:
    from datetime import datetime, timezone

    raw = spec.get('deadline')
    if not isinstance(raw, str):
        raise ValueError('45-minute recovery deadline required')
    try:
        deadline = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('invalid recovery deadline') from exc
    now = datetime.now(timezone.utc) if reference_epoch is None else datetime.fromtimestamp(reference_epoch, timezone.utc)
    if deadline.tzinfo is None or not 0 < (deadline - now).total_seconds() <= 2700:
        raise ValueError('45-minute recovery deadline required')


def check_bootstrap_archive(stage: Path, dispatch: dict[str, Any], expected: str) -> None:
    """Join the declared new bootstrap to the authenticated physical snapshot."""
    import hashlib
    import tarfile

    archive = stage.parent / 'source.tar'
    if (not _sha(expected) or fixed.file_sha(archive) != dispatch.get('code', {}).get('snapshot_sha256')
            or fixed.file_sha(Path(__file__).resolve().parent / 'w8_receiver_bootstrap.py') != expected):
        raise ValueError('witness source snapshot or bootstrap differs')
    with tarfile.open(archive) as tar:
        entry = tar.getmember('experiments/tennis/w8_receiver_bootstrap.py')
        stream = tar.extractfile(entry) if entry.isfile() else None
        if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected:
            raise ValueError('bootstrap member differs from authenticated snapshot')


def check_cpu_gate(gate: dict[str, Any], *, gate_sha: str, expected_sha: str,
                   source: dict[str, Any], environment_sha: str, host_name: str | None) -> None:
    """Check the exact root-pinned envelope after primary review of raw proofs."""
    if (not _sha(gate_sha) or gate_sha != expected_sha or gate.get('schema') != 1
            or gate.get('scope') != CPU_SCOPE or gate.get('passed') is not True
            or not _sha(source.get('source_sha256')) or not _sha(source.get('snapshot_sha256'))
            or not re.fullmatch(r'[0-9a-f]{40}', str(source.get('git_head')))
            or not _sha(environment_sha) or host_name not in ('gpu5', 'gpu6')
            or not _sha(gate.get('bootstrap_sha256'))
            or gate.get('source_sha256') != source.get('source_sha256')
            or gate.get('snapshot_sha256') != source.get('snapshot_sha256')
            or gate.get('revision') != source.get('git_head')
            or gate.get('environment_sha256') != environment_sha or gate.get('host') != host_name
            or not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', str(gate.get('job_id')))
            or not _sha(gate.get('spec_sha256'))):
        raise ValueError('same-source CPU proof envelope identity differs')
    raw, checks = gate.get('raw_receipt_sha256'), gate.get('checks')
    if (not isinstance(raw, dict) or set(raw) != set(REQUIRED_PROOFS)
            or any(not _sha(raw[name]) for name in REQUIRED_PROOFS)
            or not isinstance(checks, dict) or set(checks) != set(REQUIRED_CHECKS)
            or any(checks[name] is not True for name in REQUIRED_CHECKS)):
        raise ValueError('independent lifecycle, native and strict source proof references required')


def load_inputs(stage: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    """Bind staged proof and every frozen receiver input to the fleet dispatch."""
    paths, spec = host.stage_inputs(stage, witness=True)
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    proof = json.loads(paths['cpu_proof'].read_text())
    code = dispatch.get('code', {})
    environment = dispatch.get('staged_environment', {})
    host_name = (dispatch.get('environment') or {}).get('alias')
    check_cpu_gate(proof, gate_sha=fixed.file_sha(paths['cpu_proof']),
                   expected_sha=spec['witness_cpu_proof_sha256'], source=code,
                   environment_sha=environment.get('sha256'), host_name=host_name)
    if (spec.get('witness_bootstrap_sha256') != proof.get('bootstrap_sha256')
            or fixed.file_sha(Path(__file__).resolve().parent / 'w8_receiver_bootstrap.py') != proof['bootstrap_sha256']):
        raise ValueError('actual witness bootstrap differs from CPU proof')
    check_bootstrap_archive(stage, dispatch, proof['bootstrap_sha256'])
    return paths, spec, dispatch


def load_released_inputs(stage: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    """Use immutable staged-source records after fleet scratch has been released."""
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    declared = {x['name']: x for x in spec['staged_inputs']}
    staged = {x['name']: x for x in staging['inputs']}
    expected = {'packet': fixed.WINDOW_SHA, 'profile': fixed.PROFILE_SHA, 'oracle': host.ORACLE_SHA,
                'base': host.BASE_ARCHIVE_SHA, 'aa': host.receiver.AA_ARCHIVE_SHA,
                **{key: host.receiver.PINNED_WEIGHTS[name] for key, name in host.WEIGHT_NAMES.items()},
                'cpu_proof': spec['witness_cpu_proof_sha256']}
    promoted_profiles = ('w8-context-unit1-v1', 'w8-full-unit0-v1', 'w8-full-unit1-v1',
                         'w8-context-qualification-v1', 'w8-full-finish-v1')
    if spec.get('w8_witness_profile') in promoted_profiles:
        expected.update(promotion=spec['witness_promotion_sha256'], ledger=spec['witness_ledger_sha256'])
    elif spec.get('w8_witness_profile') == 'w8-context-pilot-v1':
        expected.update(ledger=spec['witness_ledger_sha256'], admission=spec['witness_admission_sha256'])
    if spec.get('w8_witness_profile') in ('w8-context-qualification-v1', 'w8-full-finish-v1'):
        from experiments.tennis.w8_receiver_finish import SAM_SHA
        expected.update(sam_weights=SAM_SHA,
                        native_inventory='cd6bb8cf2a239d94f595fd92f823cf6406c018e1fbda745e81cd0c0e91d1b59b',
                        **{name: spec['witness_' + name + '_sha256']
                           for name in ('producer_audit', 'producer0', 'producer1')})
    if (len(declared) != len(spec['staged_inputs']) or len(staged) != len(staging['inputs'])
            or set(declared) != set(expected) | {'preflight', 'identity'} or set(staged) != set(declared)):
        raise ValueError('released staged asset membership differs')
    for name in ('preflight', 'identity'):
        expected[name] = declared[name]['sha256']
    for name, digest in expected.items():
        if (declared[name].get('sha256') != digest or staged[name].get('sha256') != digest
                or declared[name].get('extract', False) or staged[name].get('extract', False)
                or staged[name].get('source') != declared[name].get('path')):
            raise ValueError('released staged asset identity differs')
    small_names = ['preflight', 'identity', 'oracle', 'cpu_proof']
    if spec.get('w8_witness_profile') in promoted_profiles:
        small_names.extend(('promotion', 'ledger'))
    elif spec.get('w8_witness_profile') == 'w8-context-pilot-v1':
        small_names.extend(('ledger', 'admission'))
    if spec.get('w8_witness_profile') in ('w8-context-qualification-v1', 'w8-full-finish-v1'):
        small_names.append('producer_audit')
    paths = {name: Path(staged[name]['source']) for name in small_names}
    from experiments.tennis.w8_protected_metadata import reject_protected_alias
    for path in paths.values():
        reject_protected_alias(path)
    if any(fixed.file_sha(path) != expected[name] for name, path in paths.items()):
        raise ValueError('released proof or preflight source differs')
    proof = json.loads(paths['cpu_proof'].read_text())
    check_cpu_gate(proof, gate_sha=fixed.file_sha(paths['cpu_proof']),
                   expected_sha=spec['witness_cpu_proof_sha256'], source=dispatch.get('code', {}),
                   environment_sha=dispatch.get('staged_environment', {}).get('sha256'),
                   host_name=(dispatch.get('environment') or {}).get('alias'))
    if spec.get('witness_bootstrap_sha256') != proof.get('bootstrap_sha256'):
        raise ValueError('released witness bootstrap differs from CPU proof')
    check_bootstrap_archive(stage, dispatch, proof['bootstrap_sha256'])
    return paths, spec, dispatch


def check_promotion(gate: dict[str, Any], *, gate_sha: str, expected_sha: str,
                    source: dict[str, Any], cpu_gate_sha: str, prior_artifact_sha: str,
                    ledger_sha: str, gpu_remaining: float, cpu_remaining: float,
                    reference_epoch: float | None = None) -> None:
    """Check a root-pinned complete remaining-work forecast, not a release action."""
    import math
    import time

    if (not _sha(gate_sha) or gate_sha != expected_sha
            or gate.get('scope') != 'root-reviewed W8 remaining-work forecast'
            or gate.get('schema') != 1 or gate.get('cpu_proof_sha256') != cpu_gate_sha
            or gate.get('prior_artifact_sha256') != prior_artifact_sha
            or gate.get('ledger_sha256') != ledger_sha
            or not all(_sha(value) for value in (cpu_gate_sha, prior_artifact_sha, ledger_sha))
            or gate.get('source_sha256') != source.get('source_sha256')
            or gate.get('snapshot_sha256') != source.get('snapshot_sha256')
            or not _sha(source.get('source_sha256')) or not _sha(source.get('snapshot_sha256'))
            or type(gate.get('safety_factor')) not in (int, float)
            or not math.isfinite(gate['safety_factor']) or gate['safety_factor'] < 1.5
            or not _sha(gate.get('raw_measurement_audit_sha256'))):
        raise ValueError('root-reviewed promotion evidence differs')
    issued, expires = gate.get('issued_epoch'), gate.get('expires_epoch')
    now = time.time() if reference_epoch is None else reference_epoch
    if (isinstance(issued, bool) or not isinstance(issued, (int, float)) or not math.isfinite(issued)
            or isinstance(expires, bool) or not isinstance(expires, (int, float)) or not math.isfinite(expires)):
        raise ValueError('promotion attestation is not fresh')
    if not issued <= now < expires or expires - issued > 2700:
        raise ValueError('promotion attestation is not fresh')
    if (gate.get('gpu_remaining_seconds') != gpu_remaining or gate.get('cpu_remaining_seconds') != cpu_remaining):
        raise ValueError('forecast budget differs from reviewed ledger attestation')
    required = {'unit1_context', 'unit0_full', 'unit1_full', 'native_sam_compose', 'postseal_score',
                'offline_validation', 'external_staging_audits', 'failure_headroom'}
    components = gate.get('remaining_components')
    completed = gate.get('completed_artifact_sha256', {})
    if (not isinstance(components, dict) or set(components) != required or not isinstance(completed, dict)
            or not set(completed).issubset(required - {'external_staging_audits', 'failure_headroom'})
            or any(not _sha(v) for v in completed.values())):
        raise ValueError('complete remaining-work components required')
    for resource, remaining in (('gpu_seconds', gpu_remaining), ('cpu_seconds', cpu_remaining)):
        if (isinstance(remaining, bool) or not isinstance(remaining, (int, float))
                or not math.isfinite(remaining) or remaining <= 0):
            raise ValueError('positive current budget required')
        values = [components[name].get(resource) if isinstance(components[name], dict) else None for name in required]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('complete forecast does not fit current budget')
        if any((components[name][resource] == 0) != (name in completed) for name in required):
            raise ValueError('completed and pending forecast identities differ')
        if sum(float(v) for v in values if v is not None) > remaining:
            raise ValueError('complete forecast does not fit current budget')


def promotion_ledger_remaining(ledger: dict[str, Any]) -> tuple[float, float]:
    """Use the exact current remaining-unspent W8 reservation once."""
    import math

    if ledger.get('reservation_amount_basis') != 'remaining_unspent':
        raise ValueError('explicit remaining-unspent ledger basis required')
    active: dict[str, dict[str, Any]] = {}
    for row in ledger['reservations']:
        status = row.get('status')
        if not isinstance(status, str) or not status:
            raise ValueError('reservation status missing')
        if status.lower().startswith(('closed', 'released', 'superseded', 'cancelled')):
            continue
        phase = row.get('phase')
        if not isinstance(phase, str) or not phase or phase in active:
            raise ValueError('active reservation phase missing or duplicated')
        active[phase] = row
    phase = 'w8-full-witness-staged-20261106'
    owned = active.get(phase)
    if owned is None or owned.get('workstream') != 'w8':
        raise ValueError('active W8 witness allocation required')
    result = []
    for resource in ('gpu', 'cpu'):
        field = resource + '_seconds'
        values = [row.get(field, 0) for row in ledger['charges']] + [row.get(field) for row in active.values()]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid charged or remaining reservation amount')
        total = sum(float(row.get(field, 0)) for row in ledger['charges'])
        total += sum(float(row[field]) for row in active.values())
        w8_total = sum(float(row.get(field, 0)) for row in ledger['charges'] if row.get('workstream') == 'w8')
        w8_total += sum(float(row[field]) for row in active.values() if row.get('workstream') == 'w8')
        global_limit = ledger[resource + '_total_seconds']
        workstream_limit = ledger['workstreams']['w8'][resource + '_limit_seconds']
        if total > global_limit or w8_total > workstream_limit or owned[field] <= 0:
            raise ValueError('W8 or global reservation exceeds ledger ceiling')
        result.append(float(owned[field]))
    return result[0], result[1]




def check_first_pilot_ledger(ledger: dict[str, Any], receipt: dict[str, Any], *,
                             ledger_sha: str, cpu_proof_sha: str) -> None:
    """Allow only attributed preparation for this first attempt; count it once."""
    gpu, cpu = promotion_ledger_remaining(ledger)
    nonce = receipt.get('attempt_nonce')
    gpu_cap, cpu_cap = receipt.get('gpu_ceiling_seconds'), receipt.get('cpu_ceiling_seconds')
    if (receipt.get('schema') != 1 or receipt.get('scope') != 'root-reviewed first W8 qualification pilot'
            or not isinstance(nonce, str) or not re.fullmatch(r'[0-9a-f]{32}', nonce)
            or receipt.get('ledger_sha256') != ledger_sha or receipt.get('cpu_proof_sha256') != cpu_proof_sha
            or type(gpu_cap) is not int or type(cpu_cap) is not int or not 250 <= gpu_cap <= 900
            or not 341 <= cpu_cap <= 1000 or gpu_cap > gpu or cpu_cap > cpu):
        raise ValueError('first pilot receipt or remaining reservation differs')
    rows = [row for row in ledger['charges'] if row.get('parent_phase') == 'w8-full-witness-staged-20261106']
    if any(type(row.get('witness_qualification')) is not bool for row in rows):
        raise ValueError('every parent charge needs explicit qualification attribution')
    prep = [row for row in rows if row['witness_qualification']]
    phases = [row.get('phase') for row in prep]
    if (any(row.get('qualification_charge_kind') != 'pilot_preparation'
            or row.get('qualification_attempt_nonce') != nonce for row in prep)
            or any(not isinstance(phase, str) or not phase for phase in phases)
            or len(phases) != len(set(phases)) or sorted(phases) != receipt.get('preparation_phases')):
        raise ValueError('first pilot preparation is mixed, unattributed or follows an earlier execution')
    for resource, cap, limit in (('gpu', gpu_cap, 900), ('cpu', cpu_cap, 1000)):
        spent = sum(row[resource + '_seconds'] for row in prep)
        if receipt.get(resource + '_spent_seconds') != spent or spent + cap > limit:
            raise ValueError('first pilot cumulative qualification ceiling exceeded')



def check_first_pilot_stage(paths: dict[str, Path], spec: dict[str, Any]) -> None:
    receipt = json.loads(paths['admission'].read_text())
    check_first_pilot_ledger(json.loads(paths['ledger'].read_text()), receipt,
        ledger_sha=spec['witness_ledger_sha256'], cpu_proof_sha=spec['witness_cpu_proof_sha256'])
    if (receipt['gpu_ceiling_seconds'] != spec['witness_gpu_ceiling_seconds']
            or receipt['cpu_ceiling_seconds'] != spec['witness_cpu_ceiling_seconds']):
        raise ValueError('first pilot stage ceilings differ from reviewed attempt')

def check_qualification_subcap(ledger: dict[str, Any], gate: dict[str, Any], *,
                               gpu_cap: float, cpu_cap: float) -> None:
    """Charge tagged qualification entries once against its cumulative ceiling."""
    rows = [row for row in ledger['charges'] if row.get('parent_phase') == 'w8-full-witness-staged-20261106']
    if any(type(row.get('witness_qualification')) is not bool for row in rows):
        raise ValueError('every parent charge needs explicit qualification attribution')
    qualified = [row for row in rows if row['witness_qualification']]
    phases = [row.get('phase') for row in qualified]
    if (any(not isinstance(phase, str) or not phase for phase in phases)
            or len(phases) != len(set(phases)) or sorted(phases) != gate.get('qualification_charge_phases')):
        raise ValueError('qualification charge attribution differs from reviewed gate')
    for resource, cap, ceiling in (('gpu', gpu_cap, 900), ('cpu', cpu_cap, 1000)):
        spent = sum(row[resource + '_seconds'] for row in qualified)
        if gate.get('qualification_' + resource + '_spent_seconds') != spent or spent + cap > ceiling:
            raise ValueError('qualification cumulative ceiling exceeded')

def _unit1_promotion(paths: dict[str, Path], spec: dict[str, Any], dispatch: dict[str, Any], *,
                     historical: bool = False) -> None:
    gate = json.loads(paths['promotion'].read_text())
    ledger_sha = fixed.file_sha(paths['ledger'])
    ledger = json.loads(paths['ledger'].read_text())
    gpu_remaining, cpu_remaining = promotion_ledger_remaining(ledger)
    if (spec.get('witness_gpu_remaining_seconds') != gpu_remaining
            or spec.get('witness_cpu_remaining_seconds') != cpu_remaining):
        raise ValueError('spec budget differs from active ledger reservation')
    check_promotion(gate, gate_sha=fixed.file_sha(paths['promotion']),
        expected_sha=spec['witness_promotion_sha256'], source=dispatch['code'],
        cpu_gate_sha=spec['witness_cpu_proof_sha256'], prior_artifact_sha=spec['witness_prior_artifact_sha256'],
        ledger_sha=ledger_sha, gpu_remaining=gpu_remaining, cpu_remaining=cpu_remaining,
        reference_epoch=dispatch['started'] if historical else None)
    profile = spec.get('w8_witness_profile')
    if profile in ('w8-context-unit1-v1', 'w8-context-qualification-v1'):
        check_qualification_subcap(ledger, gate, gpu_cap=spec['witness_gpu_ceiling_seconds'],
                                   cpu_cap=spec['witness_cpu_ceiling_seconds'])
    completed = gate.get('completed_artifact_sha256', {})
    if profile == 'w8-context-unit1-v1' and (gate.get('qualification_next_gpu_ceiling_seconds') != spec['witness_gpu_ceiling_seconds']
            or gate.get('qualification_next_cpu_ceiling_seconds') != spec['witness_cpu_ceiling_seconds']):
        raise ValueError('unit-1 stage ceilings differ from reviewed forecast')
    if profile in ('w8-full-unit0-v1', 'w8-full-unit1-v1'):
        expected_limits = {'full_seconds': spec['full']['seconds'], 'campaign_seconds': spec['budget_seconds'],
            'gpu_ceiling_seconds': spec['witness_gpu_ceiling_seconds'], 'cpu_ceiling_seconds': spec['witness_cpu_ceiling_seconds'],
            'generator_cpu': spec['witness_caps']['generator_cpu'], 'generator_wall': spec['witness_caps']['generator_wall']}
        if gate.get('candidate_next_limits') != expected_limits:
            raise ValueError('producer stage limits differ from reviewed forecast')
    if profile == 'w8-full-finish-v1':
        expected_limits = {'full_seconds': spec['full']['seconds'], 'campaign_seconds': spec['budget_seconds'],
            'gpu_ceiling_seconds': spec['witness_gpu_ceiling_seconds'], 'cpu_ceiling_seconds': spec['witness_cpu_ceiling_seconds'],
            'worker_cpu': spec['witness_caps']['worker_cpu'], 'worker_wall': spec['witness_caps']['worker_wall']}
        if gate.get('finish_next_limits') != expected_limits:
            raise ValueError('finish stage limits differ from reviewed forecast')
    if profile == 'w8-context-qualification-v1' and (gate.get('qualification_next_gpu_ceiling_seconds') != spec['witness_gpu_ceiling_seconds']
            or gate.get('qualification_next_cpu_ceiling_seconds') != spec['witness_cpu_ceiling_seconds']):
        raise ValueError('qualification finish ceilings differ from reviewed forecast')
    if profile == 'w8-context-unit1-v1' and 'unit1_context' in completed:
        raise ValueError('unit-1 context already completed; no replay')
    if profile in ('w8-full-unit0-v1', 'w8-full-unit1-v1'):
        target = 'unit1_full' if profile == 'w8-full-unit1-v1' else 'unit0_full'
        if (target in completed or 'unit1_context' not in completed
                or not _sha(gate.get('qualification_receipt_sha256'))):
            raise ValueError('qualification incomplete or producer already completed')
    if profile in ('w8-context-qualification-v1', 'w8-full-finish-v1'):
        needed = {'unit1_context'} if profile == 'w8-context-qualification-v1' else {'unit0_full', 'unit1_full'}
        if not needed.issubset(completed) or 'native_sam_compose' in completed:
            raise ValueError('finish producers incomplete or finish already completed')
        bindings = ({'unit1_context': spec['witness_producer1_sha256']}
                    if profile == 'w8-context-qualification-v1' else
                    {'unit0_full': spec['witness_producer0_sha256'], 'unit1_full': spec['witness_producer1_sha256']})
        if any(completed.get(name) != digest for name, digest in bindings.items()):
            raise ValueError('promotion does not bind the exact producer archives')
    component = ('native_sam_compose' if profile in ('w8-context-qualification-v1', 'w8-full-finish-v1') else
                 'unit1_context' if profile == 'w8-context-unit1-v1' else
                 'unit1_full' if profile == 'w8-full-unit1-v1' else 'unit0_full')
    if (gate['remaining_components'][component]['gpu_seconds'] < spec['witness_gpu_ceiling_seconds']
            or gate['remaining_components'][component]['cpu_seconds'] < spec['witness_cpu_ceiling_seconds']):
        raise ValueError('forecast does not reserve this stage inclusive caps')


def _pilot_spec(spec: dict[str, Any], *, unit1: bool = False, reference_epoch: float | None = None) -> None:
    """Reject every profile other than the explicitly selected context stage."""
    profile = 'w8-context-unit1-v1' if unit1 else 'w8-context-pilot-v1'
    phase = 'unit1' if unit1 else 'pilot'
    _check_deadline(spec, reference_epoch=reference_epoch)
    if (spec.get('w8_witness_profile') != profile
            or spec.get('entrypoint') != ['-m', 'experiments.tennis.w8_witness_job']
            or spec.get('arguments') != ['{phase}']
            or spec.get('scale') != {'phase': {'smoke': phase, 'full': 'noop'}}
            or spec.get('validator') != ['{python}', '-m', 'experiments.tennis.w8_witness_job', 'validate']
            or spec.get('device') != 'gpu' or spec.get('gpu_models') != ['RTX 6000 Ada']
            or spec.get('gpu_sharing') != {'mode': 'exclusive'}
            or spec.get('witness_cpu_proof_host') not in ('gpu5', 'gpu6')
            or not _sha(spec.get('witness_bootstrap_sha256'))
            or spec.get('hosts') != [spec.get('witness_cpu_proof_host')]
            or spec.get('single_cpu_campaign') is not True or spec.get('cpu_threads') != 1
            or spec.get('contention', {}).get('policy') != 'stop'
            or spec.get('contention', {}).get('resume_attempts') != 0
            or not 0 < spec.get('smoke', {}).get('seconds', 0) <= 600
            or spec.get('full', {}).get('seconds') != 5
            or not 0 < spec.get('budget_seconds', 0) <= 900
            or type(spec.get('witness_external_gpu_seconds')) is not int
            or spec['witness_external_gpu_seconds'] <= 0
            or spec['budget_seconds'] + spec['witness_external_gpu_seconds'] > spec.get('witness_gpu_ceiling_seconds', 0)
            or not 0 < spec.get('witness_gpu_ceiling_seconds', 0) <= 900
            or not 0 < spec.get('validator_seconds', 0) <= 120
            or not 0 < spec.get('witness_validator_cpu_seconds', 0) <= 120
            or not 0 < spec.get('witness_cpu_ceiling_seconds', 0) <= 1000):
        raise ValueError('unit-0 witness pilot spec differs')
    if not unit1 and (not _sha(spec.get('witness_ledger_sha256')) or not _sha(spec.get('witness_admission_sha256'))):
        raise ValueError('first pilot requires a pinned ledger')
    if unit1 and (not _sha(spec.get('witness_promotion_sha256')) or not _sha(spec.get('witness_ledger_sha256'))
                  or not _sha(spec.get('witness_prior_artifact_sha256'))):
        raise ValueError('unit-1 requires pinned prior artifact, promotion and ledger')
    caps = spec.get('witness_caps')
    names = {'parent_cpu', 'packet_cpu', 'generator_cpu', 'generator_wall', 'external_cpu'}
    if (not isinstance(caps, dict) or set(caps) != names
            or any(type(caps[x]) is not int or caps[x] <= 0 for x in names)
            or caps['generator_wall'] > 500 or caps['packet_cpu'] > 100 or caps['parent_cpu'] > 60
            or caps['generator_cpu'] > 500 or caps['external_cpu'] > 150
            or spec['budget_seconds'] + caps['external_cpu'] > spec['witness_cpu_ceiling_seconds']
            or caps['parent_cpu'] + caps['packet_cpu'] + caps['generator_cpu']
               + spec['witness_validator_cpu_seconds'] + caps['external_cpu'] > spec['witness_cpu_ceiling_seconds']):
        raise ValueError('pilot CPU or wall caps differ')


def _candidate_spec(spec: dict[str, Any], *, unit1: bool, reference_epoch: float | None = None) -> None:
    profile = 'w8-full-unit1-v1' if unit1 else 'w8-full-unit0-v1'
    phase = 'candidate1' if unit1 else 'candidate0'
    _check_deadline(spec, reference_epoch=reference_epoch)
    if (spec.get('w8_witness_profile') != profile or spec.get('hosts') != [spec.get('witness_cpu_proof_host')]
            or spec.get('witness_cpu_proof_host') not in ('gpu5', 'gpu6')
            or spec.get('entrypoint') != ['-m', 'experiments.tennis.w8_witness_job']
            or spec.get('arguments') != ['{phase}']
            or spec.get('scale') != {'phase': {'smoke': 'readiness', 'full': phase}}
            or spec.get('validator') != ['{python}', '-m', 'experiments.tennis.w8_witness_job', 'readiness-validate']
            or spec.get('gpu_models') != ['RTX 6000 Ada'] or spec.get('gpu_sharing') != {'mode': 'exclusive'}
            or spec.get('single_cpu_campaign') is not True or spec.get('cpu_threads') != 1
            or spec.get('contention', {}).get('policy') != 'stop'
            or spec.get('contention', {}).get('resume_attempts') != 0
            or not 0 < spec.get('smoke', {}).get('seconds', 0) <= 60
            or not 0 < spec.get('full', {}).get('seconds', 0) <= 1300
            or not 0 < spec.get('budget_seconds', 0) <= 1500
            or spec['budget_seconds'] < 60 + spec['full']['seconds'] + spec.get('validator_seconds', 0)
            or type(spec.get('witness_external_gpu_seconds')) is not int
            or spec['witness_external_gpu_seconds'] <= 0
            or spec['budget_seconds'] + spec['witness_external_gpu_seconds'] > spec.get('witness_gpu_ceiling_seconds', 0)
            or not 0 < spec.get('witness_gpu_ceiling_seconds', 0) <= 1500
            or not all(_sha(spec.get(x)) for x in ('witness_promotion_sha256', 'witness_ledger_sha256',
                                                    'witness_prior_artifact_sha256'))):
        raise ValueError('full producer spec differs')
    caps = spec.get('witness_caps')
    names = {'parent_cpu', 'packet_cpu', 'generator_cpu', 'generator_wall', 'external_cpu'}
    if (not isinstance(caps, dict) or set(caps) != names or any(type(caps[x]) is not int or caps[x] <= 0 for x in names)
            or caps['parent_cpu'] > 60 or caps['packet_cpu'] > 100 or caps['generator_cpu'] > 1200
            or caps['generator_wall'] > 1200 or caps['external_cpu'] > 200
            or spec['full']['seconds'] < 150 + caps['generator_wall']
            or spec['budget_seconds'] + caps['external_cpu'] > spec.get('witness_cpu_ceiling_seconds', 0)
            or not 0 < spec.get('witness_validator_cpu_seconds', 0) <= 120
            or caps['parent_cpu'] + caps['packet_cpu'] + caps['generator_cpu']
               + spec['witness_validator_cpu_seconds'] + caps['external_cpu'] > spec.get('witness_cpu_ceiling_seconds', 0)):
        raise ValueError('full producer CPU or wall cap differs')


def run_pilot(stage: Path, *, unit1: bool = False, candidate: bool = False) -> None:
    """One selected context or full candidate, each separately gated."""
    import os
    import resource
    import sys
    import time
    from experiments.tennis import w8_witness_profile as witness

    expected_stage = 'full' if candidate else 'smoke'
    if stage.name != expected_stage or os.environ.get('PS_STAGE') != expected_stage:
        raise ValueError('generation is not in its declared stage')
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('one-core Linux campaign required')
    started = time.monotonic()
    paths, spec, dispatch = load_inputs(stage)
    if candidate:
        _candidate_spec(spec, unit1=unit1)
    else:
        _pilot_spec(spec, unit1=unit1)
    if unit1 or candidate:
        _unit1_promotion(paths, spec, dispatch)
    else:
        check_first_pilot_stage(paths, spec)
    caps = spec['witness_caps']
    # Preserve the existing cost launcher's hard limit so each child can
    # independently install its reviewed soft CPU cap before isolation.
    resource.setrlimit(resource.RLIMIT_CPU, (caps['parent_cpu'], 3600))
    if (spec.get('clock_policy_profile') != host.w8_clock_profile.PROFILE
            or spec.get('clock_reference_audit_sha256') != host.CLOCK_AUDIT_SHA
            or spec.get('materialization_profile') != host.folder_profile.PROFILE
            or spec.get('folder_reference_audit_sha256') != host.FOLDER_AUDIT_SHA):
        raise ValueError('qualification requires frozen clock and materialization proofs')
    host.require_gpu_uuid_api(__import__('torch'))
    digests = host.StableDigests()
    prepared = host.prepare_receiver_inputs(paths, spec, packet_cpu=caps['packet_cpu'],
                                             materialize_profile=True, clock_profile=True,
                                             digests=digests, witness=True)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    unit = fixed.UNITS[1 if unit1 else 0]
    result = witness.execute_generation_units(prepared, dispatch, visible=visible,
        worker_cpu=caps['generator_cpu'], worker_wall=caps['generator_wall'],
        mode='generate' if candidate else 'context_qualification', units=(unit,))
    work, output = prepared['model_work'], result['outputs'][unit]
    retained = host.retained_path(stage)
    retained.mkdir()
    named = {'source-manifest.json': None, 'loader-identity.json': prepared['loader_receipt'],
             'protected-identity.json': prepared['protected_path'],
             'packet-config.json': prepared['decoder_work'] / 'worker-config.json',
             'generator-config.json': work / 'worker-config.json',
             'packet-namespace.json': prepared['decoder_work'] / 'namespace-receipt.json',
             'generator-namespace.json': work / 'namespace-receipt.json',
             'packet-namespace.json.started': prepared['decoder_work'] / 'namespace-receipt.json.started',
             'generator-namespace.json.started': work / 'namespace-receipt.json.started',
             'decoded-manifest.json': prepared['decoder_manifest'], 'decoded.pt': prepared['decoder_out'] / 'decoded.pt',
             'packet-unit.npz': prepared['decoder_out'] / 'publish' / 'units' / f'{unit}.npz',
             'packet-unit-other.npz': prepared['decoder_out'] / 'publish' / 'units' / f'{fixed.UNITS[0 if unit1 else 1]}.npz',
             'packet-cpu-only.json': prepared['decoder_out'] / 'cpu-only.json',
             'gpu-identity.json': output / 'gpu-identity.json', 'materialization.json': output / 'materialization.json',
             'sealed.json': output / 'sealed.json', f'{unit}@match5.npz': output / f'{unit}@match5.npz',
             f'{unit}@match5.json': output / f'{unit}@match5.json'}
    with (retained / 'source-manifest.json').open('x') as stream:
        json.dump(prepared['code_manifest'], stream, sort_keys=True)
    for name, path in named.items():
        if path is not None:
            host.copy_new(path, retained / name)
    import tarfile
    archive = retained / 'witness-context-artifacts.tar'
    if sum((retained / name).stat().st_size for name in named) > 512 * 1024 * 1024:
        raise ValueError('pilot retained data exceeds fixed bound')
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw, mode='w|') as tar:
            for name in sorted(named):
                tar.add(retained / name, arcname=name, recursive=False)
        raw.flush()
        os.fsync(raw.fileno())
    summary = {'scope': 'candidate' if candidate else 'context_qualification_only', 'unit': unit, 'quality_claim': False,
               'automatic_promotion': False, 'expected': result['expected'][unit],
               'validation': result['validation'][unit], 'generation_usage': result['usage'][unit],
               'packet_usage': prepared['packet_usage'], 'cpu_gate_sha256': fixed.file_sha(paths['cpu_proof']),
               **({'promotion_sha256': fixed.file_sha(paths['promotion']),
                   'ledger_sha256': fixed.file_sha(paths['ledger']),
                   'prior_artifact_sha256': spec['witness_prior_artifact_sha256']} if unit1 or candidate else {}),
               'source_manifest_sha256': fixed.file_sha(retained / 'source-manifest.json'),
               'retained': str(retained), 'archive_sha256': fixed.file_sha(archive),
               'archive_bytes': archive.stat().st_size,
               'members': {name: fixed.file_sha(retained / name) for name in named},
               'elapsed_seconds': time.monotonic() - started,
               'note': ('240-frame producer only; no receiver or quality claim' if candidate else
                        '24-frame context; no full-run prefix, quality or second-unit claim')}
    if set(summary['members']) != set(named) or any(not _sha(x) for x in summary['members'].values()):
        raise ValueError('pilot retained artifacts incomplete')
    publish = prepared['scratch'] / 'publish'
    publish.mkdir()
    with (publish / 'summary.json').open('x') as stream:
        json.dump(summary, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())




def check_stage_evidence(stage: Path, dispatch: dict[str, Any], *, terminal: bool) -> None:
    """Reject contaminated timing and require canonical terminal state for post-stage reuse."""
    execution = json.loads((stage / 'execution.json').read_text())
    if (execution.get('exit_code') != 0 or execution.get('timing_contaminated') is not False
            or execution.get('contention') != [] or execution.get('gpu_sharing') != {'mode': 'exclusive'}):
        raise ValueError('stage execution is incomplete or GPU timing contaminated')
    if terminal:
        from experiments.jobs.inbox import digest

        state = json.loads((stage.parent / 'state.json').read_text())
        gate = json.loads((stage.parent / 'gate.json').read_text())
        spec = json.loads((stage.parent / 'spec.json').read_text())
        if (state.get('status') != 'complete' or state.get('job_id') != stage.parent.name
                or gate.get('passed') is not True or gate.get('inputs') != spec.get('inputs')
                or gate.get('spec_sha256') != digest(spec)
                or gate.get('source_sha256') != dispatch.get('code', {}).get('source_sha256')):
            raise ValueError('canonical terminal completion or smoke gate differs')

def require_original_host(dispatch: dict[str, Any]) -> None:
    """PID/start-tick absence is meaningful only on the producer's host."""
    import socket

    recorded = (dispatch.get('environment') or {}).get('host')
    if not isinstance(recorded, str) or not recorded or socket.getfqdn().lower() != recorded.lower():
        raise ValueError('offline audit must run on the original fleet host')

def validate_pilot(stage: Path, target: Path) -> None:
    """Independently reopen the sealed context or full producer artifact."""
    import os
    import resource
    import tarfile
    from experiments.tennis import w8_receiver_generate as generation
    from experiments.tennis import w8_witness_profile as witness

    try:
        paths, spec, dispatch = load_released_inputs(stage)
        check_stage_evidence(stage, dispatch, terminal=stage.name == 'full')
        profile = spec.get('w8_witness_profile')
        candidate = profile in ('w8-full-unit0-v1', 'w8-full-unit1-v1')
        unit1 = profile in ('w8-context-unit1-v1', 'w8-full-unit1-v1')
        if candidate:
            _candidate_spec(spec, unit1=unit1, reference_epoch=dispatch['started'])
        else:
            _pilot_spec(spec, unit1=unit1, reference_epoch=dispatch['started'])
        if unit1 or candidate:
            _unit1_promotion(paths, spec, dispatch, historical=True)
        else:
            check_first_pilot_stage(paths, spec)
        resource.setrlimit(resource.RLIMIT_CPU, (spec['witness_validator_cpu_seconds'],
                                                 spec['witness_validator_cpu_seconds']))
        publication = stage / 'published.tar'
        staging = json.loads((stage / 'staging.json').read_text())
        if fixed.file_sha(publication) != staging['published']['sha256']:
            raise ValueError('published pilot archive SHA differs')
        with tarfile.open(publication) as tar:
            files = [entry for entry in tar.getmembers() if entry.isfile()]
            if len(files) != 1 or files[0].name != 'publish/summary.json' or files[0].size > 2_000_000:
                raise ValueError('unexpected pilot publication')
            stream = tar.extractfile(files[0])
            if stream is None:
                raise ValueError('pilot summary missing')
            summary = json.load(stream)
        unit = fixed.UNITS[1 if unit1 else 0]
        expected_scope = 'candidate' if candidate else 'context_qualification_only'
        if (summary.get('scope') != expected_scope or summary.get('unit') != unit
                or summary.get('quality_claim') is not False or summary.get('automatic_promotion') is not False
                or summary.get('cpu_gate_sha256') != spec['witness_cpu_proof_sha256']):
            raise ValueError('pilot scope or CPU proof differs')
        if (unit1 or candidate) and (summary.get('promotion_sha256') != spec['witness_promotion_sha256']
                      or summary.get('ledger_sha256') != spec['witness_ledger_sha256']
                      or summary.get('prior_artifact_sha256') != spec['witness_prior_artifact_sha256']):
            raise ValueError('unit-1 promotion references differ')
        import math
        caps = spec['witness_caps']
        for key, cpu_cap, wall_cap in (('packet_usage', caps['packet_cpu'], 130),
                                       ('generation_usage', caps['generator_cpu'], caps['generator_wall'])):
            usage = summary.get(key, {})
            if (usage.get('status') != 0 or usage.get('timed_out') is not False
                    or usage.get('canaries_preserved') is not True
                    or any(isinstance(usage.get(name), bool) or not isinstance(usage.get(name), (int, float))
                           or not math.isfinite(usage[name]) or usage[name] < 0 for name in ('wall_seconds', 'cpu_seconds'))
                    or usage['wall_seconds'] > wall_cap or usage['cpu_seconds'] > cpu_cap):
                raise ValueError('pilot worker bound or canary differs')
        retained = host.retained_path(stage)
        if summary.get('retained') != str(retained) or retained.is_symlink():
            raise ValueError('pilot retained path differs')
        members = summary.get('members')
        required = {'source-manifest.json', 'loader-identity.json', 'protected-identity.json',
                    'packet-config.json', 'generator-config.json', 'packet-namespace.json', 'generator-namespace.json',
                    'packet-namespace.json.started', 'generator-namespace.json.started',
                    'decoded-manifest.json', 'decoded.pt', 'packet-unit.npz', 'packet-unit-other.npz',
                    'packet-cpu-only.json', 'gpu-identity.json', 'materialization.json', 'sealed.json',
                    f'{unit}@match5.npz', f'{unit}@match5.json'}
        if (not isinstance(members, dict) or set(members) != required
                or any(not _sha(v) or fixed.file_sha(retained / name) != v for name, v in members.items())):
            raise ValueError('pilot retained member identities differ')
        archive = retained / 'witness-context-artifacts.tar'
        if (archive.is_symlink() or archive.stat().st_size != summary['archive_bytes']
                or fixed.file_sha(archive) != summary['archive_sha256']):
            raise ValueError('pilot durable archive differs')
        with tarfile.open(archive) as tar:
            entries = tar.getmembers()
            if (len(entries) != len(required) or {entry.name for entry in entries} != required
                    or any(not entry.isfile() for entry in entries)):
                raise ValueError('pilot durable archive membership differs')
            import hashlib
            for entry in entries:
                member = tar.extractfile(entry)
                if member is None:
                    raise ValueError('pilot archive member unavailable')
                digest = hashlib.sha256()
                while chunk := member.read(1024 * 1024):
                    digest.update(chunk)
                if digest.hexdigest() != members[entry.name]:
                    raise ValueError('pilot durable archive member differs')
        manifest = json.loads((retained / 'source-manifest.json').read_text())
        if fixed.file_sha(retained / 'source-manifest.json') != summary['source_manifest_sha256']:
            raise ValueError('pilot source manifest differs')
        config = json.loads((retained / 'generator-config.json').read_text())
        source_root = Path(config['allow_paths'][0]).resolve()
        scratch = Path(staging['scratch']).resolve()
        if source_root != scratch / 'receiver-source':
            raise ValueError('pilot source staging path differs')
        files = manifest.get('files')
        if (not isinstance(files, dict) or not files
                or any(not name.endswith('.py') or not name.startswith(('experiments/', 'src/', 'tools/')) for name in files)):
            raise ValueError('pilot source manifest membership differs')
        import hashlib
        if hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != manifest.get('manifest_sha256'):
            raise ValueError('pilot source manifest digest differs')
        source_archive = stage.parent / 'source.tar'
        if fixed.file_sha(source_archive) != dispatch['code']['snapshot_sha256']:
            raise ValueError('pilot fleet source snapshot differs')
        with tarfile.open(source_archive) as tar:
            for name, digest in files.items():
                member = tar.getmember(name)
                stream = tar.extractfile(member) if member.isfile() else None
                if stream is None or hashlib.sha256(stream.read()).hexdigest() != digest:
                    raise ValueError('pilot source member differs')
        host.check_preflight_bundle(json.loads(paths['preflight'].read_text()), paths['identity'], witness_metadata=True)
        fingerprints = host.read_oracle(paths['oracle'])
        decoded = json.loads((retained / 'decoded-manifest.json').read_text())
        packet_config = json.loads((retained / 'packet-config.json').read_text())
        preflight = json.loads(paths['preflight'].read_text())
        loader_path = retained / 'loader-identity.json'
        host.packet_audit.check_grants(packet_config, preflight['runtimes']['main'])
        host.packet_audit.check_isolation(packet_config, decoded['isolation'], json.loads(loader_path.read_text()),
                                          manifest, fixed.file_sha(loader_path))
        from experiments.tennis import w8_protected_metadata
        protected_metadata = json.loads((retained / 'protected-identity.json').read_text())
        historical = w8_protected_metadata.read_historical(paths['identity'], preflight['identity_sha256'])
        w8_protected_metadata.validate(protected_metadata, historical, check_current=True)
        if protected_metadata['historical_receipt_sha256'] != preflight['identity_sha256']:
            raise ValueError('historical protected receipt differs')
        cpu_only = json.loads((retained / 'packet-cpu-only.json').read_text())
        if (cpu_only.get('cuda_visible_devices') != '' or cpu_only.get('cuda_initialized_after_decode') is not False
                or cpu_only.get('torch_version') != preflight['receipts']['main']['torch']
                or cpu_only.get('pre_serialization', {}).get('state_tensors') != 480
                or cpu_only.get('pre_serialization', {}).get('restored_tensors') != 480):
            raise ValueError('packet CPU-only qualification differs')
        if (decoded.get('packet_sha256') != fixed.WINDOW_SHA or decoded.get('profile_sha256') != fixed.PROFILE_SHA
                or decoded.get('match_tensors') != fingerprints or decoded.get('unit_sha256') != generation.ORACLE_UNIT_SHA
                or fixed.file_sha(retained / 'packet-unit.npz') != generation.ORACLE_UNIT_SHA[unit]
                or fixed.file_sha(retained / 'packet-unit-other.npz') != generation.ORACLE_UNIT_SHA[fixed.UNITS[0 if unit1 else 1]]
                or generation.verify_serialized_match(retained / 'decoded.pt', fingerprints) != decoded['checkpoint_sha256']):
            raise ValueError('pilot packet and decoded checkpoint join differs')
        expected = {'unit': unit, 'packet_sha256': fixed.WINDOW_SHA, 'profile_sha256': fixed.PROFILE_SHA,
                    'tensor_fingerprints': fingerprints, 'files': {k: v['sha256'] for k, v in config['files'].items()}}
        if (expected != summary['expected'] or expected['files']['checkpoint'] != decoded['checkpoint_sha256']
                or expected['files']['decoder_manifest'] != fixed.file_sha(retained / 'decoded-manifest.json')):
            raise ValueError('pilot worker input identities differ')
        validation = (generation.validate_sealed(retained, expected) if candidate
                      else generation.validate_context_qualification(retained, expected))
        if validation != summary['validation']:
            raise ValueError('pilot context validation differs')
        seal = json.loads((retained / 'sealed.json').read_text())
        witness.check_worker_isolation(config, seal['isolation'],
            runtime=preflight['runtimes']['jcv'], source_manifest=manifest,
            loader=json.loads(loader_path.read_text()), loader_sha=fixed.file_sha(loader_path))
        gpu = json.loads((retained / 'gpu-identity.json').read_text())
        if (gpu.get('sealed_sha256') != fixed.file_sha(retained / 'sealed.json')
                or gpu.get('config_sha256') != fixed.file_sha(retained / 'generator-config.json')
                or gpu.get('materialization_profile') != host.folder_profile.PROFILE
                or gpu.get('materialization_sha256') != fixed.file_sha(retained / 'materialization.json')):
            raise ValueError('pilot physical GPU identity receipt differs')
        claimed = dispatch['environment']['gpu']['uuid']
        host.check_gpu_binding(gpu, seal, claimed, dispatch, manifest['files'])
        require_original_host(dispatch)
        group = json.loads((stage.parent / 'run/status.json').read_text())['process_group_id']
        outside = os.readlink('/proc/self/ns/pid')
        for name, module, usage_key in (('packet-namespace.json', 'experiments.tennis.w8_strict_receiver_worker', 'packet_usage'),
                                       ('generator-namespace.json', 'experiments.tennis.w8_strict_host_model_worker', 'generation_usage')):
            ns = witness.namespace_completion(retained / name, expected_pgid=group,
                                               expected_outside=outside, expected_target=module)
            if (summary[usage_key].get('namespace_receipt_sha256') != fixed.file_sha(retained / name)
                    or summary[usage_key]['cpu_seconds'] + 0.05 < ns['children_cpu_seconds']):
                raise ValueError('pilot namespace CPU or receipt join differs')
        if summary['validation']['scope'] != expected_scope:
            raise ValueError('producer output scope differs')
        with target.open('x') as validation_stream:
            json.dump({'passed': True, 'scope': f'{unit} {expected_scope} producer only', 'automatic_promotion': False,
                       'checks': {'source_packet_gpu_namespace_and_context': True}}, validation_stream, sort_keys=True)
    except Exception as exc:
        with target.open('x') as validation_stream:
            json.dump({'passed': False, 'scope': 'unit-0 context qualification only', 'error': str(exc)}, validation_stream)
        raise


def make_pilot_spec(*, deadline: str, proof_local: Path, proof_remote: str,
                    ledger_local: Path | None = None, ledger_remote: str | None = None,
                    admission_local: Path | None = None, admission_remote: str | None = None,
                    _promoted_base: bool = False) -> dict[str, Any]:
    """Prepare a review-only spec after a real, independently reviewed CPU gate exists."""
    digest = fixed.file_sha(proof_local)
    gate = json.loads(proof_local.read_text())
    if (gate.get('schema') != 1 or gate.get('scope') != CPU_SCOPE or gate.get('passed') is not True
            or not _sha(digest) or not Path(proof_remote).is_absolute()
            or not Path(proof_remote).is_relative_to(Path('/home/itec/emanuele/Datasets/pointstream-data'))):
        raise ValueError('actual root-reviewed proof gate and shared path required')
    check_cpu_gate(gate, gate_sha=digest, expected_sha=digest,
                   source={'source_sha256': gate.get('source_sha256'), 'snapshot_sha256': gate.get('snapshot_sha256'),
                           'git_head': gate.get('revision')},
                   environment_sha=gate.get('environment_sha256'), host_name=gate.get('host'))
    spec = host.make_materialized_spec(deadline)
    spec.pop('w8_paired_cost_profile', None)
    spec['entrypoint'] = ['-m', 'experiments.tennis.w8_witness_job']
    spec['arguments'] = ['{phase}']
    spec['scale'] = {'phase': {'smoke': 'pilot', 'full': 'noop'}}
    spec['smoke'] = {'seconds': 580, 'representative_basis':
        'One actual original-recipe 30-step 24-frame context; no candidate or automatic promotion.'}
    spec['full'] = {'seconds': 5}
    spec['budget_seconds'] = 700
    spec['witness_external_gpu_seconds'] = 200
    spec['witness_gpu_ceiling_seconds'] = 900
    spec['validator_seconds'] = 60
    spec['witness_validator_cpu_seconds'] = 60
    spec['witness_cpu_ceiling_seconds'] = 1000
    spec['witness_caps'] = {'parent_cpu': 30, 'packet_cpu': 100, 'generator_cpu': 300,
                            'generator_wall': 310, 'external_cpu': 150}
    spec['w8_witness_profile'] = 'w8-context-pilot-v1'
    spec['witness_cpu_proof_host'] = gate['host']
    spec['hosts'] = [gate['host']]
    spec['witness_cpu_proof_sha256'] = digest
    spec['witness_bootstrap_sha256'] = gate['bootstrap_sha256']
    spec['clock_policy_profile'] = host.w8_clock_profile.PROFILE
    spec['clock_reference_audit_sha256'] = host.CLOCK_AUDIT_SHA
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.w8_witness_job', 'validate']
    spec['gpu_models'] = ['RTX 6000 Ada']
    spec['staged_inputs'].append({'name': 'cpu_proof', 'path': proof_remote, 'sha256': digest, 'extract': False})
    spec['inputs'].append({'path': proof_remote, 'sha256': digest})
    if not _promoted_base:
        if ledger_local is None or ledger_remote is None or admission_local is None or admission_remote is None:
            raise ValueError('first pilot requires the current ledger and reviewed attempt receipt')
        for value in (ledger_remote, admission_remote):
            remote = Path(value)
            if (not remote.is_absolute() or '..' in remote.parts
                    or not remote.is_relative_to(Path('/home/itec/emanuele/Datasets/pointstream-data'))):
                raise ValueError('first-pilot inputs require registered data paths')
        ledger_sha, admission_sha = fixed.file_sha(ledger_local), fixed.file_sha(admission_local)
        receipt = json.loads(admission_local.read_text())
        check_first_pilot_ledger(json.loads(ledger_local.read_text()), receipt, ledger_sha=ledger_sha, cpu_proof_sha=digest)
        spec['witness_ledger_sha256'], spec['witness_admission_sha256'] = ledger_sha, admission_sha
        gpu_cap, cpu_cap = receipt['gpu_ceiling_seconds'], receipt['cpu_ceiling_seconds']
        spec['witness_gpu_ceiling_seconds'], spec['witness_cpu_ceiling_seconds'] = gpu_cap, cpu_cap
        spec['witness_external_gpu_seconds'] = 50
        spec['budget_seconds'] = min(gpu_cap - 50, cpu_cap - 150)
        spec['smoke']['seconds'] = min(580, spec['budget_seconds'] - 65)
        spec['witness_caps']['generator_wall'] = min(310, spec['smoke']['seconds'] - 145)
        spec['witness_caps']['generator_cpu'] = min(300, cpu_cap - 340)
        for name, path, sha in (('ledger', ledger_remote, ledger_sha), ('admission', admission_remote, admission_sha)):
            spec['staged_inputs'].append({'name': name, 'path': path, 'sha256': sha, 'extract': False})
            spec['inputs'].append({'path': path, 'sha256': sha})
        _pilot_spec(spec)
    from experiments.jobs.inbox import validate_spec
    validate_spec(spec)
    return spec


def make_unit1_spec(*, deadline: str, proof_local: Path, proof_remote: str,
                    promotion_local: Path, promotion_remote: str, ledger_local: Path,
                    ledger_remote: str, prior_artifact_sha: str, _for_candidate: bool = False) -> dict[str, Any]:
    """Build only after independent pilot audit and a root-pinned forecast exist."""
    spec = make_pilot_spec(deadline=deadline, proof_local=proof_local, proof_remote=proof_remote, _promoted_base=True)
    promotion_sha, ledger_sha = fixed.file_sha(promotion_local), fixed.file_sha(ledger_local)
    gate = json.loads(promotion_local.read_text())
    for remote in (promotion_remote, ledger_remote):
        if not Path(remote).is_absolute() or not Path(remote).is_relative_to(Path('/home/itec/emanuele/Datasets/pointstream-data')):
            raise ValueError('promotion and ledger must have shared-data paths')
    ledger = json.loads(ledger_local.read_text())
    gpu_remaining, cpu_remaining = promotion_ledger_remaining(ledger)
    check_promotion(gate, gate_sha=promotion_sha, expected_sha=promotion_sha,
                    source={'source_sha256': gate.get('source_sha256'), 'snapshot_sha256': gate.get('snapshot_sha256')},
                    cpu_gate_sha=spec['witness_cpu_proof_sha256'], prior_artifact_sha=prior_artifact_sha,
                    ledger_sha=ledger_sha, gpu_remaining=gpu_remaining, cpu_remaining=cpu_remaining)
    spec['w8_witness_profile'] = 'w8-context-unit1-v1'
    spec['scale'] = {'phase': {'smoke': 'unit1', 'full': 'noop'}}
    spec['witness_promotion_sha256'] = promotion_sha
    spec['witness_ledger_sha256'] = ledger_sha
    spec['witness_prior_artifact_sha256'] = prior_artifact_sha
    spec['witness_gpu_remaining_seconds'] = gate['gpu_remaining_seconds']
    spec['witness_cpu_remaining_seconds'] = gate['cpu_remaining_seconds']
    if not _for_candidate:
        gpu_cap = gate.get('qualification_next_gpu_ceiling_seconds')
        cpu_cap = gate.get('qualification_next_cpu_ceiling_seconds')
        if (type(gpu_cap) is not int or type(cpu_cap) is not int or not 250 <= gpu_cap <= 900
                or not 341 <= cpu_cap <= 1000):
            raise ValueError('explicit bounded qualification stage ceilings required')
        spec['witness_gpu_ceiling_seconds'], spec['witness_cpu_ceiling_seconds'] = gpu_cap, cpu_cap
        spec['witness_external_gpu_seconds'] = 50
        spec['budget_seconds'] = min(gpu_cap - 50, cpu_cap - 150)
        spec['smoke']['seconds'] = min(580, spec['budget_seconds'] - 65)
        spec['witness_caps']['generator_wall'] = min(310, spec['smoke']['seconds'] - 145)
        spec['witness_caps']['generator_cpu'] = min(300, cpu_cap - 340)
        if spec['witness_caps']['generator_wall'] < 11:
            raise ValueError('qualification stage wall ceiling too small')
        check_qualification_subcap(ledger, gate, gpu_cap=spec['witness_gpu_ceiling_seconds'],
                                   cpu_cap=spec['witness_cpu_ceiling_seconds'])
    if not _for_candidate and (gate['remaining_components']['unit1_context']['gpu_seconds'] < spec['witness_gpu_ceiling_seconds']
            or gate['remaining_components']['unit1_context']['cpu_seconds'] < spec['witness_cpu_ceiling_seconds']):
        raise ValueError('forecast does not reserve unit-1 inclusive caps')
    for name, path, digest in (('promotion', promotion_remote, promotion_sha), ('ledger', ledger_remote, ledger_sha)):
        spec['staged_inputs'].append({'name': name, 'path': path, 'sha256': digest, 'extract': False})
        spec['inputs'].append({'path': path, 'sha256': digest})
    _pilot_spec(spec, unit1=True)
    from experiments.jobs.inbox import validate_spec
    validate_spec(spec)
    return spec


def make_candidate_spec(*, deadline: str, proof_local: Path, proof_remote: str,
                        promotion_local: Path, promotion_remote: str, ledger_local: Path,
                        ledger_remote: str, qualification_artifact_sha: str, unit1: bool) -> dict[str, Any]:
    """Build a separately reviewed full producer after qualification and cost audit."""
    spec = make_unit1_spec(deadline=deadline, proof_local=proof_local, proof_remote=proof_remote,
        promotion_local=promotion_local, promotion_remote=promotion_remote, ledger_local=ledger_local,
        ledger_remote=ledger_remote, prior_artifact_sha=qualification_artifact_sha, _for_candidate=True)
    spec['w8_witness_profile'] = 'w8-full-unit1-v1' if unit1 else 'w8-full-unit0-v1'
    spec['scale'] = {'phase': {'smoke': 'readiness', 'full': 'candidate1' if unit1 else 'candidate0'}}
    spec['smoke'] = {'seconds': 60, 'representative_basis':
                     'Bounded exact-source, CPU-proof and reviewed remaining-budget readiness; no GPU inference.'}
    spec['full'] = {'seconds': 1300}
    spec['budget_seconds'] = 1480
    spec['witness_external_gpu_seconds'] = 20
    spec['witness_gpu_ceiling_seconds'] = 1500
    spec['validator_seconds'] = 60
    spec['witness_validator_cpu_seconds'] = 60
    spec['witness_cpu_ceiling_seconds'] = 1630
    spec['witness_caps'] = {'parent_cpu': 30, 'packet_cpu': 100, 'generator_cpu': 1100,
                            'generator_wall': 1100, 'external_cpu': 150}
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.w8_witness_job', 'readiness-validate']
    gate = json.loads(promotion_local.read_text())
    limits = gate.get('candidate_next_limits')
    keys = {'full_seconds', 'campaign_seconds', 'gpu_ceiling_seconds', 'cpu_ceiling_seconds',
            'generator_cpu', 'generator_wall'}
    if not isinstance(limits, dict) or set(limits) != keys or any(type(limits[k]) is not int or limits[k] <= 0 for k in keys):
        raise ValueError('reviewed producer stage limits required')
    spec['full']['seconds'] = limits['full_seconds']
    spec['budget_seconds'] = limits['campaign_seconds']
    spec['witness_gpu_ceiling_seconds'] = limits['gpu_ceiling_seconds']
    spec['witness_cpu_ceiling_seconds'] = limits['cpu_ceiling_seconds']
    spec['witness_caps']['generator_cpu'] = limits['generator_cpu']
    spec['witness_caps']['generator_wall'] = limits['generator_wall']
    _candidate_spec(spec, unit1=unit1)
    gate = json.loads(promotion_local.read_text())
    completed = gate.get('completed_artifact_sha256', {})
    target = 'unit1_full' if unit1 else 'unit0_full'
    if ('unit1_context' not in completed or target in completed or not _sha(gate.get('qualification_receipt_sha256'))):
        raise ValueError('qualification incomplete or producer already completed')
    component = gate['remaining_components'][target]
    if (component['gpu_seconds'] < spec['witness_gpu_ceiling_seconds']
            or component['cpu_seconds'] < spec['witness_cpu_ceiling_seconds']):
        raise ValueError('forecast does not reserve full producer inclusive caps')
    from experiments.jobs.inbox import validate_spec
    validate_spec(spec)
    return spec


def candidate_readiness(stage: Path, *, released: bool = False) -> dict[str, Any]:
    """Validate root-pinned source, completed qualification and remaining budget."""
    paths, spec, dispatch = load_released_inputs(stage) if released else load_inputs(stage)
    unit1 = spec.get('w8_witness_profile') == 'w8-full-unit1-v1'
    reference = dispatch['started'] if released else None
    _candidate_spec(spec, unit1=unit1, reference_epoch=reference)
    _unit1_promotion(paths, spec, dispatch, historical=released)
    return {'scope': 'full producer metadata readiness only', 'unit': fixed.UNITS[1 if unit1 else 0],
            'source_sha256': dispatch['code']['source_sha256'],
            'snapshot_sha256': dispatch['code']['snapshot_sha256'],
            'cpu_proof_sha256': spec['witness_cpu_proof_sha256'],
            'promotion_sha256': spec['witness_promotion_sha256'],
            'ledger_sha256': spec['witness_ledger_sha256'],
            'qualification_artifact_sha256': spec['witness_prior_artifact_sha256'],
            'model_execution': False}


def run_readiness(stage: Path) -> None:
    import os

    if stage.name != 'smoke' or os.environ.get('PS_STAGE') != 'smoke':
        raise ValueError('readiness must be the smoke stage')
    record = candidate_readiness(stage)
    output = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
    output.mkdir()
    with (output / 'readiness.json').open('x') as stream:
        json.dump(record, stream, sort_keys=True)


def validate_readiness(stage: Path, target: Path) -> None:
    """Reopen actual published readiness and recheck the bound promotion."""
    import tarfile

    try:
        expected = candidate_readiness(stage, released=True)
        archive = stage / 'published.tar'
        staging = json.loads((stage / 'staging.json').read_text())
        if fixed.file_sha(archive) != staging['published']['sha256']:
            raise ValueError('readiness publication differs')
        with tarfile.open(archive) as tar:
            files = [entry for entry in tar.getmembers() if entry.isfile()]
            if len(files) != 1 or files[0].name != 'publish/readiness.json' or files[0].size > 100_000:
                raise ValueError('readiness publication membership differs')
            stream = tar.extractfile(files[0])
            if stream is None or json.load(stream) != expected:
                raise ValueError('readiness evidence differs')
        with target.open('x') as output:
            json.dump({'passed': True, 'scope': expected['scope'],
                       'checks': {'source_cpu_proof_and_remaining_budget': True}}, output)
    except Exception as exc:
        with target.open('x') as output:
            json.dump({'passed': False, 'scope': 'full producer readiness', 'error': str(exc)}, output)
        raise


def _finish_spec(spec: dict[str, Any], *, qualification: bool, reference_epoch: float | None = None) -> None:
    """Reject altered finish profiles, budgets and dispatch before running any work."""
    _check_deadline(spec, reference_epoch=reference_epoch)
    profile = 'w8-context-qualification-v1' if qualification else 'w8-full-finish-v1'
    if qualification:
        gpu = spec.get('witness_gpu_ceiling_seconds')
        cpu = spec.get('witness_cpu_ceiling_seconds')
        caps = spec.get('witness_caps', {})
        if (type(gpu) is not int or type(cpu) is not int or not 350 <= gpu <= 900
                or not 350 <= cpu <= 1000 or not isinstance(caps, dict)
                or set(caps) != {'parent_cpu', 'packet_cpu', 'native_cpu', 'native_wall', 'worker_cpu', 'worker_wall', 'external_cpu'}
                or caps.get('parent_cpu') != 30 or caps.get('packet_cpu') != 100
                or caps.get('native_cpu') != 30 or caps.get('native_wall') != 40 or caps.get('external_cpu') != 100
                or type(caps.get('worker_cpu')) is not int or not 1 <= caps['worker_cpu'] <= 100
                or type(caps.get('worker_wall')) is not int or not 11 <= caps['worker_wall'] <= 110
                or spec.get('budget_seconds') != gpu - 30 or gpu - 30 + 100 > cpu
                or spec.get('full', {}).get('seconds') != spec['budget_seconds'] - 110
                or 195 + 3 * caps['worker_wall'] > spec['full']['seconds']):
            raise ValueError('qualification finish limits do not fit inclusive ceilings')
        full_seconds, campaign, ceiling = spec['full']['seconds'], spec['budget_seconds'], gpu
    else:
        full_seconds, campaign = spec.get('full', {}).get('seconds'), spec.get('budget_seconds')
        full_ceiling, full_cpu = spec.get('witness_gpu_ceiling_seconds'), spec.get('witness_cpu_ceiling_seconds')
        caps = spec.get('witness_caps', {})
        if (type(full_seconds) is not int or type(campaign) is not int or type(full_ceiling) is not int
                or type(full_cpu) is not int or not isinstance(caps, dict) or set(caps) != {'parent_cpu', 'packet_cpu', 'native_cpu',
                     'native_wall', 'worker_cpu', 'worker_wall', 'external_cpu'}
                or caps.get('parent_cpu') != 30 or caps.get('packet_cpu') != 100 or caps.get('native_cpu') != 30
                or caps.get('native_wall') != 40 or caps.get('external_cpu') != 100
                or type(caps.get('worker_cpu')) is not int or not 1 <= caps['worker_cpu'] <= 450
                or type(caps.get('worker_wall')) is not int or not 11 <= caps['worker_wall'] <= 460
                or not 1 <= full_seconds <= 1700 or not 1 <= campaign <= 1850
                or campaign < full_seconds + 110 or full_ceiling < campaign + 50 or full_ceiling > 1900
                or full_cpu < campaign + 100 or full_cpu > 1950 or full_seconds < 195 + 3 * caps['worker_wall']):
            raise ValueError('full finish limits do not fit inclusive ceilings')
        ceiling = full_ceiling
    if (spec.get('w8_witness_profile') != profile or spec.get('entrypoint') != ['-m', 'experiments.tennis.w8_witness_job']
            or spec.get('arguments') != ['{phase}']
            or spec.get('scale') != {'phase': {'smoke': 'finish-readiness', 'full': 'finish-context' if qualification else 'finish-candidate'}}
            or spec.get('smoke', {}).get('seconds') != 60 or spec.get('full', {}).get('seconds') != full_seconds
            or spec.get('budget_seconds') != campaign or spec.get('validator_seconds') != 50
            or spec.get('witness_gpu_ceiling_seconds') != ceiling
            or spec.get('witness_external_gpu_seconds') != (30 if qualification else 50)
            or spec.get('witness_caps') != caps or spec.get('witness_cpu_proof_host') not in ('gpu5', 'gpu6')
            or spec.get('hosts') != [spec['witness_cpu_proof_host']] or spec.get('gpu_models') != ['RTX 6000 Ada']
            or spec.get('device') != 'gpu' or spec.get('gpu_sharing') != {'mode': 'exclusive'}
            or spec.get('single_cpu_campaign') is not True or spec.get('cpu_threads') != 1
            or spec.get('contention') != {'policy': 'stop', 'resume_attempts': 0}
            or spec.get('validator') != ['{python}', '-m', 'experiments.tennis.w8_witness_job', 'finish-readiness-validate']):
        raise ValueError('finish fixed dispatch or inclusive bounds differ')
    if any(not _sha(spec.get('witness_' + name + '_sha256')) for name in ('producer_audit', 'producer0', 'producer1')):
        raise ValueError('exact audited producer artifacts required')


def make_finish_spec(*, deadline: str, proof_local: Path, proof_remote: str, promotion_local: Path,
                     promotion_remote: str, ledger_local: Path, ledger_remote: str,
                     producer_audit_local: Path, producer_audit_remote: str, producer0_local: Path,
                     producer0_remote: str, producer1_local: Path, producer1_remote: str,
                     sam_remote: str, native_inventory_remote: str, qualification: bool) -> dict[str, Any]:
    """Prepare a review-only stage, requiring a separately pinned producer audit."""
    from experiments.tennis.w8_receiver_finish import SAM_SHA
    audit_sha = fixed.file_sha(producer_audit_local)
    spec = make_unit1_spec(deadline=deadline, proof_local=proof_local, proof_remote=proof_remote,
        promotion_local=promotion_local, promotion_remote=promotion_remote, ledger_local=ledger_local,
        ledger_remote=ledger_remote, prior_artifact_sha=audit_sha, _for_candidate=True)
    spec['w8_witness_profile'] = 'w8-context-qualification-v1' if qualification else 'w8-full-finish-v1'
    spec['scale'] = {'phase': {'smoke': 'finish-readiness', 'full': 'finish-context' if qualification else 'finish-candidate'}}
    spec['smoke'] = {'seconds': 60, 'representative_basis': 'Exact reviewed source, producer and remaining-budget readiness.'}
    spec['full'] = {'seconds': 500 if qualification else 1700}
    spec['budget_seconds'] = 570 if qualification else 1850
    spec['validator_seconds'] = 50
    spec['witness_gpu_ceiling_seconds'] = 600 if qualification else 1900
    spec['witness_cpu_ceiling_seconds'] = 600 if qualification else 1950
    spec['witness_external_gpu_seconds'] = 30 if qualification else 50
    spec['witness_caps'] = {'parent_cpu': 30, 'packet_cpu': 100, 'native_cpu': 30, 'native_wall': 40,
                            'worker_cpu': 100 if qualification else 450,
                            'worker_wall': 110 if qualification else 460, 'external_cpu': 100}
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.w8_witness_job', 'finish-readiness-validate']
    if qualification:
        gate = json.loads(promotion_local.read_text())
        gpu = gate.get('qualification_next_gpu_ceiling_seconds')
        cpu = gate.get('qualification_next_cpu_ceiling_seconds')
        if type(gpu) is not int or type(cpu) is not int or not 350 <= gpu <= 900 or not 350 <= cpu <= 1000:
            raise ValueError('explicit qualification finish ceilings required')
        spec['witness_gpu_ceiling_seconds'], spec['witness_cpu_ceiling_seconds'] = gpu, cpu
        spec['budget_seconds'] = gpu - 30
        spec['full']['seconds'] = spec['budget_seconds'] - 110
        spec['witness_caps']['worker_wall'] = min(110, (spec['full']['seconds'] - 195) // 3)
        spec['witness_caps']['worker_cpu'] = min(100, spec['witness_caps']['worker_wall'])
    else:
        gate = json.loads(promotion_local.read_text())
        limits = gate.get('finish_next_limits')
        keys = {'full_seconds', 'campaign_seconds', 'gpu_ceiling_seconds', 'cpu_ceiling_seconds',
                'worker_cpu', 'worker_wall'}
        if not isinstance(limits, dict) or set(limits) != keys or any(type(limits[k]) is not int or limits[k] <= 0 for k in keys):
            raise ValueError('reviewed full finish stage limits required')
        spec['full']['seconds'], spec['budget_seconds'] = limits['full_seconds'], limits['campaign_seconds']
        spec['witness_gpu_ceiling_seconds'] = limits['gpu_ceiling_seconds']
        spec['witness_cpu_ceiling_seconds'] = limits['cpu_ceiling_seconds']
        spec['witness_caps']['worker_cpu'], spec['witness_caps']['worker_wall'] = limits['worker_cpu'], limits['worker_wall']
    additions = [('producer_audit', producer_audit_remote, audit_sha),
                 ('producer0', producer0_remote, fixed.file_sha(producer0_local)),
                 ('producer1', producer1_remote, fixed.file_sha(producer1_local)),
                 ('sam_weights', sam_remote, SAM_SHA),
                 ('native_inventory', native_inventory_remote, 'cd6bb8cf2a239d94f595fd92f823cf6406c018e1fbda745e81cd0c0e91d1b59b')]
    for name, path, digest in additions:
        permitted = (Path('/home/itec/emanuele/Models') if name == 'sam_weights'
                     else Path('/home/itec/emanuele/Datasets/pointstream-data'))
        if not Path(path).is_absolute() or not Path(path).is_relative_to(permitted):
            raise ValueError('finish input requires its registered storage root')
        if name.startswith('producer'):
            spec['witness_' + name + '_sha256'] = digest
        spec['staged_inputs'].append({'name': name, 'path': path, 'sha256': digest, 'extract': False})
        spec['inputs'].append({'path': path, 'sha256': digest})
    _finish_spec(spec, qualification=qualification)
    gate = json.loads(promotion_local.read_text())
    if qualification:
        check_qualification_subcap(json.loads(ledger_local.read_text()), gate,
                                   gpu_cap=spec['witness_gpu_ceiling_seconds'],
                                   cpu_cap=spec['witness_cpu_ceiling_seconds'])
    completed = gate.get('completed_artifact_sha256', {})
    needed = {'unit1_context'} if qualification else {'unit0_full', 'unit1_full'}
    component = gate['remaining_components']['native_sam_compose']
    bindings = {'unit1_context': spec['witness_producer1_sha256']} if qualification else {
        'unit0_full': spec['witness_producer0_sha256'], 'unit1_full': spec['witness_producer1_sha256']}
    if any(completed.get(name) != digest for name, digest in bindings.items()):
        raise ValueError('promotion does not bind the exact producer archives')
    if (not needed.issubset(completed) or 'native_sam_compose' in completed
            or any(component[k] < spec['witness_' + k.replace('_seconds', '_ceiling_seconds')]
                   for k in ('gpu_seconds', 'cpu_seconds'))):
        raise ValueError('finish forecast or required producers incomplete')
    from experiments.jobs.inbox import validate_spec
    validate_spec(spec)
    return spec

def finish_readiness(stage: Path, *, loaded: tuple[dict[str, Path], dict[str, Any], dict[str, Any]] | None = None) -> dict[str, Any]:
    paths, spec, dispatch = load_inputs(stage) if loaded is None else loaded
    qualification = spec.get('w8_witness_profile') == 'w8-context-qualification-v1'
    _finish_spec(spec, qualification=qualification)
    _unit1_promotion(paths, spec, dispatch)
    audit = json.loads(paths['producer_audit'].read_text())
    expected_scope = 'context_qualification_only' if qualification else 'candidate'
    if (audit.get('scope') != 'root-reviewed W8 producer pair' or audit.get('producer_scope') != expected_scope
            or audit.get('source_sha256') != dispatch['code']['source_sha256']
            or audit.get('snapshot_sha256') != dispatch['code']['snapshot_sha256']
            or fixed.file_sha(paths['producer_audit']) != spec['witness_prior_artifact_sha256']
            or set(audit.get('producers', {})) != set(fixed.UNITS)):
        raise ValueError('producer audit and source binding differ')
    for index, unit in enumerate(fixed.UNITS):
        if audit['producers'][unit].get('archive_sha256') != spec[f'witness_producer{index}_sha256']:
            raise ValueError('producer archive differs from pinned audit')
    completed = json.loads(paths['promotion'].read_text()).get('completed_artifact_sha256', {})
    bindings = {'unit1_context': spec['witness_producer1_sha256']} if qualification else {
        'unit0_full': spec['witness_producer0_sha256'], 'unit1_full': spec['witness_producer1_sha256']}
    if any(completed.get(name) != digest for name, digest in bindings.items()):
        raise ValueError('promotion does not bind the exact producer archives')
    return {'scope': expected_scope, 'source_sha256': dispatch['code']['source_sha256'],
            'producer_audit_sha256': fixed.file_sha(paths['producer_audit']),
            'producer_archives': [spec['witness_producer0_sha256'], spec['witness_producer1_sha256']]}


def run_finish_readiness(stage: Path) -> None:
    import os
    if stage.name != 'smoke' or os.environ.get('PS_STAGE') != 'smoke':
        raise ValueError('finish readiness belongs to the smoke stage')
    result = finish_readiness(stage)
    output = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
    output.mkdir()
    with (output / 'summary.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True)



def read_finish_publication(path: Path, *, maximum: int) -> dict[str, Any]:
    """Read the exact archive produced by the fleet publication helper."""
    import tarfile

    with tarfile.open(path) as tar:
        members = tar.getmembers()
        if (len(members) != 2 or {entry.name for entry in members} != {'publish', 'publish/summary.json'}
                or not next(entry for entry in members if entry.name == 'publish').isdir()):
            raise ValueError('finish publication membership differs')
        entry = next(entry for entry in members if entry.name == 'publish/summary.json')
        if not entry.isfile() or entry.size > maximum:
            raise ValueError('finish publication summary differs')
        content = tar.extractfile(entry)
        if content is None:
            raise ValueError('finish publication summary missing')
        result = json.load(content)
        if not isinstance(result, dict):
            raise ValueError('finish publication summary is not an object')
        return result

def validate_finish_readiness(stage: Path, target: Path) -> None:
    import os
    paths, spec, dispatch = load_released_inputs(stage)
    qualification = spec.get('w8_witness_profile') == 'w8-context-qualification-v1'
    _finish_spec(spec, qualification=qualification, reference_epoch=dispatch['started'])
    _unit1_promotion(paths, spec, dispatch, historical=True)
    check_stage_evidence(stage, dispatch, terminal=False)
    staging = json.loads((stage / 'staging.json').read_text())
    publication = stage / 'published.tar'
    if fixed.file_sha(publication) != staging['published']['sha256']:
        raise ValueError('finish readiness publication differs')
    result = read_finish_publication(publication, maximum=10000)
    if (result.get('producer_audit_sha256') != spec['witness_producer_audit_sha256']
            or result.get('producer_archives') != [spec['witness_producer0_sha256'], spec['witness_producer1_sha256']]
            or result.get('source_sha256') != dispatch['code']['source_sha256']
            or result.get('scope') != ('context_qualification_only' if qualification else 'candidate')):
        raise ValueError('finish readiness summary differs')
    with target.open('x') as output_stream:
        json.dump({'passed': True, 'scope': 'readiness only',
                   'checks': {'source_producers_and_budget': True}}, output_stream)
        output_stream.flush()
        os.fsync(output_stream.fileno())


def run_finish(stage: Path, *, qualification: bool) -> None:
    import os
    import resource
    import sys
    import time
    from experiments.tennis import w8_witness_profile as witness

    if stage.name != 'full' or os.environ.get('PS_STAGE') != 'full' or sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('one-core Linux finish full stage required')
    started = time.monotonic()
    paths, spec, dispatch = load_inputs(stage)
    _finish_spec(spec, qualification=qualification)
    _unit1_promotion(paths, spec, dispatch)
    finish_readiness(stage, loaded=(paths, spec, dispatch))
    caps = spec['witness_caps']
    resource.setrlimit(resource.RLIMIT_CPU, (caps['parent_cpu'], 3600))
    host.require_gpu_uuid_api(__import__('torch'))
    prepared = host.prepare_receiver_inputs(paths, spec, packet_cpu=caps['packet_cpu'],
        materialize_profile=True, clock_profile=True, digests=host.StableDigests(), witness=True)
    archives = {unit: paths[f'producer{index}'] for index, unit in enumerate(fixed.UNITS)}
    generations = witness.recover_audited_producers(prepared, json.loads(paths['producer_audit'].read_text()),
                                                   archives, qualification=qualification, source=dispatch['code'])
    native = witness.execute_native_base(prepared, inventory_path=paths['native_inventory'],
        parent_cpu=caps['native_cpu'], worker_cpu=caps['native_cpu'], child_cpu=caps['native_cpu'], wall=caps['native_wall'])
    result = witness.execute_qualification_finish(prepared, generations, identity_path=paths['identity'],
        sam_weights=paths['sam_weights'], native_output=native['output'], dispatch=dispatch,
        visible=os.environ.get('CUDA_VISIBLE_DEVICES', ''), worker_cpu=caps['worker_cpu'],
        worker_wall=caps['worker_wall'], qualification=qualification)
    retained = stage / 'w8-finish-retained'
    retained.mkdir()
    named = {'producer-audit.json': paths['producer_audit'], 'protected-identity.json': prepared['protected_path'],
             'loader-identity.json': prepared['loader_receipt'], 'native-sealed.json': native['output'] / 'sealed.json',
             'packet-config.json': prepared['decoder_work'] / 'worker-config.json',
             'packet-decoded.json': prepared['decoder_manifest'],
             'packet-cpu-only.json': prepared['decoder_out'] / 'cpu-only.json',
             'packet-checkpoint.pt': prepared['decoder_out'] / 'decoded.pt',
             **{f'packet/{unit}.npz': prepared['decoder_out'] / 'publish' / 'units' / f'{unit}.npz'
                for unit in fixed.UNITS},
             'packet-namespace.json': prepared['decoder_work'] / 'namespace-receipt.json',
             'packet-namespace.json.started': prepared['decoder_work'] / 'namespace-receipt.json.started'}
    for name, path in native['receipt_paths'].items():
        basename = {'namespace_started': 'namespace.json.started', 'namespace_final': 'namespace.json'}.get(name, name + '.json')
        named['native/' + basename] = Path(path)
    for unit in fixed.UNITS:
        for name in ('sealed.json', f'{unit}@match5.npz', f'{unit}@match5.json', 'decoded-manifest.json'):
            named[f'{unit}/{name}'] = generations['outputs'][unit] / name
    for name, output in result['outputs'].items():
        output_files = {'rgb': ('sealed.json', 'base.npy'),
                        'sam': ('sealed.json', *(f'{unit}-masks.npz' for unit in fixed.UNITS)),
                        'compose': ('sealed.json', 'composite.npy')}[name]
        for child in output_files:
            named[f'{name}/{child}'] = output / child
        work = prepared['scratch'] / f'witness-finish-{name}'
        for record in ('worker-config.json', 'namespace-receipt.json', 'namespace-receipt.json.started'):
            named[f'{name}/{record}'] = work / record
    for name, source in named.items():
        target = retained / name
        target.parent.mkdir(parents=True, exist_ok=True)
        host.copy_new(source, target)
    with (retained / 'source-manifest.json').open('x') as stream:
        json.dump(prepared['code_manifest'], stream, sort_keys=True)
    named['source-manifest.json'] = retained / 'source-manifest.json'
    summary = {'scope': 'context_qualification_only' if qualification else 'candidate',
               'producer_audit_sha256': fixed.file_sha(paths['producer_audit']),
               'members': {name: fixed.file_sha(retained / name) for name in named},
               'expected': generations['expected'], 'trusted': result['trusted'], 'validation': result['validation'],
               'usage': result['usage'], 'packet_usage': prepared['packet_usage'], 'native_usage': native['usage'],
               'native_namespace_sha256': native['namespace_receipt_sha256'],
               'source_sha256': dispatch['code']['source_sha256'], 'snapshot_sha256': dispatch['code']['snapshot_sha256'],
               'elapsed_seconds': time.monotonic() - started, 'automatic_promotion': False}
    output = prepared['scratch'] / 'publish'
    output.mkdir()
    with (output / 'summary.json').open('x') as stream:
        json.dump(summary, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())

def validate_finish(stage: Path, target: Path) -> None:
    """Reopen retained bytes and independently bind producers, workers and finish."""
    import hashlib
    import os
    import tarfile
    from experiments.tennis import w8_receiver_generate as generation
    from experiments.tennis import w8_witness_namespace as namespace
    from experiments.tennis import w8_witness_profile as witness

    paths, spec, dispatch = load_released_inputs(stage)
    qualification = spec.get('w8_witness_profile') == 'w8-context-qualification-v1'
    _finish_spec(spec, qualification=qualification, reference_epoch=dispatch['started'])
    _unit1_promotion(paths, spec, dispatch, historical=True)
    check_stage_evidence(stage, dispatch, terminal=True)
    staging = json.loads((stage / 'staging.json').read_text())
    publication = stage / 'published.tar'
    if fixed.file_sha(publication) != staging['published']['sha256']:
        raise ValueError('finish publication SHA differs')
    summary = read_finish_publication(publication, maximum=5_000_000)
    retained = stage / 'w8-finish-retained'
    members = summary.get('members')
    if (retained.is_symlink() or not retained.is_dir() or not isinstance(members, dict) or not members
            or any(not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts
                   or not _sha(digest) or (retained / name).is_symlink()
                   or fixed.file_sha(retained / name) != digest for name, digest in members.items())
            or summary.get('scope') != ('context_qualification_only' if qualification else 'candidate')
            or summary.get('automatic_promotion') is not False
            or summary.get('source_sha256') != dispatch['code']['source_sha256']
            or summary.get('snapshot_sha256') != dispatch['code']['snapshot_sha256']
            or summary.get('producer_audit_sha256') != spec['witness_producer_audit_sha256']
            or fixed.file_sha(retained / 'producer-audit.json') != spec['witness_producer_audit_sha256']):
        raise ValueError('finish retained member or source binding differs')
    import math
    elapsed = summary.get('elapsed_seconds')
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed) or not 0 < elapsed <= spec['full']['seconds']):
        raise ValueError('finish elapsed wall exceeds its fixed stage')
    caps = spec['witness_caps']
    for name in ('rgb', 'sam', 'compose'):
        usage = summary.get('usage', {}).get(name, {})
        if (usage.get('status') != 0 or usage.get('timed_out') is not False
                or usage.get('canaries_preserved') is not True
                or any(isinstance(usage.get(key), bool) or not isinstance(usage.get(key), (int, float))
                       or not math.isfinite(usage[key]) or usage[key] < 0 for key in ('cpu_seconds', 'wall_seconds'))
                or usage['cpu_seconds'] > caps['worker_cpu'] or usage['wall_seconds'] > caps['worker_wall']):
            raise ValueError('finish worker bounds or canary differ')
    native_usage = summary.get('native_usage', {})
    if (native_usage.get('status') != 0 or native_usage.get('timed_out') is not False
            or not isinstance(native_usage.get('wall_seconds'), (int, float))
            or not 0 <= native_usage['wall_seconds'] <= caps['native_wall'] + 25):
        raise ValueError('native finish bounds differ')
    manifest = json.loads((retained / 'source-manifest.json').read_text())
    files = manifest['files']
    if hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != manifest['manifest_sha256']:
        raise ValueError('finish source manifest differs')
    with tarfile.open(stage.parent / 'source.tar') as source_tar:
        if fixed.file_sha(stage.parent / 'source.tar') != dispatch['code']['snapshot_sha256']:
            raise ValueError('finish source archive differs')
        for name, digest in files.items():
            entry = source_tar.getmember(name)
            stream = source_tar.extractfile(entry) if entry.isfile() else None
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != digest:
                raise ValueError('finish source member differs')
    preflight = json.loads(paths['preflight'].read_text())
    host.check_preflight_bundle(preflight, paths['identity'], witness_metadata=True)
    from experiments.tennis import w8_protected_metadata
    identity = w8_protected_metadata.read_historical(paths['identity'], preflight['identity_sha256'])
    protected_metadata = json.loads((retained / 'protected-identity.json').read_text())
    w8_protected_metadata.validate(protected_metadata, identity, check_current=True)
    if protected_metadata['historical_receipt_sha256'] != preflight['identity_sha256']:
        raise ValueError('historical protected receipt differs')
    fingerprints = host.read_oracle(paths['oracle'])
    decoded = json.loads((retained / 'packet-decoded.json').read_text())
    packet_config = json.loads((retained / 'packet-config.json').read_text())
    loader_path = retained / 'loader-identity.json'
    loader = json.loads(loader_path.read_text())
    host.packet_audit.check_grants(packet_config, preflight['runtimes']['main'])
    host.packet_audit.check_isolation(packet_config, decoded['isolation'], loader, manifest, fixed.file_sha(loader_path))
    cpu_only = json.loads((retained / 'packet-cpu-only.json').read_text())
    if (cpu_only.get('cuda_visible_devices') != '' or cpu_only.get('cuda_initialized_after_decode') is not False
            or cpu_only.get('pre_serialization', {}).get('state_tensors') != 480
            or cpu_only.get('pre_serialization', {}).get('restored_tensors') != 480
            or decoded.get('match_tensors') != fingerprints or decoded.get('unit_sha256') != generation.ORACLE_UNIT_SHA
            or generation.verify_serialized_match(retained / 'packet-checkpoint.pt', fingerprints) != decoded['checkpoint_sha256']):
        raise ValueError('finish packet CPU and decoded checkpoint differ')
    audit = json.loads((retained / 'producer-audit.json').read_text())
    expected_scope = 'context_qualification_only' if qualification else 'candidate'
    if (audit.get('scope') != 'root-reviewed W8 producer pair' or audit.get('producer_scope') != expected_scope
            or audit.get('source_sha256') != dispatch['code']['source_sha256']
            or audit.get('snapshot_sha256') != dispatch['code']['snapshot_sha256']
            or set(audit.get('producers', {})) != set(fixed.UNITS)):
        raise ValueError('finish producer audit differs')
    expected = summary['expected']
    staged = {item['name']: item for item in staging['inputs']}
    for index, unit in enumerate(fixed.UNITS):
        producer = Path(staged[f'producer{index}']['source'])
        if fixed.file_sha(producer) != spec[f'witness_producer{index}_sha256']:
            raise ValueError('audited producer archive bytes differ')
        with tarfile.open(producer) as tar:
            for name in ('sealed.json', f'{unit}@match5.npz', f'{unit}@match5.json', 'decoded-manifest.json'):
                entry = tar.getmember(name)
                source = tar.extractfile(entry) if entry.isfile() else None
                if source is None:
                    raise ValueError('audited producer member unavailable')
                digest = hashlib.sha256()
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                if digest.hexdigest() != fixed.file_sha(retained / unit / name):
                    raise ValueError('retained producer differs from audited archive')
            entry = tar.getmember('generator-config.json')
            source = tar.extractfile(entry) if entry.isfile() else None
            if source is None:
                raise ValueError('audited producer config missing')
            producer_config = json.load(source)
            if {key: value['sha256'] for key, value in producer_config['files'].items()} != expected[unit]['files']:
                raise ValueError('producer input identities differ')
        if (not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', str(audit['producers'][unit].get('job_id')))
                or not _sha(audit['producers'][unit].get('terminal_audit_sha256'))
                or audit['producers'][unit].get('archive_sha256') != spec[f'witness_producer{index}_sha256']
                or expected[unit]['tensor_fingerprints'] != fingerprints
                or expected[unit]['files']['checkpoint'] != decoded['checkpoint_sha256']
                or expected[unit]['files']['unit'] != generation.ORACLE_UNIT_SHA[unit]
                or fixed.file_sha(retained / 'packet' / f'{unit}.npz') != generation.ORACLE_UNIT_SHA[unit]
                or expected[unit]['files']['decoder_manifest'] != fixed.file_sha(retained / unit / 'decoded-manifest.json')):
            raise ValueError('finish producer and fresh packet identities differ')
    from experiments.tennis.w8_cuda_system_leaves import canonical_gpu_uuid

    trusted = summary['trusted']
    if (trusted.get('sam_record_sha256') != preflight['identity_sha256']
            or trusted.get('protected_record_sha256') != fixed.file_sha(retained / 'protected-identity.json')
            or trusted.get('sam_verified_files') != identity['sam_distribution']['verified_files']
            or trusted.get('native_receipt') != json.loads((retained / 'native-sealed.json').read_text())
            or trusted.get('gpu_name') != dispatch['environment']['gpu']['name']
            or canonical_gpu_uuid(trusted.get('claimed_gpu_uuid', ''))
               != canonical_gpu_uuid(dispatch['environment']['gpu']['uuid'])):
        raise ValueError('finish independent native, SAM or GPU identity differs')
    require_original_host(dispatch)
    outside = os.readlink('/proc/self/ns/pid')
    group = json.loads((stage.parent / 'run/status.json').read_text())['process_group_id']
    for name in ('rgb', 'sam', 'compose'):
        config = json.loads((retained / name / 'worker-config.json').read_text())
        receipt = json.loads((retained / name / 'sealed.json').read_text())
        witness.check_worker_isolation(config, receipt['isolation'], runtime=preflight['runtimes']['main'],
                                       source_manifest=manifest, loader=loader, loader_sha=fixed.file_sha(loader_path))
        ns = witness.namespace_completion(retained / name / 'namespace-receipt.json', expected_pgid=group,
            expected_outside=outside, expected_target='experiments.tennis.w8_strict_receiver_worker')
        if (summary['usage'][name]['cpu_seconds'] + 0.05 < ns['children_cpu_seconds']
                or summary['usage'][name].get('namespace_receipt_sha256') != fixed.file_sha(retained / name / 'namespace-receipt.json')):
            raise ValueError('finish worker CPU or namespace receipt differs')
    packet_ns = witness.namespace_completion(retained / 'packet-namespace.json', expected_pgid=group,
        expected_outside=outside, expected_target='experiments.tennis.w8_strict_receiver_worker')
    packet_usage = summary.get('packet_usage', {})
    if (packet_usage.get('status') != 0 or packet_usage.get('timed_out') is not False
            or packet_usage.get('canaries_preserved') is not True
            or packet_usage.get('cpu_seconds', float('inf')) > caps['packet_cpu']
            or packet_usage.get('wall_seconds', float('inf')) > 130
            or packet_usage.get('cpu_seconds', -1) + 0.05 < packet_ns['children_cpu_seconds']
            or packet_usage.get('namespace_receipt_sha256') != fixed.file_sha(retained / 'packet-namespace.json')):
        raise ValueError('finish packet cost or namespace receipt differs')
    native_ns = witness.namespace_completion(retained / 'native/namespace.json', expected_pgid=group,
        expected_outside=outside, expected_target=namespace.NATIVE_TARGET)
    if (summary.get('native_namespace_sha256') != fixed.file_sha(retained / 'native/namespace.json')
            or native_usage.get('waited_cpu_seconds', -1) + 0.05 < native_ns['children_cpu_seconds']):
        raise ValueError('native finish namespace or waited CPU differs')
    checked = witness.check_finish(generation_dirs={u: retained / u for u in fixed.UNITS}, expected=expected,
        base_dir=retained / 'rgb', sam_dir=retained / 'sam', compose_dir=retained / 'compose',
        trusted=trusted, qualification=qualification)
    if checked != summary['validation']:
        raise ValueError('independent finish validation differs')
    with target.open('x') as output_stream:
        json.dump({'passed': True, 'scope': expected_scope, 'source_sha256': dispatch['code']['source_sha256'],
                   'producer_audit_sha256': summary['producer_audit_sha256'], 'members': members,
                   'automatic_promotion': False}, output_stream, sort_keys=True)
        output_stream.flush()
        os.fsync(output_stream.fileno())



def run_producer_audit(stage: Path, target: Path) -> None:
    """Account the standalone producer audit under the stage validator CPU cap."""
    import os
    import resource
    import time

    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise ValueError('producer audit requires verified one-core affinity')
    spec = json.loads((stage.parent / 'spec.json').read_text())
    limit = spec.get('witness_validator_cpu_seconds')
    if type(limit) is not int or not 1 <= limit <= 120:
        raise ValueError('producer audit CPU limit differs')
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
    started = time.monotonic()
    try:
        validate_pilot(stage, target)
        status = 'complete'
    except Exception:
        status = 'failed'
        raise
    finally:
        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        with target.with_name(target.name + '.usage.json').open('x') as stream:
            json.dump({'scope': 'one-core bounded producer audit', 'status': status, 'cpu_cap_seconds': limit,
                       'self_cpu_seconds': own.ru_utime + own.ru_stime,
                       'waited_children_cpu_seconds': children.ru_utime + children.ru_stime,
                       'affinity': affinity, 'pid': os.getpid(), 'pgid': os.getpgrp(),
                       'wall_seconds': time.monotonic() - started}, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())

def run_finish_audit(stage: Path, target: Path) -> None:
    """One-core, hard-capped post-stage audit; retain actual process CPU usage."""
    import os
    import resource
    import time

    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise ValueError('finish audit requires verified one-core affinity')
    resource.setrlimit(resource.RLIMIT_CPU, (100, 100))
    started = time.monotonic()
    try:
        validate_finish(stage, target)
        status = 'complete'
    except Exception:
        status = 'failed'
        raise
    finally:
        self_cpu = resource.getrusage(resource.RUSAGE_SELF)
        child_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        with target.with_name(target.name + '.usage.json').open('x') as stream:
            json.dump({'scope': 'one-core bounded finish audit', 'status': status, 'cpu_cap_seconds': 100,
                       'self_cpu_seconds': self_cpu.ru_utime + self_cpu.ru_stime,
                       'waited_children_cpu_seconds': child_cpu.ru_utime + child_cpu.ru_stime,
                       'affinity': affinity, 'pid': os.getpid(), 'pgid': os.getpgrp(),
                       'wall_seconds': time.monotonic() - started}, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())

def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    make = sub.add_parser('make-spec')
    make.add_argument('--deadline', required=True)
    make.add_argument('--cpu-proof-local', type=Path, required=True)
    make.add_argument('--cpu-proof-remote', required=True)
    make.add_argument('--ledger-local', type=Path, required=True)
    make.add_argument('--ledger-remote', required=True)
    make.add_argument('--admission-local', type=Path, required=True)
    make.add_argument('--admission-remote', required=True)
    make.add_argument('--output', type=Path, required=True)
    unit1 = sub.add_parser('make-unit1-spec')
    for arg in ('deadline', 'cpu-proof-remote', 'promotion-remote', 'ledger-remote', 'prior-artifact-sha'):
        unit1.add_argument('--' + arg, required=True)
    for arg in ('cpu-proof-local', 'promotion-local', 'ledger-local', 'output'):
        unit1.add_argument('--' + arg, type=Path, required=True)
    candidate = sub.add_parser('make-candidate-spec')
    for arg in ('deadline', 'cpu-proof-remote', 'promotion-remote', 'ledger-remote', 'qualification-artifact-sha'):
        candidate.add_argument('--' + arg, required=True)
    for arg in ('cpu-proof-local', 'promotion-local', 'ledger-local', 'output'):
        candidate.add_argument('--' + arg, type=Path, required=True)
    candidate.add_argument('--unit1', action='store_true')
    finish_make = sub.add_parser('make-finish-spec')
    for arg in ('deadline', 'cpu-proof-remote', 'promotion-remote', 'ledger-remote',
                'producer-audit-remote', 'producer0-remote', 'producer1-remote', 'sam-remote',
                'native-inventory-remote'):
        finish_make.add_argument('--' + arg, required=True)
    for arg in ('cpu-proof-local', 'promotion-local', 'ledger-local', 'producer-audit-local',
                'producer0-local', 'producer1-local', 'output'):
        finish_make.add_argument('--' + arg, type=Path, required=True)
    finish_make.add_argument('--qualification', action='store_true')
    finish_audit = sub.add_parser('audit-finish')
    finish_audit.add_argument('--stage', type=Path, required=True)
    finish_audit.add_argument('--output', type=Path, required=True)
    audit = sub.add_parser('audit-candidate')
    audit.add_argument('--stage', type=Path, required=True)
    audit.add_argument('--output', type=Path, required=True)
    for action in ('pilot', 'unit1', 'readiness', 'readiness-validate', 'candidate0', 'candidate1', 'noop', 'validate', 'finish-readiness', 'finish-readiness-validate',
                   'finish-context', 'finish-candidate'):
        sub.add_parser(action)
    args = parser.parse_args()
    if args.action == 'make-spec':
        spec = make_pilot_spec(deadline=args.deadline, proof_local=args.cpu_proof_local,
                               proof_remote=args.cpu_proof_remote, ledger_local=args.ledger_local,
                               ledger_remote=args.ledger_remote, admission_local=args.admission_local,
                               admission_remote=args.admission_remote)
        with args.output.open('x') as stream:
            json.dump(spec, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return
    if args.action == 'make-unit1-spec':
        spec = make_unit1_spec(deadline=args.deadline, proof_local=args.cpu_proof_local,
            proof_remote=args.cpu_proof_remote, promotion_local=args.promotion_local,
            promotion_remote=args.promotion_remote, ledger_local=args.ledger_local,
            ledger_remote=args.ledger_remote, prior_artifact_sha=args.prior_artifact_sha)
        with args.output.open('x') as stream:
            json.dump(spec, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return
    if args.action == 'make-candidate-spec':
        spec = make_candidate_spec(deadline=args.deadline, proof_local=args.cpu_proof_local,
            proof_remote=args.cpu_proof_remote, promotion_local=args.promotion_local,
            promotion_remote=args.promotion_remote, ledger_local=args.ledger_local,
            ledger_remote=args.ledger_remote, qualification_artifact_sha=args.qualification_artifact_sha,
            unit1=args.unit1)
        with args.output.open('x') as stream:
            json.dump(spec, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return
    if args.action == 'make-finish-spec':
        keys = ('deadline', 'cpu_proof_local', 'cpu_proof_remote', 'promotion_local', 'promotion_remote',
                'ledger_local', 'ledger_remote', 'producer_audit_local', 'producer_audit_remote',
                'producer0_local', 'producer0_remote', 'producer1_local', 'producer1_remote',
                'sam_remote', 'native_inventory_remote', 'qualification')
        spec = make_finish_spec(**{key: getattr(args, key) for key in keys})
        with args.output.open('x') as stream:
            json.dump(spec, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return
    if args.action == 'audit-finish':
        run_finish_audit(args.stage, args.output)
        return
    if args.action == 'audit-candidate':
        run_producer_audit(args.stage, args.output)
        return
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.action == 'finish-readiness':
        run_finish_readiness(stage)
    elif args.action == 'finish-readiness-validate':
        validate_finish_readiness(stage, Path(os.environ['PS_VALIDATION_PATH']))
    elif args.action in ('finish-context', 'finish-candidate'):
        run_finish(stage, qualification=args.action == 'finish-context')
    elif args.action == 'pilot':
        run_pilot(stage)
    elif args.action == 'unit1':
        run_pilot(stage, unit1=True)
    elif args.action == 'readiness':
        run_readiness(stage)
    elif args.action == 'readiness-validate':
        validate_readiness(stage, Path(os.environ['PS_VALIDATION_PATH']))
    elif args.action in ('candidate0', 'candidate1'):
        run_pilot(stage, unit1=args.action == 'candidate1', candidate=True)
    elif args.action == 'validate':
        validate_pilot(stage, Path(os.environ['PS_VALIDATION_PATH']))
    else:
        if os.environ.get('PS_STAGE') != 'full':
            raise ValueError('pilot no-op is full-stage only')
        output = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
        output.mkdir()
        with (output / 'noop.json').open('x') as stream:
            json.dump({'scope': 'no additional scientific execution'}, stream)


if __name__ == '__main__':
    main()

"""Prepare a review-only K5d full-cohort cost draft from actual fleet evidence.

Runs no jobs and grants no authority. Output deliberately cannot pass the
full-study worker's approval gate until the primary reviewer signs off.
"""
from __future__ import annotations

import json
import math
import tarfile
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_checkpoint_fleet as cf, k5d_fleet as fleet
from experiments.tennis import k5d_full_study as study, k5d_source_oracle as oracle
from experiments.tennis import k5d_study_evidence as proof

CONFIRMATION = ('tracknet/game9/Clip5', 'tracknet/game10/Clip12')


def number(value: Any, label: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError('invalid numeric value: ' + label)
    return float(value)


def model(s16: float, s48: float, margin: float, stage_reserve: float) -> dict[str, float]:
    s16, s48 = number(s16, '16-frame', positive=True), number(s48, '48-frame', positive=True)
    margin = number(margin, 'margin', positive=True)
    stage_reserve = number(stage_reserve, 'stage reserve', positive=True)
    if s48 < s16 or margin < 1.5:
        raise ValueError('non-monotonic timings or insufficient margin')
    marginal = (s48 - s16) / 32
    fixed = s16 - 16 * marginal
    if fixed < 0:
        raise ValueError('negative startup cost')
    return {'seconds16': s16, 'seconds48': s48, 'fixed_seconds': fixed,
            'marginal_seconds_per_frame': marginal, 'margin': margin, 'reserve_seconds': stage_reserve}


def projection(costs: dict[str, dict[str, float]], allowances: dict[str, Any],
               budgets: dict[str, Any], *, profile: str = 'legacy') -> dict[str, Any]:
    """Forecast the legacy GPU smoke or opt-in CPU readiness campaign."""
    if profile not in ('legacy', 'readiness'):
        raise ValueError('unknown study profile')
    if profile == 'readiness':
        readiness = number(allowances['gpu']['readiness_seconds'], 'readiness', positive=True)
        if not 20 <= readiness <= 60:
            raise ValueError('readiness bound must be between 20 and 60 seconds')
    else:
        readiness = 0.0
    stage_limit, campaign_limit, smoke_limit = (2670, 3000, 60) if profile == 'readiness' else (2400, 2700, 240)
    rows = study.selection()
    schedule = list(CONFIRMATION) + sorted(set(rows) - set(CONFIRMATION))
    if len(schedule) != 30 or sum(rows[x]['label_rows'] for x in schedule) != 5675:
        raise ValueError('registered full-study cohort differs')
    details = []
    gpu_compute = 0.0
    stage_caps_fit = True
    for clip in schedule:
        n = rows[clip]['label_rows']
        for phase in ('prepare', 'score'):
            c = costs[phase]
            smoke = readiness if profile == 'readiness' else c['margin'] * c['seconds16']
            full = c['margin'] * (c['fixed_seconds'] + c['marginal_seconds_per_frame'] * n) + c['reserve_seconds']
            stage_caps_fit = stage_caps_fit and smoke <= smoke_limit and full <= stage_limit
            details.append({'clip': clip, 'frames': n, 'phase': phase,
                            'smoke_seconds': smoke, 'full_seconds': full})
            gpu_compute += smoke + full
    extras: dict[str, float] = {}
    budget_fits: dict[str, bool] = {}
    for resource in ('gpu', 'cpu'):
        stage_io = number(allowances[resource]['per_stage_staging_seconds'], 'staging', positive=True)
        validator = number(allowances[resource]['per_campaign_validator_seconds'], 'validator', positive=True)
        audit = number(allowances[resource]['per_campaign_audit_seconds'], 'audit', positive=True)
        failure = number(allowances[resource]['failed_start_reserve_seconds'], 'failed starts', positive=True)
        stage_caps_fit = stage_caps_fit and all(
            row['smoke_seconds'] + row['full_seconds'] + 2 * stage_io + validator <= campaign_limit for row in details)
        extra = 120 * stage_io + 60 * validator + 60 * audit + failure
        extras[resource] = extra
        prior = number(budgets[resource]['spent_and_reserved_seconds'], 'prior spent and active reservation')
        cap = number(budgets[resource]['ball_limit_seconds'], 'ball limit', positive=True)
        global_left = number(budgets[resource]['global_remaining_seconds'], 'global remaining', positive=True)
        projected = gpu_compute if resource == 'gpu' else 0.0
        budget_fits[resource] = (resource == 'gpu' and projected + extra <= global_left
                                 and prior + projected + extra <= cap)
    return {'profile': profile, 'gpu_smoke_inferences': 0 if profile == 'readiness' else 60,
            'stage_limit_seconds': stage_limit, 'campaign_limit_seconds': campaign_limit,
            'clips': 30, 'unique_frames': 5675, 'full_native_frame_visits': 11350,
            'campaigns': 60, 'smoke_stages': 60, 'full_stages': 60,
            'stage_caps_fit': stage_caps_fit, 'budget_fits': budget_fits,
            'confirmation_order': list(CONFIRMATION), 'ordered_stages': details,
            'gpu_compute_seconds': gpu_compute, 'cpu_compute_seconds': None,
            'gpu_noncommand_seconds': extras['gpu'], 'cpu_noncommand_seconds': extras['cpu'],
            'gpu_incremental_seconds': gpu_compute + extras['gpu'],
            'cpu_incremental_seconds': None,
            'cpu_method': 'unqualified wall placeholder, not a process-tree CPU upper bound; CPU budget fit cannot be asserted',
            'limitation': '16/48 single-clip fixed-marginal fit, not established cross-clip or 586-frame cost; confirmations precede the other 28 clips'}



def qualified_remaining_draft(receipt: dict[str, Any], *, identity: dict[str, str],
                              inventory_path: Path, inventory_sha: str,
                              noncommand_path: Path, noncommand_sha: str,
                              ledger_path: Path, ledger_sha: str,
                              attestation_path: Path, attestation_sha: str) -> dict[str, Any]:
    """Recompute only verified pending work; draft never grants admission."""
    from experiments.tennis import k5d_control_study as control, k5d_study_wave as wave

    if receipt.get('consumer_execution_identity') != identity:
        raise ValueError('consumer execution identity differs')
    inventory = wave.inventory(study.checked_json(inventory_path, inventory_sha), identity)
    if any(inventory[clip][phase]['state'] != 'complete'
           for clip in wave.CONFIRMATIONS for phase in ('prepare', 'score')):
        raise ValueError('four verified confirmation stages required')
    noncommand = wave.verified_noncommand(study.checked_json(noncommand_path, noncommand_sha), identity, inventory)
    ledger = study.checked_json(ledger_path, ledger_sha)
    attestation = study.checked_json(attestation_path, attestation_sha)
    control.check_budget(ledger, attestation, ledger_sha, attestation_sha, receipt)
    forecast = control.programme_projection(receipt, inventory=inventory, noncommand=noncommand)
    return {'schema': 1, 'approved': False, 'status': 'draft pending independent review',
            'execution_identity': identity, 'inventory_sha256': inventory_sha,
            'noncommand_evidence_sha256': noncommand_sha, 'budget_ledger_sha256': ledger_sha,
            'budget_attestation_sha256': attestation_sha, 'forecast': forecast,
            'scope': 'verified pending-stage estimate; not a resource release'}

def metadata(stage_entry: dict[str, Any]) -> tuple[dict[str, Any], str]:
    job, root, stage = stage_entry['job'], Path(stage_entry['directory']), stage_entry['stage']
    if root.name != job or stage not in ('smoke', 'full'):
        raise ValueError('job or phase path differs')
    records = {}
    hashes = {}
    for name, path in (('spec', root / 'spec.json'), ('dispatch', root / stage / 'dispatch.json'),
                       ('staging', root / stage / 'staging.json'), ('execution', root / stage / 'execution.json')):
        hashes[name] = oracle.sha(path)
        records[name] = json.loads(path.read_text())
    published = records['staging']['published']['sha256']
    entry = {'job': job, 'directory': str(root), 'sha256': hashes}
    expected = {'job': job, 'stage': stage, 'published_sha256': published, 'sha256': hashes.copy()}
    return {'entry': entry, 'expected': expected, 'records': records}, published


def timing_sha(archive: Path, archive_sha: str) -> str:
    if oracle.sha(archive) != archive_sha:
        raise ValueError('scored publication changed')
    import hashlib
    with tarfile.open(archive) as tar:
        found = [m for m in tar if m.name == 'publish/timing.json']
        if len(found) != 1 or not found[0].isfile() or found[0].size > 2_000_000:
            raise ValueError('scoring timing absent or ambiguous')
        stream = tar.extractfile(found[0])
        if stream is None:
            raise ValueError('scoring timing unavailable')
        return hashlib.sha256(stream.read()).hexdigest()


def build(inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify actual artifacts and return drafts; input contains no future placeholders."""
    if set(inputs['stages']) != {'prepare16', 'prepare48', 'score16', 'score48'}:
        raise ValueError('four actual fleet stage records required')
    if oracle.sha(fleet.SOURCE) != fleet.SOURCE_SHA:
        raise ValueError('selection hash differs')
    jobs = inputs['calibration_jobs']
    if (set(jobs) != {'prepare', 'score'} or jobs['prepare'] == jobs['score']
            or any(not isinstance(job, str) or not job.startswith('2026') for job in jobs.values())
            or any(inputs['stages'][phase + str(n)]['job'] != jobs[phase]
                   for phase in ('prepare', 'score') for n in (16, 48))):
        raise ValueError('distinct paired producer and scorer job identities required')
    metadata_rows = {name: metadata(entry) for name, entry in inputs['stages'].items()}
    identity = inputs['execution_identity']
    if (set(identity) != {'source_sha256', 'snapshot_sha256', 'environment_sha256'}
            or any(not isinstance(v, str) or not cf.digest_string(v) for v in identity.values())):
        raise ValueError('frozen source identity missing')
    actual: dict[str, float] = {}
    archives = inputs['archives']
    for frames in (16, 48):
        prep, score = 'prepare' + str(frames), 'score' + str(frames)
        for name in (prep, score):
            meta, pub_sha = metadata_rows[name]
            archive = Path(archives[name])
            if oracle.sha(archive) != pub_sha:
                raise ValueError('fleet publication identity mismatch')
            actual[name] = proof.fleet_stage(meta['entry'], meta['expected'], identity,
                                             'prepare' if name == prep else 'score', frames, pub_sha,
                                             metadata_rows[prep][1])
        ref = inputs['reference' + str(frames)]
        proof.parity(Path(archives[score]), metadata_rows[score][1], Path(archives[prep]), metadata_rows[prep][1],
                     Path(ref['local']), ref['sha256'], frames, identity,
                     timing_sha(Path(archives[score]), metadata_rows[score][1]))
    controls = inputs['review_parameters']
    costs = {phase: model(actual[phase + '16'], actual[phase + '48'], controls[phase]['margin'],
                          controls[phase]['stage_reserve_seconds']) for phase in ('prepare', 'score')}
    ledger_path = Path(inputs['budget_ledger']['local'])
    if oracle.sha(ledger_path) != inputs['budget_ledger']['sha256']:
        raise ValueError('budget ledger identity mismatch')
    ledger = json.loads(ledger_path.read_text())
    attestation_path = Path(inputs['budget_attestation']['local'])
    if oracle.sha(attestation_path) != inputs['budget_attestation']['sha256']:
        raise ValueError('budget attestation identity mismatch')
    attestation = json.loads(attestation_path.read_text())
    if (attestation.get('reviewed') is not True or attestation.get('ledger_sha256') != inputs['budget_ledger']['sha256']
            or attestation.get('active_reservations_reconciled') is not True):
        raise ValueError('prior charges and outstanding reservations need reviewed reconciliation')
    for resource in ('gpu', 'cpu'):
        total_charged = sum(number(x.get(resource + '_seconds', 0), 'ledger charge') for x in ledger['charges'] if x['workstream'] == 'ball')
        all_charged = sum(number(x.get(resource + '_seconds', 0), 'global ledger charge') for x in ledger['charges'])
        if (attestation[resource]['global_charged_seconds'] != all_charged
                or attestation[resource]['global_remaining_seconds'] > ledger[resource + '_total_seconds'] - all_charged
                or attestation[resource]['ball_charged_seconds'] != total_charged
                or attestation[resource]['ball_limit_seconds'] != ledger['workstreams']['ball'][resource + '_limit_seconds']
                or attestation[resource]['spent_and_reserved_seconds'] < total_charged):
            raise ValueError('attested prior spending differs from actual ledger')
    forecast = projection(costs, inputs['allowances'], attestation, profile=inputs.get('profile', 'legacy'))
    evidence = {'schema': 1, 'passed': False, 'status': 'draft pending root independent review',
                'execution_identity': identity, 'gpu_model': 'RTX A6000', 'selection_sha256': fleet.SOURCE_SHA,
                'cost': costs, 'fleet_stages': {k: v[0]['expected'] for k, v in metadata_rows.items()},
                'preparation_archives': {str(n): metadata_rows['prepare' + str(n)][1] for n in (16, 48)},
                'parity_archives': {str(n): metadata_rows['score' + str(n)][1] for n in (16, 48)},
                'score_timing_sha256': {str(n): timing_sha(Path(archives['score' + str(n)]), metadata_rows['score' + str(n)][1]) for n in (16, 48)},
                'budget_ledger_sha256': inputs['budget_ledger']['sha256'], 'budget_attestation_sha256': inputs['budget_attestation']['sha256'],
                'allowances': inputs['allowances'], 'forecast': forecast}
    gate = {'schema': 1, 'approved': False, 'purpose': 'full-retrospective-benchmark',
            'selection_sha256': fleet.SOURCE_SHA, 'execution_identity': identity, 'gpu_models': ['RTX A6000'],
            'release': 'cost-confirmation', 'allowed_clips': list(CONFIRMATION),
            'fidelity_only_for_unqualified': True, 'historical_task_failures_preserved': True,
            'historical_source_evidence_sha256': proof.history.HISTORICAL_SHA,
            'historical_task_failures': sorted(proof.history.HISTORICAL_FAILURES),
            'status': 'draft only; cost receipt SHA and approval must be added after independent review'}
    return evidence, gate


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--cost-draft', required=True, type=Path)
    parser.add_argument('--gate-draft', required=True, type=Path)
    args = parser.parse_args()
    evidence, gate = build(json.loads(args.config.read_text()))
    for path, record in ((args.cost_draft, evidence), (args.gate_draft, gate)):
        with path.open('x') as output:
            json.dump(record, output, indent=2, allow_nan=False)
            output.write('\n')


if __name__ == '__main__':
    main()

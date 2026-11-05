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
from experiments.tennis import k5d_study_evidence as evidence

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


def programme_projection(receipt: dict[str, Any]) -> dict[str, Any]:
    """Recompute all 60 GPU full stages and CPU-readiness campaigns, once each."""
    rows = study.selection()
    if len(rows) != 30 or sum(row['label_rows'] for row in rows.values()) != 5675:
        raise ValueError('frozen population differs')
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
        if (costs[phase]['margin'] < 1.5 or costs[phase]['reserve_seconds'] <= 0
                or costs[phase]['seconds16'] <= 0 or costs[phase]['seconds48'] < costs[phase]['seconds16']):
            raise ValueError('invalid calibration bounds')
    full_compute = 0.0
    max_full = 0.0
    for row in rows.values():
        for phase in ('prepare', 'score'):
            estimate = study.forecast(receipt, phase, row['label_rows'], costs[phase]['seconds16'])
            full_compute += estimate
            max_full = max(max_full, estimate)
    if max_full > 2670:
        raise ValueError('full stage exceeds recovery bound')
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
        if max_full + d['readiness_seconds'] + 2 * d['stage_staging_seconds'] + d['validator_seconds'] > 3000:
            raise ValueError('longest campaign exceeds 3000-second recovery bound')
        incremental = (full_compute + 60 * d['readiness_seconds'] + 120 * d['stage_staging_seconds']
                       + 60 * d['validator_seconds'] + 60 * d['audit_seconds'] + d['failed_start_reserve_seconds'])
        results[resource] = {'incremental_seconds': incremental,
                             'fits': incremental <= min(d['ball_remaining_seconds'], d['global_remaining_seconds'])}
    return {'campaigns': 60, 'full_stages': 60, 'gpu_smoke_inferences': 0,
            'full_frames_per_phase': 5675, 'max_full_seconds': max_full,
            'gpu': results['gpu'], 'cpu': results['cpu'],
            'cpu_method': 'one-core elapsed wall upper bound; no cumulative RUSAGE summation'}


def check_budget(ledger: dict[str, Any], attestation: dict[str, Any], ledger_sha: str,
                 attestation_sha: str, receipt: dict[str, Any]) -> None:
    if (attestation.get('reviewed') is not True or attestation.get('ledger_sha256') != ledger_sha
            or attestation.get('active_reservations_reconciled') is not True
            or receipt.get('budget_ledger_sha256') != ledger_sha
            or receipt.get('budget_attestation_sha256') != attestation_sha):
        raise ValueError('budget snapshot and active reservations not bound')
    for resource in ('gpu', 'cpu'):
        charged = sum(row.get(resource + '_seconds', 0) for row in ledger['charges'] if row['workstream'] == 'ball')
        global_charged = sum(row.get(resource + '_seconds', 0) for row in ledger['charges'])
        row = attestation[resource]
        programme = receipt['programme'][resource]
        if (row['ball_charged_seconds'] != charged or row['global_charged_seconds'] != global_charged
                or row['ball_limit_seconds'] != ledger['workstreams']['ball'][resource + '_limit_seconds']
                or row['spent_and_reserved_seconds'] < charged
                or row['global_remaining_seconds'] > ledger[resource + '_total_seconds'] - global_charged
                or programme['ball_remaining_seconds'] > row['ball_limit_seconds'] - row['spent_and_reserved_seconds']
                or programme['global_remaining_seconds'] > row['global_remaining_seconds']):
            raise ValueError('programme budget exceeds reconciled ledger')


def approved(gate_path: Path, gate_sha: str, cost_path: Path, cost_sha: str, bridge_path: Path, bridge_sha: str,
             old_source: Path, new_source: Path, ledger_path: Path, ledger_sha: str,
             budget_path: Path, budget_sha: str, identity: dict[str, str], clip: str, phase: str) -> tuple[dict[str, Any], int]:
    gate = study.checked_json(gate_path, gate_sha)
    receipt = study.checked_json(cost_path, cost_sha)
    signed_bridge = study.checked_json(bridge_path, bridge_sha)
    if (gate.get('schema') != 1 or gate.get('approved') is not True
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
            or not set(allowed).issubset(set(clips)) or gate.get('release') not in ('cost-confirmation', 'full-study')
            or (gate['release'] == 'cost-confirmation' and not set(allowed).issubset(study.CONFIRMATION))):
        raise ValueError('clip is not admitted by this release')
    calibration = signed_bridge['calibration_execution_identity']
    if signed_bridge.get('approved') is not True or signed_bridge.get('consumer_execution_identity') != identity:
        raise ValueError('unapproved or wrong source bridge')
    actual_bridge = bridge.bridge(old_source, calibration, new_source, identity)
    for name in ('calibration_execution_identity', 'consumer_execution_identity', 'calibration_snapshot_sha256',
                 'consumer_snapshot_sha256', 'changed_paths', 'unchanged_files'):
        if signed_bridge.get(name) != actual_bridge[name]:
            raise ValueError('source bridge archived manifest differs: ' + name)
    if (receipt.get('schema') != 1 or receipt.get('passed') is not True
            or receipt.get('selection_sha256') != fleet.SOURCE_SHA
            or receipt.get('calibration_execution_identity') != calibration
            or receipt.get('consumer_execution_identity') != identity
            or receipt.get('bridge_sha256') != bridge_sha
            or receipt.get('gpu_model') != 'RTX A6000' or receipt.get('gpu_capability') != [8, 6]
            or receipt.get('calibration_jobs') != {'prepare': '20261105T025241Z-fe451f9e', 'score': '20261105T032433Z-1423f92a'}
            or receipt.get('stage_caps_fit') is not True or receipt.get('budget_fits') != {'gpu': True, 'cpu': True}
            or any(receipt['cost'][kind]['margin'] < 1.5 for kind in ('prepare', 'score'))):
        raise ValueError('full programme cost evidence does not authorize this profile')
    ledger = study.checked_json(ledger_path, ledger_sha)
    attestation = study.checked_json(budget_path, budget_sha)
    check_budget(ledger, attestation, ledger_sha, budget_sha, receipt)
    projection = programme_projection(receipt)
    if not projection['gpu']['fits'] or not projection['cpu']['fits']:
        raise ValueError('recomputed full programme exceeds available budgets')
    if phase not in ('prepare', 'score'):
        raise ValueError('invalid stage')
    return receipt, clips[clip]['label_rows']


def spec(base: dict[str, Any], *, phase: str, clip: str, identity: dict[str, str], assets: dict[str, dict[str, str]],
         deadline: str, readiness_seconds: int, full_seconds: int, campaign_seconds: int) -> dict[str, Any]:
    if (type(readiness_seconds) is not int or not 20 <= readiness_seconds <= 60
            or type(full_seconds) is not int or not 0 < full_seconds <= 2670
            or type(campaign_seconds) is not int or campaign_seconds > 3000
            or campaign_seconds < readiness_seconds + full_seconds + 30):
        raise ValueError('invalid readiness/full campaign bounds')
    for name, item in assets.items():
        if name == 'checkpoint' and item.get('archive_sha256') != item.get('sha256'):
            raise ValueError('checkpoint archive SHA aliases differ')
        cf.asset(name, item['remote'], item['sha256'], Path(item['local']), extract=False)
    for name in ('full_gate', 'cost_receipt', 'bridge', 'calibration_source', 'consumer_source', 'fidelity_gate',
                 'budget_ledger', 'budget_attestation'):
        if name not in assets:
            raise ValueError('missing approved evidence asset: ' + name)
    receipt, frames = approved(Path(assets['full_gate']['local']), assets['full_gate']['sha256'],
                               Path(assets['cost_receipt']['local']), assets['cost_receipt']['sha256'],
                               Path(assets['bridge']['local']), assets['bridge']['sha256'],
                               Path(assets['calibration_source']['local']), Path(assets['consumer_source']['local']),
                               Path(assets['budget_ledger']['local']), assets['budget_ledger']['sha256'],
                               Path(assets['budget_attestation']['local']), assets['budget_attestation']['sha256'],
                               identity, clip, phase)
    predicted = study.forecast(receipt, phase, frames, receipt['cost'][phase]['seconds16'])
    if predicted > full_seconds:
        raise ValueError('fixed/marginal full-stage forecast exceeds declared cap')
    extra = [cf.asset(name, item['remote'], item['sha256'], Path(item['local']), extract=False)
             for name, item in assets.items() if name != 'checkpoint']
    args = [phase, *base['arguments'], '--clip', clip, '--mode', '{mode}',
            '--full-gate', '{staged:full_gate}', '--full-gate-sha256', assets['full_gate']['sha256'],
            '--cost-receipt', '{staged:cost_receipt}', '--cost-receipt-sha256', assets['cost_receipt']['sha256'],
            '--bridge', '{staged:bridge}', '--bridge-sha256', assets['bridge']['sha256'],
            '--calibration-source', '{staged:calibration_source}', '--consumer-source', '{staged:consumer_source}',
            '--budget-ledger', '{staged:budget_ledger}', '--budget-ledger-sha256', assets['budget_ledger']['sha256'],
            '--budget-attestation', '{staged:budget_attestation}', '--budget-attestation-sha256', assets['budget_attestation']['sha256'],
            '--fidelity-gate', '{staged:fidelity_gate}', '--fidelity-gate-sha256', assets['fidelity_gate']['sha256']]
    if phase == 'score':
        if 'checkpoint' not in assets:
            raise ValueError('actual full-context checkpoint required')
        item = assets['checkpoint']
        extra.append(study.checkpoint(item, 'checkpoint', clip, frames, identity))
        args += ['--checkpoint', '{staged:checkpoint}', '--checkpoint-sha256', item['checkpoint_sha256']]
    result = cf.common_spec(base, extra, deadline)
    result.update(entrypoint=['-m', 'experiments.tennis.k5d_control_study'], arguments=args,
                  gpu_models=['RTX A6000'], budget_seconds=campaign_seconds, validator_seconds=30)
    result['scale']['frames'] = {'smoke': frames, 'full': frames}
    result['scale']['mode'] = {'smoke': 'readiness', 'full': 'full'}
    result['smoke'] = {'seconds': readiness_seconds, 'representative_basis': 'CPU validation of the complete source timeline, model identities, source bridge and prepared checkpoint.'}
    result['full'] = {'seconds': full_seconds}
    result['validator'] = ['{python}', '-m', 'experiments.tennis.k5d_control_study', 'validate']
    from experiments.jobs.inbox import validate_spec
    validate_spec(result)
    return result


def verify_gate(args: argparse.Namespace, identity: dict[str, str]) -> int:
    _, frames = approved(Path(args.full_gate), args.full_gate_sha256, Path(args.cost_receipt), args.cost_receipt_sha256,
                         Path(args.bridge), args.bridge_sha256, Path(args.calibration_source), Path(args.consumer_source),
                         Path(args.budget_ledger), args.budget_ledger_sha256, Path(args.budget_attestation),
                         args.budget_attestation_sha256, identity, args.clip, args.action)
    if args.frames != frames:
        raise ValueError('full native timeline differs')
    return frames


def run(args: argparse.Namespace) -> None:
    identity = cp.execution_identity()
    verify_gate(args, identity)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'score'))
    for name in ('archive', 'balltrack-code', 'balltrack-weight', 'wasb-code', 'wasb-weights', 'wasb-config',
                 'image-ckpt', 'video-ckpt', 'lpips-backbone', 'dists-backbone', 'clip', 'full-gate', 'full-gate-sha256',
                 'cost-receipt', 'cost-receipt-sha256', 'bridge', 'bridge-sha256', 'calibration-source', 'consumer-source',
                 'fidelity-gate', 'fidelity-gate-sha256', 'budget-ledger', 'budget-ledger-sha256',
                 'budget-attestation', 'budget-attestation-sha256', 'mode'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--frames', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'score' and (not args.checkpoint or not args.checkpoint_sha256):
        parser.error('score requires actual checkpoint and SHA')
    run(args)


if __name__ == '__main__':
    main()

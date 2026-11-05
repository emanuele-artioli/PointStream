"""Fail-closed, per-clip full-study preparation and coalesced scoring builders.

This module does not submit jobs. Receipts are prospective inputs supplied by
an independent admission review; this code does not grant scientific approval.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_benchmark as benchmark, k5d_checkpoint as cp
from experiments.tennis import k5d_checkpoint_fleet as cf, k5d_coalesced as co
from experiments.tennis import k5d_cost_pilot as cost, k5d_encoder as enc
from experiments.tennis import k5d_fleet as fleet, k5d_source_oracle as oracle
from experiments.tennis import k5d_study_evidence as proof

CONFIRMATION = {'tracknet/game9/Clip5', 'tracknet/game10/Clip12'}


def checked_json(path: Path, sha: str) -> dict[str, Any]:
    if not cf.digest_string(sha) or oracle.sha(path) != sha:
        raise ValueError('receipt identity mismatch')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('receipt must be an object')
    return value


def selection() -> dict[str, Any]:
    if oracle.sha(fleet.SOURCE) != fleet.SOURCE_SHA:
        raise ValueError('selection identity mismatch')
    data = json.loads(fleet.SOURCE.read_text())['tracknet']
    clips = data['clips']
    counts = [sum(c['visibility_counts'][str(i)] for c in clips) for i in range(4)]
    if (len(clips) != 30 or len({c['clip_id'] for c in clips}) != 30
            or sum(c['label_rows'] for c in clips) != 5675 or counts != [197, 5472, 0, 6]):
        # The selection groups easy/hard into two source labels; check below instead.
        if (len(clips) != 30 or len({c['clip_id'] for c in clips}) != 30
                or sum(c['label_rows'] for c in clips) != 5675 or counts[0] != 197
                or counts[1] + counts[2] != 5472 or counts[3] != 6):
            raise ValueError('selection coverage differs')
    return {c['clip_id']: c for c in clips}


def admission(gate_path: Path, gate_sha: str, evidence_path: Path, evidence_sha: str,
              clip: str, identity: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any], int]:
    if (set(identity) != {'source_sha256', 'snapshot_sha256', 'environment_sha256'}
            or any(not isinstance(v, str) or not cf.digest_string(v) for v in identity.values())):
        raise ValueError('invalid frozen execution identity')
    clips = selection()
    if clip not in clips:
        raise ValueError('clip absent from frozen selection')
    gate = checked_json(gate_path, gate_sha)
    evidence = checked_json(evidence_path, evidence_sha)
    if (gate.get('schema') != 1 or gate.get('approved') is not True
            or gate.get('purpose') != 'full-retrospective-benchmark'
            or gate.get('selection_sha256') != fleet.SOURCE_SHA
            or gate.get('cost_receipt_sha256') != evidence_sha
            or gate.get('execution_identity') != identity or gate.get('gpu_models') != ['RTX A6000']
            or gate.get('fidelity_only_for_unqualified') is not True
            or gate.get('historical_task_failures_preserved') is not True):
        raise ValueError('full-scope admission receipt mismatch')
    proof.historical(gate)
    allowed = gate.get('allowed_clips')
    if (not isinstance(allowed, list) or len(set(allowed)) != len(allowed) or clip not in allowed
            or not set(allowed).issubset(set(clips))
            or (gate.get('release') == 'cost-confirmation' and not set(allowed).issubset(CONFIRMATION))
            or gate.get('release') not in ('cost-confirmation', 'full-study')):
        raise ValueError('clip is not admitted by release')
    if (evidence.get('schema') != 1 or evidence.get('passed') is not True
            or evidence.get('execution_identity') != identity or evidence.get('gpu_model') != 'RTX A6000'
            or evidence.get('selection_sha256') != fleet.SOURCE_SHA):
        raise ValueError('coalesced cost and parity receipt mismatch')
    return gate, evidence, clips[clip]['label_rows']


def forecast(evidence: dict[str, Any], kind: str, frames: int, current16: float) -> float:
    """Extrapolate measured 16/48 envelope with observed slowdown and explicit reserve."""
    x = evidence['cost'][kind]
    values = [x[k] for k in ('seconds16', 'seconds48', 'margin', 'reserve_seconds')]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in [*values, current16]):
        raise ValueError('non-finite cost evidence')
    s16, s48, margin, reserve = values
    if not (0 < s16 <= s48 and margin >= 1 and reserve > 0 and current16 > 0 and frames >= 16):
        raise ValueError('invalid measured cost envelope')
    marginal = (s48 - s16) / 32
    fixed = s16 - 16 * marginal
    if fixed < 0:
        raise ValueError('negative measured fixed cost')
    return margin * (fixed + marginal * frames) * max(1., current16 / s16) + reserve


def checked_publication(entry: dict[str, str], name: str) -> dict[str, Any]:
    return cf.asset(name, entry['remote'], entry['archive_sha256'], Path(entry['local']), extract=True)


def checkpoint(entry: dict[str, str], name: str, clip: str, frames: int,
               identity: dict[str, str]) -> dict[str, Any]:
    import hashlib
    import tarfile

    item = checked_publication(entry, name)
    if not cf.digest_string(entry['checkpoint_sha256']):
        raise ValueError('checkpoint digest required')
    with tarfile.open(entry['local']) as bundle:
        members = [m for m in bundle if m.name == 'publish/checkpoint.json']
        if len(members) != 1 or not members[0].isfile() or members[0].size > 1_000_000:
            raise ValueError('checkpoint manifest missing or ambiguous')
        stream = bundle.extractfile(members[0])
        if stream is None:
            raise ValueError('checkpoint manifest unavailable')
        raw = stream.read()
    record = json.loads(raw)
    if (hashlib.sha256(raw).hexdigest() != entry['checkpoint_sha256'] or record.get('schema') != 1
            or record.get('clip') != clip or record.get('frames') != frames
            or record.get('execution_identity') != identity or record.get('points') != [list(p) for p in enc.POINTS]):
        raise ValueError('checkpoint scope or identity mismatch')
    return item


def spec(base: dict[str, Any], *, kind: str, clip: str, gate: dict[str, str], evidence: dict[str, str],
         fidelity: dict[str, str], identity: dict[str, str], deadline: str,
         parity16: dict[str, str], parity48: dict[str, str],
         execution_receipts: dict[str, dict[str, Any]],
         preparation16: dict[str, str], preparation48: dict[str, str],
         reference16: dict[str, str], reference48: dict[str, str],
         small: dict[str, str] | None = None, full: dict[str, str] | None = None) -> dict[str, Any]:
    gate_item = cf.asset('study_gate', gate['remote'], gate['sha256'], Path(gate['local']), extract=False)
    evidence_item = cf.asset('cost_receipt', evidence['remote'], evidence['sha256'], Path(evidence['local']), extract=False)
    _, receipt, frames = admission(Path(gate['local']), gate['sha256'], Path(evidence['local']), evidence['sha256'], clip, identity)
    if set(execution_receipts) != {'prepare16', 'prepare48', 'score16', 'score48'}:
        raise ValueError('four measured execution receipts required')
    parity_items = []
    for n, publication, preparation, reference in ((16, parity16, preparation16, reference16),
                                                    (48, parity48, preparation48, reference48)):
        item = cf.asset(f'parity{n}', publication['remote'], publication['archive_sha256'],
                        Path(publication['local']), extract=False)
        cf.asset(f'calibration_prepare{n}', preparation['remote'], preparation['archive_sha256'],
                 Path(preparation['local']), extract=False)
        cf.asset(f'calibration_reference{n}', reference['remote'], reference['archive_sha256'],
                 Path(reference['local']), extract=False)
        if (receipt['parity_archives'][str(n)] != publication['archive_sha256']
                or receipt['preparation_archives'][str(n)] != preparation['archive_sha256']):
            raise ValueError('approved calibration publication differs')
        for label, artifact in (('prepare', preparation), ('score', publication)):
            name = f'{label}{n}'
            measured = proof.fleet_stage(execution_receipts[name], receipt['fleet_stages'][name], identity,
                                         label, n, artifact['archive_sha256'], preparation['archive_sha256'])
            if receipt['cost'][label][f'seconds{n}'] < measured:
                raise ValueError('cost sample is below measured execution')
        proof.parity(Path(publication['local']), publication['archive_sha256'], Path(preparation['local']),
                     preparation['archive_sha256'], Path(reference['local']), reference['archive_sha256'],
                     n, identity, receipt['score_timing_sha256'][str(n)])
        parity_items.append(item)
    fidelity_item = cf.asset('fidelity_gate', fidelity['remote'], fidelity['sha256'], Path(fidelity['local']), extract=False)
    enc.fidelity_gate(Path(fidelity['local']), fidelity['sha256'], fleet.EXPECTED['archive'])
    if kind not in ('prepare', 'score'):
        raise ValueError('unknown study stage')
    # Even the optimistic measured baseline must fit; the smoke validator applies observed slowdown.
    if forecast(receipt, kind, frames, receipt['cost'][kind]['seconds16']) > 2400:
        raise ValueError('full stage exceeds 2400-second admission cap')
    extra = [gate_item, evidence_item, fidelity_item, *parity_items]
    args = [kind, *base['arguments'], '--clip', clip, '--study-gate', '{staged:study_gate}',
            '--study-gate-sha256', gate['sha256'], '--cost-receipt', '{staged:cost_receipt}',
            '--cost-receipt-sha256', evidence['sha256'], '--fidelity-gate', '{staged:fidelity_gate}',
            '--fidelity-gate-sha256', fidelity['sha256']]
    if kind == 'score':
        if small is None or full is None:
            raise ValueError('both actual checkpoint publications required')
        extra.extend([checkpoint(small, 'prepare16', clip, 16, identity), checkpoint(full, 'prepare_full', clip, frames, identity)])
        args += ['--prepare-16', '{staged:prepare16}', '--prepare-full', '{staged:prepare_full}',
                 '--checkpoint-16-sha256', small['checkpoint_sha256'], '--checkpoint-full-sha256', full['checkpoint_sha256']]
    if base.get('cpu_threads') != 1 or base.get('single_cpu_campaign') is not True or base.get('device') != 'gpu':
        raise ValueError('single-core GPU campaign required')
    result = json.loads(json.dumps(base))
    result.update(entrypoint=['-m', 'experiments.tennis.k5d_full_study'], arguments=args,
                  gpu_models=['RTX A6000'], deadline=deadline, validator_seconds=20,
                  verify_staged_inputs=True, budget_seconds=2700)
    result['scale']['frames'] = {'smoke': 16, 'full': frames}
    result['smoke'] = {'seconds': 240, 'representative_basis': 'Complete six-point 16-frame path with measured full-stage forecast.'}
    result['full'] = {'seconds': 2400}
    result['staged_inputs'].extend(extra)
    result['inputs'].extend([{k: item[k] for k in ('path', 'sha256')} for item in extra])
    result['validator'] = ['{python}', '-m', 'experiments.tennis.k5d_full_study', 'validate']
    return result


def run(args: argparse.Namespace) -> None:
    identity = cp.execution_identity()
    _, _, frames = admission(Path(args.study_gate), args.study_gate_sha256, Path(args.cost_receipt),
                             args.cost_receipt_sha256, args.clip, identity)
    stage = os.environ.get('PS_STAGE', '')
    if args.frames != {'smoke': 16, 'full': frames}.get(stage):
        raise ValueError('clip stage does not cover its registered timeline')
    data = benchmark.build(args)
    if args.action == 'prepare':
        cp.prepare(data, identity)
    else:
        suffix = '16' if stage == 'smoke' else 'full'
        data.checkpoint = str(Path(getattr(args, 'prepare_' + suffix)) / 'publish')
        co.score_all(data, getattr(args, 'checkpoint_' + suffix + '_sha256'), identity)


def validate() -> None:
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        stage = Path(os.environ['PS_STAGE_DIR'])
        staging = json.loads((stage / 'staging.json').read_text())
        entries = {x['name']: x for x in staging['inputs']}
        spec_data = json.loads((stage.parent / 'spec.json').read_text())
        args = spec_data['arguments']
        clip = cf.option(args, '--clip')
        identity = cp.execution_identity()
        _, receipt, frames = admission(Path(entries['study_gate']['path']), cf.option(args, '--study-gate-sha256'),
                                       Path(entries['cost_receipt']['path']), cf.option(args, '--cost-receipt-sha256'), clip, identity)
        action = args[0]
        with cost.unpack(stage / 'published.tar', staging['published']['sha256']) as output:
            if action == 'prepare':
                cp.verify(output, oracle.sha(output / 'checkpoint.json'), identity)
                report = json.loads((output / 'pilot.json').read_text())
            elif action == 'score':
                result = enc.validate(output / 'assembled')
                report = json.loads((output / 'assembled' / 'pilot.json').read_text())
                if not result['full_metrics'] or result['points'] != 6:
                    raise ValueError('incomplete all-arm scoring')
            else:
                raise ValueError('unknown stage')
            if report['clip'] != clip or report['frames'] != 16:
                raise ValueError('smoke timeline differs')
        execution = json.loads((stage / 'execution.json').read_text())
        dispatch = json.loads((stage / 'dispatch.json').read_text())
        now = time.monotonic()
        clock = dispatch['campaign_clock']
        if (execution.get('exit_code') != 0 or not 0 < clock['started'] <= clock['phase_started'] <= now < clock['budget_end']):
            raise ValueError('invalid measured stage execution')
        estimate = forecast(receipt, action, frames, execution['seconds'])
        if estimate > 2400 or estimate + spec_data['validator_seconds'] > clock['budget_end'] - now:
            raise ValueError('full stage forecast exceeds stage or campaign')
        target.write_text(json.dumps({'passed': True, 'checks': {'complete_smoke': True, 'forecast': True},
                                      'forecast_seconds': estimate, 'clip': clip}) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'study': False}, 'error': str(exc)}) + '\n')
        raise


def main() -> None:
    if sys.argv[1:] == ['validate']:
        validate()
        return
    if sys.argv[1:2] == ['--make-spec']:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True, type=Path)
        parser.add_argument('--output', required=True, type=Path)
        options = parser.parse_args(sys.argv[2:])
        data = json.loads(options.config.read_text())
        result = spec(json.loads(cf.BASE.read_text()), **data)
        from experiments.jobs.inbox import validate_spec
        validate_spec(result)
        cp.write_new(options.output, result)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'score'))
    for name in ('archive', 'balltrack-code', 'balltrack-weight', 'wasb-code', 'wasb-weights', 'wasb-config',
                 'image-ckpt', 'video-ckpt', 'lpips-backbone', 'dists-backbone', 'clip', 'study-gate',
                 'study-gate-sha256', 'cost-receipt', 'cost-receipt-sha256', 'fidelity-gate', 'fidelity-gate-sha256'):
        parser.add_argument('--' + name, required=True)
    for name in ('prepare-16', 'prepare-full', 'checkpoint-16-sha256', 'checkpoint-full-sha256'):
        parser.add_argument('--' + name)
    parser.add_argument('--frames', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'score' and any(getattr(args, name) is None for name in
                                      ('prepare_16', 'prepare_full', 'checkpoint_16_sha256', 'checkpoint_full_sha256')):
        parser.error('score requires both prepared publications')
    run(args)


if __name__ == '__main__':
    main()

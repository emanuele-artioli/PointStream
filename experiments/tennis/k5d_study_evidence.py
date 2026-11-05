"""Verify actual fleet and parity artifacts before full-study cost admission."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

from experiments.jobs import inbox
from experiments.tennis import k5d_benchmark_analysis as history, k5d_checkpoint as cp
from experiments.tennis import k5d_checkpoint_fleet as cf, k5d_coalesced_fleet as cofleet
from experiments.tennis import k5d_cost_pilot as cost, k5d_encoder as enc, k5d_source_oracle as oracle


def historical(gate: dict[str, Any]) -> None:
    if oracle.sha(history.HISTORICAL_EVIDENCE) != history.HISTORICAL_SHA:
        raise ValueError('historical evidence identity mismatch')
    data = json.loads(history.HISTORICAL_EVIDENCE.read_text())
    failures = {row['clip_id'] for row in data['source_job']['per_clip_gate']['failing_clips']}
    if (failures != history.HISTORICAL_FAILURES or gate.get('historical_source_evidence_sha256') != history.HISTORICAL_SHA
            or set(gate.get('historical_task_failures', [])) != failures
            or len(gate.get('historical_task_failures', [])) != 3):
        raise ValueError('historical task failure evidence differs')


def fleet_stage(entry: dict[str, Any], expected: dict[str, Any], identity: dict[str, str],
                kind: str, frames: int, published_sha: str, prepared_sha: str | None = None) -> float:
    """Read one saved campaign's actual spec and stage records with approved hashes."""
    stage = 'smoke' if frames == 16 else 'full'
    job = entry['job']
    root = Path(entry['directory'])
    if root.name != job or not job.startswith('2026') or not cf.digest_string(published_sha):
        raise ValueError('job directory identity mismatch')
    records = {}
    for name, path in (('spec', root / 'spec.json'), ('dispatch', root / stage / 'dispatch.json'),
                       ('staging', root / stage / 'staging.json'), ('execution', root / stage / 'execution.json')):
        digest = entry['sha256'][name]
        if not cf.digest_string(digest) or oracle.sha(path) != digest or digest != expected['sha256'][name]:
            raise ValueError('fleet record identity mismatch: ' + name)
        records[name] = json.loads(path.read_text())
    if expected['job'] != job or expected['stage'] != stage or expected['published_sha256'] != published_sha:
        raise ValueError('approved job, stage or publication mismatch')
    spec, dispatch, staging, execution = (records[x] for x in ('spec', 'dispatch', 'staging', 'execution'))
    code = dispatch['code']
    if (code['spec_sha256'] != inbox.digest(spec) or code['source_sha256'] != identity['source_sha256']
            or code['snapshot_sha256'] != identity['snapshot_sha256']
            or dispatch['staged_environment']['sha256'] != identity['environment_sha256']
            or staging['environment']['sha256'] != identity['environment_sha256']
            or staging['published']['sha256'] != published_sha
            or dispatch['staged_inputs'] != staging['inputs'] or dispatch['input_identity'] != spec['inputs']
            or spec['gpu_models'] != ['RTX A6000'] or spec['cpu_threads'] != 1
            or spec.get('single_cpu_campaign') is not True):
        raise ValueError('fleet source, environment, spec, staging or publication mismatch')
    published_path = Path(staging['published']['path'])
    if published_path.parts[-3:] != (job, stage, 'published.tar'):
        raise ValueError('published path is not the claimed job stage')
    if kind == 'score':
        inputs = [x for x in staging['inputs'] if x['name'] == ('prepare16' if frames == 16 else 'prepare48')]
        if len(inputs) != 1 or inputs[0]['sha256'] != prepared_sha:
            raise ValueError('score staged checkpoint is not the measured preparation')
    gpu = dispatch['environment']['gpu']['name']
    if 'RTX A6000' not in gpu or 'Ada' in gpu:
        raise ValueError('fleet GPU class mismatch')
    command = dispatch['command']
    module = 'experiments.tennis.k5d_checkpoint_fleet' if kind == 'prepare' else 'experiments.tennis.k5d_coalesced_fleet'
    if ('-m' not in command or command[command.index('-m') + 1:command.index('-m') + 3] != [module, kind]
            or cf.option(command, '--frames') != str(frames)
            or spec['scale']['frames'][stage] != frames):
        raise ValueError('fleet command, stage or timeline mismatch')
    seconds = execution['seconds']
    if (execution['exit_code'] != 0 or execution.get('timing_contaminated') is not False
            or isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('failed or contaminated execution timing')
    return float(seconds)


def parity(score: Path, score_sha: str, prepared: Path, prepared_sha: str, reference: Path,
           reference_sha: str, frames: int, identity: dict[str, str], timing_sha: str) -> None:
    expected_ref = cf.D8_SMOKE_SHA if frames == 16 else cofleet.FULL_SHA
    if reference_sha != expected_ref:
        raise ValueError('reference archive is not the pinned d8b5 publication')
    with (cost.unpack(score, score_sha) as proof, cost.unpack(prepared, prepared_sha) as checkpoint,
          cost.unpack(reference, reference_sha) as prior):
        checkpoint_sha = oracle.sha(checkpoint / 'checkpoint.json')
        cp.verify(checkpoint, checkpoint_sha, identity)
        record = json.loads((proof / 'coalesced-validation.json').read_text())
        if record['checkpoint_sha256'] != checkpoint_sha or record['execution_identity'] != identity:
            raise ValueError('score checkpoint lineage differs')
        points = record['points']
        expected_names = [f'point-{codec}-{point}.json' for codec, point in enc.POINTS]
        if len(points) != 6 or [p['name'] for p in points] != expected_names:
            raise ValueError('six ordered point proofs required')
        point_files = []
        for point in points:
            path = proof / point['name']
            if oracle.sha(path) != point['sha256']:
                raise ValueError('point proof hash mismatch')
            point_files.append({'path': str(path), 'sha256': point['sha256']})
        with tempfile.TemporaryDirectory(prefix='k5d-study-proof-') as tmp:
            assembled = Path(tmp) / 'assembled'
            result = cp.assemble(checkpoint, checkpoint_sha, point_files, assembled, identity)
            if result != record['result'] or oracle.sha(assembled / 'pilot.json') != oracle.sha(proof / 'assembled' / 'pilot.json'):
                raise ValueError('published assembly differs from six point proofs')
        comparison = cp.compare_equivalence(prior, proof / 'assembled', oracle.sha(prior / 'pilot.json'))
        if comparison != json.loads((proof / 'comparison.json').read_text()) or comparison['frames'] != frames:
            raise ValueError('recomputed pinned reference comparison differs')
        report = json.loads((proof / 'assembled' / 'pilot.json').read_text())
        source = json.loads((proof / 'assembled' / 'source-qualification.json').read_text())
        if len(source['rows']) != frames or any(row['label_kind'] != 1 for row in source['rows']):
            raise ValueError('cost reference requires every frame visible')
        if any(p['streak']['guard_off'] is not False for p in report['points']):
            raise ValueError('cost reference requires all 12 arms evaluated')
        if oracle.sha(proof / 'timing.json') != timing_sha:
            raise ValueError('scoring timing identity mismatch')
        events = json.loads((proof / 'timing.json').read_text())['events']
        required = {f'{kind}.{codec}.{point}.{arm}' for codec, point in enc.POINTS
                    for kind in ('metrics', 'judges') for arm in ('baseline', 'streak')}
        for name in required:
            found = [event for event in events if event['name'] == name]
            if len(found) != 1 or found[0]['completed'] is not True:
                raise ValueError('missing or duplicate outer metric/judge event')

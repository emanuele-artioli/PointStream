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
                     identity: dict[str, str], preparation: dict[str, Any] | None = None) -> None:
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
    spec = read(job_dir / 'spec.json', receipt['records']['spec_sha256'])
    if spec.get('entrypoint') not in (['-m', 'experiments.tennis.k5d_control_study'],
                                       ['-m', 'experiments.tennis.k5d_full_study']):
        raise ValueError('unsupported study entrypoint')
    for stage in names:
        records = receipt['records'][stage]
        dispatch = read(job_dir / stage / 'dispatch.json', records['dispatch_sha256'])
        staging = read(job_dir / stage / 'staging.json', records['staging_sha256'])
        execution = read(job_dir / stage / 'execution.json', records['execution_sha256'])
        artifact = publications[stage]
        control = spec.get('entrypoint') == ['-m', 'experiments.tennis.k5d_control_study']
        expected_frames = frames if control or stage == 'full' else 16
        code = dispatch['code']
        gpu = dispatch['environment']['gpu']['name']
        if (execution.get('exit_code') != 0 or code['spec_sha256'] != inbox.digest(spec)
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
            log_path = job_dir / stage / 'command.log'
            if oracle.sha(log_path) != records['command_log_sha256']:
                raise ValueError('CPU command log SHA mismatch')
            accounting.read_receipt(log_path.read_text(), phase=phase, stage=stage, clip=clip,
                                    frames=frames, identity=identity)
        if control and stage == 'smoke':
            with unpack_publication(Path(artifact['local_path']), artifact['sha256']) as output:
                readiness = json.loads((output / 'readiness.json').read_text())
                if (readiness.get('identity') != identity or readiness.get('source', {}).get('frames') != frames
                        or readiness['source'].get('labels') != frames
                        or readiness.get('scope') != 'CPU asset, source header and checkpoint readiness; no GPU scoring'):
                    raise ValueError('readiness artifact differs')
    validate_full(clip, phase, frames, identity, publications['full'], preparation, dispatch)



def validate_full(clip: str, phase: str, frames: int, identity: dict[str, str],
                  artifact: dict[str, Any], preparation: dict[str, Any] | None,
                  dispatch: dict[str, Any]) -> None:
    """Reconstruct score from the exact producer and validate native data and metrics."""
    with unpack_publication(Path(artifact['local_path']), artifact['sha256']) as output:
        if phase == 'prepare':
            record = cp.verify(output, artifact['checkpoint_sha256'], identity)
            enc.validate(output, prepared=True)
            if record['clip'] != clip or record['frames'] != frames:
                raise ValueError('prepared full timeline differs')
            return
        if preparation is None:
            raise ValueError('score requires exact prepared publication')
        prepared = preparation['publications']['full']
        checkpoint_sha = prepared['checkpoint_sha256']
        option = '--checkpoint-sha256' if 'experiments.tennis.k5d_control_study' in dispatch['command'] else '--checkpoint-full-sha256'
        if (cf.option(dispatch['command'], option) != checkpoint_sha
                or not any(x.get('sha256') == prepared['sha256'] for x in dispatch.get('staged_inputs', []))):
            raise ValueError('score dispatch is not bound to prepared publication')
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



def verified_cpu_calibration(samples: dict[str, Any], identity: dict[str, str]) -> dict[str, dict[str, float]]:
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
            if phase == 'score':
                if not sample.get('preparation'):
                    raise ValueError('score calibration needs its actual producer job')
                verify_completed(clip, 'prepare', sample['preparation'], frames, identity)
            verify_completed(clip, phase, entry, frames, identity, sample.get('preparation'))
            receipt = read(Path(entry['receipt_path']), entry['receipt_sha256'])
            log = Path(receipt['job_directory']) / 'full' / 'command.log'
            seconds = accounting.read_receipt(log.read_text(), phase=phase, stage='full', clip=clip,
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

def inventory(ledger: dict[str, Any], identity: dict[str, str]) -> dict[str, dict[str, Any]]:
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
                verify_completed(clip, phase, entry, frames, identity, row['prepare'] if phase == 'score' else None)
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
            now: datetime) -> dict[str, Any]:
    """Return at most one bounded prepare/score job, never submit it."""
    if now.tzinfo is None:
        raise ValueError('timezone required')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
           for x in (gpu_budget_seconds, cpu_budget_seconds)):
        raise ValueError('positive explicitly remaining budgets required')
    rows = inventory(read(ledger_path, ledger_sha), identity)
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
        _, evidence, frames = study.admission(gate_path, gate_sha, evidence_path, evidence_sha, clip, identity)
        cost = evidence['cost'][phase]
        if isinstance(cost.get('margin'), bool) or not isinstance(cost.get('margin'), (int, float)) or not math.isfinite(cost['margin']) or cost['margin'] < 1.5:
            raise ValueError('cost margin must be at least 1.5')
        raise ValueError('process-tree CPU bound is not established; no wave proposal')
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

"""Review-only source bridge for CPU-readiness orchestration.

This never changes checkpoint identities or authorizes an experiment. It binds
both actual fleet source archives, recomputes their complete manifests and
permits only the reviewed control-only additions to differ.
"""
from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from experiments.jobs.inbox import digest

CALIBRATION_SOURCE = '8a3b51c487cf9b456e304fdf24300f230ece3f3c7b2c8569ab6681fbd8166f13'
CONTROL_ADDITIONS = frozenset({'experiments/tennis/k5d_control_bridge.py',
                               'experiments/tennis/k5d_control_readiness.py',
                               'experiments/tennis/k5d_control_study.py',
                               'experiments/tennis/k5d_readiness_cost_probe.py',
                               'tests/tennis/test_k5d_control_bridge.py',
                               'tests/tennis/test_k5d_control_readiness.py',
                               'tests/tennis/test_k5d_control_study.py',
                               'tests/tennis/test_k5d_readiness_cost_probe.py',
                               'experiments/tennis/k5d_full_cost_builder.py',
                               'experiments/tennis/k5d_study_wave.py',
                               'experiments/tennis/k5d_cpu_accounting.py',
                               'tests/tennis/test_k5d_cpu_accounting.py',
                               'tests/tennis/test_k5d_full_cost_builder.py',
                               'tests/tennis/test_k5d_study_wave.py'})


def manifest(archive: Path, expected_sha: str) -> tuple[str, dict[str, list[str]]]:
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('source archive SHA mismatch')
    records: dict[str, list[str]] = {}
    with tarfile.open(archive) as bundle:
        for item in bundle:
            name = PurePosixPath(item.name)
            if name.is_absolute() or '..' in name.parts or not name.parts:
                raise ValueError('unsafe source archive member')
            if item.isdir():
                continue
            key = name.as_posix()
            if key in records or '__pycache__' in name.parts or name.suffix == '.pyc':
                raise ValueError('duplicate or transient source member')
            if item.isfile():
                stream = bundle.extractfile(item)
                if stream is None:
                    raise ValueError('source member unreadable')
                h = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
                records[key] = [key, h.hexdigest()]
            elif item.issym():
                target = PurePosixPath(item.linkname)
                if target.is_absolute() or '..' in target.parts:
                    raise ValueError('unsafe source symlink')
                records[key] = [key, 'symlink', item.linkname]
            else:
                raise ValueError('unsupported source archive member')
    ordered = sorted(records.values(), key=lambda record: Path(record[0]).parts)
    return digest(ordered), records


def bridge(calibration_archive: Path, calibration: dict[str, str], consumer_archive: Path,
           consumer: dict[str, str]) -> dict[str, Any]:
    for item in (calibration, consumer):
        if set(item) != {'source_sha256', 'snapshot_sha256', 'environment_sha256'}:
            raise ValueError('complete execution identities required')
    if (calibration['source_sha256'] != CALIBRATION_SOURCE
            or calibration['environment_sha256'] != consumer['environment_sha256']
            or calibration == consumer):
        raise ValueError('calibration identity, environment or distinctness differs')
    old_identity, old = manifest(calibration_archive, calibration['snapshot_sha256'])
    new_identity, new = manifest(consumer_archive, consumer['snapshot_sha256'])
    if old_identity != calibration['source_sha256'] or new_identity != consumer['source_sha256']:
        raise ValueError('source tree identity differs from execution record')
    changed = {name for name in set(old) | set(new) if old.get(name) != new.get(name)}
    if changed != CONTROL_ADDITIONS or any(name in old for name in changed):
        raise ValueError('source changes extend beyond approved new control modules')
    return {'schema': 1, 'status': 'draft; independent scientific review required', 'approved': False,
            'calibration_execution_identity': calibration, 'consumer_execution_identity': consumer,
            'calibration_snapshot_sha256': calibration['snapshot_sha256'],
            'consumer_snapshot_sha256': consumer['snapshot_sha256'],
            'changed_paths': sorted(changed), 'unchanged_files': len(set(old) & set(new)),
            'scope': 'complete archived source manifests; no checkpoint identity substitution'}

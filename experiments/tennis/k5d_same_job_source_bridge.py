"""Exact archive bridge for same-job packet receiver checkpoint admission."""
from __future__ import annotations

import difflib
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_control_bridge as base
from experiments.tennis import k5d_parent8_child2_source_bridge as resource

from experiments.tennis import k5d_first292_source_bridge as prior

PATCHES = {**prior.PATCHES,
    'experiments/tennis/k5d_checkpoint.py': (
        'b457daeca3402f8089fc169743163496dd122937daf32cd517410b50239b387c',
        'de3f67118291bff711ba65b8b7db4e683dfb22df5a423718b0034f24f89a5736',
        'e2b2f5b035c3c806f9bd191429ee3ade35a932356684a7ff109b6208dc48c789'),
    'experiments/tennis/k5d_coalesced.py': (
        'b0037aea99d5c4551c784fdde7ed2b3b706ffb84b5bd2f57c0d755107567a0ef',
        '87737b766eb7883ff1a3d6747873fefc837d585a951d5c5ecbf545aecfb16a6f',
        '1e2172ca4ca377b9c8f18879109776a3d6e292356b44b6ad37ed0cc4f4f90b98'),
}
ADDITIONS = prior.ADDITIONS | {
    'experiments/tennis/k5d_same_job_receiver.py',
    'experiments/tennis/k5d_same_job_source_bridge.py',
    'tests/tennis/test_k5d_same_job_receiver.py',
    'tests/tennis/test_k5d_same_job_coalesced.py',
}



def bridge(old_archive: Path, old_identity: dict[str, str], new_archive: Path,
           new_identity: dict[str, str]) -> dict[str, Any]:
    for identity in (old_identity, new_identity):
        if set(identity) != {'source_sha256', 'snapshot_sha256', 'environment_sha256'}:
            raise ValueError('complete execution identities required')
    if (old_identity['source_sha256'] != base.CALIBRATION_SOURCE
            or old_identity['environment_sha256'] != new_identity['environment_sha256'] or old_identity == new_identity):
        raise ValueError('calibration identity or environment differs')
    old_source, old = base.manifest(old_archive, old_identity['snapshot_sha256'])
    new_source, new = base.manifest(new_archive, new_identity['snapshot_sha256'])
    if old_source != old_identity['source_sha256'] or new_source != new_identity['source_sha256']:
        raise ValueError('source identity differs')
    changed = {name for name in set(old) | set(new) if old.get(name) != new.get(name)}
    if changed != ADDITIONS | PATCHES.keys() or any(name in old for name in ADDITIONS):
        raise ValueError('source changes extend beyond exact scoped additions and patches')
    if new.get(resource.FIXTURE) != [resource.FIXTURE, resource.FIXTURE_SHA]:
        raise ValueError('audit fixture differs')
    with tarfile.open(old_archive) as before_tar, tarfile.open(new_archive) as after_tar:
        for name, expected in PATCHES.items():
            streams = [bundle.extractfile(name) for bundle in (before_tar, after_tar)]
            if any(stream is None for stream in streams):
                raise ValueError('patch member unavailable')
            before, after = (stream.read() for stream in streams if stream is not None)
            patch = b''.join(difflib.diff_bytes(difflib.unified_diff, before.splitlines(keepends=True),
                                              after.splitlines(keepends=True), fromfile=('old/' + name).encode(),
                                              tofile=('new/' + name).encode()))
            if tuple(hashlib.sha256(data).hexdigest() for data in (before, after, patch)) != expected:
                raise ValueError('pinned old, new or patch digest differs: ' + name)
    return {'bridge_type': 'same-job-receiver-v1', 'calibration_execution_identity': old_identity,
            'consumer_execution_identity': new_identity, 'calibration_snapshot_sha256': old_identity['snapshot_sha256'],
            'consumer_snapshot_sha256': new_identity['snapshot_sha256'], 'changed_paths': sorted(changed),
            'unchanged_files': sum(old[name] == new.get(name) for name in old),
            'patches': {name: list(digests) for name, digests in PATCHES.items()}}


def verify_signed(record_path: Path, record_sha: str, old_archive: Path, new_archive: Path,
                  identity: dict[str, str]) -> None:
    if hashlib.sha256(record_path.read_bytes()).hexdigest() != record_sha:
        raise ValueError('scoped bridge receipt hash differs')
    signed = json.loads(record_path.read_text())
    if signed.get('approved') is not True or signed.get('consumer_execution_identity') != identity:
        raise ValueError('scoped bridge not approved for this source')
    actual = bridge(old_archive, signed['calibration_execution_identity'], new_archive, identity)
    if any(signed.get(key) != value for key, value in actual.items()):
        raise ValueError('scoped bridge archived identities differ')

"""Fresh Ada reference bridge; numerical producer and scorer stay fixed."""
from __future__ import annotations

import difflib
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_control_bridge as base
from experiments.tennis import k5d_parent8_child2_source_bridge as resource

from experiments.tennis import k5d_same_job_source_bridge as prior

PATCHES = {**prior.PATCHES,
    'experiments/tennis/k5d_cost_pilot.py': (
        '8a9f6bd2d7c30ce3c17197565abb6a624b6ee23a84d28de7250daab7900e657b',
        '4f506dc2b01c5d6125572216a47d8e79a1a0f1a1b8b55f4279a5aab4a7b8a1d1',
        'e2a48cc5a9c2cc1e3eb1a37eaab4440437163e26571a73d63561c8ec752ed0fa'),
    'tests/tennis/test_k5d_cost_pilot.py': (
        '8b8d74614e08debbbbf35b5049d53d62a4b95ab5c28b0e2f6871c1180b1a751d',
        '81cd4d65e61df9593e4743fe08dacaf249cc553c388ec732bbecd8740c99cc5d',
        'd07c5caa0a09687d734f55378b0d191e48aed043123fac34624c877d978c2fd8'),
    'experiments/tennis/k5d_checkpoint_fleet.py': (
        '9a355efffafa9ab65049537247081dbfb63d167f9dab1ac535b86d8376134090',
        'c902606860b3b10f7659acadd6864fcb63c09e1db01f2f7664ec4a7a0477ba12',
        '301b9f072ed58b22b7087949d6d56d45e4c0f2384902d608988779cd726b963e'),
}
ADDITIONS = prior.ADDITIONS | {'experiments/tennis/k5d_ada_source_bridge.py',
                               'tests/tennis/test_k5d_ada_source_bridge.py',
                               'experiments/tennis/k5d_ada_fresh_source_bridge.py',
                               'experiments/tennis/k5d_ada_fresh.py',
                               'experiments/tennis/k5d_ada_fresh_compare.py',
                               'tests/tennis/test_k5d_ada_fresh_compare.py'}



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
    return {'bridge_type': 'same-job-ada-fresh-reference-v1', 'calibration_execution_identity': old_identity,
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

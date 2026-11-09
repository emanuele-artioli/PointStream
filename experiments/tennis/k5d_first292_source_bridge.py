"""Exact archive bridge for the scoped first-292 admission-only source changes."""
from __future__ import annotations

import difflib
import hashlib
import tarfile
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_control_bridge as base
from experiments.tennis import k5d_parent8_child2_source_bridge as resource

PATCHES = {
    resource.MEMBER: (resource.OLD_SHA, resource.NEW_SHA, resource.PATCH_SHA),
    'experiments/tennis/k5d_encoder.py': (
        'e98895cee0999d97db09ba27f54b1857c73f0c8eceda3be391c40aaf2cce0621',
        '7602effa778c18053c03fe71f51be57ebdbc554a73b83dabdea38779e2421d27',
        '917b3f5055e8860c6b305eb431e9b2990116e31102d34a67bca2642dc68a919c'),
    'experiments/tennis/k5d_benchmark.py': (
        '3377e5b4370e19f25dc3e011a0a292004fe004ccca54acff3a2585d351b86abd',
        'a95935bdeadbbb4e0df9fdeedb75e0cfd7ff310343a2b0353c68cd25e241db7c',
        'f7a1fcaa388ad3dbf5a4170f99fe3e1ffde461ab389f8da510cbea23ed002932'),
}
ADDITIONS = resource.ADDITIONS | {'experiments/tennis/k5d_first292_source_bridge.py'}


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
    return {'bridge_type': 'first292-admission-v1', 'calibration_execution_identity': old_identity,
            'consumer_execution_identity': new_identity, 'calibration_snapshot_sha256': old_identity['snapshot_sha256'],
            'consumer_snapshot_sha256': new_identity['snapshot_sha256'], 'changed_paths': sorted(changed),
            'unchanged_files': sum(old[name] == new.get(name) for name in old),
            'patches': {name: list(digests) for name, digests in PATCHES.items()}}

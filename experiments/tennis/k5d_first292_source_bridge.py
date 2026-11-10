"""Exact archive bridge for the scoped first-292 admission-only source changes."""
from __future__ import annotations

import difflib
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_control_bridge as base
from experiments.tennis import k5d_parent8_child2_source_bridge as resource

PATCHES = {
    resource.MEMBER: (resource.OLD_SHA, resource.NEW_SHA, resource.PATCH_SHA),
    'experiments/tennis/k5d_encoder.py': (
        'e98895cee0999d97db09ba27f54b1857c73f0c8eceda3be391c40aaf2cce0621',
        'aa06febbf2429ec8b81cd4640fb6b39cf777e5557bf7b7ee05f391e33cd8eb08',
        '11dd724975b52831c832fa9703700b208f9117021bbe0f31b53a8266a9e61bd0'),
    'experiments/tennis/k5d_benchmark.py': (
        '3377e5b4370e19f25dc3e011a0a292004fe004ccca54acff3a2585d351b86abd',
        '6a6929a68db95db0e6ee54e41102e58883583b468c4bbe3737c0fa36fa254699',
        'e7a0bcd5122a050be169392f8ee48b2892dd7f7f3b9bf63d574efc125c5dc33a'),
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

"""Review-only bridge for the single pinned parent8-child2 shared-parity allocator patch.

This does not approve a run, replace a CPU proof, or alter the legacy bridge.
"""
from __future__ import annotations

import difflib
import hashlib
import tarfile
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_control_bridge as base

MEMBER = 'src/codecs/dcvc_uf_worker.py'
OLD_SHA = 'd189b733c4fe18fa383ff4f193d7c2dc9ddc7a844275a8d59bc48684eee108e4'
NEW_SHA = 'cb2652db6fdd91fe837e94f1c4ec57a0e80a409e044326e79c4c88a9e63d695f'
PATCH_SHA = 'ebe029991a1d7dbf33eb9dff1763b0cac11c6340276d0a2f5637c1ce73d7361e'
FIXTURE = 'docs/experiments/k5d-inputs/controls-3ec159d8-independent-audit-20261105.json'
FIXTURE_SHA = 'f438382b94d44dcd81bc3a3bc17405d4ad2d78fd6866691ab20655ef11f5c0ad'
ADDITIONS = base.CONTROL_ADDITIONS | frozenset({
    'experiments/tennis/k5d_shared_source_bridge.py',
    'tests/tennis/test_k5d_shared_source_bridge.py',
    'tests/tennis/test_k5d_shared_parity.py',
    'experiments/tennis/k5d_low_memory_source_bridge.py',
    'tests/tennis/test_k5d_low_memory_source_bridge.py',
    'experiments/tennis/k5d_parent8_child2_source_bridge.py',
    'tests/tennis/test_k5d_parent8_child2_source_bridge.py',
    FIXTURE,
})


def member_bytes(archive: Path) -> bytes:
    with tarfile.open(archive) as bundle:
        item = bundle.getmember(MEMBER)
        stream = bundle.extractfile(item)
        if not item.isfile() or stream is None:
            raise ValueError('codec must be a regular file')
        return stream.read()


def bridge(old_archive: Path, old_identity: dict[str, str], new_archive: Path,
           new_identity: dict[str, str]) -> dict[str, Any]:
    for identity in (old_identity, new_identity):
        if set(identity) != {'source_sha256', 'snapshot_sha256', 'environment_sha256'}:
            raise ValueError('complete execution identities required')
    if (old_identity['source_sha256'] != base.CALIBRATION_SOURCE
            or old_identity['environment_sha256'] != new_identity['environment_sha256']
            or old_identity == new_identity):
        raise ValueError('calibration identity, environment or distinctness differs')
    old_source, old = base.manifest(old_archive, old_identity['snapshot_sha256'])
    new_source, new = base.manifest(new_archive, new_identity['snapshot_sha256'])
    if old_source != old_identity['source_sha256'] or new_source != new_identity['source_sha256']:
        raise ValueError('source tree identity differs')
    changed = {name for name in set(old) | set(new) if old.get(name) != new.get(name)}
    if changed != ADDITIONS | {MEMBER} or any(name in old for name in ADDITIONS):
        raise ValueError('source changes extend beyond the pinned resource patch and control additions')
    if new.get(FIXTURE) != [FIXTURE, FIXTURE_SHA]:
        raise ValueError('pinned audit fixture digest differs')
    before, after = member_bytes(old_archive), member_bytes(new_archive)
    patch = b''.join(difflib.diff_bytes(difflib.unified_diff, before.splitlines(keepends=True),
                                      after.splitlines(keepends=True),
                                      fromfile=b'old/src/codecs/dcvc_uf_worker.py',
                                      tofile=b'new/src/codecs/dcvc_uf_worker.py'))
    if tuple(hashlib.sha256(data).hexdigest() for data in (before, after, patch)) != (OLD_SHA, NEW_SHA, PATCH_SHA):
        raise ValueError('pinned old, new or patch digest differs')
    return {'schema': 1, 'approved': False, 'scope': 'resource-only source bridge; fresh CPU proof required',
            'old_identity': old_identity, 'new_identity': new_identity, 'old_file_sha256': OLD_SHA,
            'new_file_sha256': NEW_SHA, 'patch_sha256': PATCH_SHA, 'changed_paths': sorted(changed)}

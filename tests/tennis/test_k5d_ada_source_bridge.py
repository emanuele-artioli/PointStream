import difflib
import hashlib

import pytest

from experiments.tennis import k5d_ada_source_bridge as bridge
from experiments.tennis import k5d_control_bridge as base
from experiments.tennis import k5d_parent8_child2_source_bridge as resource
from tests.tennis.test_k5d_parent8_child2_source_bridge import archive


def test_bridge_rejects_unreviewed_numerical_or_profile_patch(tmp_path, monkeypatch):
    old, new = tmp_path / 'old.tar', tmp_path / 'new.tar'
    old_files = {name: (name + ' old\n').encode() for name in bridge.PATCHES}
    new_files = {name: (name + ' new\n').encode() for name in bridge.PATCHES}
    expected = {}
    for name in bridge.PATCHES:
        a, b = old_files[name], new_files[name]
        patch = b''.join(difflib.diff_bytes(difflib.unified_diff, a.splitlines(keepends=True),
                         b.splitlines(keepends=True), fromfile=('old/' + name).encode(),
                         tofile=('new/' + name).encode()))
        expected[name] = tuple(hashlib.sha256(x).hexdigest() for x in (a, b, patch))
    monkeypatch.setattr(bridge, 'PATCHES', expected)
    monkeypatch.setattr(resource, 'FIXTURE_SHA', hashlib.sha256(b'control').hexdigest())
    additions = {name: b'control' for name in bridge.ADDITIONS}
    old_sha = archive(old, {**old_files, 'numerical.py': b'unchanged'})
    new_sha = archive(new, {**new_files, 'numerical.py': b'unchanged', **additions})
    old_id, _ = base.manifest(old, old_sha)
    new_id, _ = base.manifest(new, new_sha)
    monkeypatch.setattr(base, 'CALIBRATION_SOURCE', old_id)
    a = {'source_sha256': old_id, 'snapshot_sha256': old_sha, 'environment_sha256': 'e' * 64}
    b = {'source_sha256': new_id, 'snapshot_sha256': new_sha, 'environment_sha256': 'e' * 64}
    assert bridge.bridge(old, a, new, b)['bridge_type'] == 'same-job-ada-class-profile-v1'
    for changes in ({'numerical.py': b'changed'}, {next(iter(new_files)): b'other patch'}):
        changed_sha = archive(new, {**new_files, 'numerical.py': b'unchanged', **additions, **changes})
        changed_id, _ = base.manifest(new, changed_sha)
        with pytest.raises(ValueError):
            bridge.bridge(old, a, new, {**b, 'source_sha256': changed_id, 'snapshot_sha256': changed_sha})

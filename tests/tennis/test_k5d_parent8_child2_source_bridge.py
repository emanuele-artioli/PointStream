import difflib
import hashlib
import io
import tarfile

import pytest

from experiments.tennis import k5d_control_bridge as base
from experiments.tennis import k5d_parent8_child2_source_bridge as bridge


def archive(path, files):
    with tarfile.open(path, 'w') as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_resource_patch_rejects_any_other_original_change(tmp_path, monkeypatch):
    old, new = tmp_path / 'old.tar', tmp_path / 'new.tar'
    before, after = b'legacy\n', b'legacy\nresource guard\n'
    patch = b''.join(difflib.diff_bytes(difflib.unified_diff, before.splitlines(keepends=True),
                                      after.splitlines(keepends=True),
                                      fromfile=b'old/src/codecs/dcvc_uf_worker.py',
                                      tofile=b'new/src/codecs/dcvc_uf_worker.py'))
    for name, value in [('OLD_SHA', before), ('NEW_SHA', after), ('PATCH_SHA', patch)]:
        monkeypatch.setattr(bridge, name, hashlib.sha256(value).hexdigest())
    monkeypatch.setattr(bridge, 'FIXTURE_SHA', hashlib.sha256(b'control').hexdigest())
    additions = {name: b'control' for name in bridge.ADDITIONS}
    old_sha = archive(old, {bridge.MEMBER: before, 'model.py': b'unchanged'})
    new_sha = archive(new, {bridge.MEMBER: after, 'model.py': b'unchanged', **additions})
    old_id, _ = base.manifest(old, old_sha)
    new_id, _ = base.manifest(new, new_sha)
    monkeypatch.setattr(base, 'CALIBRATION_SOURCE', old_id)
    a = {'source_sha256': old_id, 'snapshot_sha256': old_sha, 'environment_sha256': 'e' * 64}
    b = {'source_sha256': new_id, 'snapshot_sha256': new_sha, 'environment_sha256': 'e' * 64}
    assert bridge.bridge(old, a, new, b)['approved'] is False
    modified = archive(new, {bridge.MEMBER: after, 'model.py': b'changed', **additions})
    changed, _ = base.manifest(new, modified)
    with pytest.raises(ValueError, match='source changes'):
        bridge.bridge(old, a, new, {**b, 'source_sha256': changed, 'snapshot_sha256': modified})
    modified = archive(new, {bridge.MEMBER: b'other patch', 'model.py': b'unchanged', **additions})
    changed, _ = base.manifest(new, modified)
    with pytest.raises(ValueError, match='digest'):
        bridge.bridge(old, a, new, {**b, 'source_sha256': changed, 'snapshot_sha256': modified})
    modified = archive(new, {bridge.MEMBER: after, 'model.py': b'unchanged', **{**additions, bridge.FIXTURE: b'wrong'}})
    changed, _ = base.manifest(new, modified)
    with pytest.raises(ValueError, match='fixture'):
        bridge.bridge(old, a, new, {**b, 'source_sha256': changed, 'snapshot_sha256': modified})


def test_first_pair_bridge_rejects_extra_original_change(tmp_path, monkeypatch):
    from experiments.tennis import k5d_first292_source_bridge as first

    old, new = tmp_path / 'old.tar', tmp_path / 'new.tar'
    old_files = {name: (name + ' old\n').encode() for name in first.PATCHES}
    new_files = {name: (name + ' new\n').encode() for name in first.PATCHES}
    expected = {}
    for name in first.PATCHES:
        before, after = old_files[name], new_files[name]
        patch = b''.join(difflib.diff_bytes(difflib.unified_diff, before.splitlines(keepends=True),
                                          after.splitlines(keepends=True), fromfile=('old/' + name).encode(),
                                          tofile=('new/' + name).encode()))
        expected[name] = tuple(hashlib.sha256(x).hexdigest() for x in (before, after, patch))
    monkeypatch.setattr(first, 'PATCHES', expected)
    monkeypatch.setattr(bridge, 'FIXTURE_SHA', hashlib.sha256(b'control').hexdigest())
    additions = {name: b'control' for name in first.ADDITIONS}
    old_sha = archive(old, {**old_files, 'model.py': b'unchanged'})
    new_sha = archive(new, {**new_files, 'model.py': b'unchanged', **additions})
    old_id, _ = base.manifest(old, old_sha)
    new_id, _ = base.manifest(new, new_sha)
    monkeypatch.setattr(base, 'CALIBRATION_SOURCE', old_id)
    a = {'source_sha256': old_id, 'snapshot_sha256': old_sha, 'environment_sha256': 'e' * 64}
    b = {'source_sha256': new_id, 'snapshot_sha256': new_sha, 'environment_sha256': 'e' * 64}
    assert first.bridge(old, a, new, b)['bridge_type'] == 'first292-admission-v1'
    changed_sha = archive(new, {**new_files, 'model.py': b'changed', **additions})
    changed_id, _ = base.manifest(new, changed_sha)
    with pytest.raises(ValueError, match='source changes'):
        first.bridge(old, a, new, {**b, 'source_sha256': changed_id, 'snapshot_sha256': changed_sha})

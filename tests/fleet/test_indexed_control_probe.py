import hashlib
import json
from pathlib import Path

import pytest

from experiments.jobs import indexed_control_probe as probe


def test_probe_rejects_binding_or_child_differences() -> None:
    ctx = {'binding': {'selected': [3]}}
    data = {'affinity': [3], 'cpu': 3, 'cwd': '/source', 'imported_inbox': '/source/inbox.py',
            'imported_inbox_sha256': 'a' * 64}
    probe.check(data, data.copy(), ctx)
    with pytest.raises(ValueError, match='one CPU'):
        probe.check(data, {**data, 'affinity': [4]}, ctx)
    with pytest.raises(ValueError, match='one CPU'):
        probe.check(data, {**data, 'cwd': '/other'}, ctx)


def test_control_release_is_verified_separately_from_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / 'scientific'
    source.mkdir()
    release = tmp_path / 'release'
    inbox = release / 'experiments/jobs/inbox.py'
    inbox.parent.mkdir(parents=True)
    inbox.write_bytes(b'indexed_cpu_affinity_v1')
    sha = hashlib.sha256(inbox.read_bytes()).hexdigest()
    job = tmp_path / 'job'
    stage = job / 'smoke'
    stage.mkdir(parents=True)
    (job / 'run').mkdir()
    (job / 'spec.json').write_text(json.dumps({'cpu_affinity_index': 1}))
    (job / 'run/cpu-affinity.json').write_text(json.dumps({'index': 1, 'available': [4, 7], 'selected': [7]}))
    (stage / 'dispatch.json').write_text(json.dumps({'cpu_affinity': [7],
        'fleet_control': {'root': str(release), 'files': {'experiments/jobs/inbox.py': sha}}}))
    monkeypatch.setattr(Path, 'cwd', classmethod(lambda cls: source))
    assert probe.context(stage)['binding']['selected'] == [7]
    inbox.write_text('modified')
    with pytest.raises(ValueError, match='identity mismatch'):
        probe.context(stage)

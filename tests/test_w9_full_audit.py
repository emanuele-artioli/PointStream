import io
import json
import tarfile

import pytest

from experiments.jobs.inbox import digest
from experiments.tennis import w9_full_audit


def test_audit_checks_published_summary_and_manifest_before_validation(tmp_path, monkeypatch):
    job = tmp_path / 'job'
    stage = job / 'full'
    stage.mkdir(parents=True)
    manifest = tmp_path / 'selection.json'
    manifest.write_text(json.dumps({'window': 'nad000', 'points': [], 'controls': []}))
    hybrid = tmp_path / 'hybrid.tar'
    hybrid.write_bytes(b'hybrid')
    windows = tmp_path / 'windows.json'
    windows.write_text('{}')
    spec = {'staged_inputs': [{'name': 'repair_manifest', 'sha256': w9_full_audit.sha(manifest)},
                              {'name': 'windows', 'sha256': w9_full_audit.sha(windows)},
                              {'name': 'hybrid', 'sha256': w9_full_audit.sha(hybrid)}]}
    (job / 'spec.json').write_text(json.dumps(spec))
    summary = {'repair_manifest': json.loads(manifest.read_text()), 'ticks': list(range(0, 240, 3)),
               'inputs': {'repair_manifest': w9_full_audit.sha(manifest), 'hybrid': w9_full_audit.sha(hybrid)},
               'window': 'nad000', 'window_id': 'match/000@120'}
    (stage / 'execution.json').write_text(json.dumps({'exit_code': 0}))
    (stage / 'score-summary.json').write_text(json.dumps(summary))
    archive = stage / 'published.tar'
    with tarfile.open(archive, 'w') as t:
        data = json.dumps(summary).encode()
        info = tarfile.TarInfo('publish/score.json')
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))

    def fake_validate():
        import os
        from pathlib import Path
        Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps({'passed': True, 'checks': {'scores_complete': True}}))

    from types import SimpleNamespace
    monkeypatch.setattr(w9_full_audit.w1_score, 'validate', fake_validate)
    monkeypatch.setattr(w9_full_audit.w1_score, 'audit_repair_controls', lambda *args: {'control': {'full_lpips': 0.0}})
    monkeypatch.setattr(w9_full_audit.window_manifest, 'load', lambda _: [SimpleNamespace(name='nad000', id='match/000@120', safe_id='match__000@120')])
    output = tmp_path / 'audit'
    checksum = w9_full_audit.sha(archive)
    w9_full_audit.audit(job, manifest, hybrid, windows, output, checksum, digest(spec))
    assert json.loads((output / 'audit.json').read_text())['passed']
    assert w9_full_audit.sha(archive) == checksum
    with pytest.raises(FileExistsError):
        w9_full_audit.audit(job, manifest, hybrid, windows, output, checksum, digest(spec))
    with pytest.raises(ValueError, match='archive SHA'):
        w9_full_audit.audit(job, manifest, hybrid, windows, tmp_path / 'other', '0' * 64, digest(spec))
    (stage / 'score-summary.json').write_text(json.dumps({**summary, 'ticks': []}))
    with pytest.raises(ValueError, match='summaries disagree'):
        w9_full_audit.audit(job, manifest, hybrid, windows, tmp_path / 'different', checksum, digest(spec))

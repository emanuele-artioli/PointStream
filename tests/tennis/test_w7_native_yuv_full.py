import json
from pathlib import Path

import pytest

from experiments.tennis import w7_native_yuv_full as full


def test_full75_roles_reject_omissions_or_duplicates():
    inventory = json.loads(Path('docs/experiments/w7-inputs/mkv-53-inventory-20261105.json').read_text())
    assert len(full.rows(inventory)) == 75
    inventory['new_points'][1] = inventory['new_points'][0]
    with pytest.raises(ValueError, match='coverage|duplicate'):
        full.rows(inventory)


def test_full_forecast_accounts_repeated_smoke_and_remaining_charge():
    assert full.forecast(1, 138, 75, 60) == 177
    with pytest.raises(ValueError, match='exceeds'):
        full.forecast(5, 138, 75, 60)
    with pytest.raises(ValueError, match='charge'):
        full.forecast(1, 75, 75, 60)


def test_pilot_evidence_rejects_unfinished_status(tmp_path):
    with pytest.raises(ValueError, match='completed pilot'):
        full.pilot_evidence({'status': 'running'}, tmp_path, 50)


def test_full_worker_retains_all_75_distinct_logical_roles(tmp_path, monkeypatch):
    import os
    inventory = json.loads(Path('docs/experiments/w7-inputs/mkv-53-inventory-20261105.json').read_text())
    chosen = full.rows(inventory)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    stage = tmp_path / 'full'
    stage.mkdir()
    paths = {'ffmpeg': tmp_path / 'ffmpeg', 'ffprobe': tmp_path / 'ffprobe', 'dav1d': tmp_path / 'dav1d'}
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.setattr(full, 'context', lambda stage: (chosen, paths, {}, {}))
    monkeypatch.setattr(full, 'source', lambda row, paths: tmp_path / 'native')
    def pretend_prove(row, native, work, ffmpeg, ffprobe, **kwargs):
        work.mkdir()
        (work / 'proof.json').write_text(json.dumps({'window': row['window'], 'name': row['name']}))
    monkeypatch.setattr(full.bp, 'prove', pretend_prove)
    monkeypatch.setattr(full, 'independent_check', lambda row, *args: {
        'delivered_mkv_sha256': 'mkv', 'reconstructed_ivf_sha256': 'returned'})
    full.run(stage, scratch)
    coverage = json.loads((scratch / 'publish/coverage.json').read_text())['completed']
    assert len(coverage) == 75 and [row['index'] for row in coverage] == list(range(75))
    assert len({(row['window'], row['name']) for row in coverage}) == 75
    assert not os.path.exists(scratch / 'publish/role-075')


def test_full_saved_artifact_audit_checks_all_bytes_and_native_receipts(tmp_path, monkeypatch):
    import tarfile
    from test_w7_byteproof import fake_ivf
    inventory = json.loads(Path('docs/experiments/w7-inputs/mkv-53-inventory-20261105.json').read_text())
    selected = full.rows(inventory)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    stage = tmp_path / 'full'
    stage.mkdir()
    paths = {'ffmpeg': tmp_path / 'ffmpeg', 'ffprobe': tmp_path / 'ffprobe', 'dav1d': tmp_path / 'dav1d'}
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.setattr(full, 'context', lambda stage: (selected, paths, {}, {}))
    monkeypatch.setattr(full, 'source', lambda row, paths: tmp_path / 'native')
    def pretend_prove(row, native, work, ffmpeg, ffprobe, **kwargs):
        work.mkdir()
        fake_ivf(work / 'decoder-scratch.ivf')
        (work / 'delivered.mkv').write_bytes(row['name'].encode())
        record = {**row, 'profile': full.profile.PROFILE, 'native_yuv_proven': True,
            'native_decoder_sha256': full.profile.DAV1D_SHA, 'native_yuv_sha256': row['historical_native_yuv_sha256'],
            'ordered_payloads_equal': True, 'timestamps_equal': True,
            'delivered_mkv_bytes': (work / 'delivered.mkv').stat().st_size,
            'delivered_mkv_sha256': full.bp.digest(work / 'delivered.mkv'),
            'reconstructed_ivf_sha256': full.bp.digest(work / 'decoder-scratch.ivf')}
        (work / 'proof.json').write_text(json.dumps(record))
    monkeypatch.setattr(full.bp, 'prove', pretend_prove)
    monkeypatch.setattr(full, 'independent_check', lambda row, native, work, *args: {
        'delivered_mkv_sha256': full.bp.digest(work / 'delivered.mkv'),
        'reconstructed_ivf_sha256': full.bp.digest(work / 'decoder-scratch.ivf')})
    full.run(stage, scratch)
    archive = tmp_path / 'published.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(scratch / 'publish', arcname='publish')
    assert full.audit_full(archive, full.bp.digest(archive), inventory)['logical_roles'] == 75
    (scratch / 'publish/role-074/delivered.mkv').write_bytes(b'tampered')
    bad = tmp_path / 'bad.tar'
    with tarfile.open(bad, 'w') as tar:
        tar.add(scratch / 'publish', arcname='publish')
    with pytest.raises(ValueError, match='billed MKV'):
        full.audit_full(bad, full.bp.digest(bad), inventory)

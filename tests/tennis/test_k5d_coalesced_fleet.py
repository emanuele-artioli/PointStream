from pathlib import Path

import pytest

from experiments.tennis import k5d_coalesced_fleet as adapter


def test_score_spec_requires_real_checkpoint_artifacts(tmp_path: Path) -> None:
    entry = {'remote': '/home/itec/emanuele/Datasets/pointstream-data/missing.tar',
             'archive_sha256': '0' * 64, 'checkpoint_sha256': '1' * 64,
             'local': str(tmp_path / 'missing.tar')}
    with pytest.raises((ValueError, FileNotFoundError)):
        adapter.score_spec({}, small=entry, full=entry, reference16_local=tmp_path / 'ref16',
                           reference48_local=tmp_path / 'ref48', deadline='2027-01-01T00:00:00Z')


def test_score_spec_has_bounded_all_point_campaign(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(adapter.cf, 'checkpoint_archive', lambda name, *args: {'name': name})
    monkeypatch.setattr(adapter.cf, 'asset', lambda name, *args, **kwargs: {'name': name})
    monkeypatch.setattr(adapter.cf, 'common_spec', lambda base, extra, deadline: {})
    entry = {'remote': 'x', 'archive_sha256': '0' * 64, 'checkpoint_sha256': '1' * 64, 'local': 'x'}
    spec = adapter.score_spec({'arguments': ['--frames', '{frames}']}, small=entry, full=entry,
                              reference16_local=tmp_path, reference48_local=tmp_path, deadline='deadline')
    assert spec['smoke']['seconds'] == 200
    assert spec['full']['seconds'] == 420
    assert spec['budget_seconds'] == 650
    assert '--codec' not in spec['arguments']
    assert '{staged:reference16}' in spec['arguments']
    assert '{staged:reference48}' in spec['arguments']

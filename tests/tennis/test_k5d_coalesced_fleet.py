from pathlib import Path
from typing import Any

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


def test_resized_scoring_caps_and_forecast_remain_strict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json

    monkeypatch.setattr(adapter.cf, 'checkpoint_archive', lambda name, *args: {'name': name})
    monkeypatch.setattr(adapter.cf, 'asset', lambda name, *args, **kwargs: {'name': name})
    monkeypatch.setattr(adapter.cf, 'common_spec', lambda base, extra, deadline: {})
    entry = {'remote': 'x', 'archive_sha256': '0' * 64, 'checkpoint_sha256': '1' * 64, 'local': 'x'}
    kwargs: dict[str, Any] = dict(small=entry, full=entry, reference16_local=tmp_path, reference48_local=tmp_path, deadline='deadline')
    spec = adapter.score_spec({'arguments': []}, **kwargs, smoke_seconds=220, full_seconds=750, budget_seconds=1100)
    assert (spec['smoke']['seconds'], spec['full']['seconds'], spec['budget_seconds']) == (220, 750, 1100)
    for bad in (True, 751, 0):
        with pytest.raises(ValueError, match='bounded'):
            adapter.score_spec({'arguments': []}, **kwargs, smoke_seconds=220, full_seconds=bad, budget_seconds=1100)
    stage = tmp_path / 'smoke'
    stage.mkdir()
    (stage / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 174.02550172805786}))
    (stage / 'dispatch.json').write_text(json.dumps({'campaign_clock': {'started': 1, 'phase_started': 2, 'budget_end': 1100}}))
    (tmp_path / 'spec.json').write_text(json.dumps({'full': {'seconds': 750}}))
    assert adapter.score_forecast(stage, 200)['estimate_seconds'] == pytest.approx(672.595631480217)
    (tmp_path / 'spec.json').write_text(json.dumps({'full': {'seconds': 420}}))
    with pytest.raises(ValueError, match='forecast'):
        adapter.score_forecast(stage, 200)
    (tmp_path / 'spec.json').write_text(json.dumps({'full': {'seconds': 751}}))
    with pytest.raises(ValueError, match='stage cap'):
        adapter.score_forecast(stage, 200)

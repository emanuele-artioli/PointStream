import json

import pytest

from experiments.tennis import k5d_fleet


def timings(path, seconds=50, end=3000):
    (path / 'execution.json').write_text(json.dumps({'seconds': seconds, 'exit_code': 0}))
    (path / 'dispatch.json').write_text(json.dumps({'campaign_clock': {
        'started': 1000, 'phase_started': 1010, 'budget_end': end}}))


def test_forecast_requires_full_188_frame_run_to_fit(tmp_path):
    timings(tmp_path)
    result = k5d_fleet.projection(tmp_path, 1100)
    assert result['projected_full_seconds_including_60s'] == pytest.approx(794.375)
    timings(tmp_path, seconds=160)
    with pytest.raises(ValueError, match='forecast'):
        k5d_fleet.projection(tmp_path, 1100)
    timings(tmp_path, seconds=50, end=1800)
    with pytest.raises(ValueError, match='forecast'):
        k5d_fleet.projection(tmp_path, 1100)


def test_forecast_rejects_failed_smoke(tmp_path):
    timings(tmp_path)
    (tmp_path / 'execution.json').write_text(json.dumps({'seconds': 10, 'exit_code': 1}))
    with pytest.raises(ValueError, match='inconsistent'):
        k5d_fleet.projection(tmp_path, 1100)


def test_tar_tree_identity_matches_extracted_files_and_detects_tamper(tmp_path):
    import tarfile

    from experiments.tennis.k5d_source_oracle import tree_sha

    source = tmp_path / 'source'
    source.mkdir()
    (source / 'model.py').write_text('x = 1\n')
    (source / 'config.yaml').write_text('a: 1\n')
    (source / 'ignored.txt').write_text('not in digest')
    archive = tmp_path / 'code.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(source, arcname='.')
    assert k5d_fleet.archive_tree_sha(archive) == tree_sha(source)
    (source / 'model.py').write_text('x = 2\n')
    assert k5d_fleet.archive_tree_sha(archive) != tree_sha(source)


def test_snapshot_head_from_fleet_dispatch(tmp_path):
    code = {'git_head': 'a' * 40, 'source_sha256': 'b' * 64,
            'snapshot_sha256': 'c' * 64, 'spec_sha256': 'd' * 64}
    (tmp_path / 'dispatch.json').write_text(json.dumps({'code': code}))
    assert k5d_fleet.snapshot_head(tmp_path) == 'a' * 40
    code['git_head'] = 'not-a-sha'
    (tmp_path / 'dispatch.json').write_text(json.dumps({'code': code}))
    with pytest.raises(ValueError, match='Git HEAD'):
        k5d_fleet.snapshot_head(tmp_path)

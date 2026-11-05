from pathlib import Path
from typing import Any

import pytest

from experiments.tennis import w1_roi as roi, w7_shared_score as worker


def points() -> list[dict]:
    controls = [{'window': w, 'name': f'control-s{s}', 'kind': 'control', 'scale': s, 'crf': 63, 'offset': 0}
                for w in worker.WINDOWS for s in (8, 4, 2)]
    new = [{'window': w, 'name': f'new-{c}', 'kind': 'new', 'scale': 2, 'crf': c, 'offset': -20}
           for w in ('med000', 'med019') for c in (47, 70)]
    return controls + new


def test_required_22_points_and_fixed_controls() -> None:
    rows = points()
    worker.cohort(rows)
    rows[0]['scale'] = 2
    with pytest.raises(ValueError, match='cohort'):
        worker.cohort(rows)


def test_pilot_cost_includes_six_contexts_22_points_and_pilot() -> None:
    cost = {'setup': 20, 'source': 10, 'points': [7, 9], 'pilot_total': 55}
    pilot = {'passed': True, 'window': 'med000', 'point_count': 2, 'cost': {'wall': cost, 'cpu': cost}}
    source_bounds = {w: {'wall': 10, 'cpu': 10} for w in worker.WINDOWS}
    source_bounds['al028'] = {'wall': 15, 'cpu': 15}
    opts: dict[str, Any] = dict(source_bounds=source_bounds, gpu_remaining=1000, cpu_remaining=1000,
                gpu_staging=12, cpu_staging=13, gpu_reserve=20, cpu_reserve=30)
    values = worker.forecast(pilot, **opts)
    assert values['wall_seconds'] == pytest.approx(1.5 * (20 + 65 + 22 * 9) + 12 + 20 + 55)
    opts['gpu_remaining'] = 100
    with pytest.raises(ValueError, match='budget'):
        worker.forecast(pilot, **opts)
    del source_bounds['al028']
    with pytest.raises(ValueError, match='six separately'):
        worker.forecast(pilot, **opts)


def test_point_requires_native_yuv_validation_and_scorer_bridge(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / 'point.mkv'
    path.write_bytes(b'mkv')
    monkeypatch.setitem(worker.w7_analysis.W7_INPUTS, 'al', {'score': {'job': 'historical'}})
    bridge = {'repair_w1_score_sha256': 'scorer', 'historical_w1_score_sha256': 'historical-sha',
              'original_score_jobs': [{'job': 'historical', 'files': {'experiments/tennis/w1_score.py': 'historical-sha'}}]}
    row = {'window': 'al', 'name': 'x', 'scale': 2, 'offset': 0, 'crf': 63, 'kind': 'control', 'score_job': 'historical',
           'score_archive_sha256': 's', 'encode_archive_sha256': 'e', 'ivf_member': 'publish/a.ivf',
           'original_ivf_sha256': 'original', 'historical_native_yuv_sha256': 'decoded'}
    proof: dict[str, Any] = {**row, 'fps': str(roi.AL.fps), 'frames': 240, 'profile': worker.profile.PROFILE, 'exact_ivf_bytes': False,
             'ordered_payloads_equal': True, 'timestamps_equal': True, 'native_yuv_proven': True,
             'native_yuv_sha256': 'decoded', 'native_decoder_sha256': worker.profile.DAV1D_SHA,
             'reconstructed_ivf_sha256': 'different-header', 'delivered_mkv_sha256': worker.old.digest(path),
             'delivered_mkv_bytes': 3, 'wire_kbps': 3 * 8 * roi.AL.fps / 240 / 1000, 'source_and_scorer': {'source_archive_sha256': roi.AL.source_sha,
             'masks_archive_sha256': 'mask', 'hybrid_archive_sha256': 'hybrid', 'lpips_sha256': roi.WEIGHTS['lpips'],
             'dists_sha256': roi.WEIGHTS['dists'], 'ticks': roi.TICKS, 'score_source_identity': None}}
    row['source_and_scorer'] = dict(proof['source_and_scorer'])
    validation = {'checks': {'reextract': True, 'payloads': True, 'timestamps': True, 'native_yuv': True},
                  'source_ivf_sha256': 'original', 'returned_ivf_sha256': 'different-header',
                  'delivered_mkv_sha256': worker.old.digest(path), 'native_yuv_sha256': 'decoded',
                  'native_decoder_sha256': worker.profile.DAV1D_SHA}
    worker.check_point(row, proof, validation, path, roi.AL, roi.AL.source_sha, 'mask', 'hybrid', 'scorer', bridge)
    validation['native_yuv_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='profile'):
        worker.check_point(row, proof, validation, path, roi.AL, roi.AL.source_sha, 'mask', 'hybrid', 'scorer', bridge)
    validation['native_yuv_sha256'] = 'decoded'
    with pytest.raises(ValueError, match='scorer identity'):
        worker.check_point(row, proof, validation, path, roi.AL, roi.AL.source_sha, 'mask', 'hybrid', 'other', bridge)


def test_provisional_inventory_binds_actual_22_and_fails_closed_on_missing_scorer() -> None:
    path = Path(__file__).resolve().parents[2] / 'docs/experiments/w7-inputs/mkv-53-inventory-20261105.json'
    if not path.exists():
        pytest.skip('provisional inventory is independently produced')
    rows = worker.inventory(path, '867d9fff8974f4ec5320c9b830138d4aac2ee22c285d7de45aebb04dff3bab11')
    assert len(rows) == 22
    assert all(r['source_and_scorer']['score_source_identity'] is None for r in rows)


def test_pilot_builder_refuses_missing_actual_artifacts() -> None:
    from experiments.tennis import w7_shared_fleet

    with pytest.raises(ValueError, match='invalid inclusive pilot bounds'):
        w7_shared_fleet.pilot_spec(assets={}, deadline='2030-01-01T00:00:00Z', smoke_seconds=500,
                                   campaign_seconds=550, gpu_budget_seconds=350, cpu_budget_seconds=450)


def test_point_assets_must_be_members_of_bound_proof_publication(tmp_path: Path) -> None:
    import hashlib
    import io
    import json
    import tarfile
    row = {'window': 'med000', 'name': 'test', 'proof_asset': 'proof', 'validation_asset': 'validation', 'mkv_asset': 'mkv'}
    blobs = {'proof': b'{"valid":true}', 'validation': b'{"checks":{}}', 'mkv': b'actual matroska'}
    inputs = {}
    for name, data in blobs.items():
        path = tmp_path / name
        path.write_bytes(data)
        inputs[name] = path
    coverage = {'completed': [{'window': 'med000', 'name': 'test', 'index': 0,
        'proof_sha256': hashlib.sha256(blobs['proof']).hexdigest(),
        'independent_validation_sha256': hashlib.sha256(blobs['validation']).hexdigest()}]}
    archive = tmp_path / 'publication.tar'
    with tarfile.open(archive, 'w') as tar:
        for name, data in [('publish/coverage.json', json.dumps(coverage).encode()),
                           ('publish/role-000/proof.json', blobs['proof']),
                           ('publish/role-000/independent-validation.json', blobs['validation']),
                           ('publish/role-000/delivered.mkv', blobs['mkv'])]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    worker.check_proof_publication(archive, worker.old.digest(archive), [row], inputs)
    inputs['mkv'].write_bytes(b'different')
    with pytest.raises(ValueError, match='do not match'):
        worker.check_proof_publication(archive, worker.old.digest(archive), [row], inputs)


def test_pilot_spec_uses_actual_whole_argument_stage_placeholder(tmp_path, monkeypatch):
    import json
    from experiments.tennis import w7_shared_fleet as fleet
    names = ('inventory', 'bindings', 'windows', 'state', 'lpips', 'dists', 'med000_source',
             'med000_masks', 'med000_hybrid', 'byte_publication', 'p1', 'v1', 'm1', 'p2', 'v2', 'm2')
    assets = {}
    for name in names:
        path = tmp_path / name
        path.write_text('{}')
        assets[name] = {'local': str(path), 'remote': '/data/' + name, 'sha256': 'a' * 64}
    assets['byte_publication']['sha256'] = '55739efe3cbf6937b6251e26c2ed1322663f8d1d2d1ee5f36bfb1ace593cdb59'
    rows = [{'window': 'med000', 'kind': 'control', 'scale': 8, 'crf': 63, 'proof_asset': 'p1',
             'validation_asset': 'v1', 'mkv_asset': 'm1'},
            {'window': 'med000', 'kind': 'new', 'scale': 2, 'crf': 70, 'proof_asset': 'p2',
             'validation_asset': 'v2', 'mkv_asset': 'm2'}]
    monkeypatch.setattr(fleet.cf, 'asset', lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, 'inventory', lambda *args: rows)
    monkeypatch.setattr(worker, 'bound_rows', lambda r, b: r)
    monkeypatch.setattr(roi, 'preload_windows', lambda *args: None)
    monkeypatch.setitem(roi.WINDOWS, 'med000', roi.AL)
    monkeypatch.setattr(worker, 'check_point', lambda *args: None)
    monkeypatch.setattr(worker, 'materialize_proofs', lambda archive, sha, chosen, destination: {name: tmp_path / name for name in ('p1', 'v1', 'm1', 'p2', 'v2', 'm2')})
    monkeypatch.setattr(worker, 'check_proof_publication', lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, 'source_bridge', lambda: {})
    monkeypatch.setattr(worker.old, 'digest', lambda path: roi.WEIGHTS[path.name] if path.name in roi.WEIGHTS else 'f' * 64)
    result = fleet.pilot_spec(assets=assets, deadline='2099-01-01T00:00:00Z', smoke_seconds=100,
                              campaign_seconds=150, gpu_budget_seconds=200, cpu_budget_seconds=250)
    assert result['scale']['items'] == {'smoke': 2, 'full': 0}
    assert result['arguments'][result['arguments'].index('--items') + 1] == '{items}'
    assert json.loads((tmp_path / 'p1').read_text()) == {}


def test_materialize_checks_full_coverage_and_selected_bytes(tmp_path: Path) -> None:
    import hashlib
    import io
    import json
    import tarfile
    row = {'window': 'med000', 'name': 'control', 'proof_asset': 'p', 'validation_asset': 'v', 'mkv_asset': 'm'}
    proof, valid, mkv = b'{}', b'{"checks":{}}', b'matroska'
    coverage = [{'window': 'med000', 'name': 'control', 'index': 0,
                 'proof_sha256': hashlib.sha256(proof).hexdigest(),
                 'independent_validation_sha256': hashlib.sha256(valid).hexdigest()}]
    coverage += [{'window': f'w{i}', 'name': 'n', 'index': i} for i in range(1, 75)]
    archive = tmp_path / 'delivery.tar'
    with tarfile.open(archive, 'w') as tar:
        for name, data in [('publish/coverage.json', json.dumps({'completed': coverage}).encode()),
                           ('publish/role-000/proof.json', proof),
                           ('publish/role-000/independent-validation.json', valid),
                           ('publish/role-000/delivered.mkv', mkv)]:
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    files = worker.materialize_proofs(archive, worker.old.digest(archive), [row], tmp_path / 'selected')
    assert files['m'].read_bytes() == mkv
    with pytest.raises(ValueError, match='hash differs'):
        worker.materialize_proofs(archive, '0' * 64, [row], tmp_path / 'bad')

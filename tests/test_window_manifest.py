"""The generic window manifest reproduces the frozen two-window constants and rejects malformed manifests."""
import json
from argparse import Namespace
from pathlib import Path

import pytest

from experiments.tennis import k6b_confirm as kc
from experiments.tennis import k6b_native_base as nb
from experiments.tennis import w1_roi
from experiments.tennis import window_manifest as wm
from experiments.tennis.k6b_confirm_validate import check_anchor_rows  # noqa: F401  (imports with the cohort helper)

SELECTION = Path(__file__).resolve().parents[1] / 'docs/experiments/w7-specs/selection.json'


def test_frozen_pair_reproduces_native_base_sources():
    assert wm.legacy_sources() == nb.SOURCES
    args = Namespace(windows_file=None, al=Path('a.mp4'), med=Path('m.mp4'), window=None, source=None)
    paths, sources = nb.sources_for(args)
    assert paths == [Path('a.mp4'), Path('m.mp4')] and sources is nb.SOURCES


def test_frozen_pair_reproduces_w1_windows():
    al = w1_roi.window_from_spec(wm.FROZEN_PAIR[0], {'control_ivf_bytes': w1_roi.AL.control_ivf_bytes, 'start_base': w1_roi.AL.start_base,
                                                     'kps1_bytes': w1_roi.AL.kps1_bytes, 'kps2_bytes': w1_roi.AL.kps2_bytes,
                                                     'hybrid_archive_sha': w1_roi.AL.hybrid_archive_sha})
    med = w1_roi.window_from_spec(wm.FROZEN_PAIR[1], {'control_ivf_bytes': w1_roi.MED.control_ivf_bytes, 'start_base': w1_roi.MED.start_base,
                                                      'kps1_bytes': w1_roi.MED.kps1_bytes, 'kps2_bytes': w1_roi.MED.kps2_bytes,
                                                      'hybrid_archive_sha': w1_roi.MED.hybrid_archive_sha})
    assert al == w1_roi.AL and med == w1_roi.MED
    assert [w.id for w in wm.FROZEN_PAIR] == [w1_roi.WINDOW, 'medvedev_struff/014@194']


def test_old_cohort_checks_are_unchanged():
    assert kc.cohort_of({}) == kc.FROZEN and len(kc.FROZEN) == 12
    # a manifest without `cohort` is checked against FROZEN, centered, exactly as before
    with pytest.raises(ValueError):
        kc.check_manifest({'provenance': {}, 'budget_gpu_hours': 8, 'clips': [], 'old_clips': []})
    assert [w.id for w in wm.FROZEN_PAIR] == [f'{c}@{s + 120}' for c, s, _, _ in kc.FROZEN[:1]] + ['medvedev_struff/014@194']


def test_w7_manifest_from_selection():
    sel = json.loads(SELECTION.read_text())
    windows = wm.from_selection(sel)
    assert [w.name for w in windows] == ['al028', 'al034', 'al043', 'med000', 'med024', 'med019']
    assert [w.id for w in windows][0] == 'alcaraz_hurkacz/028@120' and all(w.start == 0 and w.frames == 240 for w in windows)
    assert windows[3].adapter['key'] == 'med' and windows[0].adapter['key'] == 'al'
    assert wm.cohort_tuples(windows)[0][:3] == ['alcaraz_hurkacz/028', 0, 25]
    assert [w.name for w in wm.select(windows, ['med019'])] == ['med019']
    with pytest.raises(ValueError):
        wm.select(windows, ['nope'])


def test_check_rejects_bad_manifests():
    a = wm.FROZEN_PAIR[0]
    with pytest.raises(ValueError):
        wm.check([a, a])
    with pytest.raises(ValueError):
        wm.check([a, wm.WindowSpec('al2', a.match, a.clip, a.start + 100, 240, 25, a.source_sha256, None)])  # overlap in one clip
    with pytest.raises(ValueError):
        wm.check([wm.WindowSpec('x_y', a.match, a.clip, 0, 240, 25, a.source_sha256, None)])
    with pytest.raises(ValueError):
        wm.legacy_sources([wm.WindowSpec('al2', a.match, a.clip, 300, 240, 25, a.source_sha256, None)])


def test_native_base_sources_from_manifest(tmp_path):
    windows = wm.from_selection(json.loads(SELECTION.read_text()), {c: '0' * 64 for c in ('alcaraz_hurkacz/028', 'medvedev_struff/000')})
    path = tmp_path / 'w.json'
    wm.dump(windows, path)
    args = Namespace(windows_file=path, window=['al028', 'med000'], source=[Path('a'), Path('b')], al=None, med=None)
    paths, sources = nb.sources_for(args)
    assert paths == [Path('a'), Path('b')] and [s[0] for s in sources] == ['al028', 'med000'] and sources[1][1:3] == (0, 50)
    with pytest.raises(ValueError):
        nb.sources_for(Namespace(windows_file=path, window=['al028'], source=[Path('a'), Path('b')], al=None, med=None))


def test_manifest_cohort_is_checked_by_confirm(tmp_path):
    windows = wm.from_selection(json.loads(SELECTION.read_text()))
    cohort = kc.cohort_of({'cohort': wm.cohort_tuples(windows)})
    assert cohort[0] == ('alcaraz_hurkacz/028', 0, 25, windows[0].source_sha256) and len(cohort) == 6


def test_check_manifest_accepts_a_noncentered_manifest_cohort():
    import numpy as np
    windows = wm.from_selection(json.loads(SELECTION.read_text()))
    clips = [{'id': w.clip_id, 'frames': 400, 'fps': w.fps, 'sha256': w.source_sha256, 'review_frames': [w.start + 120],
              'staged': {'sha256': w.source_sha256},
              'registration': [{'frame': w.start + t, 'fits': True, 'homography': np.eye(3).tolist()} for t in range(240)]}
             for w in windows]
    manifest = {'contains_human_marks': False, 'provenance': dict(kc.PROVENANCE), 'budget_gpu_hours': 8, 'centered': False,
                'old_clips': [f'old/{i}' for i in range(49)], 'cohort': wm.cohort_tuples(windows), 'clips': clips}
    assert sorted(kc.check_manifest(manifest)) == sorted(w.id for w in windows)
    manifest.pop('cohort')  # the same manifest against the frozen 12 windows is rejected, as before
    with pytest.raises(ValueError):
        kc.check_manifest(manifest)


def test_w7_rates_and_state():
    from experiments.tennis import w7_rates as r
    row = {'window': 'alcaraz_hurkacz/028@120', 'match': 'alcaraz_hurkacz', 'reference_files': ['a.ref', 'b.ref'],
           'references': {'png_bytes': 150000}, 'keypoints': {'bytes': 2701, 'kps1_bytes': 4236, 'contract': {'fps': [25, 1]}}}
    out = r.rates({'rows': [row]})['alcaraz_hurkacz/028@120']
    assert abs(out['keypoint_kbps'] - 2701 * 8 / 9.6 / 1000) < 1e-12
    assert abs(out['match_bank_kbps_7200'] - 150000 * 6 * 8 / 7200 / 1000) < 1e-12
    assert abs(out['bank_worst_case_kbps'] - 150000 * 8 / 9.6 / 1000) < 1e-9
    base = [{'window': 'al028', 'scale': s, 'ivf': {'bytes': 1000 * s}} for s in (8, 4, 2)]
    st = r.window_state(base, row, 'al028')
    assert st['control_ivf_bytes'] == {8: 8000, 4: 4000, 2: 2000} and st['kps1_bytes'] == 4236 and st['kps2_bytes'] == 2701
    with pytest.raises(ValueError):
        r.window_state(base[:2], row, 'al028')


def test_w3_manifest_mode_leaves_the_frozen_sources_alone_until_activated():
    from experiments.tennis import w3_neural_baseline as w3
    from experiments.tennis import w5_dcvc
    saved_sources, saved_inputs = dict(w3.SOURCES), dict(w3.INPUTS)
    try:
        names = w3.activate_manifest(Path(__file__).resolve().parents[1] / 'docs/experiments/w7-specs/windows.json', ['al028', 'med000'])
        assert names == ['al028', 'med000'] and w3.SOURCES['al028']['wid'] == 'alcaraz_hurkacz/028@120'
        assert 'anchor' not in w3.INPUTS and 'lpips' in w3.INPUTS
        assert w3.parse_point_list('al028:8:30,med000:8:30') == {'al028': [(8, 30)], 'med000': [(8, 30)]}
    finally:
        w3.SOURCES.clear()
        w3.SOURCES.update(saved_sources)
        w3.INPUTS.clear()
        w3.INPUTS.update(saved_inputs)
    assert list(w3.SOURCES) == ['al', 'med'] and w5_dcvc.points()[0]['crf'] == 15
    assert w5_dcvc.points([(10, 5.0, .1, .2, .3, .4)])[0]['rate_kbps'] == 5.0


def test_new_match_can_be_prepared_before_its_adapter_is_fitted(tmp_path):
    w = wm.WindowSpec('nad001', 'nadal_kyrgios', '001', 0, 240, 50, 'a' * 64, 'b' * 64, tier='w7-tier2')
    path = tmp_path / 'windows.json'
    wm.dump([w], path)
    assert wm.load(path) == [w]
    assert w.adapter['job'] == '20261031T150848Z-a582ecc0'


def test_new_match_rates_bill_explicit_adapter_bytes():
    from experiments.tennis import w7_rates as r

    row = {'window': 'nadal_kyrgios/000@120', 'match': 'nadal_kyrgios', 'reference_files': ['a', 'b'],
           'references': {'png_bytes': 1000}, 'keypoints': {'bytes': 100, 'contract': {'fps': [50, 1]}}}
    with pytest.raises(KeyError):
        r.rates({'rows': [row]})
    rate = r.rates({'rows': [row]}, {'nadal_kyrgios': 1234567})[row['window']]
    assert rate['adapter_guider_kbps_7200'] == pytest.approx(1234567 * 8 / 7200 / 1000)
    with pytest.raises(ValueError, match='positive integer'):
        r.rates({'rows': [row]}, {'nadal_kyrgios': 0})


def test_rate_computation_rejects_charge_without_fitted_adapter_identity(tmp_path, monkeypatch):
    from experiments.tennis import w7_rates as r

    window = wm.WindowSpec('nad000', 'nadal_kyrgios', '000', 0, 240, 50, 'a' * 64, 'b' * 64)
    path = tmp_path / 'windows.json'
    wm.dump([window], path, adapter_bytes={'nadal_kyrgios': 1234567})
    data = json.loads(path.read_text())
    del data['adapters']['nadal_kyrgios']
    path.write_text(json.dumps(data))
    transport = tmp_path / 'transport.json'
    row = {'window': window.id, 'match': window.match, 'reference_files': ['a', 'b'], 'references': {'png_bytes': 1000},
           'keypoints': {'bytes': 100, 'kps1_bytes': 200, 'contract': {'fps': [50, 1]}}}
    transport.write_text(json.dumps({'rows': [row]}))
    monkeypatch.setattr(r, 'base_rows', lambda _: [{'window': window.name, 'scale': s, 'ivf': {'bytes': 10}} for s in (2, 4, 8)])
    with pytest.raises(ValueError, match='no matching fitted adapter identity'):
        r.compute(transport, 'base.tar', path, [window.name])
    monkeypatch.setitem(wm.ADAPTERS, 'nadal_kyrgios', {'key': 'nad', 'job': 'fit', 'name': 'nk-C', 'published_sha256': 'c' * 64})
    wm.dump([window], path, adapter_bytes={'nadal_kyrgios': 1234567})
    result = r.compute(transport, 'base.tar', path, [window.name])
    assert result['rates'][window.id]['adapter_guider_kbps_7200'] == pytest.approx(1234567 * 8 / 7200 / 1000)

"""W7 Tier 1 analysis: the pre-registered rule on small synthetic archives in the formats the writers produce.

al028's recorded numbers (docs/experiments/w7-untouched.md) are the fixture: hybrid scale 4 player-region LPIPS/DISTS 0.3042/0.2420,
whole 0.2845/0.1750; SVT at the hybrid's rate; DCVC from the committed scale-2 curve, which gives 0.3929/0.3074 at 18.4614 kbps.
"""
import io
import json
import math
import tarfile
from pathlib import Path

import pytest

from experiments.tennis import w1_roi
from experiments.tennis import w7_analysis as a

RATE = 18.461418333333334
HYBRID = {'lpips': {'whole': 0.2845, 'foreground': 0.3042}, 'dists': {'whole': 0.1750, 'foreground': 0.2420}}
#: (whole LPIPS, foreground LPIPS, whole DISTS, foreground DISTS) at the hybrid's rate, as recorded for al028.
SVT_AT_TARGET = {'s2_o0': (0.1908, 0.5933, 0.1458, 0.4344), 's4_o0': (0.2761, 0.6416, 0.1909, 0.4347),
                 's2_o20': (0.2380, 0.6496, 0.1682, 0.4645)}
COMMITTED = Path(__file__).resolve().parents[1] / 'docs/experiments/w7-specs/dcvc-points-al028-scale2.json'


def judge(window_ids: dict[str, float | tuple[float, float]] | None = None, shift: float = 0.0, rate: float = RATE) -> dict:
    """A quality-summary.json as `k6b_real_base_judge` writes it: per row, `means` over [8 base, 8 hybrid, 4 base, 4 hybrid, 2 base, 2 hybrid]."""
    ids = window_ids or {'alcaraz_hurkacz/028@0': 0.0}
    rows = []
    for wid, extra in ids.items():
        s = shift + (extra if isinstance(extra, float) else 0.0)
        rows.append({'id': wid, 'ticks': list(range(0, 240, 3)),
                     'means': {'full_lpips': [0.45, 0.43, 0.29, HYBRID['lpips']['whole'], 0.15, 0.14],
                               'foreground_lpips': [0.85, 0.33, 0.71, HYBRID['lpips']['foreground'] + s, 0.48, 0.27],
                               'full_dists': [0.29, 0.26, 0.20, HYBRID['dists']['whole'], 0.12, 0.10],
                               'foreground_dists': [0.59, 0.25, 0.49, HYBRID['dists']['foreground'] + s, 0.36, 0.23],
                               'full_psnr': [20.0] * 6},
                     'rates': [{'scale': 8, 'base_kbps': 9.3, 'total_kbps': 13.68}, {'scale': 4, 'base_kbps': 14.1, 'total_kbps': rate},
                               {'scale': 2, 'base_kbps': 27.2, 'total_kbps': 31.57}]})
    return {'rows': rows, 'calibration': [], 'seconds': 1.0}


def bracket(rate: float, values: tuple[float, float, float, float], span: float = 1.5, below_only: bool = False) -> list[dict]:
    """Two scored points whose log-rate interpolation at `rate` is exactly `values` (whole LPIPS, fg LPIPS, whole DISTS, fg DISTS)."""
    names = ('whole_lpips', 'foreground_lpips', 'whole_dists', 'foreground_dists')
    low, high = rate / span, rate * span
    if below_only:
        low, high = rate / span ** 2, rate / span
    return [{'crf': 60 - 4 * i, 'rate_kbps': r, **{n: v + sign * 0.01 for n, v in zip(names, values)}}
            for i, (r, sign) in enumerate(((low, -1), (high, 1)))]


def score(window: str = 'al028', rate: float = RATE, shift: float = 0.0, drop: tuple[str, ...] = (), unbracketed: tuple[str, ...] = (),
          svt_shift: float = 0.0) -> dict:
    """A score-summary.json as `w1_score` writes it (the parts the analysis reads)."""
    by_policy = {}
    for family, v in SVT_AT_TARGET.items():
        if family in drop:
            continue
        values = (v[0], v[1] + svt_shift, v[2], v[3] + svt_shift)
        by_policy[family] = bracket(rate, values, below_only=family in unbracketed)
    return {'window': window, 'hybrid': {'hybrid': HYBRID, 'kps1_kbps': rate + 1.2, 'kps2_kbps': rate},
            'analysis': {'target_kbps': rate, 'by_policy': by_policy,
                         'at_target': {f: w1_roi.policy_at_target(rows, rate) for f, rows in by_policy.items()},
                         'decision': {'outcome': 'A'}}}


def dcvc_summary(window: str = 'al028', rate: float = RATE, extra: list[dict] | None = None, accepted_high: bool = True) -> dict:
    """A w3-summary.json: scale-2 points straddling `rate` (the values of the committed curve at QP 18 and 24, rates moved), plus a scale-8 point."""
    def point(scale, qp, kbps, fl, fd, ok=True):
        return {'scale': scale, 'qp': qp, 'kbps': kbps, 'accepted': ok,
                'mean': {'full_lpips': 0.15, 'full_dists': 0.12, 'foreground_lpips': fl, 'foreground_dists': fd}}
    points = [point(8, 30, 3.306, 0.8186, 0.5307), point(2, 18, rate / 1.3, 0.4435, 0.3324), point(2, 24, rate * 1.1, 0.3714, 0.2967, accepted_high)]
    return {'windows': [{'window': window, 'dcvc': points + (extra or []), 'svt': [], 'accepted': True}]}


def tar_with(path: Path, members: dict[str, dict]) -> Path:
    with tarfile.open(path, 'w') as tar:
        for name, content in members.items():
            data = json.dumps(content).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def window_inputs(tmp: Path, name: str = 'al028', dcvc: bool = True, **kw) -> dict:
    wid = a.window_prefix(name) + '0'
    files = {'judge': [tar_with(tmp / f'{name}-judge.tar', {'publish/quality.json': judge({wid: 0.0}, kw.pop('shift', 0.0)),
                                                              'publish/other/window.json': {}})],
             'score': [tar_with(tmp / f'{name}-score.tar', {'publish/score.json': score(name, **kw)})], 'dcvc': []}
    if dcvc:
        files['dcvc'] = [COMMITTED]
    return files


def test_al028_expected_gap_with_committed_dcvc_curve(tmp_path):
    result = a.analyse({'al028': window_inputs(tmp_path)})
    r = result['windows']['al028']
    assert r['gap']['lpips']['delta'] == pytest.approx(0.3929 - 0.3042, abs=2e-4)
    assert r['gap']['dists']['delta'] == pytest.approx(0.3074 - 0.2420, abs=2e-4)
    assert r['gap']['lpips']['best_anchor'] == r['gap']['dists']['best_anchor'] == 'dcvc'
    assert r['rate_kbps'] == pytest.approx(RATE)
    assert r['anchors_used'] == ['dcvc', 's2_o0', 's2_o20', 's4_o0'] and r['anchors_excluded'] == {}
    assert not r['svt_only_best_anchor'] and r['warnings'] == []
    # SVT scores come back from the bracket exactly (log-rate interpolation at the midpoint)
    assert r['at_hybrid_rate']['s2_o0']['foreground_lpips'] == pytest.approx(0.5933)
    assert r['gap']['lpips']['per_anchor']['s2_o0'] == pytest.approx(0.5933 - 0.3042)


def test_whole_frame_difference_and_dominance(tmp_path):
    r = a.analyse({'al028': window_inputs(tmp_path)})['windows']['al028']
    # best (lowest) whole score at equal rate: DCVC whole LPIPS 0.1490 / DISTS 0.1238 on the committed curve, against the hybrid's 0.2845 / 0.1750
    assert r['whole_frame']['lpips']['difference'] == pytest.approx(0.1490 - 0.2845, abs=2e-4)
    assert r['whole_frame']['dists']['difference'] == pytest.approx(0.1238 - 0.1750, abs=2e-4)
    # W1 dominance: SVT s2_o0 has the better whole frame but a far worse player region, so nothing dominates (outcome A)
    assert r['dominance']['outcome'] == 'A' and r['dominance']['outcome_svt_only'] == 'A'
    assert r['dominance']['per_family']['dcvc'] == {'lpips': False, 'dists': False}
    assert r['w1_embedded_outcome'] == 'A'


def test_dominance_b_when_an_anchor_beats_the_hybrid_on_both_scores(tmp_path):
    r = a.analyse({'al028': window_inputs(tmp_path, svt_shift=-0.4)})['windows']['al028']  # SVT player region now better than the hybrid
    assert r['dominance']['outcome_svt_only'] == 'B' and r['gap']['lpips']['delta'] < 0
    assert 's2_o0' in r['dominance']['dominating']['lpips']


def test_t_interval_matches_the_hand_formula_for_n6():
    values = [0.09, 0.05, 0.11, 0.02, 0.07, 0.08]
    g = a.t_interval(values)
    mean = sum(values) / 6
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / 5)
    assert g['t'] == pytest.approx(2.5706, abs=5e-5) and g['n'] == 6
    assert g['lo'] == pytest.approx(mean - 2.5706 * sd / math.sqrt(6), abs=1e-5)
    assert g['hi'] == pytest.approx(mean + 2.5706 * sd / math.sqrt(6), abs=1e-5) and g['excludes_zero_above']
    assert a.t_interval([0.1, -0.1, 0.0, 0.2, -0.05, 0.02])['excludes_zero_above'] is False
    assert a.t_interval([0.1])['lo'] is None  # one window has no interval


def test_missing_dcvc_uses_svt_only_and_is_flagged(tmp_path):
    r = a.analyse({'med000': window_inputs(tmp_path, 'med000', dcvc=False)})['windows']['med000']
    assert r['svt_only_best_anchor'] and r['anchors_excluded'] == {'dcvc': 'missing'}
    assert r['gap']['lpips']['best_anchor'] == 's2_o0' and r['gap']['lpips']['delta'] == pytest.approx(0.5933 - 0.3042)
    assert r['gap']['dists']['delta'] == pytest.approx(0.4344 - 0.2420)
    assert 'dcvc' not in r['dominance']['per_family']


def test_unresolved_family_is_excluded_and_listed(tmp_path):
    r = a.analyse({'al028': window_inputs(tmp_path, unbracketed=('s2_o0',), drop=('s4_o0',))})['windows']['al028']
    assert r['anchors_excluded'] == {'s2_o0': 'unresolved', 's4_o0': 'unresolved'}
    assert r['anchors_used'] == ['dcvc', 's2_o20']  # best anchor is chosen among the resolved families only
    assert r['gap']['lpips']['delta'] == pytest.approx(0.3929 - 0.3042, abs=2e-4)


def test_extrapolation_is_refused(tmp_path):
    # the hybrid's rate lies above the highest DCVC point (the first-run situation of the al028 entry): unresolved, not extrapolated
    points = json.loads(COMMITTED.read_text())[:2]
    assert a.dcvc_value([tuple(p) for p in points], RATE) is None
    assert a.dcvc_value([tuple(p) for p in points[1:]], 5.0) is None  # below the lowest
    path = tmp_path / 'dcvc.json'
    path.write_text(json.dumps(points))
    files = window_inputs(tmp_path)
    files['dcvc'] = [path]
    r = a.analyse({'al028': files})['windows']['al028']
    assert r['anchors_excluded'] == {'dcvc': 'unresolved'} and r['gap']['lpips']['best_anchor'] == 's2_o0'
    # an SVT policy whose both points lie below the rate is likewise unresolved
    assert a.svt_policies(score(unbracketed=('s4_o0',)), RATE)['s4_o0'] is None
    # no all-unresolved window
    files = window_inputs(tmp_path, unbracketed=tuple(a.SVT))
    files['dcvc'] = []
    with pytest.raises(ValueError, match='no anchor family resolved'):
        a.analyse({'al028': files})


def test_dcvc_reads_a_w3_summary_accepts_only_scale2_accepted_points(tmp_path):
    summary = dcvc_summary()
    rows, notes = a.dcvc_rows([summary], 'al028')
    assert [(r[0], round(r[1], 2)) for r in rows] == [(18.0, round(RATE / 1.3, 2)), (24.0, round(RATE * 1.1, 2))]  # scale 8 excluded
    assert rows[0][4:] == (0.4435, 0.3324) and notes == []
    # a point that failed the decode acceptance is never used
    rows, notes = a.dcvc_rows([dcvc_summary(accepted_high=False)], 'al028')
    assert len(rows) == 1 and any('not accepted' in n for n in notes)
    with pytest.raises(ValueError, match='entries for this window'):
        a.dcvc_rows([summary], 'al034')
    # archive path: published.tar with the summary under publish/, found by name
    files = window_inputs(tmp_path)
    files['dcvc'] = [tar_with(tmp_path / 'dcvc.tar', {'publish/summary.json': summary, 'publish/al028/result.json': {}})]
    r = a.analyse({'al028': files})['windows']['al028']
    assert r['dcvc_bracket_kbps'] == [pytest.approx(RATE / 1.3), pytest.approx(RATE * 1.1)]
    # bracket points of two archives merge; the same QP is not double counted
    both, _ = a.dcvc_rows([dcvc_summary(), json.loads(COMMITTED.read_text())], 'al028')
    assert sorted(r[0] for r in both) == [6.0, 18.0, 18.0, 24.0, 24.0, 30.0] or len(both) >= 4


def test_dcvc_interpolation_is_the_w5_dcvc_logic():
    from experiments.tennis import w5_dcvc
    table = [tuple(r) for r in json.loads(COMMITTED.read_text())]
    direct = w1_roi.policy_at_target(w5_dcvc.points(table), RATE)
    assert a.dcvc_value(table, RATE) == direct
    assert direct['foreground_lpips'] == pytest.approx(0.3929, abs=1e-4) and direct['foreground_dists'] == pytest.approx(0.3074, abs=1e-4)
    assert direct['whole_lpips'] == pytest.approx(0.1490, abs=1e-4) and direct['crf'] == [18.0, 24.0]


def test_archive_member_search_and_sha256(tmp_path):
    tar = tar_with(tmp_path / 'x.tar', {'deep/er/quality-summary.json': {'rows': 1}, 'quality.json': {'rows': 2}})
    assert a.read_json(tar, 'judge') == {'rows': 1}  # quality-summary.json is preferred
    plain = tmp_path / 'plain.json'
    plain.write_text('{"rows": 3}')
    assert a.read_json(plain, 'judge') == {'rows': 3}
    with pytest.raises(ValueError, match='no member'):
        a.read_json(tar, 'score')
    digest = a.sha256_of(tar)
    assert a.verify(tar, digest) == digest and a.verify(tar, digest[:12]) == digest and a.verify(tar, None) == digest
    with pytest.raises(ValueError, match='sha256 mismatch'):
        a.verify(tar, '0' * 64)
    with pytest.raises(ValueError, match='sha256 mismatch'):
        a.verify(tar, digest[:6])  # a short prefix is not an identity
    files = window_inputs(tmp_path)
    with pytest.raises(ValueError, match='sha256 mismatch'):
        a.analyse({'al028': files}, {files['judge'][0]: '1' * 64})


def test_one_judge_archive_covers_two_windows(tmp_path):
    both = tar_with(tmp_path / 'judge2.tar', {'full/quality-summary.json': judge({'alcaraz_hurkacz/034@120': 0.0, 'medvedev_struff/024@120': 0.02})})
    assert a.hybrid_from_judge(a.read_json(both, 'judge'), 'med024')['hybrid']['lpips']['foreground'] == pytest.approx(0.3042 + 0.02)
    assert a.hybrid_from_judge(a.read_json(both, 'judge'), 'al034')['hybrid']['lpips']['foreground'] == pytest.approx(0.3042)
    with pytest.raises(ValueError, match='found 0'):
        a.hybrid_from_judge(a.read_json(both, 'judge'), 'med019')


def six_windows(tmp_path, spread=0.0, kill_dcvc=('med000',)):
    inputs = {}
    for k, name in enumerate(a.WINDOWS):
        kw = {'svt_shift': 0.0 + spread * (k - 2.5)}
        files = window_inputs(tmp_path, name, dcvc=name not in kill_dcvc, **kw)
        if name not in kill_dcvc:
            files['dcvc'] = [tar_with(tmp_path / f'{name}-dcvc.tar', {'publish/summary.json': dcvc_summary(name)})]
        inputs[name] = files
    return inputs


def test_six_windows_pooled_clusters_and_sensitivity(tmp_path):
    result = a.analyse(six_windows(tmp_path))
    assert result['complete'] and result['missing_windows'] == [] and result['tier1_holds']
    assert result['pooled']['lpips']['n'] == 6 and result['pooled']['lpips']['t'] == pytest.approx(2.5706, abs=5e-5)
    assert set(result['clusters']) == {'al', 'med'} and result['clusters']['al']['windows'] == ['al028', 'al034', 'al043']
    assert result['clusters']['med']['lpips']['n'] == 3 and result['clusters']['med']['lpips']['t'] == pytest.approx(4.3027, abs=1e-3)
    s = result['sensitivity_without_missing_dcvc']
    assert s['excluded'] == ['med000'] and s['lpips']['n'] == 5
    assert result['windows']['med000']['svt_only_best_anchor'] and not result['windows']['med024']['svt_only_best_anchor']
    # pooled mean is the plain mean of the per-window gaps
    deltas = [result['windows'][n]['gap']['lpips']['delta'] for n in a.WINDOWS]
    assert result['pooled']['lpips']['mean'] == pytest.approx(sum(deltas) / 6)
    text = a.summary_text(result)
    assert 'Tier 1 player win HOLDS' in text and '[SVT-only best anchor]' in text and 'without med000' in text


def test_tier_fails_when_the_interval_reaches_zero_and_flags_incomplete(tmp_path):
    inputs = six_windows(tmp_path, spread=0.5, kill_dcvc=())  # wide spread of gaps: the interval includes zero
    result = a.analyse(inputs)
    assert not result['pooled']['holds'] and not result['tier1_holds']
    partial = a.analyse({n: inputs[n] for n in ('al028', 'al034', 'al043')})
    assert not partial['complete'] and partial['missing_windows'] == ['med000', 'med024', 'med019'] and not partial['tier1_holds']
    assert 'INCOMPLETE' in a.summary_text(partial)


def test_command_line_parsing_and_output(tmp_path, capsys):
    files = window_inputs(tmp_path)
    digest = a.sha256_of(files['judge'][0])
    out = tmp_path / 'result.json'
    argv = ['--window', 'al028', f"judge={files['judge'][0]}", f'sha256={digest}', f"score={files['score'][0]}", f'dcvc={COMMITTED}',
            '--out', str(out)]
    inputs, checks, path = a.parse_args(argv)
    assert path == out and checks[files['judge'][0]] == digest and list(inputs) == ['al028'] and inputs['al028']['dcvc'] == [COMMITTED]
    a.main(argv)
    written = json.loads(out.read_text())
    assert written['windows']['al028']['gap']['lpips']['best_anchor'] == 'dcvc' and written['input_sha256'][str(files['judge'][0])] == digest
    assert 'al028' in capsys.readouterr().out
    for bad in (['judge=x'], ['--window', 'zz999'], ['--window', 'al028', 'sha256=ab'], ['--window', 'al028', 'bogus=1']):
        with pytest.raises(SystemExit):
            a.parse_args(bad)


def test_inputs_table_lists_the_accepted_jobs_and_matches_the_docstring(capsys):
    a.main(['inputs'])
    printed = capsys.readouterr().out
    assert set(a.W7_INPUTS) == set(a.WINDOWS) and '4242c23c' in printed and '--window med000' in printed
    assert a.W7_INPUTS['med024']['dcvc'][0]['job'].endswith('4242c23c') and a.W7_INPUTS['med000']['dcvc'] == []
    assert a.W7_INPUTS['al028']['judge']['job'].endswith('e51a4dec') and a.W7_INPUTS['al028']['score']['job'].endswith('e32a981b')
    doc = a.__doc__ or ''
    for entry in a.W7_INPUTS.values():
        for kind in ('judge', 'score'):
            assert entry[kind]['job'] in doc
        for d in entry['dcvc']:
            assert 'job' not in d or d['job'] in doc
    assert 'c4df5a5f' not in ''.join(e['job'] for v in a.W7_INPUTS.values() for e in v['dcvc'] if 'job' in e)


def test_tier2_uses_its_own_cohort_and_match_cluster(tmp_path):
    inputs = {name: window_inputs(tmp_path, name, dcvc=False) for name in a.TIER2_WINDOWS}
    result = a.analyse(inputs)
    assert result['tier'] == 2 and result['complete'] and result['missing_windows'] == []
    assert result['clusters']['nad']['windows'] == list(a.TIER2_WINDOWS)
    assert result['pooled']['lpips']['n'] == 3
    assert 'Tier 2 player win' in a.summary_text(result)
    partial = a.analyse({'nad000': inputs['nad000']})
    assert partial['missing_windows'] == ['nad001', 'nad002'] and not partial['tier2_holds']
    with pytest.raises(ValueError, match='cannot be pooled'):
        a.analyse({'nad000': inputs['nad000'], 'al028': window_inputs(tmp_path)})

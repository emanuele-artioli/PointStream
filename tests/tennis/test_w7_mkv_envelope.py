import copy

import numpy as np
import pytest

from experiments.tennis import w7_mkv_envelope as subject


def fixture():
    values = np.full((4, 80), 0.1)
    digest = subject.score_digest(values)
    inventory = {'points': [], 'controls': [], 'new_points': []}
    reviews = {'rerate': [], 'control': [], 'new': []}
    rows = []
    duplicates = []
    for role, total, group in [('rerate', 53, 'points'), ('control', 18, 'controls'), ('new', 4, 'new_points')]:
        for i in range(total):
            window = subject.w7_analysis.WINDOWS[i % 6]
            name = f'{role}{i}'
            ident = f'{role}{i}'
            if role == 'control' and i < 4:
                ident = f'rerate{i}'
                duplicates.append({'window': window, 'control': name, 'roi': f'rerate{i}'})
            base = {'window': window, 'name': name, 'original_ivf_sha256': ident,
                    'native_yuv_sha256': ident, 'mkv_sha256': name, 'mkv_bytes': 100, 'fps': 25,
                    'gpu_class': 'A6000', 'score_job': 'synthetic'}
            inventory[group].append({'window': window, 'name': name, 'original_ivf_sha256': ident,
                                     'historical_native_yuv_sha256': ident, 'policy': 's2_o0', 'crf': 63})
            reviews[role].append({**base, 'score_array_sha256': digest})
            rows.append({**base, 'role': role, 'values': values.copy()})
    return (inventory, rows, {'approved_historical_reuse': True, 'rows': reviews['rerate']},
            {'approved_historical_control_reuse': True, 'rows': reviews['control'], 'duplicates': duplicates},
            {'approved_new_scores': True, 'rows': reviews['new']})


def test_complete_inventory_deduplicates_only_proven_equal_roles():
    args = fixture()
    views = subject.verified_views(*args)
    assert (len(views['primary']), len(views['historical_family_sensitivity']), len(views['deduplicated'])) == (71, 57, 4)
    altered = copy.deepcopy(args)
    altered[1][53]['mkv_bytes'] += 1
    with pytest.raises(ValueError):
        subject.verified_views(*altered)


def test_unapproved_or_incomplete_new_scores_cannot_form_hull():
    args = fixture()
    args[4]['approved_new_scores'] = False
    with pytest.raises(ValueError, match='not accepted'):
        subject.verified_views(*args)
    args[4]['approved_new_scores'] = True
    with pytest.raises(ValueError, match='75'):
        subject.verified_views(args[0], args[1][:-1], *args[2:])


def test_nonoverlap_never_invents_bd_rate():
    value = subject.compare_curves([(1.0, 0.0), (2.0, 0.1)], [(1.0, 0.5), (2.0, 0.6)])
    assert value == {'bd_rate': None, 'basis': 'no overlapping quality range'}


def test_bank_horizons_and_no_extrapolation():
    row = {'rate_kbps': 20.0, 'billing_correction': {'intended_bank_bytes': 1_000_000}}
    assert subject.bank_rate(row, 'al028', 7200) == 20.0
    assert subject.bank_rate(row, 'al028', 3600) > 20.0
    assert subject.bank_rate(row, 'al028', 1800) > subject.bank_rate(row, 'al028', 3600)
    points = [(10.0, -0.4), (20.0, -0.2)]
    assert subject.achievable_at_rate([], 5.0) == (None, 'absent')
    assert subject.achievable_at_rate(points, 5.0) == (None, 'below_minimum_rate')
    assert subject.achievable_at_rate(points, 25.0) == (-0.2, 'affordable_endpoint')
    assert subject.achievable_at_rate(points, 15.0)[1] == 'interpolated_bracket'
    assert subject.achievable_at_rate(points, 15.0)[0] <= subject.achievable_at_rate(points, 25.0)[0]


def test_review_sha_must_match_before_loading_archives(tmp_path):
    old = tmp_path / 'old.json'
    old.write_text('{}')
    control = tmp_path / 'control.json'
    control.write_text('{}')
    with pytest.raises(ValueError, match='SHA mismatch'):
        subject.load_historical(old, control, {})


def test_decision_preserves_missing_brackets_and_18_scenarios():
    views = subject.verified_views(*fixture())
    billing = {name: {'rate_kbps': 20.0, 'billing_correction': {'intended_bank_bytes': 1_000_000},
                      'dcvc_points': [], 'hybrid': {m: {'whole': .5, 'foreground': .5}
                                                   for m in ('lpips', 'dists')}}
               for name in subject.w7_analysis.WINDOWS}
    sizes = {(r['role'], r['window'], r['name']): 100 for r in views['all_roles']}
    report = subject.framing_reports(views, billing, sizes)['primary_ivf']
    assert len(report['scenarios']) == 18
    historical = [case for key, case in report['scenarios'].items() if key.endswith('historical_family_sensitivity')]
    assert all(case['holds'] is None for case in historical)
    primary = [case for key, case in report['scenarios'].items() if key.endswith('primary')]
    assert all(all(w['codec_coverage_by_metric']['lpips']['svt'] == 'affordable_endpoint'
                       and w['codec_coverage_by_metric']['lpips']['dcvc'] == 'absent'
                       for w in case['windows'].values()) for case in primary)
    assert len(report['roles']) == 75


def test_final_gate_rejects_missing_or_wrong_source_before_loading(tmp_path):
    with pytest.raises(ValueError, match='before artifact loading'):
        subject.check_final_gate(None, None, tmp_path)
    path = tmp_path / 'gate.json'
    path.write_text('{"approved":true,"max_cpu_seconds":500}')
    with pytest.raises(ValueError, match='exact 60-second cap'):
        subject.check_final_gate(path, subject._file_sha(path), tmp_path)


def test_gate_rejects_changed_scientific_dependency_identity(tmp_path, monkeypatch):
    import json
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps({'approved': True, 'max_cpu_seconds': 60,
                                'analysis_source_sha256': subject._file_sha(subject.Path(subject.__file__)),
                                'new_score_audit_sha256': subject.NEW_REVIEW_SHA,
                                'scientific_source_identity': {'tracked_python_manifest_sha256': 'old'}}))
    monkeypatch.setattr(subject, 'source_identity', lambda root: {'tracked_python_manifest_sha256': 'changed'})
    with pytest.raises(ValueError, match='source identity'):
        subject.check_final_gate(gate, subject._file_sha(gate), tmp_path)


def test_strict_historical_family_keeps_original_no_bracket_rule():
    points = [{'crf': 1, 'rate_kbps': 10, 'whole_lpips': .3, 'foreground_lpips': .4,
               'whole_dists': .3, 'foreground_dists': .4},
              {'crf': 2, 'rate_kbps': 20, 'whole_lpips': .2, 'foreground_lpips': .3,
               'whole_dists': .2, 'foreground_dists': .3}]
    assert subject.policy_at_target(points, 25) is None
    assert subject.achievable_at_rate([(10, -.4), (20, -.3)], 25) == (-.3, 'affordable_endpoint')


def test_primary_best_achievable_error_never_rises_with_target_rate():
    svt = [(10.0, -.6), (20.0, -.4)]
    dcvc = [(12.0, -.5), (18.0, -.3)]
    targets = (12.0, 16.0, 21.0, 30.0)
    best = []
    for target in targets:
        values = [v for points in (svt, dcvc) if (v := subject.achievable_at_rate(points, target)[0]) is not None]
        best.append(-max(values))
    assert best == sorted(best, reverse=True)
    assert best[-1] == best[-2]

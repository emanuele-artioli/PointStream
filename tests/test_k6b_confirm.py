"""Corruption must stop the diagnostic before expensive feature extraction."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.tennis.k6b_confirm import FROZEN, PROVENANCE, check_feature_rows, check_manifest, gate


def manifest():
    return {'contains_human_marks': False, 'provenance': dict(PROVENANCE), 'budget_gpu_hours': 8,
            'old_clips': [f'old/{i}' for i in range(49)],
            'clips': [{'id': c, 'frames': 240 + 2*s, 'fps': fps, 'sha256': sha,
                       'review_frames': [s + 120], 'staged': {'sha256': sha},
                       'registration': [{'frame': s+t, 'fits': True, 'homography': np.eye(3).tolist()} for t in range(240)]}
                      for c, s, fps, sha in FROZEN]}


def test_gate_boundary_and_invalid_geometry():
    rows = manifest()['clips'][0]['registration']
    rows[227:] = [None] * 13
    assert len(gate(rows)) == 227
    rows[227] = {'fits': True, 'homography': np.eye(3).tolist()}
    assert len(gate(rows)) == 228
    for h in ([[0]*3]*3, [[float('nan')]*3]*3, [[float('inf')]*3]*3, [1, 2]):
        rows[227] = {'fits': True, 'homography': h}
        assert len(gate(rows)) == 227
    rows[227] = {'registered': True, 'homography': np.eye(3).tolist()}
    assert len(gate(rows)) == 227
    with pytest.raises(ValueError):
        gate(rows[:239])


def test_manifest_rejects_scope_source_overlap_and_placement_corruption():
    source = manifest()
    assert len(check_manifest(source)) == 12
    for key, value in [('fps', 50), ('sha256', '0'*64), ('review_frames', [0]), ('frames', 999)]:
        m = deepcopy(source)
        m['clips'][0][key] = value
        with pytest.raises(ValueError):
            check_manifest(m)
    for mutate in ('duplicate', 'overlap', 'scope', 'gate', 'budget', 'registration_frame'):
        m = deepcopy(source)
        if mutate == 'duplicate':
            m['clips'][1] = m['clips'][0]
        elif mutate == 'overlap':
            m['old_clips'][0] = FROZEN[0][0]
        elif mutate == 'scope':
            m['provenance']['claim_scope'] = 'independent_confirmation'
        elif mutate == 'gate':
            m['clips'][0]['registration'][:13] = [None]*13
        elif mutate == 'budget':
            m['budget_gpu_hours'] = 9
        else:
            m['clips'][0]['registration'][0]['frame'] += 1
        with pytest.raises(ValueError):
            check_manifest(m)


def test_feature_output_requires_exact_window_and_full_decode():
    from experiments.tennis.k4b_features import plan_windows

    m = manifest()
    row = next(iter(plan_windows(m).values()))
    row['conversion'] = {'frames': 240}
    summary = {'requested': [row['id']], 'configured_windows': 1, 'windows': [row]}
    check_feature_rows(m, summary)
    for key, value in [('start', row['start'] + 1), ('conversion', {'frames': 239}), ('fps', 50)]:
        s = deepcopy(summary)
        s['windows'][0][key] = value
        with pytest.raises(ValueError):
            check_feature_rows(m, s)
    summary['windows'] = []
    with pytest.raises(ValueError):
        check_feature_rows(m, summary)


def test_anchor_envelope_never_extrapolates_or_awards_a_missing_cell():
    from experiments.tennis.k6b_confirm_validate import strict_won

    anchors = {'a': {'kbps': 1., 'lpips': .4}, 'b': {'kbps': 10., 'lpips': .2},
               'c': {'kbps': 5., 'lpips': .35}}
    assert strict_won(anchors, 'lpips', 5.5, .28)['won']
    assert not strict_won(anchors, 'lpips', 5.5, .31)['won']
    for rate in (0., 11.):
        cell = strict_won(anchors, 'lpips', rate, .01)
        assert cell['won'] is None and not cell['resolved']
    with pytest.raises(ValueError):
        strict_won(anchors, 'lpips', float('nan'), .1)
    with pytest.raises(ValueError):
        strict_won({}, 'lpips', 1, .1)


def test_generated_recipe_and_target_coverage_corruption():
    from experiments.tennis.k6b_confirm_validate import ADAPTER_SHA, BASE_SHA, check_generated

    prepared = {'windows': [{'units': [{'unit': 'u', 'window': 'alcaraz_hurkacz/031@129'}]}]}
    result = {'profiles': ['match5'], 'steps': 30, 'from_t': 0,
              'settings': {'size': [512, 784], 'cfg': {'match5': 2.0}, 'context': [24, 4]},
              'units': ['u'], 'max_frames': 0,
              'fits': {'match5': {'sha256': ADAPTER_SHA['alcaraz_hurkacz'], 'bits': 4}},
              'models': {'match5': {'fit': {'base': {'sha256': BASE_SHA},
                         'billing': {'pose_guider_billed': True, 'pose_guider_bytes': 4, 'bytes': 8}}}}}
    a = {'t': np.array([0, 1, 2, 3]), 'run': np.zeros(4), 'judged': np.array([True, False, False, True])}
    g = {'t': np.array([0, 3]), 'crops': np.zeros((2, 784, 512, 3), dtype=np.uint8)}
    check_generated(result, prepared, {'u': (a, g)})
    for key, value in [('steps', 29), ('profiles', ['original']), ('units', ['u', 'u'])]:
        bad = deepcopy(result)
        bad[key] = value
        with pytest.raises(ValueError):
            check_generated(bad, prepared, {'u': (a, g)})
    g['t'] = np.array([0, 0])
    with pytest.raises(ValueError):
        check_generated(result, prepared, {'u': (a, g)})



def test_reference_uses_first_fitted_scored_source_and_retains_missingness():
    from experiments.tennis.k6b_confirm_validate import reference_index

    row = {'scored': [1, 2, 3], 'judged': [3]}
    poses = {'p': [{'fitted': True}, None, {'fitted': True}, {'fitted': True}]}
    assert reference_index(row, poses, 'p') == 2
    with pytest.raises(ValueError):
        reference_index({'scored': [1]}, poses, 'p')


def test_descriptive_verdict_keeps_all_cells_and_never_passes_early():
    from experiments.tennis.k6b_confirm_report import verdict

    def rows(n, billed_wins, unbilled_wins):
        return [{'match': 'alcaraz_hurkacz' if i < 6 else 'medvedev_struff',
                 'billed': {m: {'won': 2*i+j < billed_wins} for j,m in enumerate(('lpips','dists'))},
                 'unbilled': {m: {'won': 2*i+j < unbilled_wins} for j,m in enumerate(('lpips','dists'))}}
                for i in range(n)]
    assert verdict(rows(9, 18, 18))['decision'] == 'incomplete; no interim pass'
    assert verdict(rows(12, 18, 18))['decision'] == 'descriptive billed bar reached'
    assert verdict(rows(12, 17, 18))['decision'] == 'user judgment: unbilled bar only'
    assert verdict(rows(12, 17, 17))['decision'] == 'descriptive criterion failed'
    v = verdict(rows(6, 0, 0))
    assert v['decision'].startswith('futility:')
    assert v['scores']['unbilled']['maximum_possible_wins'] == 12

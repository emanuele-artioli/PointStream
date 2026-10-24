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
                       'registration': [{'fits': True, 'homography': np.eye(3).tolist()} for _ in range(240)]}
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
    for mutate in ('duplicate', 'overlap', 'scope', 'gate', 'budget'):
        m = deepcopy(source)
        if mutate == 'duplicate':
            m['clips'][1] = m['clips'][0]
        elif mutate == 'overlap':
            m['old_clips'][0] = FROZEN[0][0]
        elif mutate == 'scope':
            m['provenance']['claim_scope'] = 'independent_confirmation'
        elif mutate == 'gate':
            m['clips'][0]['registration'][:13] = [None]*13
        else:
            m['budget_gpu_hours'] = 9
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

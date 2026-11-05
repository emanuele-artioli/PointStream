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
                    'native_yuv_sha256': ident, 'mkv_sha256': name, 'mkv_bytes': 100, 'fps': 25}
            inventory[group].append({'window': window, 'name': name, 'original_ivf_sha256': ident,
                                     'historical_native_yuv_sha256': ident})
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

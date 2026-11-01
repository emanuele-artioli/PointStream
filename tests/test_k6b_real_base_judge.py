import numpy as np
import pytest

from experiments.tennis.k6b_real_base_judge import check_diagnostic_coverage, prediction_ticks, unit_mask


def test_diagnostic_accepts_a_superset_but_requires_each_selected_window():
    diag = {'source': [{'window': w, 'source_matches': True, 'ticks': list(range(80))} for w in ('match/0', 'match/1')],
            'base': [{'window': w, 'scale': s} for w in ('w0', 'w1') for s in (8, 4, 2)]}
    check_diagnostic_coverage(diag, ['match/0'], ['w0'])
    check_diagnostic_coverage(diag, ['match/1'], ['w1'])
    with pytest.raises(ValueError):
        check_diagnostic_coverage(diag, ['match/2'], ['w1'])
    diag['base'].append({'window': 'w0', 'scale': 2})
    with pytest.raises(ValueError):
        check_diagnostic_coverage(diag, ['match/0'], ['w0'])


def test_frozen_pair_expects_the_full_grid_and_a_manifest_window_its_recorded_ticks():
    grid = list(range(0, 240, 3))
    assert prediction_ticks({"ticks": grid}, manifest=False) == grid
    # A manifest unit without a pose at some frames was predicted on fewer ticks (job 20261031T102650Z-8d510571: 77 of 80).
    recorded = [t for t in grid if t != 84][:77]
    assert prediction_ticks({"ticks": recorded}, manifest=True) == recorded
    assert prediction_ticks({"ticks": grid}, manifest=True) == grid


def test_unit_mask_is_empty_only_for_an_uncropped_frame():
    # Judged frame 84 has no crop for this unit (job 20261031T104417Z-d22c8fb0 failed on it at composing).
    mask = np.ones((4, 6), bool)
    assert unit_mask({3: mask}, True, 3, (4, 6)) is mask
    assert not unit_mask({3: mask}, False, 84, (4, 6)).any()
    with pytest.raises(ValueError):
        unit_mask({}, True, 3, (4, 6))

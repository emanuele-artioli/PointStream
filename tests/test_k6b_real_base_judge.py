import numpy as np
import pytest

from experiments.tennis.k6b_real_base_judge import prediction_ticks, unit_mask


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

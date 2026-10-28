import numpy as np
import pytest

from experiments.tennis.parallel_baselines import cost_projection, roi_text, union_regions


def test_roi_blocks_keep_foreground_and_raise_background():
    region = np.zeros((2, 128, 128), bool)
    region[0, 0, 0] = True
    region[1, 127, 127] = True
    assert roi_text(region, 128, 128).splitlines() == ['0 0 32 32 32', '1 32 32 32 0']


def test_projection_scales_native_and_scores_separately():
    assert cost_projection(10, 2, 3, 4) == pytest.approx(1.35 * (10 + 20 + 80 + 16))
    with pytest.raises(ValueError):
        cost_projection(0, float('nan'), 1, 1)


def test_half_grid_player_union_preserves_both_players():
    regions = np.zeros((240, 2, 540, 960), bool)
    regions[0, 0, 0, 0] = True
    regions[0, 1, 539, 959] = True
    union = union_regions(regions, {'output_shape': [1080, 1920], 'native_grid_factor': 1})
    assert union.shape == (240, 540, 960)
    assert union.sum() == 2
    first = [int(x) for x in roi_text(union[:1], 480, 272).split()]
    assert first[0] == 0 and len(first) == 1 + 5 * 8
    assert first[1] == 0
    with pytest.raises(ValueError):
        union_regions(regions[:, :1], {'output_shape': [1080, 1920], 'native_grid_factor': 1})

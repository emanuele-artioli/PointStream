"""No oracle decoder or claimed inverse may recover information absent in RGB."""
import numpy as np
import pytest

from experiments.tennis.k6b_edge_ramp import alpha_ramp, ambiguity, decode_luma, decode_rgb, diagnostics


def test_hard_key_survives_dark_pixels_without_oracle_alpha():
    pixels = np.array([[[0, 0, 0], [32, 32, 32], [255, 255, 255]]], np.uint8)
    rgb, mask = decode_rgb(pixels)
    assert mask.tolist() == [[False, True, True]]
    assert rgb.tolist() == [[[0, 0, 0], [0, 0, 0], [255, 255, 255]]]
    assert decode_luma(np.array([127, 128], np.uint8)).tolist() == [False, True]
    assert not ambiguity()['unique_alpha_or_color']


def test_small_ramp_threshold_is_a_boundary_estimate_not_rgb_unmixing():
    mask = np.zeros((12, 12), bool)
    mask[:, 6:] = True
    assert np.array_equal(alpha_ramp(mask, 0), mask)
    for width in (2, 4):
        ramp = alpha_ramp(mask, width)
        assert np.array_equal(ramp >= .5, mask)
        assert np.any((ramp > 0) & (ramp < 1))
        # Alpha is not recoverable from mixed arbitrary RGB even without a codec.
        assert .5 * 128 == 1 * 64
    with pytest.raises(ValueError):
        alpha_ramp(mask, 8)


def test_edge_displacement_diagnostic_detects_one_pixel_shift():
    truth = np.zeros((2, 16, 16), bool)
    truth[:, 4:12, 4:12] = True
    decoded = np.roll(truth, 1, axis=2)
    result = diagnostics(truth, decoded, None, np.zeros((2, 16, 16, 3), np.uint8))
    assert result['iou'] < 1
    assert 0 < result['boundary_mean_px'] <= 1
    assert result['boundary_p95_px'] == 1
    assert result['temporal_boundary_error_flips_per_true_boundary_pixel'] == 0


def test_wire_player_ids_map_to_saved_units_without_using_list_position():
    from experiments.tennis.k6b_edge_ramp import player_index
    units = [{'name': 'alcaraz__031_at129__person_000000000006'},
             {'name': 'alcaraz__031_at129__person_000000000001'}]
    assert player_index(units, 'person:000000000001') == 1
    assert player_index(units, 'person:000000000006') == 0
    with pytest.raises(ValueError, match='identity'):
        player_index(units, 'person:000000000002')

"""Contracts with visible failure modes for the proposed transport."""
import numpy as np
import pytest

from experiments.tennis.k6b_reference_video import (
    FLOORS, aligned_packet, black_key, decode_aligned, overlap, recover_key,
)


def test_reserved_black_distinguishes_black_foreground_before_codec():
    image = np.zeros((2, 8, 10, 3), np.uint8)
    alpha = np.zeros(image.shape[:-1], bool)
    alpha[:, 2:6, 3:8] = True
    for floor in FLOORS:
        coded = black_key(image, alpha, floor)
        assert np.all(coded[~alpha] == 0)
        assert np.array_equal(recover_key(coded, floor), alpha)
    with pytest.raises(ValueError, match='Undeclared'):
        black_key(image, alpha, 0)


def test_one_level_guard_band_is_not_a_lossy_codec_guarantee():
    image = np.ones((1, 4, 4, 3), np.uint8)
    mask = np.zeros((1, 4, 4), bool)
    mask[:, 1:3, 1:3] = True
    coded = black_key(image, mask, 1)
    # One level of reconstruction error can erase a shoe and invent background.
    coded[:, 1, 1] = 0
    coded[:, 0, 0] = 1
    result = overlap(mask, recover_key(coded, 1))
    assert result['false_positive'] == result['false_negative'] == 1


def test_alignment_reversible_for_empty_and_edge_crossing_shapes():
    masks = np.zeros((6, 16, 24), bool)
    masks[1, :4, :5] = True
    masks[2, -4:, -5:] = True
    masks[3, :, :2] = True
    masks[4, 0, 0] = masks[4, -1, -1] = True
    masks[5] = True
    blob = aligned_packet(masks)
    assert np.array_equal(decode_aligned(blob), masks)
    with pytest.raises((ValueError, RuntimeError)):
        decode_aligned(blob[:-1])


def test_affine_guard_band_preserves_dark_colors_with_rounding_only():
    from experiments.tennis.k6b_reference_affine import lift, unlift
    rgb = np.arange(256, dtype=np.uint8).reshape(1, 16, 16, 1).repeat(3, -1)
    mask = np.ones(rgb.shape[:-1], bool)
    encoded = lift(rgb, mask)
    assert encoded.min() == 32
    assert np.array_equal(recover_key(encoded, 32), mask)
    assert np.max(np.abs(unlift(encoded).astype(int)-rgb.astype(int))) <= 1
    assert not lift(rgb, ~mask).any()

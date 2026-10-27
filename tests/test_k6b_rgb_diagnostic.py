"""Difference summaries retain signedness-safe per-pixel and channel counts."""
import numpy as np

from experiments.tennis.k6b_rgb_diagnostic import pixel_diff


def test_pixel_diff_counts_and_channel_means():
    first = np.array([[[0, 5, 255], [9, 10, 11]]], dtype=np.uint8)
    second = np.array([[[255, 4, 0], [9, 10, 11]]], dtype=np.uint8)
    row = pixel_diff(first, second)
    assert row['max'] == 255
    assert row['sum'] == 511
    assert row['different_pixels'] == 1
    assert row['channel_sum'] == [255, 1, 255]
    assert row['channel_mean_absolute'] == [127.5, 0.5, 127.5]

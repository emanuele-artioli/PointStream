"""Lossy matte packet correctness and footprint regressions."""
import numpy as np
import pytest

from experiments.tennis.k6b_matte_cpu import coverage, decode, encode
from src.pointstream import mask as mk
from src.pointstream.composite import OBJECT


@pytest.mark.parametrize("mode", ["h5", "rle"])
def test_actual_packet_preserves_planes_identity_and_partial_edges(mode):
    planes = np.zeros((240, 2, 34, 60), bool)
    planes[3:239, 0, 1:7, 4:9] = True
    planes[:, 1, -1, -1] = True
    blob = encode(planes, 32, ["near", "far"], mode)
    meta, decoded = decode(blob)
    assert meta["units"] == ["near", "far"]
    assert meta["frame_ticks"] == [0, 240]
    assert np.array_equal(planes, decoded)
    damaged = bytearray(blob)
    damaged[-1] ^= 1
    with pytest.raises(Exception):
        decode(bytes(damaged))


def test_exact_double_grid_area_reduction_matches_full_frame():
    source = np.zeros((2, 15, 19), np.uint8)
    source[:, 1:13, 2:12] = OBJECT
    full = np.repeat(np.repeat(source, 2, 1), 2, 2)
    for scale in (8, 16, 32):
        assert np.array_equal(mk.reduce(full, scale), mk.reduce(source, scale // 2))


def test_crop_footprint_never_grants_rgb_outside_saved_rectangle():
    mask = np.zeros((1080, 1920), bool)
    mask[3:6, 3:6] = True
    assert coverage(mask, np.array([4., 4., 3., 3.])) == 5
    assert coverage(mask, None) == 9
    assert coverage(mask, np.array([-10., -10., 2., 2.])) == 9

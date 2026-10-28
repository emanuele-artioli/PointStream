import numpy as np
import pytest

from experiments.tennis.single_restoration import align, fuse, header, parse_header


def test_current_to_neighbour_mapping_and_outside_pixels_are_rejected():
    current = np.repeat((np.arange(32, dtype=np.uint8)+100)[None, :, None], 24, 0)
    current = np.repeat(current, 3, 2)
    neighbour = np.zeros_like(current)
    neighbour[:, 2:] = current[:, :-2]
    flow = np.zeros((24, 32, 2), np.float32)
    flow[..., 0] = 2
    backward = -flow
    warped, weight = align(current, neighbour, flow, backward)
    assert np.array_equal(warped[:, :-2], current[:, :-2])
    assert np.all(weight[:, :-2] == 1)
    assert np.all(weight[:, -2:] == 0)
    assert np.array_equal(fuse(current, [(warped, weight)], .6), current)


def test_occlusion_and_photometric_change_do_not_blur_the_current_frame():
    current = np.full((24, 32, 3), 100, np.uint8)
    flow = np.zeros((24, 32, 2), np.float32)
    bad_reverse = np.full_like(flow, 10)
    warped, weight = align(current, current, flow, bad_reverse)
    assert not weight.any()
    assert np.array_equal(fuse(current, [(warped, weight)], .6), current)
    _, weight = align(current, np.full_like(current, 200), flow, flow)
    assert not weight.any()


def test_weighted_restoration_is_bounded_and_profile_is_delivered():
    current = np.full((24, 32, 3), 100, np.uint8)
    neighbour = np.full_like(current, 110)
    result = fuse(current, [(neighbour, np.ones((24, 32), np.float32))], .6)
    assert np.all(result == 103)
    assert np.array_equal(fuse(current, [], .6), current)
    assert len(header(2, 96)) == 8
    assert parse_header(header(2, 96)) == (.6, 96)
    with pytest.raises(ValueError):
        parse_header(header(2, 96)+b'x')
    with pytest.raises(ValueError):
        header(3, 96)
    flow = np.zeros((24, 32, 2), np.float32)
    flow[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        align(current, neighbour, flow, flow)

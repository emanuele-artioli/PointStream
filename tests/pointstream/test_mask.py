import numpy as np
import pytest

from src.pointstream import mask as mk
from src.pointstream.composite import HAND, OBJECT


def moving_classes(n: int = 6, h: int = 120, w: int = 200) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    out = np.zeros((n, h, w), np.uint8)
    for t in range(n):
        out[t][(yy - 60) ** 2 + (xx - 60 - 4 * t) ** 2 < 30**2] = OBJECT
        out[t][(yy - 60) ** 2 + (xx - 120 - 2 * t) ** 2 < 25**2] = HAND
    return out


@pytest.mark.parametrize("scale", [1, 2, 4, 8])
def test_round_trip_is_lossless_on_the_reduced_grid(scale: int) -> None:
    classes = moving_classes()
    reduced = mk.reduce(classes, scale)
    payload = mk.encode(reduced, scale)
    back, s = mk.decode(payload)
    assert s == scale and np.array_equal(back, reduced)


def test_upsampled_mask_tracks_the_original() -> None:
    classes = moving_classes()
    for scale, min_iou in ((2, 0.96), (4, 0.94), (8, 0.88)):
        up = mk.upsample(mk.reduce(classes, scale), (200, 120))
        f, g = classes > 0, up > 0
        assert (f & g).sum() / (f | g).sum() >= min_iou
        hand = classes == HAND
        assert (up[hand & g] == HAND).mean() > 0.9  # the hand keeps its class
    assert np.array_equal(mk.upsample(mk.reduce(classes, 1), (200, 120)), classes)


def test_coarser_grids_cost_fewer_bytes() -> None:
    classes = moving_classes(n=12, h=240, w=400)
    sizes = [len(mk.encode(mk.reduce(classes, s), s)) for s in (1, 2, 4, 8)]
    assert sizes == sorted(sizes, reverse=True)


def test_empty_frames() -> None:
    classes = np.zeros((3, 40, 60), np.uint8)
    back, _ = mk.decode(mk.encode(mk.reduce(classes, 4), 4))
    assert not mk.upsample(back, (60, 40)).any()

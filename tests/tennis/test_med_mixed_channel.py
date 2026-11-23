"""Receiver regressions for conventional fallback around a paid native interval."""

import cv2
import numpy as np
import pytest

from experiments.tennis import med_mixed_channel as channel
from src.pointstream import ball
from src.pointstream.tracked_patch import yuv_to_rgb


@pytest.mark.parametrize("start", [0, 69, 92, 120])
def test_smoke_spans_both_modes_and_fallback_is_native(start):
    ticks = channel.smoke_ticks(start)
    assert len(ticks) == 4 and len(set(ticks)) == 4
    assert sum(start <= t < start + 120 for t in ticks) == 3
    fallback = next(t for t in ticks if not start <= t < start + 120)
    raw = np.random.default_rng(19).integers(16, 230, (144, 160), dtype=np.uint8)
    direct = cv2.resize(
        yuv_to_rgb(raw, 160, 96, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
    )
    track = [None] * 240
    profile = {
        "start": start,
        "stop": start + 120,
        "pos": track,
        "velocity": ball.velocities(track),
        "looks": track,
    }
    assert np.array_equal(channel.canvas(raw, fallback, profile), direct)


def test_full_native_smoke_includes_both_endpoints():
    assert channel.smoke_ticks(0, 240) == [0, 39, 119, 239]
    assert channel.smoke_ticks(69, 189) == channel.smoke_ticks(69)


@pytest.mark.parametrize("width,height", [(320, 192), (640, 384)])
def test_wider_native_canvas_uses_the_paid_dimensions(width, height):
    raw = np.random.default_rng(23).integers(16, 230, (height * 3 // 2, width), dtype=np.uint8)
    track = [None]
    decoded = dict(
        coded_size=(width, height),
        start=0,
        stop=1,
        pos=track,
        velocity=ball.velocities(track),
        looks=track,
    )
    rgb = yuv_to_rgb(raw, width, height, False)
    scene = cv2.resize(rgb, (960, 544), interpolation=cv2.INTER_CUBIC)
    expected = cv2.resize(scene, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    assert np.array_equal(channel.canvas(raw, 0, decoded), expected)
    with pytest.raises(ValueError, match="paid coded size"):
        channel.canvas(raw[:, :-2], 0, decoded)


def test_split_sections_keep_their_native_size_and_own_local_tick():
    from experiments.tennis import med_split_native as split

    sizes = [
        (0, 1, "conventional", 320, 192),
        (1, 3, "objects", 160, 96),
        (3, 5, "conventional", 320, 192),
    ]
    rng = np.random.default_rng(87)
    parts = []
    for start, stop, mode, width, height in sizes:
        raw = rng.integers(30, 190, (stop - start, height * 3 // 2, width), dtype=np.uint8)
        parts.append((dict(begin=start, end=stop, mode=mode, width=width, height=height), raw, ""))
    locs = [None] * 5
    paid = dict(start=1, stop=3, pos=locs, looks=locs, velocity=ball.velocities(locs))
    for i, (part, raw, _) in enumerate(parts):
        t = part["end"] - 1
        if i == 1:
            expected = channel.canvas(raw[-1], t, paid)
        else:
            expected = cv2.resize(
                yuv_to_rgb(raw[-1], 320, 192, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
            )
        assert np.array_equal(split.canvas(parts, t, paid), expected)
    with pytest.raises(ValueError, match="undelivered"):
        split.canvas(parts, 5, paid)

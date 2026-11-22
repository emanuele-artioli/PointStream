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

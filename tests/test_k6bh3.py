"""K6b (hands) 3: the overfit oracle's pairing and learning-curve extrapolation (docs/experiments.md, 2026-10-22)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from experiments.visor import k6bh3  # noqa: E402


def test_reference_pairs_stay_on_the_same_side_and_differ() -> None:
    sides = np.array([False, False, True, True, True, False])
    rng = np.random.default_rng(0)
    picks = np.array([0, 2, 3, 5, 4, 1])
    refs = k6bh3.reference_pairs(sides, rng, picks)
    assert (sides[refs] == sides[picks]).all() and (refs != picks).all()
    lone = k6bh3.reference_pairs(np.array([False, True]), rng, np.array([1]))
    assert list(lone) == [1]  # the side's only crop is its own reference


def test_log_time_fit_and_extrapolation() -> None:
    secs = [450.0, 900.0, 1800.0]
    values = [0.40 - 0.05 * np.log(s / 450) / np.log(2) for s in secs]  # -0.05 per doubling
    a, b = k6bh3.log_time_fit(secs, values)
    assert b == pytest.approx(-0.05 / np.log(2))
    # 0.30 at 1800 s; 0.25 one doubling later (60 min), 0.275 at 2545 s
    assert k6bh3.minutes_to_reach(secs, values, 0.25) == pytest.approx(60.0, rel=1e-6)
    assert k6bh3.minutes_to_reach(secs, values, 0.35) == pytest.approx(15.0, rel=1e-6)
    assert k6bh3.minutes_to_reach(secs, [0.4, 0.41, 0.42], 0.3) is None  # not learning
    assert k6bh3.minutes_to_reach(secs, [0.2, 0.21, 0.22], 0.3) == 30.0  # already below at the end


def test_decision_names_the_pilot_horizon() -> None:
    assert k6bh3.HULL_MINUTES == 45.0 and k6bh3.JUDGED == (0.25, 0.5, 1.0)
    assert "45 min" in k6bh3.DECISION["pilot"]


def test_look_frames_are_evenly_spaced_and_content_blind() -> None:
    hand = list(range(10, 250))
    looks = k6bh3.look_frames(hand)
    assert len(looks) == k6bh3.LOOKS and looks == sorted(looks)
    assert looks[0] > hand[0] and looks[-1] < hand[-1]  # the ends are left out
    assert k6bh3.look_frames([3, 7]) == [3, 7] and k6bh3.look_frames([]) == []


def test_flicker_reads_luma_changes_inside_r() -> None:
    frames = np.zeros((3, 4, 4, 3))
    frames[1] += 10.0  # a jump of 10 in luma between frames 0 and 1, none between 1 and 2
    frames[2] += 10.0
    regions = [np.ones((4, 4), bool)] * 3
    assert k6bh3.flicker(lambda t: frames[t], regions, [0, 1, 2]) == pytest.approx(5.0)
    assert np.isnan(k6bh3.flicker(lambda t: frames[t], regions, [0]))

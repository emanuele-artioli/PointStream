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


def _candidates() -> dict[str, np.ndarray]:
    # 6 candidate frames; frame 2 has no kept fit, frame 4 has only a dropped one
    return {"visor_frame": np.array([10, 20, 30, 40, 50, 60]), "index": np.array([9, 19, 29, 39, 49, 59]),
            "row_cand": np.array([0, 0, 1, 3, 4, 5]), "row_right": np.array([False, True, False, True, True, False]),
            "row_inside": np.array([21, 18, 15, 20, 3, 21]), "row_kp2d": np.tile(np.array([[100.0, 100.0]]), (6, 21, 1))
            + np.arange(21)[None, :, None]}


def test_transfer_picks_are_kept_frames_evenly_spaced() -> None:
    arrays = _candidates()
    assert k6bh3.transfer_picks(arrays, 10, 15) == [0, 1, 3, 5]  # every frame with a kept fit, in order
    assert k6bh3.transfer_picks(arrays, 2, 15) == [0, 5]  # the ends, by position only
    assert k6bh3.transfer_picks(arrays, 3, 15) == [0, 1, 5] or k6bh3.transfer_picks(arrays, 3, 15) == [0, 3, 5]


def test_transfer_crops_take_kept_hands_with_their_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.visor import k6bh2c
    from src.segmentation import visor

    arrays = _candidates()
    shape = (300, 400)
    left = np.zeros(shape, bool)
    left[90:140, 90:140] = True

    def masks(frame: int) -> dict[str, np.ndarray]:  # frame 10 has both hands, 40 has none, 60 only the left
        return {10: {visor.HANDS[0]: left, visor.HANDS[1]: left}, 40: {}, 60: {visor.HANDS[0]: left}}[frame]

    monkeypatch.setattr(k6bh2c, "hand_masks", masks)
    rgb = np.zeros((3, *shape, 3), np.uint8)
    crops = k6bh3.transfer_crops(arrays, [0, 3, 5], rgb, [0, 3, 5], {10: 10, 40: 40, 60: 60}, 15)
    assert [(c["t"], c["right"]) for c in crops] == [(0, False), (0, True), (5, False)]  # 40's hand has no mask
    assert all(c["image"].shape == (256, 256, 3) and c["mask"].shape == (256, 256) and c["kp"].shape == (21, 2) for c in crops)
    assert [c["index"] for c in crops] == [9, 9, 59]


def test_near_share_counts_frames_close_to_the_window() -> None:
    window = (1000, 1240)
    assert k6bh3.near_share([400, 499, 500, 1239, 1739, 1740], window, 50.0, 10.0) == pytest.approx(3 / 6)
    assert np.isnan(k6bh3.near_share([], window, 50.0, 10.0))


def test_transfer_decision_is_the_users_criterion() -> None:
    assert "unbilled" in k6bh3.TRANSFER_DECISION["billing"] and "0.75" in k6bh3.TRANSFER_DECISION["pass"]
    assert k6bh3.TRANSFER_FRAMES == 240 and k6bh3.WEIGHT_BYTES_PER_PARAM == 2

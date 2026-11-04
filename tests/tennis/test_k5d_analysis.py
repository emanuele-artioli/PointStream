from fractions import Fraction

import pytest

from experiments.tennis.k5d_analysis import Point, compare, quality_at, score_frames

D = Fraction(8, 1)


def point(codec: str, variant: str, rate: int, value: float, metric: str = "frame_psnr") -> Point:
    # Eight seconds makes the byte count equal to rate in bits/s.
    return Point(codec, variant, str(rate), rate, D, 240, metric, value)


def test_known_half_rate_and_loss_orientation() -> None:
    rows = [point("svt", "native", r, q) for r, q in [(200, 30), (400, 35), (800, 40)]]
    rows += [point("svt", "ball", r, q) for r, q in [(100, 30), (200, 35), (400, 40)]]
    result = compare(rows, 240, D, "frame_psnr", "ball", 1)
    assert result["status"] == "bd_rate"
    assert result["bd_rate"] == pytest.approx(-0.5)
    loss_rows = [Point(r.codec, r.variant, r.setting, r.byte_count, r.duration, r.frames,
                       "crop_dists", 1 - r.value / 100) for r in rows]
    assert compare(loss_rows, 240, D, "crop_dists", "ball", 0.01)["bd_rate"] == pytest.approx(-0.5)


def test_pooled_anchor_and_no_extrapolation() -> None:
    rows = [point("svt", "native", 100, 20), point("svt", "native", 400, 25),
            point("dcvc", "native", 200, 24), point("dcvc", "native", 800, 27),
            point("svt", "ball", 200, 25), point("svt", "ball", 900, 30)]
    result = compare(rows, 240, D, "frame_psnr", "ball", 1)
    candidate = result["candidate_points"]
    assert isinstance(candidate, list)
    assert candidate[0]["anchor_at_rate"] == 24
    assert candidate[1]["anchor_at_rate"] is None
    assert quality_at([(1, 1), (4, 3)], 2) == pytest.approx(2)
    assert quality_at([(1, 1), (4, 3)], 5) is None


def test_disjoint_and_too_little_overlap() -> None:
    native = [point("svt", "native", 100, 20), point("svt", "native", 200, 25)]
    better = [point("svt", "ball", 100, 30), point("svt", "ball", 200, 35)]
    assert compare(native + better, 240, D, "frame_psnr", "ball", 1)["status"] == "better_at_every_supported_rate"
    lower = [point("svt", "ball", 100, 10), point("svt", "ball", 200, 15)]
    assert compare(native + lower, 240, D, "frame_psnr", "ball", 1)["status"] == "worse_at_every_supported_rate"
    small = [point("svt", "ball", 100, 24.9), point("svt", "ball", 200, 27)]
    assert compare(native + small, 240, D, "frame_psnr", "ball", 1)["status"] == "unsupported_range"


def test_discrete_and_label_denominators() -> None:
    labels = [(1, 0., 0.), (1, 0., 0.), (1, 0., 0.), (0, 0., 0.), (-1, 0., 0.)]
    preds = [(True, 3., 4.), (True, 7., 0.), (False, 0., 0.), (True, 0., 0.), (True, 0., 0.)]
    scored = score_frames(labels, preds)
    assert (scored["visible"], scored["absent"], scored["unknown"]) == (3, 1, 1)
    assert scored["recall5"] == pytest.approx(1 / 3)
    assert scored["recall10"] == pytest.approx(2 / 3)
    assert scored["miss10"] == 1 and scored["false_per_negative"] == 1
    unknown = score_frames([(-1, 0., 0.)], [(True, 0., 0.)])
    assert unknown["recall10"] is None and unknown["false_per_negative"] is None
    rows = [point("svt", "native", 100, 0.2, "recall10"), point("svt", "native", 200, 0.4, "recall10"),
            point("svt", "ball", 150, 0.8, "recall10")]
    result = compare(rows, 240, D, "recall10", "ball", 0.01)
    assert result["status"] == "descriptive_discrete" and result["bd_rate"] is None


def test_invalid_or_duplicate_points() -> None:
    native = point("svt", "native", 100, 20)
    candidate = point("svt", "ball", 100, 22)
    for rows in ([native, native, candidate], [native, point("svt", "ball", 100, float("nan"))],
                 [native, Point("svt", "ball", "x", 100, Fraction(9), 240, "frame_psnr", 22)],
                 [native, Point("svt", "ball", "x", 100, D, 239, "frame_psnr", 22)]):
        with pytest.raises(ValueError):
            compare(rows, 240, D, "frame_psnr", "ball", 1)
    with pytest.raises(ValueError):
        score_frames([(2, 0., 0.)], [(False, 0., 0.)])


def test_reject_negative_distance_and_boolean_metric() -> None:
    for metric, value in (("crop_lpips", -0.1), ("crop_dists", -0.1), ("frame_psnr", True)):
        rows = [point("svt", "native", 100, 0.2, metric), point("svt", "ball", 100, value, metric)]
        with pytest.raises(ValueError):
            compare(rows, 240, D, metric, "ball", 0.01)


def test_constant_anchor_support_does_not_end_at_first_point() -> None:
    for metric, value in (("false_per_negative", 0.0), ("recall10", 0.0)):
        rows = [point("svt", "native", r, value, metric) for r in (100, 200, 400)]
        rows += [point("svt", "ball", r, 0.1, metric) for r in (150, 500)]
        candidates = compare(rows, 240, D, metric, "ball", 0.01)["candidate_points"]
        assert isinstance(candidates, list)
        assert candidates[0]["anchor_at_rate"] == 0
        assert candidates[1]["anchor_at_rate"] is None


def test_constant_quality_dominance_uses_only_measured_support() -> None:
    rows = [point("svt", "native", r, 20) for r in (100, 200, 400)]
    rows += [point("svt", "ball", r, 25) for r in (150, 300)]
    assert compare(rows, 240, D, "frame_psnr", "ball", 1)["status"] == "better_at_every_supported_rate"
    outside = rows[:3] + [point("svt", "ball", r, 25) for r in (500, 600)]
    assert compare(outside, 240, D, "frame_psnr", "ball", 1)["status"] == "unsupported_range"

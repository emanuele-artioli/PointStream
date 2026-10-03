from demo.experiments.holdout_hand_rd import _collapse, kbps, segment_bounds, steady_fps


def test_steady_fps_meets_realtime_at_24():
    assert steady_fps(24, 1.0) == 24.0
    assert steady_fps(300, 20.0) == 15.0


def test_segment_bounds_cover_the_holdout_without_overlap():
    bounds = segment_bounds(300, 30)
    assert bounds[0] == (0, 30)
    assert bounds[-1] == (270, 300)
    assert sum(stop - start for start, stop in bounds) == 300


def test_two_hands_share_one_duration():
    rows = [
        {"segment_frames": 30, "start": 0, "rung": "240p", "bytes": 1000, "span_frames": 30, "lpips": 0.2},
        {"segment_frames": 30, "start": 0, "rung": "240p", "bytes": 500, "span_frames": 30, "lpips": 0.4},
    ]
    collapsed = _collapse(rows)
    assert collapsed[0]["bytes"] == 1500
    assert collapsed[0]["kbps"] == kbps(1500, 30)
    assert abs(collapsed[0]["lpips"] - 0.3) < 1e-9

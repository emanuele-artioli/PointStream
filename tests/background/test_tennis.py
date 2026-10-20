"""K4 and K5: ball trajectory symbols and rate, the camera measure, keypoint symbols, BD verdicts and the rules."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from experiments.tennis import k4, k5
from experiments.tennis import windows as tw


def test_trajectory_symbols_predict_constant_velocity_and_restart_after_a_gap():
    rows = [(1, 10, 10), (1, 12, 11), (1, 14, 12), (2, 17, 13), (0, 0, 0), (3, 5, 5), (1, 50, 60), (1, 51, 60)]
    s = tw.trajectory_symbols(rows)
    assert s["visibility"] == [1, 1, 1, 1, 0, 0, 1, 1]
    # frame 1: zero-velocity residual; 2: on the line; 3: one px off in x; after the gap an absolute, then zero velocity
    assert s["residual"] == [(2, 1), (0, 0), (1, 0), (1, 0)]
    assert s["absolute"] == [2]
    rate = tw.trajectory_rate([s], 30.0)
    assert rate["visible"] == 6 and rate["frames"] == 8
    expected = tw.entropy(s["visibility"]) * 8 + tw.entropy(s["residual"]) * 4 + 2 * tw.RAW_POSITION_BITS
    assert math.isclose(rate["bits_per_frame"], expected / 8)
    assert math.isclose(rate["raw_bits_per_frame"], (8 + 6 * tw.RAW_POSITION_BITS) / 8)


def test_camera_pair_still_for_the_same_frame_and_moving_for_a_shift():
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((540, 960)) * 255).astype(np.uint8), (0, 0), 2)
    sift = cv2.SIFT.create(nfeatures=2000)
    still = tw.camera_pair(base, base.copy(), sift, 2.0)
    assert still["kind"] == "still" and still["corner_px"] < 0.5
    shifted = np.roll(base, 3, axis=1)
    moving = tw.camera_pair(base, shifted, sift, 2.0)
    assert moving["kind"] == "moving" and abs(moving["corner_px"] - 6.0) < 1.0  # 3 px at 540p is 6 px at 1080p
    flat = np.full_like(base, 128)
    assert tw.camera_pair(base, flat, sift, 2.0)["kind"] == "cut"


def test_camera_summary_counts_still_runs():
    kinds = ["still", "still", "moving", "still", "cut", "still", "still", "still"]
    s = tw.camera_summary([{"kind": k} for k in kinds], 0.5)
    assert math.isclose(s["still"], 6 / 8) and s["still_runs"] == 3
    assert s["still_run_s_median"] == 0.5 * 2


def test_mi_counts_split_each_cell_by_part():
    labels = np.zeros((8, 12), np.uint8)
    labels[:4, :4] = 1
    labels[0, 4] = 2
    c = tw.mi_counts(labels, 3)
    assert c.shape == (3, 2, 3)
    assert c[1, 0, 0] == 16 and c[2, 0, 1] == 1 and c[0, 0, 1] == 15 and c.sum() == 8 * 12


def test_pose_analysis_symbols_are_first_differences_along_a_track():
    size = (64, 48)
    mask = np.zeros((size[1], size[0]), bool)
    mask[4:44, 10:40] = True
    small = mask  # SAM size equals the window size here
    kp0 = np.zeros((17, 3), np.float32)
    kp0[:, 0], kp0[:, 1], kp0[:, 2] = 20.0, 20.0, 0.9
    dets, players = [], []
    for t in range(3):
        kp = kp0.copy()
        kp[:, 0] += t  # one px per frame
        dets.append({"boxes": np.array([[12, 6, 38, 42]], np.float32), "kps": kp[None]})
        players.append([(7, small, 0.9)])
    out = k4.pose_analysis("m", dets, players, size, 30.0)
    assert out["fitted_share"] == 1.0 and out["inside_median"] == 1.0 and out["track_starts"] == 1
    # 2 differences x 34 values; x moves 4 quantization steps at 0.25 px, y stays
    assert out["symbols"]["0.25"] == {"4": 34, "0": 34}
    assert out["symbols"]["1"] == {"1": 34, "0": 34}
    assert out["accel_box_per_frame2_median"] == 0.0


def test_k5_verdicts_bd_and_every_rate():
    anchor = [(10.0, 30.0), (20.0, 33.0), (40.0, 36.0)]
    half = [(5.0, 30.0), (10.0, 33.0), (20.0, 36.0)]
    v = k5.verdict(half, anchor, 0.5)
    assert v["basis"] == "bd" and math.isclose(v["bd_rate"], -0.5, rel_tol=1e-6)
    above = [(12.0, 45.0), (25.0, 46.0)]
    assert k5.verdict(above, anchor, 0.5)["basis"] == "better at every rate"
    below = [(12.0, 20.0), (25.0, 21.0)]
    assert k5.verdict(below, anchor, 0.5)["basis"] == "worse at every rate"


def test_k5_collapse_rate_interpolates_in_log_rate():
    def p(kbps: float, r: float) -> dict:
        return {"kbps": kbps, "decode": {"detectors": {"d": {"recall_10px": r}}}}

    rate = k5.collapse_rate([p(10, 0.2), p(100, 0.8), p(1000, 0.9)], "d", 0.9)
    target = 0.9 * 0.9
    assert rate is not None and rate > 100  # 0.81 lies between 0.8 and 0.9
    w = (target - 0.8) / 0.1
    assert math.isclose(rate, math.exp(math.log(100) + w * (math.log(1000) - math.log(100))))
    assert k5.collapse_rate([p(10, 0.9), p(100, 0.95)], "d", 0.9) is None


def test_k4_decision_reads_the_rule():
    summary = {"model": "m", "foreground_bit_share": {"48": {"player": 0.8}, "62": {"player": 0.7}},
               "pose": {"m": {"fitted_share": 0.95, "inside_median": 0.97}}, "rate_share_crf62": 0.3}
    d = k4.decide(summary)
    assert d["a_bits"] and d["b_fit"] and not d["c_rate"] and not d["passes_h1_screen"] and d["k6_gate"]
    summary["rate_share_crf62"] = 0.6
    assert not k4.decide(summary)["k6_gate"]


def test_k5_oracle_pastes_the_source_inside_the_disk_only():
    source = np.full((2, 20, 20, 3), 200, np.uint8)
    decoded = np.zeros_like(source)
    rows = [(1, 10, 10), (0, 0, 0)]
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        comp = k5.oracle(decoded, source, rows, Path(d) / "o.npy")
        assert comp[0, 10, 10, 0] == 200 and comp[0, 10, 15, 0] == 0 and comp[1].max() == 0
        q = k5.quality(comp, source, rows)
        assert q["ball_psnr"] < 100 and q["ball_psnr"] > k5.quality(decoded, source, rows)["ball_psnr"]


def test_choose_players_takes_the_two_present_tracks_that_travel_furthest():
    def blob(x: int, y: int) -> np.ndarray:
        m = np.zeros((60, 100), bool)
        m[y:y + 10, x:x + 4] = True
        return m

    found = []
    for t in range(10):
        found.append({"player": [(1, blob(10 + 3 * t, 10), 0.9), (2, blob(50, 40 - 2 * t), 0.9), (3, blob(90, 5), 0.9)]
                      + ([(4, blob(5 * t, 45), 0.9)] if t < 3 else []), "racket": [], "ball": []})
    chosen, stats = k4.choose_players(found)
    assert chosen == {1, 2}  # track 3 stands still; track 4 moves fastest but is present on 3 of 10 frames
    assert stats["3"]["body_heights_travelled"] == 0.0 and not stats["4"]["player"] and stats["4"]["presence"] == 0.3


def test_trajectory_rate_is_the_same_after_a_json_round_trip():
    import json

    s = tw.trajectory_symbols([(1, 10, 10), (1, 12, 11), (1, 14, 13), (1, 15, 13)])
    back = json.loads(json.dumps(s))
    assert tw.trajectory_rate([back], 30.0) == tw.trajectory_rate([s], 30.0)


def test_k5b_scores_the_composite_the_decoder_draws():
    from experiments.tennis import k5b
    from src.pointstream import ball

    rng = np.random.default_rng(2)
    h, w, n = 90, 160, 12
    source = rng.integers(0, 255, (n, h, w, 3), dtype=np.uint8)
    decoded = np.clip(source.astype(int) + rng.integers(-20, 20, source.shape), 0, 255).astype(np.uint8)
    track: ball.Track = [(10 + 9 * t, 40 + t) if t != 5 else None for t in range(n)]
    vel = ball.velocities(track)
    app = ball.Appearance((250, 240, 30), 4, 6, 2)
    crops = ball.Crops.build(track, source, vel)
    masks = k5b.judge_masks(crops)
    dec = [ball.crops_of(decoded, g["frames"], g["centres"], g["half"]) for g in crops.groups]
    mse = k5b.frame_mse(decoded, source)
    q = k5b.score_variant(crops, masks, dec, mse, "streak", app, (h, w))
    drawn = np.stack([ball.draw(decoded[t], p, vel[t], app) if p else decoded[t] for t, p in enumerate(track)])
    assert abs(q["frame_psnr"] - k5.mse_to_psnr(float(k5b.frame_mse(drawn, source).mean()))) < 1e-6
    rows = [(1, p[0], p[1]) if p else (0, 0, 0) for p in track]
    assert abs(q["ball_psnr"] - k5.quality(drawn, source, rows)["ball_psnr"]) < 1e-6
    base = k5b.score_variant(crops, masks, dec, mse, "decode", None, (h, w))
    assert abs(base["ball_psnr"] - k5.quality(decoded, source, rows)["ball_psnr"]) < 1e-6
    oracle = k5b.score_variant(crops, masks, dec, mse, "oracle", None, (h, w))
    with tempfile.TemporaryDirectory() as d:
        ref = k5.quality(k5.oracle(decoded, source, rows, Path(d) / "o.npy"), source, rows)
    assert abs(oracle["ball_psnr"] - ref["ball_psnr"]) < 1e-6 and abs(oracle["frame_psnr"] - ref["frame_psnr"]) < 1e-6


def test_k5b_decision_needs_both_judges_on_six_of_eight():
    from experiments.tennis import k5b

    def window(ball_ok: bool, frame_ok: bool) -> dict:
        return {"passes": {"streak-dcvc-uf-htl": {"ball": ball_ok, "frame": frame_ok},
                           "disk-dcvc-uf-htl": {"ball": False, "frame": True}}}

    six = {str(i): window(i < 6, True) for i in range(8)}
    assert k5b.decide(six)["adopted"] == {"disk": False, "streak": True}
    frame_cost = {str(i): window(True, i < 5) for i in range(8)}
    assert k5b.decide(frame_cost)["adopted"]["streak"] is False
    assert k5b.decide({str(i): window(True, True) for i in range(4)})["adopted"]["streak"] is False


def test_k5b_labels_read_only_frames_without_touching_them():
    from experiments.tennis import k5b

    frame = np.full((72, 128, 3), 50, np.uint8)
    frame.setflags(write=False)
    out = k5b.label(frame, "source")
    assert out.max() > 200 and frame.max() == 50

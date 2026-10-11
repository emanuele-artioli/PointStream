import math

import numpy as np

from experiments.visor import h3


def test_flow_box_covers_both_masks_with_even_sides():
    a = np.zeros((100, 200), bool)
    b = np.zeros((100, 200), bool)
    a[10:20, 30:41] = True
    b[50:61, 150:160] = True
    x0, y0, x1, y1 = h3.flow_box(a, b, pad=5)
    assert (x0, y0) == (25, 5) and x1 >= 165 and y1 >= 66
    assert (x1 - x0) % 2 == 0 and (y1 - y0) % 2 == 0 and x1 <= 200 and y1 <= 100


def test_flow_warps_the_reference_onto_the_target():
    rng = np.random.default_rng(0)
    import cv2

    texture = cv2.GaussianBlur(rng.integers(0, 255, (160, 200), dtype=np.uint8), (0, 0), 2.0)
    source = texture[20:140, 20:180]
    target = texture[23:143, 25:185]  # the scene moved 5 px left and 3 px up
    flow = h3.dis_flow(target, source, scale=1.0)
    centre = flow[30:90, 40:120]
    assert abs(float(np.median(centre[..., 0])) - 5) < 0.5 and abs(float(np.median(centre[..., 1])) - 3) < 0.5
    warped = h3.warp(source, flow)
    err = np.abs(warped[30:90, 40:120].astype(int) - target[30:90, 40:120].astype(int)).mean()
    assert err < np.abs(source[30:90, 40:120].astype(int) - target[30:90, 40:120].astype(int)).mean() / 4


def test_run_sends_a_reference_when_no_warp_is_within_the_margin():
    run = h3.Run(48.0, 0.05, 5)
    run.settle(0, None, 0, -1, 0.10, 100)  # empty bank
    assert run.reference[0] and run.bank == [0]
    run.settle(1, 0.14, 50, 0, 0.10, 100)  # within 0.05 of SVT-AV1
    assert not run.reference[1] and run.source[1] == 0 and run.lpips[1] == 0.14 and run.sse[1] == 50
    run.settle(2, 0.16, 50, 0, 0.10, 100)  # 0.06 worse: refresh with SVT-AV1's pixels
    assert run.reference[2] and run.lpips[2] == 0.10 and run.sse[2] == 100 and run.bank == [0, 2]
    for t in range(3, 10 - 5):
        run.bank.append(t)
    assert run.candidates() == run.bank[-h3.BANK:]


def test_at_rate_is_linear_in_log_rate_and_held_at_the_ends():
    rates = np.array([100.0, 10.0])
    values = np.array([0.1, 0.3])
    assert math.isclose(h3.at_rate(rates, values, math.sqrt(1000.0)), 0.2)
    assert h3.at_rate(rates, values, 1.0) == 0.3
    assert h3.at_rate(rates, values, 1000.0) == 0.1


def test_item_curves_rate_and_comparison():
    frames = 4
    a = {
        "pixels": np.array([0, 10, 10, 10]),
        "points": np.array([41.0, 62.0]),
        "svt_lpips": np.array([[np.nan, 0.1, 0.1, 0.1], [np.nan, 0.3, 0.3, 0.3]]),
        "svt_sse": np.full((2, frames), 30),
        "oracle_lpips": np.array([[np.nan, 0.1, 0.15, 0.15]]),
        "oracle_sse": np.full((1, frames), 30),
        "oracle_reference": np.array([[False, True, False, False]]),
        "run_point": np.array([41.0]), "run_margin": np.array([0.05]),
    }
    kbps = {"41": 90.0, "62": 10.0}
    out = h3.item_curves({"unit": "x"}, a, kbps, pose_kbps=0.0)
    row = out["oracle"][0]
    assert out["hand_frames"] == 3 and math.isclose(row["reference_share"], 1 / 3)
    assert math.isclose(row["kbps"], 30.0)  # one CRF 41 hand-frame in three
    want = 0.3 + (0.1 - 0.3) * (math.log(30) - math.log(10)) / (math.log(90) - math.log(10))
    assert math.isclose(row["svt_lpips_at_rate"], want)
    assert math.isclose(row["lpips"], (0.1 + 0.15 + 0.15) / 3)
    tripled = h3.item_curves({"unit": "x"}, a, kbps, pose_kbps=1.0, factor=3.0)["oracle"][0]
    assert math.isclose(tripled["kbps"], 91.0) and math.isclose(tripled["svt_lpips_at_rate"], 0.1)


def test_composite_replaces_only_the_mask():
    truth = np.zeros((6, 8, 3), np.uint8)
    inside = np.full((2, 4, 3), 200, np.uint8)
    mask = np.zeros((6, 8), bool)
    mask[2, 3] = mask[3, 5] = True
    out = h3.composite(truth, inside, mask, (2, 2, 6, 4))
    assert out[2, 3].tolist() == [200] * 3 and out[3, 5].tolist() == [200] * 3 and out.sum() == 2 * 3 * 200
    assert truth.sum() == 0


def test_svt_rate_at_quality_inverts_the_curve_and_refuses_extrapolation():
    svt = [{"kbps": 100.0, "lpips": 0.1}, {"kbps": 10.0, "lpips": 0.3}]
    assert math.isclose(h3.svt_rate_at_quality(svt, 0.2), math.sqrt(1000.0))
    assert h3.svt_rate_at_quality(svt, 0.35) is None and h3.svt_rate_at_quality(svt, 0.05) is None


def test_stream_curves_use_the_measured_reference_bits():
    frames = 4
    a = {
        "pixels": np.array([0, 10, 10, 10]), "points": np.array([41.0, 62.0]),
        "svt_lpips": np.array([[np.nan, 0.1, 0.1, 0.1], [np.nan, 0.3, 0.3, 0.3]]), "svt_sse": np.full((2, frames), 30),
        "oracle_lpips": np.array([[np.nan, 0.1, 0.15, 0.15]]), "oracle_sse": np.full((1, frames), 30),
        "oracle_reference": np.array([[False, True, False, False]]),
        "run_point": np.array([41.0]), "run_margin": np.array([0.05]),
        "stream_hand_bits": np.array([60000.0]), "stream_lpips": np.array([[np.nan, 0.2, 0.2, 0.2]]),
        "stream_sse": np.full((1, frames), 30), "full_points": np.array([41.0]), "full_hand_bits": np.array([180000.0]),
    }
    meta = {"unit": "x", "frames": 4, "fps": 2.0}  # a 2 s window
    out = h3.stream_curves(meta, a, {"41": 90.0, "62": 10.0}, pose_kbps=0.0)
    row = out["oracle"][0]
    assert math.isclose(row["kbps"], 30.0) and math.isclose(row["lpips"], 0.2)
    assert math.isclose(row["svt_kbps_at_quality"], 30.0) and math.isclose(row["saving"], 0.0, abs_tol=1e-12)
    assert math.isclose(out["full_rate_check"]["41"], 1.0)  # 180 kbit over 2 s against 90 kbps
    # One reference in three hand-frames: 30 kbps, against 30 kbps at the inter-frame cost.
    assert math.isclose(row["reference_cost_ratio"], 1.0)

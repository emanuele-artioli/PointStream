"""G5 oracle: clip timing, colour round trip, scores, rate-distortion verdicts and arm B's model."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from experiments.background import g5


def test_frame_rates_are_recognised():
    assert g5.frame_rate(list(np.arange(240) / 50.0)) == Fraction(50)
    assert g5.frame_rate(list(np.arange(240) * 1001 / 60000)) == Fraction(60000, 1001)
    assert g5.frame_rate(list(np.arange(1200) / 10.0)) == Fraction(10)
    with pytest.raises(ValueError):
        g5.frame_rate([0.0, 0.07, 0.14])


def test_excerpt_keeps_the_dense_window_and_fits_the_stretch():
    assert g5.excerpt_start(1200, [0, 1, 2], 240) == 0
    assert g5.excerpt_start(1200, [500, 520], 240) == 500
    assert g5.excerpt_start(1028, [900, 940], 240) == 788
    assert g5.excerpt_start(1200, [], 240) == 0


def test_references_refresh_every_tenth_of_a_second():
    assert g5.refresh_reference(12, Fraction(50)) == [None, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 10]
    assert g5.refresh_reference(8, Fraction(60000, 1001))[1:8] == [0, 0, 0, 0, 0, 0, 6]
    assert g5.refresh_reference(4, Fraction(10)) == [None, 0, 1, 2]


def test_yuv420_round_trip_is_close_on_smooth_content():
    pytest.importorskip("torch")
    y, x = np.mgrid[0:48, 0:96]
    rgb = np.stack([x * 2, y * 4, (x + y) % 256], axis=-1).astype(np.uint8)[None].repeat(2, 0)
    back = g5.yuv420_to_rgb(g5.rgb_to_yuv420(rgb))
    assert back.shape == rgb.shape and back.dtype == np.uint8
    assert np.abs(back.astype(int) - rgb.astype(int)).mean() < 2.0


def make_clip(n: int = 3, h: int = 24, w: int = 48) -> g5.Clip:
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (n, h, w, 3), dtype=np.uint8)
    keep = np.ones((n, h, w), bool)
    keep[:, :, : w // 4] = False
    clip = g5.Clip("visor-window/T", "visor-window", Fraction(50), 0, frames, keep)
    clip.scored = {"dataset": (list(range(n)), keep)}
    return clip


def test_score_reads_only_the_visible_background():
    clip = make_clip()
    decoded = clip.frames.copy()
    decoded[~clip.keep] = 0  # errors in the foreground do not count
    decoded[:, 0, -1] = 255 - decoded[:, 0, -1]  # one wrong background pixel per frame
    s = g5.score(clip, decoded)
    count = int(clip.keep[0].sum())
    diff = clip.frames[:, 0, -1].astype(int) - decoded[:, 0, -1].astype(int)
    expected = np.mean([10 * math.log10(255.0**2 / ((d**2).sum() / (3 * count))) for d in diff])
    assert s["dataset"]["psnr_v"] == pytest.approx(expected)
    assert s["dataset"]["frames"] == 3 and s["frame"]["psnr"] < s["dataset"]["psnr_v"]


def test_pareto_keeps_points_better_than_every_cheaper_one():
    assert g5.pareto([(100, 30), (200, 29), (300, 33), (50, 28)]) == [(50, 28), (100, 30), (300, 33)]


def test_bd_rate_of_half_the_rate_is_minus_half():
    anchor = [(100.0, 30.0), (200.0, 33.0), (400.0, 36.0), (800.0, 38.0)]
    half = [(r / 2, q) for r, q in anchor]
    assert g5.bd_rate(anchor, anchor) == pytest.approx(0.0, abs=1e-9)
    assert g5.bd_rate(half, anchor) == pytest.approx(-0.5, abs=1e-6)
    assert g5.bd_rate([(r * 3, q) for r, q in anchor], anchor) == pytest.approx(2.0, abs=1e-6)


def test_without_overlap_the_verdict_is_dominance_or_unclear():
    anchor = [(100.0, 34.0), (200.0, 36.0), (400.0, 38.0)]
    assert g5.clip_verdict([(150.0, 25.0), (300.0, 27.0)], anchor)["bd_rate"] == math.inf
    assert g5.clip_verdict([(150.0, 45.0), (300.0, 47.0)], anchor)["bd_rate"] == -math.inf
    # Below the anchor's cheapest rate there is nothing to compare with.
    assert g5.clip_verdict([(20.0, 25.0), (40.0, 27.0)], anchor)["basis"] == "unclear"
    # Above its range the anchor counts with its best quality.
    assert g5.clip_verdict([(900.0, 30.0), (1600.0, 33.0)], anchor)["bd_rate"] == math.inf


def test_pilot_decision():
    ruled = {c: {"bd_rate": v, "converged": True} for c, v in zip("abc", (1.2, math.inf, 3.0))}
    assert g5.pilot_decision(ruled) == "ruled out"
    assert g5.pilot_decision({**ruled, "c": {"bd_rate": 0.4, "converged": True}}) == "candidate"
    assert g5.pilot_decision({**ruled, "c": {"bd_rate": -0.1, "converged": True}}) == "candidate"
    assert g5.pilot_decision({**ruled, "c": {"bd_rate": None, "converged": True}}) == "unclear"
    assert g5.pilot_decision({**ruled, "c": {"bd_rate": 2.0, "converged": False}}) == "unclear"


def test_group_gate_reads_the_median_clip():
    verdicts = {c: {"bd_rate": v} for c, v in zip("abcd", (-0.3, -0.1, 0.2, None))}
    gate = g5.group_gate(verdicts)
    assert gate["median_bd_rate"] == pytest.approx(-0.1) and gate["passes"] and gate["unclear"] == 1


def test_training_log_curve_and_convergence():
    lines = []
    for e in range(61):
        lines.append(f"Train - Epoch {e} [30/60]    lr: 1e-3    img/s: 30    loss: 1    bpp: 0.9000    psnr: 10.0")
        lines.append(f"Train - Epoch {e} [60/60]    lr: 1e-3    img/s: 30    loss: 1    r_loss: 0.1    "
                     f"d_loss: 0.1    bpp: {0.05 - e * 1e-5:.4f}    psnr: {30 + e * 0.001:.4f}")
    curve = g5.train_curve("\n".join(lines))
    assert len(curve) == 61 and curve[0]["psnr"] == 30.0
    assert g5.converged(curve, 30)["converged"]
    rising = [{**p, "psnr": 30 + p["epoch"] * 0.05} for p in curve]
    assert not g5.converged(rising, 30)["converged"]
    # 12 steps per epoch logged every 8: the epoch's last logged step stands for it.
    short = "\n".join(f"Train - Epoch {e} [8/12]    img/s: 1    bpp: 0.5000    psnr: {15 + e:.4f}" for e in range(2))
    assert [p["psnr"] for p in g5.train_curve(short)] == [15.0, 16.0]


def test_nvrc_configs_scale_the_base_grid_with_the_frames(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    paths = g5.nvrc_configs(tmp_path, 240, {"s1": 360, "s2": 30}, 200.0)
    model = yaml.safe_load(paths["model"].read_text())
    assert model["config"]["base_encoding"]["base_grid_size"] == [80, 9, 16, 6]
    assert math.prod(model["config"]["decoder"]["scales_hw"]) == 60 and 540 % 60 == 0 and 960 % 60 == 0
    s1 = yaml.safe_load(paths["s1"].read_text())
    assert s1["epochs"] == 360 and s1["eval_epochs"] == 30
    smoke = yaml.safe_load(g5.nvrc_configs(tmp_path / "smoke", 48, {"s1": 2, "s2": 1}, 50.0)["s2"].read_text())
    assert smoke["epochs"] == 1 and smoke["eval_epochs"] == 1
    assert all(p % 60 == 0 for p in g5.NVRC_PATCH[1:])


def test_refresh_groups_reference_earlier_groups():
    from experiments.background import g5_cond

    refs = g5.refresh_reference(13, Fraction(50))
    groups = g5_cond.refresh_groups(refs)
    assert groups[0] == [0] and groups[1] == [1, 2, 3, 4, 5] and groups[2] == [6, 7, 8, 9, 10]
    seen: set[int] = set()
    for group in groups:
        r = refs[group[0]]
        assert r is None or r in seen
        seen.update(group)
    assert sorted(seen) == list(range(13))


def test_widths_grow_with_frames_and_lambda():
    from experiments.background import g5_cond

    assert g5_cond.widths_for(200.0, 240) == g5_cond.BASE_WIDTHS
    assert all(a < b for a, b in zip(g5_cond.widths_for(50.0, 240), g5_cond.widths_for(800.0, 240)))
    assert all(a < b for a, b in zip(g5_cond.widths_for(200.0, 48), g5_cond.widths_for(200.0, 240)))


def test_entropy_bits():
    from experiments.background import g5_cond

    assert g5_cond.entropy_bits(np.zeros(100, int)) == pytest.approx(0.0)
    assert g5_cond.entropy_bits(np.array([0, 1] * 50)) == pytest.approx(100.0)


def test_warp_follows_the_flow():
    torch = pytest.importorskip("torch")
    from experiments.background import g5_cond

    image = torch.zeros(1, 3, 12, 24)
    image[0, :, 5, 10] = 1.0
    flow = torch.zeros(1, 12, 24, 2)
    flow[..., 0] = 3.0  # frame_t(x) = reference(x + 3)
    warped, valid = g5_cond.warp(image, flow)
    assert float(warped[0, 0, 5, 7]) == pytest.approx(1.0)
    assert float(valid[0, 0, 5, 23]) == 0.0 and float(valid[0, 0, 5, 20]) == 1.0


def test_arm_b_fits_prices_and_rolls_out_on_a_tiny_clip():
    pytest.importorskip("torch")
    from experiments.background import g5_cond

    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (48, 108, 3), dtype=np.uint8)
    frames = np.stack([np.roll(base, k, axis=1) for k in range(6)])[:, :, :96]
    keep = np.ones(frames.shape[:3], bool)
    refs = g5.refresh_reference(6, Fraction(10))
    flows = g5_cond.oracle_flows(frames, refs, workers=1)
    out = g5_cond.fit(frames, keep, flows, refs, lamb=200.0, epochs=2, device="cpu")
    assert out["decoded"].shape == frames.shape and out["decoded"].dtype == np.uint8
    assert out["bits"]["latents"] > 0 and out["bits"]["weights"] > 0
    assert out["curve"][-1]["epoch"] == 2 and out["render_ms_per_frame"] > 0

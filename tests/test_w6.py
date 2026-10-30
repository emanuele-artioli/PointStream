from __future__ import annotations

import numpy as np
import pytest

from experiments.tennis import w6, w6_specs
from experiments.tennis.k6b_generate import keep_row


def test_log_weight_and_interpolation() -> None:
    assert w6.log_weight(10.0, 20.0, 5.0) == pytest.approx(0.5)
    assert w6.at_rate(10.0, (5.0, 2.0), (20.0, 4.0)) == pytest.approx(3.0)
    assert np.allclose(w6.at_rate(10.0, (20.0, np.array([1.0, 2.0])), (5.0, np.array([3.0, 4.0]))), [2.0, 3.0])
    with pytest.raises(ValueError):
        w6.log_weight(30.0, 20.0, 5.0)  # never extrapolated


def test_pose_check_uses_the_larger_of_anchor_and_scaled_keypoint_floor() -> None:
    assert w6.pose_check(0.10, 0.12, 0.01)["pass"]
    assert not w6.pose_check(0.10, 0.05, 0.01)["pass"]  # above the anchor and above 1.5 x 0.01
    assert w6.pose_check(0.10, 0.05, 0.07)["pass"]  # the floor 1.5 x 0.07 = 0.105 admits it
    assert w6.pose_check(0.10, 0.05, 0.07)["limit"] == pytest.approx(0.105)


def test_time_check_is_an_or_of_flicker_and_energy() -> None:
    both = w6.time_check(5.0, 3.0, 10.0, 4.0)
    assert not both["pass"] and not both["pass_flicker"] and not both["pass_energy"]
    assert w6.time_check(2.0, 3.0, 10.0, 4.0)["pass"]  # flicker alone
    assert w6.time_check(5.0, 3.0, 5.0, 4.0)["pass"]  # energy 5 <= 1.25 x 4 alone
    assert w6.time_check(5.0, 3.0, 5.01, 4.0)["pass"] is False


def test_decide_reports_each_check_and_conjunction() -> None:
    rows = {"pose": {"method": 0.1, "anchor": 0.2, "coded": 0.01},
            "time": {"flicker": 5.0, "anchor_flicker": 3.0, "energy": 10.0, "source_energy": 4.0}}
    out = w6.decide(rows)
    assert out["pose"]["pass"] and not out["time"]["pass"] and not out["pass"]


def test_block_bootstrap_brackets_the_mean_and_ignores_nan() -> None:
    values = np.concatenate([np.full(48, 1.0), [np.nan], np.full(48, 3.0)])
    lo, hi = w6.block_bootstrap(values, block=12, n=300, seed=1)
    assert lo <= 2.0 <= hi
    assert np.isnan(w6.block_bootstrap([np.nan])[0])


def test_racket_region_and_stats() -> None:
    region = np.zeros((40, 60), bool)
    region[5:35, 10:30] = True
    hull = np.zeros_like(region)
    hull[5:35, 10:20] = True
    racket = w6.racket_region(region, hull)
    assert racket.sum() == 30 * 10
    predicted = np.zeros_like(region)
    predicted[5:35, 10:25] = True
    written = np.zeros_like(region)
    stats = w6.racket_stats(racket, predicted, written)
    assert stats is not None and stats[0] == 300 and stats[1] == pytest.approx(0.5) and stats[2] == 0.0
    assert w6.racket_stats(racket[:3], predicted[:3], written[:3]) is None  # too few pixels


def test_racket_summary_counts_only_cells_with_a_racket() -> None:
    pixels = np.array([[300.0, np.nan], [200.0, 400.0]])
    covered = np.array([[0.0, np.nan], [0.6, 0.2]])
    out = w6.racket_summary(pixels, covered)
    assert out["cells_with_racket"] == 3 and out["cells"] == 4
    assert out["overlap_any"] == pytest.approx(2 / 3) and out["overlap_half"] == pytest.approx(1 / 3)


def test_reproduction_statistics() -> None:
    old = np.zeros((2, 4, 4, 3), np.uint8)
    new = old.copy()
    assert w6.reproduction(old, new)["within_one_level"]
    new[1, 0, 0, 0] = 3
    out = w6.reproduction(old, new)
    assert out["max_abs_diff"] == 3 and out["frames_identical"] == 1 and not out["within_one_level"]


def test_keep_row_keeps_ticks_smokes_and_keep_all() -> None:
    assert keep_row(True, 0, False) and not keep_row(False, 0, False)
    assert keep_row(False, 8, False) and keep_row(False, 0, True)


def test_generate_spec_keeps_the_frozen_recipe_and_adds_keep_all() -> None:
    spec = w6_specs.generate(",".join(w6_specs.UNITS))
    args = spec["arguments"]
    assert args[0] == "generate" and args[-1] == "--keep-all"
    assert args[args.index("--units") + 1] == ",".join(w6_specs.UNITS)
    assert args[args.index("--steps") + 1] == "30" and args[args.index("--from-t") + 1] == "0"
    assert args[args.index("--cfg") + 2] == "2.0" and "{frames}" in args
    assert spec["scale"]["frames"] == {"smoke": 8, "full": 0}
    assert set(spec["gpu_models"]) == {"RTX A6000", "RTX 6000 Ada"} and spec["smoke"]["seconds"] <= 600
    assert args[args.index("--gpu-class") + 1] == "RTX" and spec["full"]["seconds"] <= 2700


def test_post_spec_stages_every_input_once_and_pins_it() -> None:
    gens = [f"/d/g/published.tar@{'a' * 64}"]
    spec = w6_specs.post(gens)
    assert spec["arguments"][0] == "post" and "--clips" in spec["arguments"]
    assert len(spec["staged_inputs"]) == len(spec["inputs"]) and spec["scale"]["frames"]["full"] == 240
    assert all(len(x["sha256"]) == 64 for x in spec["staged_inputs"])
    assert spec["full"]["seconds"] <= 2700 and spec["smoke"]["seconds"] <= 600


def test_majority_vote_and_smoothing():
    a = np.array([[1, 1, 0, 0]], bool)
    b = np.array([[1, 0, 0, 1]], bool)
    c = np.array([[0, 1, 0, 1]], bool)
    assert w6.majority_vote([a, b, c]).tolist() == [[True, True, False, True]]
    assert w6.majority_vote([a, b]).tolist() == [[True, True, False, True]]  # a tie is foreground
    assert w6.majority_vote([a]).tolist() == a.tolist()
    with pytest.raises(ValueError):
        w6.majority_vote([])
    seq = {0: a, 1: b, 2: c, 3: ~a}
    assert np.array_equal(w6.smooth_masks(seq, 1), w6.majority_vote([a, b, c]))
    assert np.array_equal(w6.smooth_masks(seq, 0), w6.majority_vote([a, b]))  # the ends use the neighbours that exist
    assert np.array_equal(w6.smooth_masks(seq, 3), w6.majority_vote([c, ~a]))
    flicker = np.zeros((1, 8), bool)
    flicker[0, 3] = True  # one frame of noise is removed by the vote
    assert not w6.smooth_masks({t: np.roll(flicker, 0)[:, t:t + 1] for t in range(8)}, 3).any()


def test_split_shares_and_decision():
    shares = w6.split_shares(10.0, 4.0, 9.0)
    assert shares["mask_share"] == pytest.approx(0.6) and shares["content_share"] == pytest.approx(0.1)
    assert np.isnan(w6.split_shares(0.0, 1.0, 1.0)["mask_share"])
    assert w6.split_decision(0.6, 0.1)["fix"] == "masks"
    assert w6.split_decision(0.1, 0.6)["fix"] == "content"
    assert w6.split_decision(0.25, 0.05)["fix"] == "composite"  # the larger share is below 0.3
    assert w6.split_decision(0.5, 0.45)["fix"] == "composite"  # within 0.1 of each other
    assert w6.split_decision(0.5, 0.4)["fix"] == "masks"
    assert w6.split_decision(float("nan"), 0.4)["fix"] == "undefined"
    out = w6.excess_shares({"H": 28.0, "G+S": 22.0, "S+R": 25.0, "G+Rs": 27.0, "B": 17.0}, 18.0)
    assert out["excess"]["H"] == pytest.approx(10.0)
    assert out["mask_share"] == pytest.approx(0.6) and out["content_share"] == pytest.approx(0.3)
    assert out["decision"]["fix"] == "masks"


def test_split_bootstrap_and_control():
    rng = np.random.default_rng(1)
    h = 28 + rng.normal(0, 0.5, 96)
    series = {"H": h, "G+S": h - 6, "S+R": h - 1, "G+Rs": h - 0.5, "B": h * 0 + 17}
    boot = w6.split_bootstrap(series, np.full(96, 18.0), draws=200)
    assert boot["mask_share"][0] < 0.65 and boot["mask_share"][1] > 0.55 and boot["content_share"][1] < 0.3
    assert boot["share_mask_larger_by_lead"] == 1.0
    idx = w6.block_resamples(50, block=24, draws=5)
    assert idx.shape == (5, 50) and idx.max() < 50
    assert w6.control_max_diff(np.array([np.nan, 1.0]), np.array([np.nan, 1.0 + 1e-8])) == pytest.approx(1e-8)
    assert w6.control_max_diff(np.array([np.nan, 1.0]), np.array([0.0, 1.0])) == float("inf")

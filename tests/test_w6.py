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
    spec = w6_specs.generate(w6_specs.UNITS[0])
    args = spec["arguments"]
    assert args[0] == "generate" and args[-1] == "--keep-all"
    assert args[args.index("--units") + 1] == w6_specs.UNITS[0]
    assert args[args.index("--steps") + 1] == "30" and args[args.index("--from-t") + 1] == "0"
    assert args[args.index("--cfg") + 2] == "2.0" and "{frames}" in args
    assert spec["scale"]["frames"] == {"smoke": 8, "full": 0}
    assert spec["gpu_models"] == ["RTX 6000 Ada"] and spec["smoke"]["seconds"] <= 600
    names = {x["name"] for x in spec["staged_inputs"]}
    assert all(f"{{staged:{n}}}" in args or n.startswith("w") for n in names)


def test_masks_and_judge_specs_pin_every_staged_input() -> None:
    gens = [f"/d/g{i}/published.tar@{'a' * 63}{i}" for i in range(2)]
    spec = w6_specs.masks(gens)
    assert len(spec["staged_inputs"]) == len(spec["inputs"]) and spec["arguments"].count("--generated") == 2
    judge = w6_specs.judge(gens, f"/d/m/published.tar@{'b' * 64}")
    assert judge["scale"]["frames"]["full"] == 240 and "--clips" in judge["arguments"]
    assert all(len(x["sha256"]) == 64 for x in judge["staged_inputs"] if x["name"] != "weights")

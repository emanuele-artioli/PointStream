"""G1e: forced reference ages, the oracle, the forward-backward floor and the decision rule."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from experiments.background import camera, depth, g1d, g1e
from tests.background.test_depth import LENS, POSE_B, POSE_C, render

W, H = camera.ANALYSIS_SIZE


def test_window_targets_start_at_one_second_every_tenth_frame():
    times = np.arange(240) / 50.0
    chosen = g1e.targets(times, "visor-window")
    assert chosen[0] == 50 and np.all(np.diff(chosen) == g1e.WINDOW_STRIDE) and chosen[-1] <= 239


def test_stretch_targets_are_one_per_second():
    times = np.arange(1200) / 10.0
    chosen = g1e.targets(times, "visor-long")
    assert chosen[:3] == [10, 20, 30] and len(chosen) == 119


def test_ages_become_gaps_and_coincide_at_ten_frames_per_second():
    assert g1e.gaps(np.arange(240) / 50.0, ["frame", 0.1, 1.0]) == {1: ["frame"], 5: [0.1], 50: [1.0]}
    assert g1e.gaps(np.arange(240) / 59.94, ["frame", 0.1, 1.0]) == {1: ["frame"], 6: [0.1], 60: [1.0]}
    assert g1e.gaps(np.arange(1200) / 10.0, ["frame", 0.1, 1.0]) == {1: ["frame", 0.1], 10: [1.0]}


def test_shards_partition_the_rest():
    clips = [{"id": f"visor-window/V{i}", "group": "visor-window", "video": f"V{i}"} for i in range(6)]
    clips += [{"id": f"visor-long/V{i}", "group": "visor-long", "video": f"V{i}"} for i in range(12)]
    clips += [{"id": f"ott/test_{i}", "group": "ott", "video": f"test_{i}"} for i in range(7)]
    spec = {"clips": clips}
    rest = [c["id"] for c in g1e.select_clips(spec, "rest")]
    shards = [c["id"] for k in (1, 2, 3) for c in g1e.select_clips(spec, f"rest:{k}/3")]
    assert sorted(shards) == sorted(rest) and len(shards) == len(set(shards))
    pilot = {c["id"] for c in g1e.select_clips(spec, "pilot")}
    assert len(pilot) == 4 and not pilot & set(rest)
    assert len(rest) + len(pilot) == 6 + g1d.STRETCHES + 7


def textured(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(cv2.GaussianBlur(rng.uniform(0, 255, (H, W)).astype(np.float32), (0, 0), 2.0) * 3 - 255, 0, 255).astype(np.uint8)


def test_forward_backward_is_small_for_a_consistent_pair_and_large_for_unrelated_images():
    image = textured(3)
    region = np.ones((H, W), bool)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 1.5], [0, 1, -1.0]]), (W, H), borderMode=cv2.BORDER_REFLECT)
    same = g1e.forward_backward_p90(image, shifted, region)
    other = g1e.forward_backward_p90(image, textured(4), region)
    assert same is not None and other is not None
    assert same < 1.0 < 4.0 < other


def test_oracle_takes_the_best_part_and_dominates_each():
    parts = {"epi": {"measured": True, "covered_explained": False, "flow_p90_px": 2.5, "hole_share": 0.02},
             "planes4": {"measured": True, "covered_explained": True, "flow_p90_px": 1.6, "hole_share": 0.03},
             "tri": {"measured": True, "covered_explained": False, "flow_p90_px": 1.2, "hole_share": 0.4}}
    o = g1e.oracle(parts)
    assert o["covered_explained"] and o["explained_by"] == ["planes4"] and o["best"] == "planes4"
    assert o["flow_p90_px"] == 1.6 and not o["epi_matches"]
    assert not g1e.oracle({m: {"measured": False} for m in g1e.ORACLE_OF})["covered_explained"]


@pytest.fixture(scope="module")
def clip() -> dict[str, object]:
    """A reference, a target 0.1 s later and companions 0.5 s and 1.0 s after the reference."""
    poses = [(np.eye(3), np.zeros(3)), (POSE_B[0], POSE_B[1] * 0.4), POSE_B, POSE_C]
    gray = np.stack([render(*p)[0] for p in poses])
    return {"gray": gray, "times": np.array([0.0, 0.1, 0.5, 1.0])}


def test_a_rigid_pair_is_explained_by_the_oracle(clip):
    gray, times = clip["gray"], clip["times"]
    fg = np.zeros(gray.shape, bool)
    bg = ~fg
    state = {"cal_lens": LENS, "companions": {}}
    feats: dict[int, camera.Features] = {}
    state["companions"][0] = g1e.companions_for(LENS, 0, gray, bg, times, feats)
    assert [c["index"] for c in state["companions"][0]] == [2, 3]
    products = g1d.Products(state, gray, bg)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g1d.NEAR_FG_PX + 1, 2 * g1d.NEAR_FG_PX + 1))
    rec = g1e.evaluate_pair(LENS, gray, fg, times, 1, 0, g1e.VISOR_METHODS, products, feats, kernel)
    assert rec["matched"] and rec["age_s"] == 0.1
    m = rec["methods"]
    assert set(m) == {*g1e.VISOR_METHODS, "oracle"}
    assert m["epi"]["covered_explained"] and m["oracle"]["covered_explained"]
    assert all(m["oracle"]["covered_explained"] >= bool(m[k].get("covered_explained")) for k in g1e.ORACLE_OF)
    assert m["epi"]["fb_p90_px"] is not None and m["epi"]["fb_p90_px"] < camera.EXPLAINED_PX
    assert m["planes4"]["reference_bytes"] > 0 and m["tri"]["reference_bytes"] > 0
    assert isinstance(depth.epic_fields_lens(), camera.Lens)


def table(window: dict[str, dict[str, float]], stretch: dict[str, dict[str, float]], fb: float = 1.0,
          ott: float = 1.2) -> dict[str, object]:
    def cells(shares: dict[str, dict[str, float]]) -> dict[str, object]:
        out: dict[str, object] = {"clips": 3}
        for age, by_method in shares.items():
            by_method = {**by_method, "epi": by_method["oracle"]}
            out[age] = {m: {"explained_share_median_clip": v, "fb_p90_px_median": fb, "flow_p90_px_median": 1.0,
                            "frame_bytes_median": 128.0 if m == "planes4" else 24.0,
                            "reference_bytes_median": 900.0 if m == "planes4" else 4000.0}
                        for m, v in by_method.items()}
        return out

    return {"visor-window": cells(window), "visor-long": cells(stretch),
            "ott": {"clips": 7, **{a: {"h1": {"flow_p90_px_median": ott}} for a in ("frame", "0.1", "1.0")}}}


AGES = ["frame", 0.1, 1.0]


def test_rule_step1_when_only_one_frame_passes():
    t = table({"frame": {"oracle": 0.9, "planes4": 0.8, "tri": 0.5}, "0.1": {"oracle": 0.4, "planes4": 0.3, "tri": 0.2},
               "1.0": {"oracle": 0.1, "planes4": 0.0, "tri": 0.0}},
              {"frame": {"oracle": 0.6, "planes4": 0.5, "tri": 0.3}, "0.1": {"oracle": 0.6, "planes4": 0.5, "tri": 0.3},
               "1.0": {"oracle": 0.1, "planes4": 0.0, "tri": 0.0}})
    d = g1e.decide(t, {"dense": {}, "own": {}}, AGES)
    assert d["step"] == 1 and d["refresh_age"]["visor-window"]["oracle"] == "frame"


def test_rule_step2_picks_the_cheaper_sendable_warp_and_asks_for_refinement():
    shares = {"frame": {"oracle": 0.9, "planes4": 0.8, "tri": 0.7}, "0.1": {"oracle": 0.7, "planes4": 0.55, "tri": 0.5},
              "1.0": {"oracle": 0.3, "planes4": 0.1, "tri": 0.1}}
    t = table(shares, shares)
    d = g1e.decide(t, {"dense": {}, "own": {}}, AGES)
    assert d["step"] == 2 and d["qualifying"] == {"planes4": 0.1, "tri": 0.1} and d["chosen"] == "planes4"
    assert d["refine_at_0.3s"] and d["floor"]["1.0"]["measurable"] and d["ott_holds_at_1s"]


def test_rule_step3_and_a_failing_mask_check_uses_the_dense_masks():
    window = {"frame": {"oracle": 0.9, "planes4": 0.8, "tri": 0.6}, "0.1": {"oracle": 0.7, "planes4": 0.55, "tri": 0.4},
              "1.0": {"oracle": 0.6, "planes4": 0.2, "tri": 0.1}}
    stretch = {"frame": {"oracle": 0.5, "planes4": 0.3, "tri": 0.2}, "0.1": {"oracle": 0.5, "planes4": 0.3, "tri": 0.2},
               "1.0": {"oracle": 0.1, "planes4": 0.0, "tri": 0.0}}
    t = table(window, stretch, fb=2.5)
    own = table(window, stretch)
    dense = table(window, {a: {m: min(1.0, v + 0.3) for m, v in by.items()} for a, by in stretch.items()})
    d = g1e.decide(t, {"dense": dense, "own": own}, AGES)
    assert not d["mask_check_holds"] and d["stretch_verdict_from"].startswith("dense")
    assert d["step"] == 2 and "planes4" in d["qualifying"]
    assert not d["floor"]["0.1"]["measurable"]
    d = g1e.decide(t, {"dense": {}, "own": {}}, AGES)
    assert d["step"] == 3 and not d["qualifying"]

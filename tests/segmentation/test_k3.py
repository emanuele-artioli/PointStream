import math

import numpy as np
import pytest

from experiments.visor import h3c, k3
from experiments.visor import k3_ref as ref

NALS = [{"nal_bytes": 100, "sps_bytes": 10}, {"nal_bytes": 50, "sps_bytes": 0}]
ESTIMATE_A = [{"frames": [0], "estimated_bits": 1000.0, "estimated_region_bits": 250.0},
              {"frames": [1, 2], "estimated_bits": 400.0, "estimated_region_bits": 400.0}]


def test_arms_match_the_tree_script():
    assert k3.ARMS == ref.ARMS and set(k3.BILLED) <= set(k3.ARMS)


def test_calibration_of_the_unmodified_arm_is_h3cs_split():
    out = k3.calibrated_bits(NALS, ESTIMATE_A, ESTIMATE_A, 160 + h3c.CONTAINER_HEADER)
    split = h3c.nal_region_bits(NALS, ESTIMATE_A, 160 + h3c.CONTAINER_HEADER)
    assert math.isclose(out["region_bits"], split["region_bits"]) and math.isclose(out["bits"], split["bits"])
    assert math.isclose(out["estimate_over_actual"], split["estimate_over_actual"])


def test_every_arm_takes_the_unmodified_codecs_factor_per_nal():
    arm = [{"frames": [0], "estimated_bits": 1000.0, "estimated_region_bits": 250.0},  # the shared I frame
           {"frames": [1, 2], "estimated_bits": 200.0, "estimated_region_bits": 100.0}]  # half the P bits, a quarter on R
    out = k3.calibrated_bits(NALS, ESTIMATE_A, arm, 160)
    # NAL 0: 880 real bits for 1000 estimated; NAL 1: 400 real for 400 estimated.
    assert math.isclose(out["region_bits"], 250 * 0.88 + 100 * 1.0)
    assert math.isclose(out["bits"], 1000 * 0.88 + 200 * 1.0)
    assert out["estimated_region_bits"] == 350.0 and out["estimated_bits"] == 1200.0
    with pytest.raises(RuntimeError):
        k3.calibrated_bits(NALS, ESTIMATE_A, arm[:1], 160)
    with pytest.raises(RuntimeError):
        k3.calibrated_bits(NALS, ESTIMATE_A, [arm[0], {**arm[1], "frames": [1]}], 160)


def test_reference_targets_the_groups_first_or_middle_frame():
    assert ref.target_frame("C", list(range(9, 17))) == 9
    assert ref.target_frame("E", list(range(9, 17))) == 9
    assert ref.target_frame("D", list(range(9, 17))) == 13
    assert ref.target_frame("D", [233, 234, 235]) == 234  # a short last group


def test_reference_mask_is_the_hand_before_and_at_the_target():
    region = np.zeros((3, 4, 4), bool)
    region[0, 0, 0] = True
    region[2, 3, 3] = True
    mask = ref.reference_mask(region, 0, 2)
    assert mask.sum() == 2 and mask[0, 0] and mask[3, 3]


def _row(kbps: float, lpips: float) -> dict[str, float]:
    return {"kbps": kbps, "lpips": lpips, "dists": lpips, "psnr": 40 - 100 * lpips}


ANCHOR = [_row(1.0, 0.30), _row(4.0, 0.20), _row(16.0, 0.10)]


def test_half_the_rate_passes_and_keypoints_are_billed():
    half = {item: [{**r, "kbps": r["kbps"] / 2} for r in ANCHOR] for item in h3c.PILOT}
    anchors = {item: ANCHOR for item in h3c.PILOT}
    out = k3.versus(half, anchors)
    assert out["verdict"]["verdict"] == "passes" and math.isclose(out["verdict"]["mean_bd"], -0.5)
    assert all(b["kbps"] == r["kbps"] + 2.0 for b, r in zip(k3.billed(ANCHOR, 2.0), ANCHOR))
    # Billed 8 kbps, the same curve is worse at every point: the gate fails.
    worse = k3.versus({i: k3.billed(rows, 8.0) for i, rows in half.items()}, anchors)
    assert worse["verdict"]["verdict"] == "fails"


def test_gate_takes_the_better_of_c_and_d_and_waits_for_both():
    passes = {"verdict": {"verdict": "passes", "mean_bd": -0.2}}
    fails = {"verdict": {"verdict": "fails", "mean_bd": 0.1}}
    assert k3.decide({"C": fails, "D": passes}) | {"arms": None} == {
        "k3": "passes", "better_arm": "D", "arms": None, "then": k3.DECISION["then"]["passes"]}
    assert k3.decide({"C": fails, "D": fails})["k3"] == "fails"
    assert k3.decide({"C": passes})["k3"] == "incomplete"
    assert k3.decide({"C": passes, "D": {"verdict": {"verdict": "incomplete"}}})["k3"] == "incomplete"

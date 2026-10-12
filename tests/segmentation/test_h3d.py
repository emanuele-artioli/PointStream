import numpy as np
import pytest

from experiments.visor import h3c
from experiments.visor import h3d


def ring_regions(frames=4, size=(160, 200)):
    regions = []
    for t in range(frames):
        r = np.zeros(size, bool)
        if t != 2:  # one frame without hands
            r[40 + t:90 + t, 60:100] = True
            r[50 + t:60 + t, 70:80] = False
        regions.append(r)
    return regions


def test_patches_are_centred_on_region_pixels_and_stay_inside_the_frame():
    regions = ring_regions()
    boxes = h3d.region_boxes(regions)
    assert boxes[2] is None and boxes[0] == (40, 90, 60, 100)
    hand = [t for t, b in enumerate(boxes) if b is not None]
    patches = h3d.sample_on_region(np.random.default_rng(0), regions, boxes, hand, 200, 64)
    height, width = regions[0].shape
    for t, y, x in patches:
        assert t in hand and 0 <= y <= height - 64 and 0 <= x <= width - 64
        assert regions[t][y:y + 64, x:x + 64].any()


def test_the_network_at_step_zero_is_the_base_point():
    torch = pytest.importorskip("torch")
    from experiments.background.g5d import enhancer

    regions = ring_regions()
    rng = np.random.default_rng(1)
    upscaled = rng.integers(0, 256, (4, 160, 200, 3), dtype=np.uint8)
    torch.manual_seed(0)
    net = enhancer(2, 8).eval()
    assert np.array_equal(h3d.apply(net, upscaled, regions, "cpu"), upscaled)


def test_apply_changes_only_the_box_around_r():
    torch = pytest.importorskip("torch")

    class Plus(torch.nn.Module):
        def forward(self, x):
            return x + 10 / 255

    regions = ring_regions(size=(300, 300))
    upscaled = np.full((4, 300, 300, 3), 100, np.uint8)
    out = h3d.apply(Plus(), upscaled, regions, "cpu")
    assert (out[2] == 100).all()  # no hand: unchanged
    y0, y1, x0, x1 = h3d.region_boxes(regions)[0]
    m = h3d.APPLY_MARGIN
    assert (out[0, y0 - m:y1 + m, x0 - m:x1 + m] == 110).all()
    assert (out[0, :y0 - m] == 100).all() and (out[0, :, x1 + m:] == 100).all()


def test_flicker_of_the_source_is_zero_and_skips_frames_without_r():
    regions = ring_regions()
    truth = np.random.default_rng(2).integers(0, 256, (4, 160, 200, 3), dtype=np.uint8)
    f = h3d.flicker_flow_free(truth, truth, regions)
    assert np.isnan(f[0]) and np.isnan(f[2]) and np.isnan(f[3]) and f[1] == 0.0


def curve(rates, lpips):
    return [{"name": f"p{i}", "kbps": r, "lpips": q, "dists": q, "psnr": 40 - 30 * q} for i, (r, q) in enumerate(zip(rates, lpips))]


ANCHOR = curve([1.0, 2.0, 4.0, 8.0, 16.0], [0.30, 0.24, 0.18, 0.13, 0.10])


def test_half_the_rate_is_minus_half():
    half = [{**p, "kbps": p["kbps"] / 2} for p in ANCHOR]
    assert h3d.compare(half, ANCHOR, "lpips")["bd"] == pytest.approx(-0.5, abs=1e-9)
    assert h3d.compare(half, ANCHOR, "psnr")["bd"] == pytest.approx(-0.5, abs=1e-9)


def test_the_bar_is_the_concave_hull_not_the_staircase():
    # A point just better than its cheaper neighbour, far above the chord of its neighbours, is dropped from the bar:
    # switching between 2 and 8 kbps reaches it for less.
    staircase = ANCHOR + curve([6.0], [0.179])
    same = h3d.compare(ANCHOR, staircase, "lpips")["bd"]
    assert same == pytest.approx(0.0, abs=1e-9)


def test_disjoint_curves_are_better_or_worse_at_every_rate():
    better = curve([0.2, 0.4], [0.08, 0.06])  # better than the bar's best LPIPS, below its cheapest rate
    assert h3d.compare(better, ANCHOR, "lpips") == {"bd": None, "where": "better at every rate"}
    worse = curve([20.0, 40.0], [0.40, 0.35])
    assert h3d.compare(worse, ANCHOR, "lpips") == {"bd": None, "where": "worse at every rate"}


def items_with(results):
    return {f"{v}_0000000001": {"lpips": r} for v, r in zip(h3c.PILOT, results)}


def bd(x):
    return {"bd": x, "where": "overlap"}


@pytest.mark.parametrize("results,expected", [
    ([bd(-0.3), bd(-0.2), bd(-0.1), bd(0.05)], "passes"),
    ([bd(-0.3), bd(-0.2), {"bd": None, "where": "better at every rate"}, bd(0.05)], "passes"),
    ([bd(-0.3), bd(-0.2), bd(-0.1), {"bd": None, "where": "worse at every rate"}], "unclear"),
    ([bd(-0.05), bd(-0.02), bd(-0.01), bd(0.02)], "unclear"),
    ([bd(0.1), bd(-0.2), bd(0.3), bd(0.05)], "fails"),
    ([bd(-0.3), bd(-0.3), bd(0.4), bd(0.5)], "fails"),
])
def test_decision_rule(results, expected):
    assert h3d.arm_verdict(items_with(results))["verdict"] == expected


def test_oracle_passes_on_either_arm_and_waits_for_both():
    passes = {"verdict": {"verdict": "passes"}}
    fails = {"verdict": {"verdict": "fails"}}
    assert h3d.decide({"dcvc": passes, "svt": fails})["oracle"] == "passes"
    assert h3d.decide({"dcvc": fails, "svt": fails})["oracle"] == "fails"
    assert h3d.decide({"dcvc": {"verdict": {"verdict": "unclear"}}, "svt": fails})["oracle"] == "unclear"
    assert h3d.decide({"dcvc": passes})["oracle"] == "incomplete"


def test_payback_is_the_weights_over_the_saved_rate():
    half = [{**p, "kbps": p["kbps"] / 2} for p in ANCHOR]
    out = h3d.payback(half, ANCHOR, 1000)
    assert out["seconds"] == pytest.approx(8000 / (out["kbps_saved"] * 1000))
    assert out["kbps_saved"] > 0


def test_point_names_match_h3c():
    assert h3d.point_name("dcvc", 21.0, 4) == "dcvc@21/4" and h3d.h3c_name("dcvc", 21.0, 4) == "reference@21/4"
    assert h3d.point_name("svt", 63.0, 8) == h3d.h3c_name("svt", 63.0, 8) == "svt@63/8"
    assert len(h3d.BASES["dcvc"]) == 6 and len(h3d.BASES["svt"]) == 4

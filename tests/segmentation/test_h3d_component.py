import numpy as np
import pytest

from experiments.visor import h3c
from experiments.visor import h3d
from experiments.visor import h3d_component as comp


VIDEOS = {"P02_09": (59.94, 6000), "P02_12": (59.94, 4000), "P02_101": (50.0, 3000)}


def test_segments_are_fixed_disjoint_and_away_from_the_ends_and_the_window():
    exclude = {"P02_09": (2000, 2600)}
    a = comp.choose_segments("P02_02_0000006080", "K", VIDEOS, 24, 48, exclude)
    assert a == comp.choose_segments("P02_02_0000006080", "K", VIDEOS, 24, 48, exclude)  # content-blind, reproducible
    assert a != comp.choose_segments("P02_02_0000006080", "G", VIDEOS, 24, 48, exclude)
    assert len(a) == 24 and len({s["video"] for s in a}) == 3
    for s in a:
        fps, frames = VIDEOS[s["video"]]
        edge = round(5 * fps)
        assert s["last_index"] - s["first_index"] + 1 == 48
        assert s["first_index"] >= edge and s["last_index"] < frames - edge
        if s["video"] in exclude:
            lo, hi = exclude[s["video"]]
            assert s["last_index"] < lo or s["first_index"] > hi
    for v in VIDEOS:
        spans = sorted((s["first_index"], s["last_index"]) for s in a if s["video"] == v)
        assert all(b[0] > p[1] for p, b in zip(spans, spans[1:]))


def test_too_little_footage_is_an_error():
    with pytest.raises(SystemExit, match="only"):
        comp.choose_segments("x", "V", {"short": (50.0, 700)}, 24, 48)


def test_sizes_match_g5d_and_shrink():
    assert comp.size_parameters(*comp.SIZES["L"]) == 594371  # G5d's recorded count
    assert comp.size_parameters(*comp.SIZES["M"]) == 168915
    assert comp.size_parameters(*comp.SIZES["S"]) == 42987


def test_crop_is_centred_on_r_and_inside_the_frame():
    mask = np.zeros((540, 960), bool)
    mask[100:140, 200:260] = True  # centre (120, 230) at analysis size -> (240, 460) at 1080p
    y, x = comp.crop_box(mask, 512)
    assert (y, x) == (0, 460 - 256)
    mask[:] = False
    mask[530:540, 950:960] = True
    assert comp.crop_box(mask, 512) == (1080 - 512, 1920 - 512)


def unit(item, name, kbps_r, rows, lpips_by_row, nbytes=1_000_000):
    frames = 4
    a = {"pixels_r": np.full(frames, 10), "lpips": np.array([[v] * frames for v in lpips_by_row], float),
         "dists": np.array([[v] * frames for v in lpips_by_row], float), "sse": np.full((len(rows), frames), 100),
         "flicker": np.ones((len(rows), frames))}
    fits = {r: {"bytes_fp16": nbytes} for r in rows if r != "base"}
    meta = {"item": item, "video": item.rsplit("_", 1)[0], "name": name, "point": {"kbps_r": kbps_r}, "rows": rows, "fits": fits}
    return meta, a


def test_v_is_billed_its_bits_over_the_video_and_reported_free_beside():
    rows = ["base", "V-L", "K-L", "G-L"]
    units = [unit("P01_107_0000003049", "dcvc@21/4", 2.0, rows, [0.3, 0.1, 0.12, 0.15])]
    out = h3d.component_rows(units, {"P01_107": 100.0})
    assert out["V-L"]["P01_107_0000003049"][0]["kbps"] == pytest.approx(2.0 + 8e6 / 100 / 1000)
    assert out["V-L/free"]["P01_107_0000003049"][0]["kbps"] == 2.0
    assert out["G-L"]["P01_107_0000003049"][0]["kbps"] == 2.0 and "base" in out


def verdicts(**names):
    return {n.replace("_", "-"): {"verdict": {"verdict": v[0], "mean_bd": v[1]}} for n, v in names.items()}


def fronts(scale_by):
    """Each arm's front per item: the anchor curve with rates scaled by ``scale_by[arm]``."""
    base = [(1.0, 0.3), (2.0, 0.2), (4.0, 0.12), (8.0, 0.08)]
    return {arm: {f"{v}_0000000001": [{"kbps": r * k, "lpips": q} for r, q in base] for v in h3c.PILOT}
            for arm, k in scale_by.items()}


def test_g_is_the_coder_unless_k_beats_it_by_ten_percent():
    arms = verdicts(G_L=("passes", -0.3), K_L=("passes", -0.35), V_L=("fails", 0.5), V_S=("fails", 0.2))
    near = h3d.component_decision(arms, fronts({"G-L": 1.0, "K-L": 0.95}))
    assert near["coder"] == "G-L" and not near["K_vs_G"]["beats"]
    far = h3d.component_decision(arms, fronts({"G-L": 1.0, "K-L": 0.8}))
    assert far["coder"] == "K-L" and far["best_v"] == "V-S"


def test_v_only_when_it_beats_every_standing_shared_arm():
    arms = verdicts(G_L=("fails", 0.1), K_L=("passes", -0.2), V_L=("passes", -0.5), V_S=("passes", -0.4))
    assert h3d.component_decision(arms, fronts({"K-L": 1.0, "V-L": 0.7, "V-S": 0.8}))["coder"] == "V-L"
    assert h3d.component_decision(arms, fronts({"K-L": 1.0, "V-L": 0.95, "V-S": 0.97}))["coder"] == "K-L"
    none = verdicts(G_L=("fails", 0.1), K_L=("unclear", -0.05), V_L=("fails", 0.3))
    assert h3d.component_decision(none, fronts({}))["coder"].startswith("none")

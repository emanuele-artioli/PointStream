"""G5c: masked DISTS, flicker, model-size variants and the decision rules."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.background import g5, g5c


def test_masked_dists_with_a_full_mask_is_dists():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("torchmetrics")
    from src.codecs.quality import dists_network, masked_dists

    torch.manual_seed(0)
    net = dists_network(None)
    x, y = torch.rand(2, 3, 64, 96), torch.rand(2, 3, 64, 96)
    ones = torch.ones(2, 1, 64, 96)
    assert torch.allclose(masked_dists(net, x, y, ones), net(x, y), atol=1e-5)
    assert torch.allclose(masked_dists(net, x, x, ones), torch.zeros(2), atol=1e-5)
    half = ones.clone()
    half[:, :, :, :48] = 0
    assert not torch.allclose(masked_dists(net, x, y, half), net(x, y), atol=1e-4)


def clip_of(frames: np.ndarray, keep: np.ndarray) -> g5.Clip:
    n = len(frames)
    return g5.Clip("c", "visor-long", Fraction(10), 0, frames, keep, {"own": (list(range(n)), keep)}, {})


def test_flicker_cancels_what_the_source_does():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n, h, w = 5, 16, 24
    frames = rng.integers(0, 200, (n, h, w, 3), dtype=np.uint8)
    keep = np.ones((n, h, w), bool)
    keep[:, :4] = False
    clip = clip_of(frames, keep)
    flows = np.zeros((n, h, w, 2), np.float16)
    exact = g5c.flicker(clip, frames.copy(), flows, "cpu")
    assert exact["flicker_v"] == pytest.approx(0.0) and exact["pairs"] == n - 1
    assert exact["residual_source"] > 0
    offset = g5c.flicker(clip, frames + 10, flows, "cpu")  # a constant error does not flicker
    assert offset["flicker_v"] == pytest.approx(0.0, abs=1e-4)
    noisy = frames.copy()
    noisy[1::2] += 20  # every other frame brighter: flicker of 20 levels on luma
    out = g5c.flicker(clip, noisy, flows, "cpu")
    assert out["flicker_v"] == pytest.approx(20.0, abs=1e-3)
    outside = frames.copy()
    outside[1::2, :4] = 0  # changes only outside V are not counted
    assert g5c.flicker(clip, outside, flows, "cpu")["flicker_v"] == pytest.approx(0.0)


def test_variants_parse():
    assert g5c.parse_variant("whole-8") == ("whole", 8, 1.0)
    assert g5c.parse_variant("delta-4-s0.1") == ("delta", 4, 0.1)
    for bad in ("whole-4-s0.1", "delta-3", "half-8"):
        with pytest.raises(ValueError):
            g5c.parse_variant(bad)
    assert all(g5c.parse_variant(v) for v in g5c.VARIANTS)


def test_quantize_per_channel():
    torch = pytest.importorskip("torch")
    t = torch.tensor([[0.5, -1.0, 0.25, 0.0], [0.0, 0.0, 0.0, 0.0], [3.0, 1.0, -2.0, 0.1]])
    q, scale, deq = g5c.quantize_channels(t, 2)  # ternary
    assert set(q.unique().tolist()) <= {-1, 0, 1} and scale[1] == 0 and (deq[1] == 0).all()
    q8, s8, d8 = g5c.quantize_channels(t, 8)
    assert q8.abs().max() == 127
    assert ((d8 - t).abs() <= s8.float()[:, None] / 2 + 1e-6).all()


def test_compress_state_rebuilds_what_it_prices():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    public = {"w": torch.randn(64, 32), "b": torch.randn(64), "n": torch.tensor([3], dtype=torch.int64)}
    ft = {"w": public["w"] + 1e-3 * torch.randn(64, 32), "b": public["b"].clone(), "n": public["n"].clone()}
    whole, size16 = g5c.compress_state(ft, public, "whole-16", workers=2)
    assert torch.equal(whole["w"], ft["w"].half().float()) and size16["raw_fp16_bytes"] == 2 * (64 * 32 + 64)
    delta, size_d = g5c.compress_state(ft, public, "delta-8", workers=2)
    assert size_d["bytes"] < size16["bytes"]
    assert (delta["w"] - ft["w"]).abs().max() < 1e-4 and torch.equal(delta["b"], public["b"])
    assert torch.equal(delta["n"], public["n"])
    sparse, size_s = g5c.compress_state(ft, public, "delta-8-s0.1", workers=2)
    changed = (sparse["w"] != public["w"]).float().mean().item()
    assert 0.05 < changed <= 0.11 and size_s["bytes"] < size_d["bytes"]
    assert size_s["nonzero_share"] < 0.2
    w2, size2 = g5c.compress_state(ft, public, "whole-2", workers=2)
    assert size2["bytes"] < size16["bytes"] and size2["quantized_share"] > 0.9


def point(kbps: float, lpips: float, dists: float, flick: float, psnr: float = 30.0) -> dict:
    tier = {"lpips_v": lpips, "dists_v": dists, "psnr_v": psnr}
    return {"kbps": kbps, "score": {"dataset": tier, "own": tier,
                                    "flicker": {"flicker_v": flick, "pairs": 239, "residual_source": 5.0}}}


def test_report_applies_the_rules(tmp_path):
    pytest.importorskip("scipy")
    svt = [point(r, 0.30 - 0.05 * i, 0.20 - 0.03 * i, 2.0 - 0.3 * i) for i, r in enumerate((100, 200, 400, 800))]
    docs = []
    for clip_id, comp_flicker in (("visor-long/P26_02", 1.0), ("visor-long/P06_03", 3.0)):
        comp = [point(r / 2, 0.30 - 0.05 * i, 0.20 - 0.03 * i, comp_flicker - 0.1 * i)
                for i, r in enumerate((100, 200, 400, 800))]
        docs.append({"kind": "checks", "clip": {"id": clip_id}, "methods": [
            {"arm": "svtav1-filled", "codec": "svtav1", "points": svt},
            {"arm": "scene-lpips", "codec": "dcvc", "points": comp}]})
    rows = [{"variant": "whole-16", "size": {"bytes": 1000}, "points": [point(r, 0.3 - 0.05 * i, 0.2, 1.0)
                                                                      for i, r in enumerate((10, 20, 40, 80))]},
            {"variant": "delta-4", "size": {"bytes": 10}, "points": [point(1.02 * r, 0.3 - 0.05 * i, 0.2, 1.0)
                                                                    for i, r in enumerate((10, 20, 40, 80))]},
            {"variant": "whole-2", "size": {"bytes": 5}, "points": [point(2 * r, 0.3 - 0.05 * i, 0.2, 1.0)
                                                                   for i, r in enumerate((10, 20, 40, 80))]}]
    docs.append({"kind": "compress", "clip": {"id": "visor-long/P06_03"}, "variants": rows})
    paths = []
    for i, doc in enumerate(docs):
        (tmp_path / f"{i}.json").write_text(json.dumps(doc))
        paths.append(str(tmp_path / f"{i}.json"))
    g5c.command_report(SimpleNamespace(result=paths, out=str(tmp_path / "out")))
    report = json.loads((tmp_path / "out" / "g5c-report.json").read_text())
    comp = report["checks"]["visor-long/P06_03"]["scene-lpips"]
    assert comp["dists_v"]["bd_rate"] == pytest.approx(-0.5, abs=1e-6)
    assert report["checks_decision"] == {"dists_transfers": True, "flicker_found": True}
    size = report["size"]["visor-long/P06_03"]
    assert size["variants"]["delta-4"]["keeps_quality"] and not size["variants"]["whole-2"]["keeps_quality"]
    assert size["smallest_keeping_quality"] == "delta-4"
    assert math.isclose(size["variants"]["whole-2"]["vs_16bit"], 1.0, abs_tol=1e-6)

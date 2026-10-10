"""G5b: held-out split, training crops, convergence, payback and the decision rule."""

from __future__ import annotations

import json
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.background import g5, g5b, g5b_train


def test_held_out_split_keeps_a_guard_around_the_excerpt():
    assert g5b.fit_indices(1200, 0, 240, Fraction(10), "heldout") == list(range(240))
    scene = g5b.fit_indices(1200, 0, 240, Fraction(10), "scene")
    assert scene[0] == 290 and scene[-1] == 1199 and len(scene) == 910
    middle = g5b.fit_indices(1200, 500, 240, Fraction(10), "scene")
    assert 449 in middle and 450 not in middle and 789 not in middle and 790 in middle


def test_sequences_never_cross_a_gap():
    fit = list(range(0, 10)) + list(range(20, 25))
    assert g5b_train.sequence_starts(fit, 5) == [0, 1, 2, 3, 4, 5, 20]
    assert g5b_train.sequence_starts(fit, 11) == []


def test_crop_aligns_chroma_with_luma_and_flips_the_mask():
    torch = pytest.importorskip("torch")
    n, h, w = 3, 8, 12
    frames = np.zeros((n, h * 3 // 2, w), np.uint8)
    frames[:, :h] = np.arange(w, dtype=np.uint8)[None, None, :] * 10  # luma ramp along x
    chroma = frames[:, h:].reshape(n, 2, h // 2, w // 2)
    chroma[:, 0] = np.arange(w // 2, dtype=np.uint8)[None, None, :] * 20  # Cb ramp at half resolution
    keep = np.ones((n, h, w), bool)
    keep[:, :, :3] = False
    scene = g5b_train.Scene(frames, keep)
    x, m = scene.crop(0, 2, 2, 2, 4, 6, flip=False)
    assert x.shape == (2, 3, 4, 6) and m.shape == (2, 1, 4, 6)
    luma = (x[0, 0, 0] + 0.5) * 255
    cb = (x[0, 1, 0] + 0.5) * 255
    assert torch.allclose(luma, torch.tensor([20.0, 30, 40, 50, 60, 70]), atol=1e-3)
    assert torch.allclose(cb, torch.tensor([20.0, 20, 40, 40, 60, 60]), atol=1e-3)
    assert m[0, 0, 0].tolist() == [0, 1, 1, 1, 1, 1]
    xf, mf = scene.crop(0, 2, 2, 2, 4, 6, flip=True)
    assert torch.equal(xf, x.flip(-1)) and mf[0, 0, 0].tolist() == [1, 1, 1, 1, 1, 0]
    rng = np.random.default_rng(0)
    xb, mb, qp = scene.batch(rng, [0, 1], 2, 3, 4, 6, 64)
    assert xb.shape == (2, 3, 3, 4, 6) and mb.shape == (2, 3, 1, 4, 6) and qp.shape == (3,)


def test_monitor_convergence():
    flat = [{"step": s, "loss": 1.0 - 0.001 * s / 100} for s in range(0, 1001, 100)]
    assert g5b.monitor_converged(flat)["converged"]
    falling = [{"step": s, "loss": 1.0 - 0.5 * s / 1000} for s in range(0, 1001, 100)]
    out = g5b.monitor_converged(falling)
    assert not out["converged"] and out["from_step"] == 800
    assert not g5b.monitor_converged(flat[:2])["converged"]


def test_payback_from_rate_saved_at_equal_quality():
    anchor = [(100.0, -20.0), (400.0, -10.0)]
    assert g5b.anchor_rate(anchor, -15.0) == pytest.approx(200.0)
    assert g5b.anchor_rate(anchor, -5.0) is None
    out = g5b.payback_seconds(1_000_000, [(100.0, -15.0), (500.0, -15.0)], anchor)
    assert out[0]["seconds"] == pytest.approx(8_000_000 / (1000 * 100.0))
    assert out[1]["seconds"] is None


def test_pilot_rule_needs_both_baselines_beaten():
    def row(svt, dcvc, converged=True):
        return {"vs_svtav1": {"bd_rate": svt}, "vs_dcvc": {"bd_rate": dcvc}, "converged": converged}

    a, b = g5b.DECISION["pilot"]
    assert g5b.stage_decision({a: row(-0.1, -0.2), b: row(0.3, 0.1)})["decision"] == "passes"
    assert g5b.stage_decision({a: row(-0.1, -0.05), b: row(0.3, 0.1)})["decision"] == "unclear"
    assert g5b.stage_decision({a: row(0.2, 0.0), b: row(0.3, 0.1)})["decision"] == "fails"
    assert g5b.stage_decision({a: row(0.2, 0.0), b: row(0.3, 0.1, False)})["decision"] == "unclear"
    assert g5b.stage_decision({a: row(-0.5, -0.5)})["decision"] == "incomplete"


def test_refresh_interval_is_a_parameter():
    refs = g5.refresh_reference(25, Fraction(50), 0.3)
    assert refs[1] == 0 and refs[15] == 0 and refs[16] == 15


def point(kbps: float, lpips: float, psnr: float) -> dict:
    tier = {"lpips_v": lpips, "psnr_v": psnr}
    return {"kbps": kbps, "score": {"dataset": tier, "own": tier}}


def test_report_reads_every_kind(tmp_path):
    pytest.importorskip("scipy")
    clip = "visor-long/P26_02"
    docs = {
        "rescore": {"kind": "rescore", "clips": [{"id": clip, "points": [
            point(100, 0.30, 30), point(200, 0.25, 32), point(400, 0.20, 34), point(800, 0.15, 36)]}]},
        "dcvc": {"kind": "dcvc", "clips": [{"id": clip, "points": [
            point(110, 0.30, 30), point(220, 0.25, 32), point(440, 0.20, 34), point(880, 0.15, 36)]}]},
        "ft": {"kind": "finetune", "clips": [{"id": clip, "arm": "upper-mse", "model_bytes_fp16": 10**7,
                                              "converged": {"converged": True}, "points": [
            point(50, 0.30, 30), point(100, 0.25, 32), point(200, 0.20, 34), point(400, 0.15, 36)]}]},
    }
    paths = []
    for name, doc in docs.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(doc))
        paths.append(str(tmp_path / f"{name}.json"))
    g5b.command_report(SimpleNamespace(result=paths, out=str(tmp_path / "out")))
    report = json.loads((tmp_path / "out" / "g5b-report.json").read_text())
    upper = report["arms"]["upper-mse"]["clips"][clip]
    assert upper["lpips_v"]["vs_svtav1"]["bd_rate"] == pytest.approx(-0.5, abs=1e-6)
    assert upper["lpips_v"]["vs_dcvc"]["bd_rate"] == pytest.approx(50 / 110 - 1, abs=1e-6)
    assert report["arms"]["dcvc"]["clips"][clip]["lpips_v"]["vs_svtav1"]["bd_rate"] == pytest.approx(0.1, abs=1e-6)
    assert upper["payback_vs_svtav1"][1]["seconds"] == pytest.approx(8 * 10**7 / (1000 * 100))
    assert report["arms"]["upper-mse"]["pilot"]["decision"] == "incomplete"


def test_visual_frames_come_from_the_gate_tier():
    clip = SimpleNamespace(n=240, scored={"dataset": (list(range(10, 50)), None)})
    assert g5b.visual_frames(clip) == [10, 30, 49]
    clip = SimpleNamespace(n=5, scored={"dataset": ([], None)})
    assert g5b.visual_frames(clip) == [0, 2, 4]

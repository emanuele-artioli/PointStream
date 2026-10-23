import math

import numpy as np
import pytest

from experiments.tennis import k6b
from tests.test_k6 import _body


def test_openpose18_maps_coco_as_dwpose_does():
    kp = _body()
    pts, vis = k6b.openpose18(kp)
    assert vis.all()
    # DWPose's wholebody.py: openpose slots [1..17] take mmpose [neck, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    for slot, coco in zip([2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17], [6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]):
        assert np.allclose(pts[slot], kp[coco, :2])
    assert np.allclose(pts[1], kp[[5, 6], :2].mean(0))
    kp[6, 2] = 0.2  # one shoulder unconfident: no neck, no right shoulder
    _, vis = k6b.openpose18(kp)
    assert not vis[1] and not vis[2] and vis[5]


def test_pose_map_draws_limbs_in_dwpose_colours_where_the_keypoints_are():
    kp = _body(x0=256, y0=200, h=400)
    canvas = k6b.pose_map(kp)
    assert canvas.shape == (k6b.GEN_H, k6b.GEN_W, 3)
    # the nose's disc in its own colour (slot 0), drawn after the limbs at full strength
    assert tuple(canvas[200, 256]) == k6b.COLOURS[0]
    # the right upper arm (limb 3-4 in 1-based slots: index 2) at 0.6 strength between shoulder and elbow
    mid = ((kp[6, :2] + kp[8, :2]) / 2).astype(int)
    assert tuple(canvas[mid[1], mid[0]]) == tuple(int(c * 0.6) for c in k6b.COLOURS[2])
    assert canvas[:100].max() == 0 or canvas[:150, :100].max() == 0  # nothing far from the body
    kp[:, 2] = 0.1
    assert k6b.pose_map(kp).max() == 0


def test_run_boxes_hold_the_keypoints_at_one_size_and_follow_the_player():
    kps = [_body(x0=500 + 10 * i, y0=300, h=200) for i in range(30)]
    boxes = k6b.run_boxes(kps)
    assert boxes.shape == (30, 4)
    assert np.allclose(boxes[:, 2:], boxes[0, 2:])  # one size per run
    assert np.isclose(boxes[0, 2] / boxes[0, 3], k6b.GEN_W / k6b.GEN_H)
    for kp, (x0, y0, w, h) in zip(kps[5:-5], boxes[5:-5]):  # away from the smoothing's edges
        assert x0 < kp[:, 0].min() and kp[:, 0].max() < x0 + w
        assert y0 < kp[:, 1].min() and kp[:, 1].max() < y0 + h
    assert boxes[-1, 0] > boxes[0, 0]
    # the drawn pose sits in the crop's middle
    crop = k6b.to_crop(kps[15], boxes[15])
    assert 0.3 * k6b.GEN_W < crop[:, 0].mean() < 0.7 * k6b.GEN_W


def test_run_boxes_fill_frames_without_a_confident_keypoint():
    kps = [_body(x0=500, y0=300) for _ in range(5)]
    kps[2] = kps[2].copy()
    kps[2][:, 2] = 0.0
    boxes = k6b.run_boxes(kps)
    assert np.isfinite(boxes).all()
    blank = np.zeros((17, 3))
    with pytest.raises(ValueError):
        k6b.run_boxes([blank])


def test_inside_frame_shifts_a_box_that_fits():
    assert np.allclose(k6b.inside_frame(np.array([-50., 1000., 200., 300.])), [0., 780., 200., 300.])
    big = np.array([-10., -10., 3000., 300.])
    assert k6b.inside_frame(big)[0] == -10.  # wider than the frame: left as is


def test_crop_then_paste_returns_the_pixels_on_the_mask():
    rng = np.random.default_rng(0)
    image = np.zeros((1080, 1920, 3), np.uint8)
    image[300:700, 800:1100] = rng.integers(0, 255, (400, 300, 3), dtype=np.uint8) // 8 * 8  # blocky, survives resizing
    box = np.array([780., 280., 340 * k6b.GEN_W / k6b.GEN_H * 1.4, 340 * 1.4])
    crop = k6b.crop_image(image, box)
    assert crop.shape == (k6b.GEN_H, k6b.GEN_W, 3)
    mask = np.zeros((1080, 1920), bool)
    mask[400:600, 900:1000] = True
    canvas = np.zeros_like(image)
    written = k6b.paste(canvas, crop, box, mask)
    assert (written == mask).all()
    err = np.abs(canvas[mask].astype(int) - image[mask].astype(int)).mean()
    assert err < 20  # area down, Lanczos up
    assert not canvas[~mask].any()


def test_runs_split_at_gaps():
    assert k6b.runs([5, 1, 2, 3, 7, 8]) == [[1, 2, 3], [5], [7, 8]]


def test_won_reads_the_envelope_at_the_keypoint_rate():
    anchors = {"a": {"kbps": 1.0, "lpips": 0.40, "dists": 0.30}, "b": {"kbps": 10.0, "lpips": 0.20, "dists": 0.20},
               "c": {"kbps": 5.0, "lpips": 0.35, "dists": 0.25}}  # c is under the hull
    r = k6b.won(anchors, "lpips", 5.5, 0.28)
    assert np.isclose(r["envelope"], 0.40 - 0.20 * 4.5 / 9)
    assert r["won"]
    assert not k6b.won(anchors, "lpips", 5.5, 0.31)["won"]


def test_score_and_decision():
    def row(o_l, o_d, f_l, f_d):
        return {"gen-original": {"lpips": {"won": o_l}, "dists": {"won": o_d}},
                "gen-finetuned_tennis": {"lpips": {"won": f_l}, "dists": {"won": f_d}}}
    rows = [row(True, True, True, True)] * 3 + [row(False, True, True, True)]
    s = {v: k6b.score(rows, v) for v in k6b.GEN_VARIANTS}
    assert s["gen-original"]["score"] == pytest.approx(0.875) and s["gen-original"]["passes"]
    assert k6b.decide({"all": s}, True).startswith("pass")
    assert k6b.decide({"all": s}, False).startswith("not read")
    rows = [row(False, True, True, True)] * 4  # amendment 1: only gen-original decides
    s = {v: k6b.score(rows, v) for v in k6b.GEN_VARIANTS}
    assert k6b.decide({"all": s}, True).startswith("fail")
    rows = [row(False, True, False, True)] * 4
    s = {v: k6b.score(rows, v) for v in k6b.GEN_VARIANTS}
    assert k6b.decide({"all": s}, True).startswith("fail")


def test_unit_names_and_seeds_are_stable():
    name = k6b.unit_name("djokovic_federer/038@120", "person:000000000002")
    assert "/" not in name and ":" not in name and "@" not in name
    assert k6b.unit_seed(name) == k6b.unit_seed(name)


def test_step3_decision_reads_the_fitted_renderer():
    def row(won):
        return {"gen-fit": {"lpips": {"won": won}, "dists": {"won": won}}}
    s = {"gen-fit": k6b.score([row(True)] * 5 + [row(False)], "gen-fit")}
    assert k6b.decide({"all": s}, True, step=3).startswith("pass")
    s = {"gen-fit": k6b.score([row(True)] * 4 + [row(False)] * 2, "gen-fit")}
    assert k6b.decide({"all": s}, True, step=3).startswith("fail")
    assert k6b.decide({"all": s}, False, step=3).startswith("not read")


def test_billed_rate_and_payback():
    # 1.5 MB over 2 hours: 12 Mbit / 7200 s = 1.667 kbps
    assert k6b.billed_kbps(1.5e6, 2.0) == pytest.approx(1.6667, abs=1e-3)
    hull = [(1.0, -0.40), (10.0, -0.20)]
    # LPIPS 0.30 needs 5.5 kbps on the hull; at 2.5 kbps of keypoints the saving is 3 kbps; 1.5 MB repays in 4000 s
    assert k6b.payback_hours(hull, 0.30, 2.5, 1.5e6) == pytest.approx(12e6 / 3000 / 3600)
    assert k6b.payback_hours(hull, 0.45, 2.5, 1.5e6) == math.inf  # worse than every point: never repaid


def test_step4_verdict():
    passing, failing = {"passes": True}, {"passes": False}
    assert k6b.decide_billed(passing, passing, True).startswith("pass")
    assert k6b.decide_billed(failing, passing, True).startswith("to the user")
    assert k6b.decide_billed(failing, failing, True).startswith("fail")
    assert k6b.decide_billed(passing, passing, False).startswith("not read")


def test_adapter_quantization_matches_g5c_and_is_billed():
    torch = pytest.importorskip("torch")
    from experiments.background import g5c
    from experiments.tennis import k6b_fit

    gen = torch.Generator().manual_seed(0)
    t = torch.randn(64, 320, generator=gen)
    for a, b in zip(k6b_fit.quantize_channels(t, 8), g5c.quantize_channels(t, 8)):
        assert torch.equal(a, b)
    state = {"rank": 4, "reference_unet": {"x.lora_A.weight": torch.randn(4, 320, generator=gen)},
             "denoising_unet": {"y.lora_B.weight": torch.randn(320, 4, generator=gen), "tiny": torch.randn(8, generator=gen)}}
    rebuilt, billing = k6b_fit.quantize_adapters(state, 8)
    assert billing["adapter_values"] == 4 * 320 + 320 * 4 + 8 and not billing["pose_guider_billed"]
    assert 0 < billing["bytes"] < billing["raw_fp16_bytes"]  # 8 bits and LZMA beat fp16
    err = (rebuilt["reference_unet"]["x.lora_A.weight"].float() - state["reference_unet"]["x.lora_A.weight"]).abs().max()
    assert err < 0.02 * state["reference_unet"]["x.lora_A.weight"].abs().max()
    assert torch.equal(rebuilt["denoising_unet"]["tiny"], state["denoising_unet"]["tiny"].half())  # small tensors stay fp16


def test_guider_is_billed_as_its_quantized_change_and_fewer_bits_cost_less():
    torch = pytest.importorskip("torch")
    from experiments.tennis import k6b_fit

    gen = torch.Generator().manual_seed(1)
    base = {"conv.weight": torch.randn(16, 3, 3, 3, generator=gen), "conv.bias": torch.randn(16, generator=gen)}
    trained = {k: v + 0.01 * torch.randn(v.shape, generator=gen) for k, v in base.items()}
    lora = {"x.lora_A.weight": torch.randn(4, 320, generator=gen)}
    state = {"rank": 4, "reference_unet": lora, "denoising_unet": {}, "pose_guider": trained}
    with pytest.raises(SystemExit):
        k6b_fit.quantize_adapters(state, 8)  # a trained guider needs the decoder's weights
    rebuilt, billing = k6b_fit.quantize_adapters(state, 8, base)
    assert billing["pose_guider_billed"] and billing["pose_guider_values"] == 16 * 27 + 16
    assert 0 < billing["pose_guider_bytes"] < billing["bytes"]
    for k in base:  # the decoder rebuilds base + change, close to the trained weights
        assert (rebuilt["pose_guider"][k] - trained[k]).abs().max() < 1e-3
    _, four = k6b_fit.quantize_adapters(state, 4, base)
    assert four["bytes"] < billing["bytes"]


def test_billing_entry_reads_both_forms():
    billing = {"gen-match5": {"djokovic_federer": 1000}, "gen-player": {"hours": 20, "bytes": {"djokovic_federer": 3000}}}
    assert k6b.billing_entry(billing, "gen-match5", "djokovic_federer", 2.0) == (1000.0, 2.0)
    assert k6b.billing_entry(billing, "gen-player", "djokovic_federer", 2.0) == (3000.0, 20.0)
    assert k6b.billing_entry(billing, "gen-player", "rublev_auger-aliassime", 2.0) is None
    assert k6b.billing_entry(billing, "gen-A", "djokovic_federer", 2.0) is None


def _dev_row(**variants):
    return {v.replace("_", "-"): {m: {"value": val[i]} for i, m in enumerate(("lpips", "dists"))} |
            {"billed": {m: {"value": val[i], "envelope": 0.2, "won": val[i] < 0.2} for i, m in enumerate(("lpips", "dists"))}}
            for v, val in variants.items()}


def test_step5_development_choice():
    rows = [_dev_row(gen_A=(0.21, 0.19), gen_B=(0.19, 0.19), gen_C=(0.19, 0.18), gen_C4=(0.193, 0.183), gen_Cg=(0.18, 0.21)),
            _dev_row(gen_A=(0.25, 0.22), gen_B=(0.22, 0.19), gen_C=(0.22, 0.19), gen_C4=(0.22, 0.19), gen_Cg=(0.19, 0.18))]
    stats = {v: k6b.dev_stats(rows, v) for v in ("gen-A", "gen-B", "gen-C", "gen-C4", "gen-Cg")}
    assert stats["gen-A"]["won"] == 1 and stats["gen-B"]["won"] == 3 and stats["gen-C"]["won"] == 3
    assert k6b.choose_arm(stats, ["gen-A", "gen-B", "gen-C"]) == "gen-C"  # a tie on wins goes to the larger margin
    stats["gen-C"]["margin"] = stats["gen-B"]["margin"]
    assert k6b.choose_arm(stats, ["gen-A", "gen-B", "gen-C"]) == "gen-B"  # then to the cheaper
    assert k6b.fewer_bits_hold(rows, "gen-C", "gen-C4")
    assert not k6b.fewer_bits_hold(rows, "gen-A", "gen-B")
    assert k6b.guidance_better(stats, "gen-C", "gen-Cg")  # 3 won each, the larger margin

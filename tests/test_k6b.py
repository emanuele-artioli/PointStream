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

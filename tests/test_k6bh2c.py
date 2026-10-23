"""K6b (hands) 2c: bank selection, reference coding geometry and billing, candidates outside the window
(docs/experiments.md, 2026-10-22)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from experiments.visor import k6bh2c  # noqa: E402


def test_greedy_medoids_is_nested_and_covers_the_mass() -> None:
    rng = np.random.default_rng(0)
    crowd = rng.normal(0.0, 0.05, (200, 2))
    outlier = np.array([[10.0, 10.0]])
    poses = np.concatenate([crowd, outlier])
    one = k6bh2c.greedy_medoids(poses, 1)
    assert one != [200]  # the first medoid sits in the crowd, not on the outlier (farthest-point would start anywhere)
    assert k6bh2c.greedy_medoids(poses, 5)[:3] == k6bh2c.greedy_medoids(poses, 3)
    assert len(k6bh2c.greedy_medoids(poses[:4], 10)) == 4
    assert k6bh2c.greedy_medoids(np.zeros((0, 2)), 3) == []


def test_joints_inside_counts_the_dilated_mask() -> None:
    mask = np.zeros((100, 100), bool)
    mask[40:60, 40:60] = True
    points = np.array([[50.0, 50.0], [61.0, 50.0], [90.0, 90.0], [-5.0, 3.0]])
    assert k6bh2c.joints_inside(points, mask) == 2  # 61 is within 10% of 20 px (2 px) of the box


def test_ref_box_is_padded_even_and_inside_the_frame() -> None:
    mask = np.zeros((1080, 1920), bool)
    mask[1000:1080, 1890:1920] = True  # a forearm in the corner
    x0, y0, w, h = k6bh2c.ref_box(mask)
    assert w % 8 == 0 and h % 8 == 0 and w >= 64 and h >= 64
    assert 0 <= x0 and x0 + w <= 1920 and 0 <= y0 and y0 + h <= 1080
    assert x0 <= 1890 and y0 <= 1000


def test_smooth_fill_keeps_the_mask_and_is_smooth_outside() -> None:
    image = np.zeros((64, 64, 3), np.uint8)
    image[:, :32] = 200
    image[:, 32:] = 50
    mask = np.zeros((64, 64), bool)
    mask[20:44, 10:30] = True
    out = k6bh2c.smooth_fill(image, mask)
    assert np.array_equal(out[mask], image[mask])
    assert abs(int(out[30, 60, 0]) - 200) < 30  # outside: the mask's colour carried, not the hidden 50


def test_polygons_round_trip_and_are_billed() -> None:
    mask = np.zeros((120, 160), bool)
    yy, xx = np.mgrid[:120, :160]
    mask[(yy - 60) ** 2 + (xx - 80) ** 2 < 40 ** 2] = True
    polys, bits = k6bh2c.code_polygons(mask)
    back = k6bh2c.decode_polygons(polys, (120, 160))
    iou = (back & mask).sum() / (back | mask).sum()
    assert iou > 0.95
    vertices = sum(len(p) for p in polys)
    assert bits == 8 + 12 * len(polys) + vertices * (8 + 7)  # ceil(log2 160) + ceil(log2 120)
    assert bits < 0.05 * mask.size  # far below one bit per pixel


def test_sent_joints_are_whole_pixels_in_the_box() -> None:
    pts = np.array([[10.4, 20.6], [500.0, 5.0]] + [[30.0, 30.0]] * 19)
    sent = k6bh2c.sent_joints(pts, (8, 16, 64, 64))
    assert np.allclose(sent[0], [10.0, 21.0]) and np.allclose(sent[1], [71.0, 16.0])
    assert k6bh2c.joint_bits(64, 64) == 21 * 12
    full = k6bh2c.with_conf(sent)
    assert full.shape == (21, 3) and (full[:, 2] == 1).all() and np.array_equal(full[:, :2], sent)


def test_payback_and_hull_reading() -> None:
    hull = [(2.0, -0.30), (5.0, -0.20), (20.0, -0.10)]  # (kbps, -LPIPS), rising
    exact = k6bh2c.payback_seconds(30_000.0, 2.0, hull, 0.20)  # the hull needs 5 kbps; 3 kbps of headroom
    assert exact["bound"] == "exact" and exact["seconds"] == pytest.approx(10.0)
    assert k6bh2c.payback_seconds(1e6, 2.0, hull, 0.05)["bound"].startswith("at most")
    assert k6bh2c.payback_seconds(1e6, 2.0, hull, 0.40)["seconds"] is None
    assert k6bh2c.won(0.15, hull, 5.0)[0] and not k6bh2c.won(0.25, hull, 5.0)[0]
    assert k6bh2c.won(0.25, hull, 1.0) == (True, 0.30, "below the hull's lowest rate (its lowest point)")
    assert not k6bh2c.won(0.01, hull, 25.0)[0]  # above the hull's highest rate counts as lost


def test_candidates_leave_out_the_window_and_guard() -> None:
    def entry(n: int, names: list[str]) -> dict:
        return {"image": {"name": f"P01_01_frame_{n:010d}.jpg", "video": "P01_01", "interpolation": "r"},
                "annotations": [{"name": x, "segments": [[[0, 0], [10, 0], [10, 10]]]} for x in names]}

    doc = {"video_annotations": [entry(n, ["left hand"] if n != 300 else ["cup"]) for n in range(1, 2001)]}
    item = {"video": "P01_01", "fps": 50.0, "first_video_index": 999, "frames": 240}
    cands = k6bh2c.candidate_frames(item, doc, {}, stride=5)
    indices = [i for _, i in cands]
    assert all(n % 5 == 0 and i == n - 1 for n, i in cands)
    assert not any(999 - 50 <= i < 999 + 240 + 50 for i in indices)
    assert 299 not in indices  # VISOR frame 300 has no hand
    assert indices == sorted(indices) and len(indices) > 300


def test_candidates_restore_from_an_archive_cut_to_a_coarser_stride(tmp_path: Path) -> None:
    import tarfile

    from experiments.visor.h2 import save_unit

    frames = np.arange(5, 105, 5)  # stride 5
    arrays = {"visor_frame": frames, "index": frames - 1, "row_cand": np.array([0, 1, 1, 3, 19]),
              "row_right": np.array([False, False, True, True, False]), "row_kp2d": np.arange(5 * 42, dtype=np.float32).reshape(5, 21, 2),
              "row_inside": np.array([21, 20, 3, 21, 21])}
    meta = {"unit": "X_1", "item": "X_1", "stride": 5, "candidates": 20, "window": [0, 1], "guard_frames": 0,
            "video_record": {"duration_s": 100.0, "frames_total": 5000}}
    old = tmp_path / "old"
    save_unit(old / "publish", "k6bh2c-candidates", "X_1", arrays, meta)
    tar = tmp_path / "partial.tar"
    with tarfile.open(tar, "w") as handle:
        handle.add(old / "publish", arcname="publish")
    publish = tmp_path / "new"
    assert k6bh2c.restore_candidates(tar, publish, {"X_1"}, 10) == ["X_1"]
    (m, a), = k6bh2c.published_units(publish, "k6bh2c-candidates").values()
    assert m["stride"] == 10 and list(a["visor_frame"]) == list(range(10, 101, 10))
    # rows on frames 10 (cand 1 -> 0, twice), 20 (cand 3 -> 1) and 100 (cand 19 -> 9) survive; frame 5's goes
    assert list(a["row_cand"]) == [0, 0, 1, 9] and list(a["row_inside"]) == [20, 3, 21, 21]
    assert np.array_equal(a["row_kp2d"], arrays["row_kp2d"][1:])
    assert m["cut_from"]["stride"] == 5
    assert k6bh2c.restore_candidates(tar, publish, {"other"}, 10) == []

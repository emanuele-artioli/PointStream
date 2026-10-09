"""H1's geometry, bit accounting, rigid-motion scoring and rates on synthetic inputs (no GPU, no video)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.visor import h1


def test_wrist_split_is_a_half_plane() -> None:
    mask = np.zeros((40, 40), bool)
    mask[5:35, 10:20] = True  # a vertical arm; the fingers point up
    hand, forearm = h1.wrist_split(mask, np.array([15.0, 20.0]), np.array([15.0, 10.0]))
    assert not (hand & forearm).any() and ((hand | forearm) == mask).all()
    assert hand[5:21].sum() == hand.sum() and forearm.sum() == mask[21:35].sum()


def test_rasterize_two_triangles_fill_the_square() -> None:
    points = np.array([[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    out = h1.rasterize(points, faces, (40, 40))
    assert abs(int(out.sum()) - 21 * 21) <= 42
    assert not out[:9].any() and not out[:, :9].any()


def test_match_detections_is_one_to_one_by_mask_pixels() -> None:
    left = np.zeros((100, 100), bool)
    left[20:80, 5:40] = True
    right = np.zeros((100, 100), bool)
    right[20:80, 60:95] = True
    boxes = np.array([[62, 25, 90, 50], [8, 25, 35, 50], [40, 0, 60, 10]], float)
    assert h1.match_detections({"left hand": left, "right hand": right}, boxes) == {"left hand": 1, "right hand": 0}
    assert h1.match_detections({"left hand": left}, np.zeros((0, 4))) == {}


def test_unwrap_order_hints_across_the_modulus() -> None:
    display = list(range(0, 300))
    # Decode order of a 32-frame pyramid is not monotonic; take a jittered permutation.
    order = []
    for start in range(0, 300, 32):
        group = list(range(start, min(start + 32, 300)))
        order += [group[-1]] + group[:-1]
    assert h1.unwrap_order_hints([d % 128 for d in order]) == order
    assert sorted(order) == display


def test_parse_inspect_and_bit_density_spread_each_block() -> None:
    size_map = {"BLOCK_4X4": 0, "BLOCK_8X8": 3, "BLOCK_16X8": 5}
    sizes = [[5, 5, 5, 5], [5, 5, 5, 5], [3, 3, 0, 0], [3, 3, 0, 0]]
    frame = {"blockSizeMap": size_map, "blockSize": sizes, "orderHint": 0,
             # 16x8 block at (0,0): 64 bits; 8x8 at (0,2): 32 bits in two symbols; 4x4 at (3,3): 8 bits.
             "symbols": [[0, 0], [1, 256, 1], [3, 0], [3, 0, 1], [0, 2], [2, 128, 1], [2, 128, 1], [3, 3], [4, 64, 1]]}
    text = "[\n" + json.dumps(frame) + ",\n" + json.dumps(frame) + ",\nnull\n]"
    parsed = h1.parse_inspect(text)
    assert len(parsed) == 2
    density, total = h1.bit_density(parsed[0], h1.block_dims(size_map))
    assert total == pytest.approx(32 + 0 + 32 + 8)
    assert density.sum() == pytest.approx(total)
    assert density[0, 0] == pytest.approx(32 / 8) and density[1, 3] == pytest.approx(32 / 8)
    assert density[2, 0] == pytest.approx(32 / 4) and density[3, 3] == pytest.approx(8)
    assert density[2, 2] == 0.0


def test_mi_counts() -> None:
    labels = np.zeros((8, 8), np.uint8)
    labels[:4, :2] = 2
    counts = h1.mi_counts(labels, 3)
    assert counts.shape == (3, 2, 2)
    assert counts[2, 0, 0] == 8 and counts[0, 0, 0] == 8 and counts.sum() == 64


def textured(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (shape[0] // 4, shape[1] // 4, 3)).astype(np.uint8)
    return cv2.resize(noise, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def shifted_pair(dx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (120, 160)
    prev = textured(shape)
    cur = np.roll(prev, dx, axis=1)
    mask_prev = np.zeros(shape, bool)
    mask_prev[30:90, 30:90] = True
    mask_t = np.roll(mask_prev, dx, axis=1)
    flow = np.zeros((*shape, 2), np.float32)
    flow[..., 0] = -dx  # backward flow: where each pixel of t was at t - 1
    return prev, cur, flow, mask_prev, mask_t


def test_rigid_shift_is_explained_by_the_homography_and_the_flow() -> None:
    prev, cur, flow, mask_prev, mask_t = shifted_pair(6)
    fits = h1.fit_models(flow, mask_t)
    assert np.allclose(fits["homography"], [[1, 0, 6], [0, 1, 0], [0, 0, 1]], atol=1e-3)
    row = h1.frame_motion(prev, cur, flow, mask_prev, mask_t, np.zeros_like(mask_t), fits)
    assert row["psnr_homography"] > 60 and row["psnr_flow"] > 60 and row["psnr_similarity"] > 60
    assert row["psnr_copy"] < 25
    split = row["error_split"]
    assert split["new"] + split["hand_occlusion"] + split["deformation"] + split["appearance"] + split["motion_blur"] == \
        pytest.approx(split["sse"])
    assert row["new_share"] == 0.0


def test_frame_motion_attributes_disocclusion_by_hands() -> None:
    prev, cur, flow, mask_prev, mask_t = shifted_pair(0)
    hand = np.zeros_like(mask_t)
    hand[30:90, 30:50] = True
    mask_prev &= ~hand  # the object's left part was under the hand at t - 1
    cur = cur.copy()
    fits = h1.fit_models(flow, mask_t)
    row = h1.frame_motion(prev, cur, flow, mask_prev, mask_t, hand, fits)
    assert row["hand_occlusion_share"] == pytest.approx(20 / 60)
    assert row["new_share"] == 0.0


def test_reference_chain_holds_then_refreshes_on_new_appearance() -> None:
    shape = (120, 160)
    base = textured(shape)
    frames = np.stack([np.roll(base, 2 * t, axis=1) for t in range(5)] + [textured(shape, seed=1)])
    masks = {t: np.roll(np.pad(np.ones((60, 60), bool), ((30, 30), (30, 70))), 2 * t, axis=1) for t in range(6)}
    chain = h1.ReferenceChain(30.0, fps=50.0)
    chain.start(0)
    step = np.array([[1, 0, 2], [0, 1, 0], [0, 0, 1]], float)
    results = [chain.step(t, step, frames, lambda r: masks[r], masks[t]) for t in range(1, 6)]
    assert [r["held"] for r in results] == [True, True, True, True, False]
    summary = chain.summary(5)
    assert summary["references"] == 2 and summary["held"] == 4 and summary["frames"] == 6
    assert summary["reference_life_s"] == [pytest.approx(5 / 50), pytest.approx(1 / 50)]


def test_corner_symbols_and_entropy() -> None:
    assert h1.corner_symbols(np.eye(3), (0, 0, 10, 10)) == [0] * 8
    shift = np.array([[1, 0, 0.25], [0, 1, 0], [0, 0, 1]])
    assert h1.corner_symbols(shift, (0, 0, 10, 10)) == [2, 0] * 4
    assert h1.entropy_bits([1, 1, 1, 1]) == 0.0
    assert h1.entropy_bits([0, 1, 2, 3]) == pytest.approx(2.0)


def test_review_frames_are_content_blind_and_stable() -> None:
    a = h1.review_frames("P01_107_0000003049", 240, 2)
    assert a == h1.review_frames("P01_107_0000003049", 240, 2)
    assert len(a) == 2 and all(1 <= t < 240 for t in a)


def test_sample_map_handles_more_points_than_remap_allows_per_side() -> None:
    image = textured((300, 400))
    ys, xs = np.mgrid[0:300, 0:400]
    points = np.c_[xs.reshape(-1), ys.reshape(-1)].astype(np.float64)  # 120,000 points
    assert np.array_equal(h1.sample_map(points, image), image.reshape(-1, 3))
    mask = np.zeros((300, 400), bool)
    mask[10:20] = True
    assert np.array_equal(h1.sample_map(points, mask, nearest=True), mask.reshape(-1))

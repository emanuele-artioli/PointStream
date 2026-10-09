"""Depth-aware warps on a synthetic translating camera: two textured planes at different depths."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from experiments.background import camera, depth

W, H = camera.ANALYSIS_SIZE
LENS = depth.lens_for(70.0, -0.12)
FAR, NEAR = 4.0, 1.6  # z of the back wall and of a near panel covering x < NEAR_EDGE


def texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 255, (1024, 1024)).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), 1.6)


TEXTURES = (texture(1), texture(2))
NEAR_EDGE = -0.15


def render(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grey image and z-depth of the scene seen by camera x_cam = R X + t."""
    grid = camera.pixel_grid(W, H).reshape(-1, 2)
    d = depth.rays(LENS, grid) @ R  # world directions (R^T ray), row-wise
    centre = -R.T @ t
    best = np.full(len(d), np.inf)
    value = np.zeros(len(d), np.float32)
    for plane, z0 in enumerate((FAR, NEAR)):
        s = (z0 - centre[2]) / d[:, 2]
        X = centre + s[:, None] * d
        ok = (s > 0) & (s < best)
        if plane == 1:
            ok &= X[:, 0] < NEAR_EDGE
        tex = TEXTURES[plane]
        u = np.clip(X[:, 0] * 150 + 512, 0, 1022).astype(np.float32)
        v = np.clip(X[:, 1] * 150 + 512, 0, 1022).astype(np.float32)
        sample = cv2.remap(tex, u.reshape(H, W), v.reshape(H, W), cv2.INTER_LINEAR).ravel()
        best = np.where(ok, s, best)
        value = np.where(ok, sample, value)
    Xc = (centre + best[:, None] * d - centre) @ R.T  # camera-frame point
    return np.clip(value, 0, 255).astype(np.uint8).reshape(H, W), Xc[:, 2].reshape(H, W)


POSE_B = (camera.rotation([0.01, -0.03, 0.005]), np.array([0.12, 0.02, 0.05]))
POSE_C = (camera.rotation([-0.01, 0.02, 0.0]), np.array([-0.10, 0.03, -0.04]))


@pytest.fixture(scope="module")
def scene() -> dict[str, object]:
    ident = (np.eye(3), np.zeros(3))
    key, key_depth = render(*ident)
    target, _ = render(*POSE_B)
    comp, _ = render(*POSE_C)
    bg = np.ones((H, W), bool)
    fk, ft, fc = camera.features(key, bg), camera.features(target, bg), camera.features(comp, bg)
    return {"key": key, "depth": key_depth, "target": target, "comp": comp, "bg": bg,
            "kt": camera.match(fk, ft), "kc": camera.match(fk, fc)}


def p90(warp: camera.Warp, target: np.ndarray, source: np.ndarray, bg: np.ndarray, lens: camera.Lens = LENS) -> float:
    res = camera.residual(lens, target, source, warp, bg, bg)
    assert res["measured"]
    return float(res["flow_p90_px"])


def test_true_depth_and_pose_reproduce_the_frame(scene):
    warp, hit = depth.warp_depth(LENS, scene["depth"], scene["bg"], *POSE_B)
    assert hit.mean() > 0.85
    assert p90(warp, scene["target"], scene["key"], scene["bg"]) < 0.6


def test_one_homography_cannot(scene):
    pa, pb = scene["kt"]
    Hm, _ = depth.homography_u(LENS, pa, pb)
    assert Hm is not None
    assert p90(depth.warp_h(LENS, Hm), scene["target"], scene["key"], scene["bg"]) > 2.0


def test_pnp_recovers_the_pose(scene):
    pa, pb = scene["kt"]
    pose = depth.pnp(LENS, scene["depth"], scene["bg"], pa, pb)
    assert pose is not None
    R, t, inliers = pose
    assert camera.angle_deg(R @ POSE_B[0].T) < 0.05
    assert np.linalg.norm(t - POSE_B[1]) < 0.01
    assert inliers > 100


def test_triangulated_depth_matches_truth_up_to_scale(scene):
    pa, pc = scene["kc"]
    kd = depth.keyframe_depth(LENS, scene["key"], scene["bg"], [{"gray": scene["comp"], "bg": scene["bg"],
                                                                  "pa": pa, "pb": pc, "index": 1}])
    assert kd is not None
    both = kd.measured & np.isfinite(kd.depth)
    ratio = kd.depth[both] / scene["depth"][both]
    spread = np.percentile(ratio, 90) / np.percentile(ratio, 10)
    assert spread < 1.08
    assert kd.png_bytes > 0 and kd.png16_bytes >= kd.png_bytes
    # Rendered with the triangulated (and quantized) depth and the PnP pose, the frame is explained.
    pa, pb = scene["kt"]
    pose = depth.pnp(LENS, kd.depth, kd.measured, pa, pb)
    assert pose is not None
    warp, _ = depth.warp_depth(LENS, kd.depth, scene["bg"], pose[0], pose[1])
    assert p90(warp, scene["target"], scene["key"], scene["bg"]) < camera.EXPLAINED_PX


def test_epipolar_ceiling_explains_a_rigid_scene(scene):
    pa, pb = scene["kt"]
    Hm, _ = depth.homography_u(LENS, pa, pb)
    pair = camera.translation_pair(LENS, pa, pb)
    assert pair is not None and Hm is not None
    warp = depth.warp_epipolar(LENS, scene["target"], scene["key"], Hm, pair["F"], scene["bg"], scene["bg"])
    assert p90(warp, scene["target"], scene["key"], scene["bg"]) < camera.EXPLAINED_PX


def test_two_planes_explain_two_planes(scene):
    pa, pc = scene["kc"]
    labels = depth.plane_labels(LENS, scene["key"], scene["comp"], scene["bg"], scene["bg"], pa, pc, 2)
    assert labels is not None and set(np.unique(labels)) == {0, 1}
    pa, pb = scene["kt"]
    homographies = []
    xi = np.clip(np.rint(pa[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.rint(pa[:, 1]).astype(int), 0, H - 1)
    for i in range(2):
        sel = labels[yi, xi] == i
        Hm, _ = depth.homography_u(LENS, pa[sel], pb[sel])
        assert Hm is not None
        homographies.append(Hm)
    warp, hit = depth.warp_planes(LENS, labels, homographies)
    assert hit.mean() > 0.85
    assert p90(warp, scene["target"], scene["key"], scene["bg"]) < camera.EXPLAINED_PX


def test_splat_keeps_the_nearest_and_fills_cracks():
    positions = np.array([[10.2, 5.1], [10.4, 5.3], [30.0, 30.0]])
    sp = depth.splat(64, 48, positions, np.array([1.0, 2.0, 0.5]), np.array([7, 8, 9]))
    assert sp.winner[5, 10] == 8  # the larger key wins
    assert sp.winner[30, 30] == 9 and sp.winner[31, 31] == 9
    hole = depth.Splat(np.ones((5, 5)), np.arange(25).reshape(5, 5))
    hole.winner[2, 2], hole.key[2, 2] = -1, -np.inf
    assert depth.fill_cracks(hole).winner[2, 2] >= 0


def test_complete_extrapolates_the_map_into_holes():
    grid = camera.pixel_grid(64, 48).astype(np.float32)
    valid = np.ones((48, 64), bool)
    valid[10:30, 20:45] = False
    valid[:, :3] = False
    warp = camera.Warp(grid[..., 0] + 2.5, grid[..., 1] - 1.0, valid)
    warp.map_x[~valid], warp.map_y[~valid] = -1, -1
    done = depth.complete(warp)
    inside = ~valid & (grid[..., 0] + 2.5 <= 63) & (grid[..., 1] - 1.0 >= 0)
    assert np.allclose(done.map_x[inside], grid[..., 0][inside] + 2.5)
    assert np.allclose(done.map_y[inside], grid[..., 1][inside] - 1.0)
    assert (done.valid == valid).all()


def test_calibration_finds_the_focal_length():
    rng = np.random.default_rng(0)
    truth = depth.lens_for(78.0, -0.15)
    pairs = []
    for _ in range(10):
        X = np.stack([rng.uniform(-3, 3, 600), rng.uniform(-2, 2, 600), rng.uniform(1.0, 6.0, 600)], 1)
        R = camera.rotation(rng.normal(0, 0.03, 3))
        t = rng.normal(0, 0.15, 3)
        pa, oka = depth.project(truth, X)
        pb, okb = depth.project(truth, X @ R.T + t)
        ok = oka & okb & truth.inside(pa) & truth.inside(pb)
        noise = rng.normal(0, 0.3, (2, int(ok.sum()), 2))
        pairs.append((pa[ok] + noise[0], pb[ok] + noise[1]))
    lens, report = depth.calibrate(pairs)
    assert report["observable"]
    hfov = math.degrees(2 * math.atan(lens.width / 2 / lens.f))
    assert abs(hfov - 78.0) <= 4.0
    assert abs(lens.k1 + 0.15) <= 0.06

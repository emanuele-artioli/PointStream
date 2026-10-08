from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from experiments.background import camera
from experiments.background.camera import Lens, Pose

LENS = Lens(f=520.0, k1=-0.12)


def texture() -> np.ndarray:
    """A fixed equirectangular texture with features at several scales."""
    rng = np.random.default_rng(7)
    base = rng.random((512, 1024)).astype(np.float32)
    tex = np.zeros((2048, 4096), np.float32)
    for s, w in ((1, 0.5), (3, 1.0), (8, 1.5)):
        tex += cv2.resize(cv2.GaussianBlur(base, (0, 0), s), (4096, 2048)) * w
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    return (tex * 230 + 10).astype(np.uint8)


TEX = texture()


def render(lens: Lens, R: np.ndarray, logf: float = 0.0) -> np.ndarray:
    """The textured sphere seen by a camera that only rotates (rays = R @ world)."""
    grid = camera.pixel_grid(lens.width, lens.height)
    world = lens.rays(grid, logf) @ R
    lon = np.arctan2(world[..., 0], world[..., 2])
    lat = np.arcsin(np.clip(world[..., 1], -1, 1))
    mx = ((lon + math.pi) / (2 * math.pi) * TEX.shape[1]).astype(np.float32)
    my = ((lat + math.pi / 2) / math.pi * TEX.shape[0]).astype(np.float32)
    return cv2.remap(TEX, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def rot(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    return camera.rotation(np.radians([pitch, yaw, roll]))


def test_distortion_round_trip() -> None:
    pts = np.random.default_rng(0).uniform([0, 0], [959, 539], (200, 2))
    for k1 in (-0.3, 0.0, 0.1):
        lens = Lens(500.0, k1)
        assert np.allclose(lens.undistort(lens.distort(pts)), pts, atol=1e-6)
        rays = lens.rays(pts, 0.1)
        back, ok = lens.project(rays, 0.1)
        assert ok.all() and np.allclose(back, pts, atol=1e-6)


def test_rotation_fit_recovers_rotation_and_zoom() -> None:
    rng = np.random.default_rng(1)
    pa = rng.uniform([50, 50], [910, 490], (300, 2))
    R = rot(4.0, 1.5, 0.7)
    pb, ok = LENS.project(LENS.rays(pa) @ R.T, 0.05)
    fit = camera.fit_rotation(LENS, pa[ok], pb[ok] + rng.normal(0, 0.2, pb[ok].shape))
    assert camera.angle_deg(fit.R @ R.T) < 0.02
    assert abs(fit.s - 0.05) < 1e-3
    assert np.median(fit.errors) < 0.5


def test_lens_fit_recovers_focal_length_and_distortion() -> None:
    rng = np.random.default_rng(2)
    pairs = []
    for i in range(10):
        pa = rng.uniform([20, 20], [940, 520], (200, 2))
        R = rot(rng.uniform(-8, 8), rng.uniform(-5, 5), rng.uniform(-2, 2))
        pb, ok = LENS.project(LENS.rays(pa) @ R.T)
        keep = ok & LENS.inside(pb)
        pairs.append((pa[keep], pb[keep] + rng.normal(0, 0.2, pb[keep].shape)))
    lens, report = camera.fit_lens(pairs)
    assert report["observable"]
    assert abs(lens.f - LENS.f) / LENS.f < 0.03
    assert abs(lens.k1 - LENS.k1) < 0.03


def test_lens_unobservable_without_motion() -> None:
    pa = np.random.default_rng(3).uniform([20, 20], [940, 520], (200, 2))
    lens, report = camera.fit_lens([(pa, pa + 0.1)] * 10)
    assert not report["observable"] and lens == camera.prior_lens()


def test_translation_test_separates_rotation_from_parallax() -> None:
    rng = np.random.default_rng(4)
    lens = Lens(500.0)
    pts = rng.uniform([-3, -2, 0.6], [3, 2, 4.0], (400, 3))  # scene 0.6-4 m deep

    def view(R: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cam = (pts - c) @ R.T
        p, ok = lens.project(cam / np.linalg.norm(cam, axis=1, keepdims=True))
        return p, ok & lens.inside(p)

    pa, oa = view(np.eye(3), np.zeros(3))
    pb, ob = view(rot(3.0), np.zeros(3))
    keep = oa & ob
    still = camera.translation_pair(lens, pa[keep], pb[keep] + rng.normal(0, 0.2, pb[keep].shape))
    pc, oc = view(rot(3.0), np.array([0.08, 0.0, 0.03]))  # 8 cm sideways, as a head moves
    keep = oa & oc
    moved = camera.translation_pair(lens, pa[keep], pc[keep] + rng.normal(0, 0.2, pc[keep].shape))
    assert still is not None and moved is not None
    assert not still["prefers_f"] and still["parallax_p50_px"] < 1.0
    assert moved["prefers_f"] and moved["parallax_p50_px"] > 2.0


@pytest.fixture(scope="module")
def rotating_clip() -> tuple[list[np.ndarray], list[np.ndarray]]:
    yaws = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]
    frames = [render(LENS, rot(y, 0.5 * i)) for i, y in enumerate(yaws)]
    rotations = [rot(y, 0.5 * i) for i, y in enumerate(yaws)]
    return frames, rotations


def test_rotating_camera_registers_and_is_explained(rotating_clip: tuple[list[np.ndarray], list[np.ndarray]]) -> None:
    frames, rotations = rotating_clip
    bg = np.ones(frames[0].shape, bool)
    feats = [camera.features(f, bg) for f in frames]
    f2f: list[tuple[np.ndarray, np.ndarray] | None] = []
    for t in range(len(frames)):
        if t == 0:
            f2f.append(None)
            continue
        pa, pb = camera.match(feats[t - 1], feats[t])
        H, keep = camera.homography(pa, pb)
        f2f.append((pa[keep], pb[keep]) if H is not None else None)
    reg = camera.register(LENS, feats, f2f)
    assert [p.status for p in reg.poses].count("direct") == len(frames) - 1
    for t in range(1, len(frames)):
        truth = rotations[t] @ rotations[0].T
        assert camera.angle_deg(reg.poses[t].R @ truth.T) < 0.1
        pose, ref = reg.poses[t], reg.poses[reg.poses[t].reference]  # type: ignore[index]
        res = camera.residual(LENS, frames[t], frames[pose.reference], camera.warp_rotation(LENS, pose, ref),  # type: ignore[index]
                              bg, bg, attribute=True)
        assert res["measured"] and res["explained"], res
        assert res["flow_p90_px"] < 1.0


def test_coverage_counts_what_the_warm_up_saw() -> None:
    lens = Lens(500.0)
    # Pan right 2 degrees per frame for 20 frames, then back to the start.
    yaws = [2.0 * t for t in range(20)] + [2.0 * (19 - t) for t in range(20)]
    poses = [Pose(rot(y), 0.0, "direct", 0) for y in yaws]
    times = np.arange(len(poses), dtype=float)
    bg = np.ones((lens.height, lens.width), bool)
    warmups = np.array([0.0, 5.0, 19.0, 30.0])
    cov = camera.coverage(lens, poses, times, lambda t: bg, warmups)
    curve = dict(zip(cov["warmups_s"], cov["curve"]))
    assert curve[19.0] > 0.99  # everything after the pan was seen during it
    assert curve[0.0] < curve[5.0] < curve[19.0]
    assert cov["online"][25] > 0.99
    assert cov["warmup_99_s"] == 19.0


def test_foreground_occlusion_lowers_coverage() -> None:
    lens = Lens(500.0)
    poses = [Pose(np.eye(3), 0.0, "direct", 0) for _ in range(10)]
    times = np.arange(10, dtype=float)
    left = np.ones((lens.height, lens.width), bool)
    left[:, :480] = False  # a person covers the left half during the first five frames
    full = np.ones_like(left)
    cov = camera.coverage(lens, poses, times, lambda t: left if t < 5 else full, np.array([0.0, 4.0, 5.0]))
    curve = dict(zip(cov["warmups_s"], cov["curve"]))
    assert abs(curve[4.0] - 0.5) < 0.02 and curve[5.0] > 0.99


def test_degenerate_matches_yield_no_fit_instead_of_raising() -> None:
    # Identical points give OpenCV's USAC no model; that must read as "no fit".
    same = np.full((60, 2), 100.0, np.float32)
    H, _ = camera.homography(same, same)
    assert H is None
    assert camera.translation_pair(LENS, same, same) is None

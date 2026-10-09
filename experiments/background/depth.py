"""Depth-aware warps of a reference keyframe into a frame (PLAN step G1d).

Works on G1's analysis frames (``camera.ANALYSIS_SIZE``) and returns
``camera.Warp`` objects, so `camera.residual` measures every method the same
way. Pixels are distorted image pixels; ``lens.undistort`` maps them to an ideal
pinhole whose intrinsics are ``K(lens)`` (principal point at the image centre).

* `calibrate`: focal length and k1 of a translating camera, on a grid, by the
  median Sampson error of the calibrated essential matrix.
* `warp_h`: one homography on undistorted pixels.
* `plane_labels` and `warp_planes`: a few planes, each with its own homography;
  a label map on the keyframe, forward-splatted (the larger displacement wins).
* `warp_epipolar`: dense flow projected onto the pair's epipolar lines, the
  best any static scene can do (a ceiling, not a representation).
* `triangulate` and `keyframe_depth`: depth of a keyframe from dense flow to
  companion frames, posed by the essential matrix.
* `pnp` and `warp_depth`: a frame's pose from the keyframe's 3D points, and the
  keyframe's background forward-splatted into the frame with a z-buffer.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from experiments.background import camera

CAL_HFOV_DEG = tuple(float(x) for x in range(40, 124, 4))
CAL_K1 = tuple(round(-0.5 + 0.05 * i, 3) for i in range(13))
#: EPIC Fields' COLMAP calibration of EPIC-KITCHENS (example P28_101, OPENCV model at 456x256:
#: fx 239.61, fy 243.08, principal point at the centre, k1 -0.0013, k2 0.0002, p1 0.0009, p2 -0.0006;
#: https://github.com/epic-kitchens/epic-fields-code, example_data/P28_101.json). Distortion is negligible.
EPIC_FIELDS = {"fx_456": 239.61061989963898, "fy_456": 243.07683578049975, "width": 456}
CAL_PAIRS = 24
CAL_MIN_DISPLACEMENT = 6.0  # analysis px, median, for a pair to inform the calibration
PLANE_PX = 1.0  # analysis px (2 at 1080p, the explained bar): inlier threshold of a plane's homography
PLANE_MIN = camera.MIN_INLIERS
DENSE_STEP = 6  # px: grid of the dense correspondences the planes are fitted to
MODE_FILTER = 31  # px: box of the label vote
MIN_ANGLE_DEG = 0.5  # triangulation angle of a usable depth
TRI_REPROJECTION_PX = 1.5  # analysis px
PNP_PX = 2.0  # analysis px
PNP_MIN = 12
DEPTH_LEVELS = 255  # 8-bit inverse depth
CRACK_NEIGHBOURS = 5  # of 8: a hole pixel with this many rendered neighbours is a crack
FILL_WINDOWS = (31, 61, 121, 241)  # px: windows of the local plane fits that fill depth holes
FILL_SUPPORT = 0.15  # share of a window that must be measured for its plane to fill
FUSE_AGREEMENT = 0.05  # median |log depth ratio| for a second companion to join the primary


# ----------------------------------------------------------------- lens


def K(lens: camera.Lens) -> np.ndarray:
    return np.array([[lens.f, 0, lens.width / 2], [0, lens.f, lens.height / 2], [0, 0, 1.0]])


def lens_for(hfov_deg: float, k1: float) -> camera.Lens:
    width, height = camera.ANALYSIS_SIZE
    return camera.Lens((width / 2) / math.tan(math.radians(hfov_deg) / 2), k1, width, height)


def epic_fields_lens() -> camera.Lens:
    """EPIC Fields' focal length (mean of fx and fy) at the analysis size, without distortion."""
    width, height = camera.ANALYSIS_SIZE
    f = (EPIC_FIELDS["fx_456"] + EPIC_FIELDS["fy_456"]) / 2 * width / EPIC_FIELDS["width"]
    return camera.Lens(f, 0.0, width, height)


def normalized(lens: camera.Lens, pts: np.ndarray) -> np.ndarray:
    return (lens.undistort(pts) - lens.centre) / lens.f


def essential_error(lens: camera.Lens, pa: np.ndarray, pb: np.ndarray) -> float | None:
    """Median Sampson error (analysis px) of the calibrated essential matrix fitted to ``pa -> pb``."""
    na, nb = normalized(lens, pa), normalized(lens, pb)
    try:
        E, _ = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1.0 / lens.f)
    except cv2.error:
        return None
    if E is None or E.shape != (3, 3):
        return None
    return float(np.median(camera.sampson(E, na, nb))) * lens.f


def calibrate(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[camera.Lens, dict[str, Any]]:
    """Focal length and k1 minimising the median over pairs of the essential matrix's median Sampson error.

    Pairs are raw matches already filtered by a loose fundamental matrix (outliers
    out, distortion tolerated). A coarse grid, then a finer one around its best.
    """
    used = [(pa.astype(np.float64), pb.astype(np.float64)) for pa, pb in pairs
            if len(pa) >= 2 * camera.MIN_INLIERS
            and float(np.median(np.linalg.norm(pb - pa, axis=1))) >= CAL_MIN_DISPLACEMENT][:CAL_PAIRS]
    report: dict[str, Any] = {"pairs_offered": len(pairs), "pairs_used": len(used)}
    if len(used) < 4:
        lens = lens_for(camera.PRIOR_HFOV_DEG, 0.0)
        report.update(observable=False, hfov_deg=camera.PRIOR_HFOV_DEG, k1=0.0)
        return lens, report

    def score(hfov: float, k1: float) -> float:
        lens = lens_for(hfov, k1)
        errors = [e for pa, pb in used if (e := essential_error(lens, pa, pb)) is not None]
        return float(np.median(errors)) if errors else math.inf

    cv2.setRNGSeed(0)
    coarse = {(h, k): score(h, k) for h in CAL_HFOV_DEG for k in CAL_K1}
    h0, k0 = min(coarse, key=lambda key: coarse[key])
    fine = {(h, k): score(h, k) for h in np.arange(h0 - 3, h0 + 3.01, 1.0) for k in np.arange(k0 - 0.04, k0 + 0.041, 0.02)}
    best = min(fine, key=lambda key: fine[key])
    hfov, k1 = float(best[0]), float(np.clip(best[1], -0.6, 0.15))
    worst = max(v for v in coarse.values() if np.isfinite(v))
    report.update(observable=True, hfov_deg=round(hfov, 2), k1=round(k1, 4), error_px=round(fine[best] * camera.TO_1080, 4),
                  coarse_best=[h0, k0], coarse_worst_error_px=round(worst * camera.TO_1080, 4),
                  error_at_g1_prior_px=round(score(camera.PRIOR_HFOV_DEG, 0.0) * camera.TO_1080, 4),
                  error_at_epic_fields_px=round(score(math.degrees(2 * math.atan(epic_fields_lens().width / 2 / epic_fields_lens().f)), 0.0)
                                                * camera.TO_1080, 4))
    return lens_for(hfov, k1), report


# ----------------------------------------------------------------- warps from undistorted maps


def warp_from(lens: camera.Lens, to_source: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]) -> camera.Warp:
    """Warp whose undistorted source position of each undistorted target pixel is ``to_source``."""
    grid = camera.pixel_grid(lens.width, lens.height).reshape(-1, 2)
    src_u, ok = to_source(lens.undistort(grid))
    q = lens.distort(src_u).reshape(lens.height, lens.width, 2)
    valid = ok.reshape(lens.height, lens.width) & lens.inside(q) & np.isfinite(q).all(-1)
    q = np.where(np.isfinite(q), q, -1.0)
    return camera.Warp(q[..., 0].astype(np.float32), q[..., 1].astype(np.float32), valid)


def warp_h(lens: camera.Lens, H: np.ndarray) -> camera.Warp:
    """One homography ``H`` mapping undistorted source pixels to undistorted target pixels."""
    Hinv = np.linalg.inv(H)
    return warp_from(lens, lambda u: (camera.apply_h(Hinv, u), np.ones(len(u), bool)))


def homography_u(lens: camera.Lens, pa: np.ndarray, pb: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    return camera.homography(lens.undistort(pa).astype(np.float32), lens.undistort(pb).astype(np.float32))


def dense_correspondence(lens: camera.Lens, target: np.ndarray, source: np.ndarray, H: np.ndarray,
                         bg_target: np.ndarray, bg_source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distorted source position of every target pixel: ``H`` (source -> target, undistorted), refined by DIS flow.

    The flow is computed after a gain and offset fit, as in `camera.residual`.
    Returns positions (H, W, 2) and their validity.
    """
    base = warp_h(lens, H)
    warped = cv2.remap(source, base.map_x, base.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped_bg = cv2.remap(bg_source.astype(np.uint8), base.map_x, base.map_y, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT) > 0
    region = base.valid & bg_target & warped_bg
    s = warped.astype(np.float32)
    if region.sum() > 100:
        a, b = np.linalg.lstsq(np.stack([s[region], np.ones(int(region.sum()), np.float32)], 1),
                               target.astype(np.float32)[region], rcond=None)[0]
        s = a * s + b
    u = camera.flow(target, np.clip(s, 0, 255).astype(np.uint8))
    grid = camera.pixel_grid(lens.width, lens.height)
    px, py = grid[..., 0] + u[..., 0], grid[..., 1] + u[..., 1]
    pos = np.stack([camera.bilinear(base.map_x, px, py), camera.bilinear(base.map_y, px, py)], -1)
    ok = (camera.bilinear(base.valid.astype(np.float32), px, py) > 0.99) & region
    return pos, ok


def warp_epipolar(lens: camera.Lens, target: np.ndarray, source: np.ndarray, H: np.ndarray, F: np.ndarray,
                  bg_target: np.ndarray, bg_source: np.ndarray) -> camera.Warp:
    """Dense flow target -> source projected onto the epipolar lines of ``F`` (``x_t^T F x_s = 0``, undistorted)."""
    pos, ok = dense_correspondence(lens, target, source, H, bg_target, bg_source)
    grid = camera.pixel_grid(lens.width, lens.height).reshape(-1, 2)
    xt = np.concatenate([lens.undistort(grid), np.ones((len(grid), 1))], 1)
    line = xt @ F  # l_s = F^T x_t, row-wise
    q = lens.undistort(pos.reshape(-1, 2))
    dist = (line[:, 0] * q[:, 0] + line[:, 1] * q[:, 1] + line[:, 2]) / np.maximum(np.hypot(line[:, 0], line[:, 1]), 1e-12)
    norm = line[:, :2] / np.maximum(np.hypot(line[:, 0], line[:, 1]), 1e-12)[:, None]
    projected = q - dist[:, None] * norm
    p = lens.distort(projected).reshape(lens.height, lens.width, 2)
    valid = ok & lens.inside(p)
    return camera.Warp(p[..., 0].astype(np.float32), p[..., 1].astype(np.float32), valid)


def complete(warp: camera.Warp) -> camera.Warp:
    """The same warp with its map extrapolated into invalid pixels (validity unchanged).

    `camera.residual` measures dense flow on the whole warped image; black holes
    inside it drag the flow at their edges. Each invalid pixel takes its nearest
    valid pixel's source position plus its offset from that pixel.
    """
    invalid = ~warp.valid
    if not invalid.any() or not warp.valid.any():
        return warp
    _, labels = cv2.distanceTransformWithLabels(invalid.astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    vy, vx = np.nonzero(warp.valid)  # the zero pixels of ``invalid``, in raster order, are labels 1..N
    iy, ix = np.nonzero(invalid)
    nearest = labels[iy, ix] - 1
    ny, nx = vy[nearest], vx[nearest]
    h, w = warp.valid.shape
    map_x, map_y = warp.map_x.copy(), warp.map_y.copy()
    map_x[iy, ix] = np.clip(warp.map_x[ny, nx] + (ix - nx), 0, w - 1)
    map_y[iy, ix] = np.clip(warp.map_y[ny, nx] + (iy - ny), 0, h - 1)
    return camera.Warp(map_x, map_y, warp.valid)


# ----------------------------------------------------------------- forward splatting


@dataclass
class Splat:
    key: np.ndarray  # (H, W) float: priority of the winning source pixel, -inf where nothing landed
    winner: np.ndarray  # (H, W) int: flat index of the winning source pixel, -1 where nothing landed


def splat(width: int, height: int, positions: np.ndarray, keys: np.ndarray, sources: np.ndarray) -> Splat:
    """Each source pixel lands on the 4 target pixels around its position; the largest key wins a pixel."""
    x, y = positions[:, 0], positions[:, 1]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    tx = np.concatenate([x0, x0 + 1, x0, x0 + 1])
    ty = np.concatenate([y0, y0, y0 + 1, y0 + 1])
    k = np.tile(keys, 4)
    s = np.tile(sources, 4)
    inside = (tx >= 0) & (tx < width) & (ty >= 0) & (ty < height) & np.isfinite(k)
    flat, k, s = (ty * width + tx)[inside], k[inside], s[inside]
    order = np.lexsort((-k, flat))  # by target, then largest key first
    flat, k, s = flat[order], k[order], s[order]
    first = np.ones(len(flat), bool)
    first[1:] = flat[1:] != flat[:-1]
    key = np.full(width * height, -np.inf)
    winner = np.full(width * height, -1, np.int64)
    key[flat[first]] = k[first]
    winner[flat[first]] = s[first]
    return Splat(key.reshape(height, width), winner.reshape(height, width))


def fill_cracks(sp: Splat) -> Splat:
    """A hole with at least ``CRACK_NEIGHBOURS`` of 8 rendered neighbours takes its best neighbour (one pass)."""
    hit = sp.winner >= 0
    count = cv2.filter2D(hit.astype(np.float32), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - hit
    holes = ~hit & (count >= CRACK_NEIGHBOURS - 0.5)
    if not holes.any():
        return sp
    key, winner = sp.key.copy(), sp.winner.copy()
    ys, xs = np.nonzero(holes)
    best_k = np.full(len(ys), -np.inf)
    best_w = np.full(len(ys), -1, np.int64)
    h, w = key.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yy, xx = np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)
            better = sp.key[yy, xx] > best_k
            best_k = np.where(better, sp.key[yy, xx], best_k)
            best_w = np.where(better, sp.winner[yy, xx], best_w)
    key[ys, xs], winner[ys, xs] = best_k, best_w
    return Splat(key, winner)


# ----------------------------------------------------------------- planes


def fit_planes(lens: camera.Lens, pa: np.ndarray, pb: np.ndarray, count: int) -> list[np.ndarray]:
    """Up to ``count`` homographies (undistorted, a -> b) by sequential MAGSAC, largest first."""
    ua, ub = lens.undistort(pa).astype(np.float32), lens.undistort(pb).astype(np.float32)
    left = np.ones(len(ua), bool)
    planes: list[np.ndarray] = []
    while len(planes) < count and left.sum() >= PLANE_MIN:
        idx = np.nonzero(left)[0]
        try:
            H, inl = cv2.findHomography(ua[idx], ub[idx], cv2.USAC_MAGSAC, PLANE_PX, maxIters=5000, confidence=0.999)
        except cv2.error:
            break
        if H is None or inl is None or inl.sum() < PLANE_MIN:
            break
        planes.append(H)
        left[idx[inl.ravel().astype(bool)]] = False
    return planes


def dense_samples(key_gray: np.ndarray, pos: np.ndarray, ok: np.ndarray, step: int = DENSE_STEP) -> tuple[np.ndarray, np.ndarray]:
    """Keyframe pixels on a ``step`` grid (textured, with a correspondence) and their positions ``pos``."""
    sel = np.zeros_like(ok)
    sel[::step, ::step] = True
    sel &= ok & camera.textured(key_gray)
    ys, xs = np.nonzero(sel)
    return np.stack([xs, ys], 1).astype(np.float32), pos[ys, xs].astype(np.float32)


def plane_labels(lens: camera.Lens, key_gray: np.ndarray, comp_gray: np.ndarray, bg_key: np.ndarray,
                 bg_comp: np.ndarray, pa: np.ndarray, pb: np.ndarray, count: int) -> np.ndarray | None:
    """Label map on the keyframe (-1 off the background): planes fitted to the dense flow to the companion
    (sampled by `dense_samples`), each pixel labelled by the plane that best predicts its flow, then a mode
    filter. ``pa -> pb`` (keyframe-to-companion matches) only seed the dense flow."""
    H, _ = homography_u(lens, pa, pb)
    if H is None:
        return None
    # Companion position of every keyframe pixel (the warp maps companion -> keyframe).
    target, ok_t = dense_correspondence(lens, key_gray, comp_gray, np.linalg.inv(H), bg_key, bg_comp)
    planes = fit_planes(lens, *dense_samples(key_gray, target, ok_t), count)
    if not planes:
        return None
    grid = camera.pixel_grid(lens.width, lens.height).reshape(-1, 2)
    u = lens.undistort(grid)
    errors = np.stack([np.linalg.norm(lens.distort(camera.apply_h(H, u)) - target.reshape(-1, 2), axis=1)
                       for H in planes], 1).reshape(lens.height, lens.width, len(planes))
    raw = np.argmin(errors, -1)
    votes = []
    for i in range(len(planes)):
        votes.append(cv2.blur(((raw == i) & ok_t).astype(np.float32), (MODE_FILTER, MODE_FILTER)))
    labels = np.argmax(np.stack(votes, -1), -1).astype(np.int8)
    labels[~bg_key] = -1
    return labels


def warp_planes(lens: camera.Lens, labels: np.ndarray, homographies: list[np.ndarray]) -> tuple[camera.Warp, np.ndarray]:
    """Forward-splat the labelled keyframe pixels by their plane's homography; returns the warp and the hit map."""
    h, w = labels.shape
    grid = camera.pixel_grid(w, h).reshape(-1, 2)
    flat_labels = labels.ravel()
    src = np.nonzero(flat_labels >= 0)[0]
    u = lens.undistort(grid[src])
    dest_u = np.empty_like(u)
    for i, H in enumerate(homographies):
        sel = flat_labels[src] == i
        dest_u[sel] = camera.apply_h(H, u[sel])
    dest = lens.distort(dest_u)
    keys = np.linalg.norm(dest - grid[src], axis=1)  # nearer surfaces move more and occlude
    sp = fill_cracks(splat(w, h, dest, keys, src))
    hit = sp.winner >= 0
    label_t = np.where(hit, flat_labels[np.maximum(sp.winner, 0)].reshape(h, w), -1)
    target_u = lens.undistort(grid)
    src_u = np.full_like(target_u, -1.0)
    for i, H in enumerate(homographies):
        sel = label_t.ravel() == i
        src_u[sel] = camera.apply_h(np.linalg.inv(H), target_u[sel])
    q = lens.distort(src_u).reshape(h, w, 2)
    valid = hit & lens.inside(q)
    return camera.Warp(q[..., 0].astype(np.float32), q[..., 1].astype(np.float32), valid), hit


# ----------------------------------------------------------------- depth


def rays(lens: camera.Lens, pts: np.ndarray) -> np.ndarray:
    """Pinhole rays with z = 1 through distorted pixels."""
    n = normalized(lens, pts)
    return np.concatenate([n, np.ones(n.shape[:-1] + (1,))], -1)


def relative_pose(lens: camera.Lens, pa: np.ndarray, pb: np.ndarray) -> tuple[np.ndarray, np.ndarray, int] | None:
    """R, t (unit) with x_b ~ R x_a + t, from the essential matrix of matches ``pa -> pb``."""
    na, nb = normalized(lens, pa), normalized(lens, pb)
    try:
        E, inl = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1.0 / lens.f)
        if E is None or E.shape != (3, 3):
            return None
        good, R, t, _ = cv2.recoverPose(E, na, nb, np.eye(3), mask=inl)
    except cv2.error:
        return None
    if good < camera.MIN_INLIERS:
        return None
    return R, t.ravel(), int(good)


def triangulate(lens: camera.Lens, R: np.ndarray, t: np.ndarray, key_pts: np.ndarray,
                other_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depth along each keyframe ray (z-depth, units of ``|t|``) and the triangulation angle in degrees."""
    a = rays(lens, key_pts)
    b = rays(lens, other_pts)
    Ra = a @ R.T
    m = np.cross(b, Ra)
    n = np.cross(b, np.broadcast_to(t, b.shape))
    d = -(m * n).sum(-1) / np.maximum((m * m).sum(-1), 1e-18)
    X = d[..., None] * a
    centre = -R.T @ t
    v = X - centre
    cos = (a * v).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(v, axis=-1), 1e-18)
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return d, angle


def project(lens: camera.Lens, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = X[..., 2]
    ok = z > 1e-9
    u = lens.centre + lens.f * X[..., :2] / np.where(ok, z, 1.0)[..., None]
    return lens.distort(u), ok


@dataclass
class KeyDepth:
    depth: np.ndarray  # (H, W) float z-depth, quantized as sent; nan off the background
    measured: np.ndarray  # (H, W) bool: triangulated (not inpainted)
    info: dict[str, Any]
    png_bytes: int  # 8-bit inverse depth as PNG
    png16_bytes: int  # 16-bit inverse depth as PNG (for comparison)


def depth_from_companion(lens: camera.Lens, key_gray: np.ndarray, comp_gray: np.ndarray, bg_key: np.ndarray,
                         bg_comp: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> dict[str, Any] | None:
    """Per-pixel depth of the keyframe from one companion; ``pa -> pb`` are keyframe-to-companion matches."""
    pose = relative_pose(lens, pa, pb)
    H, _ = homography_u(lens, pa, pb)
    if pose is None or H is None:
        return None
    R, t, inliers = pose
    pos, ok = dense_correspondence(lens, key_gray, comp_gray, np.linalg.inv(H), bg_key, bg_comp)
    grid = camera.pixel_grid(lens.width, lens.height)
    d, angle = triangulate(lens, R, t, grid, pos)
    X = d[..., None] * rays(lens, grid)
    back, front = project(lens, X @ R.T + t)
    error = np.linalg.norm(back - pos, axis=-1)
    good = ok & front & (d > 0) & (angle >= MIN_ANGLE_DEG) & (error <= TRI_REPROJECTION_PX)
    return {"depth": np.where(good, d, np.nan), "angle": np.where(good, angle, 0.0), "pose_inliers": inliers,
            "good_share": float(good[bg_key].mean()) if bg_key.any() else 0.0}


def quantize_depth(depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int, int]:
    """8-bit inverse depth over the valid range (sent), its PNG size and the 16-bit PNG size beside it."""
    inv = 1.0 / depth[valid]
    lo, hi = float(np.percentile(inv, 0.5)), float(np.percentile(inv, 99.5))
    hi = max(hi, lo * 1.0001 + 1e-9)
    full = np.where(valid, 1.0 / np.where(valid, depth, 1.0), lo)
    q8 = np.clip(np.rint((full - lo) / (hi - lo) * DEPTH_LEVELS), 0, DEPTH_LEVELS).astype(np.uint8)
    q16 = np.clip(np.rint((full - lo) / (hi - lo) * 65535), 0, 65535).astype(np.uint16)
    q8[~valid], q16[~valid] = 0, 0
    b8 = len(cv2.imencode(".png", q8, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1]) + 8  # + range (2 float32)
    b16 = len(cv2.imencode(".png", q16, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1]) + 8
    inv_q = lo + q8.astype(np.float64) / DEPTH_LEVELS * (hi - lo)
    return np.where(valid, 1.0 / np.maximum(inv_q, 1e-12), np.nan), b8, b16


def fill_depth(depth: np.ndarray, measured: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Background pixels without a triangulated depth get a local plane: inverse depth is linear in image
    coordinates on a plane, so each hole pixel takes the least-squares plane of the measured inverse depth in
    a window around it (``FILL_WINDOWS``, the smallest with ``FILL_SUPPORT`` of its pixels measured); what no
    window reaches takes its nearest measured pixel's value."""
    h, w = depth.shape
    inv = np.where(measured, 1.0 / np.where(measured, depth, 1.0), 0.0)
    out = inv.copy()
    todo = bg & ~measured
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx, yy = xx / w, yy / w
    m = measured.astype(np.float64)
    for size in FILL_WINDOWS:
        if not todo.any():
            break

        def box(a: np.ndarray, size: int = size) -> np.ndarray:
            return cv2.boxFilter(a, cv2.CV_64F, (size, size), normalize=False, borderType=cv2.BORDER_CONSTANT)

        n = box(m)
        sx, sy = box(m * xx), box(m * yy)
        sxx, sxy, syy = box(m * xx * xx), box(m * xx * yy), box(m * yy * yy)
        sz, sxz, syz = box(m * inv), box(m * xx * inv), box(m * yy * inv)
        sel = todo & (n >= FILL_SUPPORT * size * size)
        if not sel.any():
            continue
        A = np.stack([np.stack([sxx[sel], sxy[sel], sx[sel]], -1), np.stack([sxy[sel], syy[sel], sy[sel]], -1),
                      np.stack([sx[sel], sy[sel], n[sel]], -1)], -2)
        b = np.stack([sxz[sel], syz[sel], sz[sel]], -1)
        A = A + np.eye(3) * 1e-9 * n[sel][:, None, None]
        coef = np.linalg.solve(A, b[..., None])[..., 0]
        value = coef[:, 0] * xx[sel] + coef[:, 1] * yy[sel] + coef[:, 2]
        lo, hi = float(inv[measured].min()), float(inv[measured].max())
        out[sel] = np.clip(value, lo, hi)
        todo = todo & ~sel
    if todo.any():
        _, labels = cv2.distanceTransformWithLabels((~measured).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        vy, vx = np.nonzero(measured)
        iy, ix = np.nonzero(todo)
        nearest = labels[iy, ix] - 1
        out[iy, ix] = inv[vy[nearest], vx[nearest]]
    ok = bg & (out > 0)
    return np.where(ok, 1.0 / np.where(ok, out, 1.0), np.nan)


def keyframe_depth(lens: camera.Lens, key_gray: np.ndarray, bg_key: np.ndarray,
                   companions: list[dict[str, Any]]) -> KeyDepth | None:
    """Fuse companions' depths, fill the background's holes with local planes (`fill_depth`), quantize.

    The companion with the most essential-matrix inliers is the primary; another
    is scale-matched to it and used only if it agrees on their shared pixels
    (median |log ratio| <= ``FUSE_AGREEMENT``); where both measure, the larger
    triangulation angle wins. Each companion: ``gray``, ``bg``, ``pa`` and ``pb``
    (keyframe-to-companion matches) and ``index``.
    """
    estimates = []
    for comp in companions:
        est = depth_from_companion(lens, key_gray, comp["gray"], bg_key, comp["bg"], comp["pa"], comp["pb"])
        if est is not None and np.isfinite(est["depth"]).sum() >= 1000:
            estimates.append({**est, "index": comp["index"]})
    if not estimates:
        return None
    estimates.sort(key=lambda e: -e["pose_inliers"])
    base = estimates[0]
    depth, angle = base["depth"].copy(), base["angle"].copy()
    info: dict[str, Any] = {"companions": [], "primary": base["index"]}
    for est in estimates:
        record = {"index": est["index"], "pose_inliers": est["pose_inliers"], "good_share": round(est["good_share"], 4)}
        info["companions"].append(record)
        if est is base:
            continue
        both = np.isfinite(depth) & np.isfinite(est["depth"])
        if both.sum() < 1000:
            record["used"] = False
            continue
        ratio = depth[both] / est["depth"][both]
        scale = float(np.median(ratio))
        disagreement = float(np.median(np.abs(np.log(ratio / scale))))
        record.update(scale=round(scale, 5), disagreement=round(disagreement, 4), used=disagreement <= FUSE_AGREEMENT)
        if not record["used"]:
            continue
        other = est["depth"] * scale
        take = np.isfinite(other) & (~np.isfinite(depth) | (est["angle"] > angle))
        depth[take], angle[take] = other[take], est["angle"][take]
    measured = np.isfinite(depth) & bg_key
    if measured.sum() < 1000:
        return None
    filled = fill_depth(depth, measured, bg_key)
    valid = np.isfinite(filled)
    quantized, b8, b16 = quantize_depth(filled, valid)
    info.update(measured_share=round(float(measured[bg_key].mean()), 4), angle_p50_deg=round(float(np.median(angle[measured])), 3))
    return KeyDepth(quantized, measured, info, b8, b16)


def pnp(lens: camera.Lens, depth: np.ndarray, usable: np.ndarray, pa: np.ndarray,
        pb: np.ndarray) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Pose (R, t) of the frame, x_frame ~ R X_key + t, from the keyframe's 3D points at matches ``pa -> pb``."""
    xi = np.clip(np.rint(pa[:, 0]).astype(int), 0, lens.width - 1)
    yi = np.clip(np.rint(pa[:, 1]).astype(int), 0, lens.height - 1)
    keep = usable[yi, xi] & np.isfinite(depth[yi, xi])
    if keep.sum() < PNP_MIN:
        return None
    X = depth[yi, xi][keep, None] * rays(lens, pa[keep])
    img = lens.undistort(pb[keep])
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(X.astype(np.float64), img.astype(np.float64), K(lens), np.zeros(4),
                                                 reprojectionError=PNP_PX, iterationsCount=2000, confidence=0.999,
                                                 flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        return None
    if not ok or inl is None or len(inl) < PNP_MIN:
        return None
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(X[inl].astype(np.float64), img[inl].astype(np.float64), K(lens), np.zeros(4), rvec, tvec)
    return camera.rotation(rvec), tvec.ravel(), int(len(inl))


def warp_depth(lens: camera.Lens, depth: np.ndarray, source_bg: np.ndarray, R: np.ndarray,
               t: np.ndarray) -> tuple[camera.Warp, np.ndarray]:
    """Forward-splat the keyframe's background (z-buffer on inverse depth), fill cracks, warp colour backwards.

    Returns the warp (target -> keyframe) and the map of target pixels a source pixel landed on.
    """
    h, w = depth.shape
    grid = camera.pixel_grid(w, h).reshape(-1, 2)
    src = np.nonzero((source_bg & np.isfinite(depth)).ravel())[0]
    X = depth.ravel()[src, None] * rays(lens, grid[src])
    Xt = X @ R.T + t
    dest, front = project(lens, Xt)
    keys = np.where(front, 1.0 / np.maximum(Xt[:, 2], 1e-12), -np.inf)
    sp = fill_cracks(splat(w, h, dest, keys, src))
    hit = (sp.winner >= 0) & np.isfinite(sp.key) & (sp.key > 0)
    zt = np.where(hit, 1.0 / np.where(hit, sp.key, 1.0), np.nan)
    Xq = zt.reshape(-1, 1) * rays(lens, grid)
    Xk = (Xq - t) @ R  # R^T (X - t), row-wise
    q, ok = project(lens, Xk)
    q = q.reshape(h, w, 2)
    valid = hit & ok.reshape(h, w) & lens.inside(q) & np.isfinite(q).all(-1)
    q = np.where(np.isfinite(q), q, -1.0)
    return camera.Warp(q[..., 0].astype(np.float32), q[..., 1].astype(np.float32), valid), hit

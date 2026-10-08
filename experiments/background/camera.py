"""Camera motion and background coverage of one clip (PLAN step G1).

Everything here works on *analysis frames*: frames downscaled to
``ANALYSIS_SIZE`` (960x540, half of 1080p), each with a background mask (the
complement of the dilated foreground). Pixel errors are reported at 1080p
(``TO_1080``). Only numpy, OpenCV and SciPy, so it runs on the CPU of a fleet
job and in unit tests on synthetic cameras.

Models, fitted on background features only (SIFT, ratio test, MAGSAC):

* *Homography*, frame to previous frame (``f2f``) and frame to its reference
  keyframe (``f2r``).
* *Rotation and zoom*: a camera that only turns about its centre and changes
  its focal length. A frame's rays are another frame's rays rotated; each frame
  has its own focal length ``f * exp(logf)``, the principal point is the image
  centre, and a one-term division model ``k1`` describes radial distortion.
  ``f`` and ``k1`` are fitted once per clip (`fit_lens`).

Frames register to keyframes (`register`): a frame is matched to the
keyframes its predicted view overlaps most, and becomes a keyframe itself when
its best overlap falls below ``KEYFRAME_OVERLAP`` or its match to the keyframe
keeps fewer than ``KEYFRAME_INLIERS`` inliers. Its pose is measured against
a reference, not chained frame to frame.

Residuals (`residual`): the reference is warped into the frame by a model; on
the background of both, a gain and offset fit removes exposure change, dense
optical flow (DIS) measures the misalignment left, and re-warping by that flow
leaves the photometric remainder (lighting, blur, noise, disocclusion). A
frame is *explained* when the 90th percentile of residual flow on textured
background is at most ``EXPLAINED_PX`` at 1080p. Misaligned pixels split into
parallax (the correspondence obeys the epipolar geometry of a translating
camera) and independent motion (it does not: water, screens, people, leaks
of the foreground mask).

Translation (`translation_pair`): GRIC (Torr) compares a homography with a
fundamental matrix on the same matches, and the parallax is the homography's
transfer error on the fundamental matrix's inliers.

Coverage (`coverage`): every registered frame's background is marked on a
spherical (azimuth, elevation) canvas through the rotation model, with the
frame that first saw it; the share of each later frame's background seen
within a warm-up of ``w`` seconds gives the warm-up curve.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

ANALYSIS_SIZE = (960, 540)  # width, height
TO_1080 = 1080 / ANALYSIS_SIZE[1]
DILATE = 8  # analysis px of foreground dilation (16 at 1080p)
BORDER = 6  # analysis px at the image border without features
MAX_FEATURES = 2000
RATIO = 0.8
RANSAC_PX = 3.0  # analysis px: MAGSAC's threshold for a correspondence set (the residual judges the fit)
FIT_SCALE = 1.0  # analysis px: robust scale of the least-squares fits
MIN_INLIERS = 30
MIN_INLIER_SHARE = 0.15
KEYFRAME_OVERLAP = 0.6
KEYFRAME_INLIERS = 100  # fewer direct inliers than this also start a keyframe
TEXTURE = 4.0  # gradient magnitude (grey levels per px) of a textured pixel
EXPLAINED_PX = 2.0  # 1080p px: p90 residual flow of an explained frame
FLOW_THRESHOLDS = (1.0, 2.0, 4.0)  # 1080p px, reported shares
MOVED_PX = 1.0  # 1080p px of residual flow that counts as misaligned
EPIPOLAR_PX = 1.0  # analysis px
GRIC_SIGMA = 0.5  # analysis px
TRANSLATION_SECONDS = 0.5  # frame gap of the translation test
MIN_REGION = 0.05  # share of the frame a residual needs
COVER_CELL = 2.0  # canvas cell in analysis px at the clip's median focal length
PRIOR_HFOV_DEG = 60.0  # focal-length prior when rotation cannot identify it
LENS_MIN_PAIRS = 8
LENS_MIN_DISPLACEMENT = 10.0  # analysis px, median, for a pair to inform the lens
RADIUS_BINS = 5


# ----------------------------------------------------------------- features


@dataclass
class Features:
    pts: np.ndarray  # (N, 2) float32, analysis px
    desc: np.ndarray  # (N, 128) uint8

    def __len__(self) -> int:
        return len(self.pts)


_SIFT: Any = None
_MATCHER: Any = None


def features(gray: np.ndarray, background: np.ndarray) -> Features:
    """SIFT on background pixels away from the border."""
    global _SIFT
    if _SIFT is None:
        _SIFT = getattr(cv2, "SIFT_create")(nfeatures=MAX_FEATURES)
    mask = background.astype(np.uint8) * 255
    mask[:BORDER], mask[-BORDER:], mask[:, :BORDER], mask[:, -BORDER:] = 0, 0, 0, 0
    keypoints, desc = _SIFT.detectAndCompute(gray, mask)
    if desc is None or not keypoints:
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8))
    pts = np.array([k.pt for k in keypoints], np.float32)
    return Features(pts, np.clip(np.rint(desc), 0, 255).astype(np.uint8))


def match(a: Features, b: Features) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one ratio-test matches, as point arrays in ``a`` and ``b``."""
    global _MATCHER
    empty = np.zeros((0, 2), np.float32)
    if len(a) < 2 or len(b) < 2:
        return empty, empty
    if _MATCHER is None:
        _MATCHER = cv2.BFMatcher(cv2.NORM_L2)
    pairs = _MATCHER.knnMatch(a.desc.astype(np.float32), b.desc.astype(np.float32), k=2)
    best: dict[int, tuple[float, int]] = {}
    for candidates in pairs:
        if len(candidates) < 2:
            continue
        m, n = candidates
        if m.distance < RATIO * n.distance and (m.trainIdx not in best or m.distance < best[m.trainIdx][0]):
            best[m.trainIdx] = (m.distance, m.queryIdx)
    if not best:
        return empty, empty
    ib = np.fromiter(best.keys(), int)
    ia = np.array([best[i][1] for i in ib], int)
    return a.pts[ia], b.pts[ib]


def homography(pa: np.ndarray, pb: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """MAGSAC homography mapping ``pa`` to ``pb``; None unless it has enough inliers."""
    none = np.zeros(len(pa), bool)
    if len(pa) < MIN_INLIERS:
        return None, none
    H, inliers = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, RANSAC_PX, maxIters=5000, confidence=0.999)
    if H is None or inliers is None:
        return None, none
    keep = inliers.ravel().astype(bool)
    if keep.sum() < MIN_INLIERS or keep.mean() < MIN_INLIER_SHARE:
        return None, keep
    return H, keep


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    q = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ H.T
    return q[:, :2] / np.where(np.abs(q[:, 2:]) < 1e-12, 1e-12, q[:, 2:])


# ----------------------------------------------------------------- lens and rotation


@dataclass(frozen=True)
class Lens:
    """Pinhole with the image centre as principal point and a division-model distortion."""

    f: float  # analysis px, at logf = 0
    k1: float = 0.0
    width: int = ANALYSIS_SIZE[0]
    height: int = ANALYSIS_SIZE[1]

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.width / 2, self.height / 2])

    @property
    def norm(self) -> float:
        return math.hypot(self.width / 2, self.height / 2)

    def undistort(self, pts: np.ndarray) -> np.ndarray:
        d = np.asarray(pts, float) - self.centre
        r2 = (d * d).sum(-1, keepdims=True) / self.norm**2
        return self.centre + d / (1 + self.k1 * r2)

    def distort(self, pts: np.ndarray) -> np.ndarray:
        d = np.asarray(pts, float) - self.centre
        r2 = (d * d).sum(-1, keepdims=True) / self.norm**2
        disc = np.maximum(1 - 4 * self.k1 * r2, 0.0)
        return self.centre + d * (2 / (1 + np.sqrt(disc)))

    def rays(self, pts: np.ndarray, logf: float = 0.0) -> np.ndarray:
        u = self.undistort(pts)
        f = self.f * math.exp(logf)
        v = np.concatenate([(u - self.centre) / f, np.ones(u.shape[:-1] + (1,))], -1)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def project(self, rays: np.ndarray, logf: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        z = rays[..., 2]
        ok = z > 1e-6
        f = self.f * math.exp(logf)
        u = self.centre + f * rays[..., :2] / np.where(ok, z, 1.0)[..., None]
        return self.distort(u), ok

    def inside(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return (
            (pts[..., 0] >= margin) & (pts[..., 0] <= self.width - 1 - margin)
            & (pts[..., 1] >= margin) & (pts[..., 1] <= self.height - 1 - margin)
        )


def prior_lens() -> Lens:
    return Lens(f=(ANALYSIS_SIZE[0] / 2) / math.tan(math.radians(PRIOR_HFOV_DEG) / 2))


def rotation(rvec: Sequence[float] | np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))[0]


def rvec_of(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1) / 2))))


def rotation_from_h(lens: Lens, H: np.ndarray, logf_a: float = 0.0) -> tuple[np.ndarray, float]:
    """Nearest rotation and zoom to a homography, ignoring distortion (an initial guess)."""
    f = lens.f * math.exp(logf_a)
    K = np.array([[f, 0, lens.width / 2], [0, f, lens.height / 2], [0, 0, 1]])
    M = np.linalg.inv(K) @ H @ K
    lam = np.linalg.norm(M[2])
    if not np.isfinite(lam) or lam < 1e-12:
        return np.eye(3), 0.0
    M = M / lam
    s = math.log(max(1e-6, float(np.linalg.norm(M[0]) + np.linalg.norm(M[1])) / 2))
    M = np.diag([math.exp(-s), math.exp(-s), 1.0]) @ M
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R, s


def _rotation_residuals(lens: Lens, x: np.ndarray, rays_a: np.ndarray, pb: np.ndarray, logf_a: float) -> np.ndarray:
    p, ok = lens.project(rays_a @ rotation(x[:3]).T, logf_a + x[3])
    r = p - pb
    r[~ok] = 1e3
    return r.ravel()


@dataclass
class RotationFit:
    R: np.ndarray  # rays_b = R @ rays_a
    s: float  # logf_b - logf_a
    errors: np.ndarray  # analysis px per correspondence


def fit_rotation(lens: Lens, pa: np.ndarray, pb: np.ndarray, *, logf_a: float = 0.0,
                 init: tuple[np.ndarray, float] | None = None) -> RotationFit:
    """Rotation and zoom mapping ``pa`` to ``pb`` (robust least squares)."""
    from scipy.optimize import least_squares

    rays_a = lens.rays(pa, logf_a)
    if init is None:
        H, _ = homography(pa, pb)
        init = rotation_from_h(lens, H, logf_a) if H is not None else (np.eye(3), 0.0)
    x0 = np.concatenate([rvec_of(init[0]), [init[1]]])
    result = least_squares(lambda x: _rotation_residuals(lens, x, rays_a, pb, logf_a), x0,
                           loss="soft_l1", f_scale=FIT_SCALE, max_nfev=60)
    errors = np.linalg.norm(result.fun.reshape(-1, 2), axis=1)
    return RotationFit(rotation(result.x[:3]), float(result.x[3]), errors)


def fit_lens(pairs: list[tuple[np.ndarray, np.ndarray]], start: Lens | None = None,
             max_points: int = 150) -> tuple[Lens, dict[str, Any]]:
    """Joint fit of ``f``, ``k1`` and each pair's rotation and zoom to inlier correspondences.

    With fewer than ``LENS_MIN_PAIRS`` pairs that move by ``LENS_MIN_DISPLACEMENT``
    the rotation cannot identify the focal length; the prior lens is returned and
    the report says so.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    start = start or prior_lens()
    rng = np.random.default_rng(0)
    used = []
    for pa, pb in pairs:
        if len(pa) < MIN_INLIERS or float(np.median(np.linalg.norm(pb - pa, axis=1))) < LENS_MIN_DISPLACEMENT:
            continue
        if len(pa) > max_points:
            pick = rng.choice(len(pa), max_points, replace=False)
            pa, pb = pa[pick], pb[pick]
        used.append((pa.astype(float), pb.astype(float)))
    report: dict[str, Any] = {"pairs_offered": len(pairs), "pairs_used": len(used)}
    if len(used) < LENS_MIN_PAIRS:
        report.update(observable=False, f=start.f, k1=start.k1)
        return start, report
    x0 = [math.log(start.f), start.k1]
    for pa, pb in used:
        fit = fit_rotation(start, pa, pb)
        x0.extend(list(rvec_of(fit.R)) + [fit.s])
    x0a = np.array(x0)
    sizes = [2 * len(pa) for pa, _ in used]
    rows = sum(sizes)
    sparsity = lil_matrix((rows, len(x0a)), dtype=int)
    row = 0
    for i, size in enumerate(sizes):
        sparsity[row:row + size, 0:2] = 1
        sparsity[row:row + size, 2 + 4 * i:6 + 4 * i] = 1
        row += size

    def residuals(x: np.ndarray) -> np.ndarray:
        lens = Lens(math.exp(x[0]), x[1], start.width, start.height)
        out = []
        for i, (pa, pb) in enumerate(used):
            out.append(_rotation_residuals(lens, x[2 + 4 * i:6 + 4 * i], lens.rays(pa), pb, 0.0))
        return np.concatenate(out)

    lo = np.full(len(x0a), -np.inf)
    hi = np.full(len(x0a), np.inf)
    lo[0], hi[0] = math.log(0.25 * start.width), math.log(8 * start.width)
    lo[1], hi[1] = -0.6, 0.15
    x0a[0] = float(np.clip(x0a[0], lo[0] + 1e-6, hi[0] - 1e-6))
    x0a[1] = float(np.clip(x0a[1], lo[1] + 1e-6, hi[1] - 1e-6))
    before = residuals(x0a)
    result = least_squares(residuals, x0a, jac_sparsity=sparsity, bounds=(lo, hi), loss="soft_l1",
                           f_scale=FIT_SCALE, x_scale="jac", max_nfev=200)
    lens = Lens(math.exp(result.x[0]), float(result.x[1]), start.width, start.height)
    angles = [angle_deg(rotation(result.x[2 + 4 * i:5 + 4 * i])) for i in range(len(used))]
    err_before = np.linalg.norm(before.reshape(-1, 2), axis=1)
    err_after = np.linalg.norm(result.fun.reshape(-1, 2), axis=1)
    report.update(observable=True, f=lens.f, k1=lens.k1, converged=bool(result.success),
                  hfov_deg=math.degrees(2 * math.atan(lens.width / 2 / lens.f)),
                  median_pair_angle_deg=float(np.median(angles)),
                  median_error_before_px=float(np.median(err_before) * TO_1080),
                  median_error_after_px=float(np.median(err_after) * TO_1080),
                  points=int(len(err_after)))
    return lens, report


# ----------------------------------------------------------------- translation


def sampson(F: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Sampson distance (px) of correspondences ``pa -> pb`` under ``pb^T F pa = 0``."""
    xa = np.concatenate([pa, np.ones((len(pa), 1))], 1)
    xb = np.concatenate([pb, np.ones((len(pb), 1))], 1)
    Fa = xa @ F.T
    Fb = xb @ F
    num = (xb * Fa).sum(1) ** 2
    den = Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Fb[:, 0] ** 2 + Fb[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def gric(errors: np.ndarray, d: int, k: int, sigma: float = GRIC_SIGMA) -> float:
    """Torr's geometric robust information criterion for 2D-2D data (r = 4)."""
    r, n = 4, len(errors)
    rho = np.minimum((errors / sigma) ** 2, 2.0 * (r - d))
    return float(rho.sum() + math.log(r) * d * n + math.log(r * n) * k)


def translation_pair(lens: Lens, pa: np.ndarray, pb: np.ndarray) -> dict[str, Any] | None:
    """Homography against fundamental matrix on undistorted matches ``pa -> pb``."""
    if len(pa) < MIN_INLIERS:
        return None
    ua = lens.undistort(pa).astype(np.float32)
    ub = lens.undistort(pb).astype(np.float32)
    H, h_in = cv2.findHomography(ua, ub, cv2.USAC_MAGSAC, RANSAC_PX, maxIters=5000, confidence=0.999)
    F, f_in = cv2.findFundamentalMat(ua, ub, cv2.USAC_MAGSAC, EPIPOLAR_PX, 0.999, 5000)
    if H is None or F is None or h_in is None or f_in is None:
        return None
    F = F[:3]
    f_in = f_in.ravel().astype(bool)
    e_h = np.linalg.norm(apply_h(H, ua) - ub, axis=1)
    e_f = sampson(F, ua, ub)
    g_h, g_f = gric(e_h, 2, 8), gric(e_f, 3, 7)
    on_f = e_h[f_in] if f_in.any() else e_h
    return {
        "matches": int(len(pa)),
        "h_inliers": int(h_in.sum()),
        "f_inliers": int(f_in.sum()),
        "gric_h": round(g_h, 2),
        "gric_f": round(g_f, 2),
        "prefers_f": bool(g_f < g_h),
        "parallax_p50_px": round(float(np.median(on_f)) * TO_1080, 3),
        "parallax_p90_px": round(float(np.percentile(on_f, 90)) * TO_1080, 3),
        "displacement_p50_px": round(float(np.median(np.linalg.norm(pb - pa, axis=1))) * TO_1080, 2),
        "F": F,
    }


# ----------------------------------------------------------------- registration


@dataclass
class Pose:
    R: np.ndarray  # camera rays = R @ world rays (world: the segment's first frame)
    logf: float
    status: str  # first, direct, chained, lost
    segment: int
    reference: int | None = None  # keyframe the pose was measured against
    inliers: int = 0
    rotation_p50_px: float | None = None  # rotation-fit error on the inliers, 1080p px
    pair: dict[str, Any] | None = None  # translation_pair on the registration matches
    H: np.ndarray | None = None  # undistorted homography reference -> frame, on the same matches
    matches: tuple[np.ndarray, np.ndarray] | None = None  # the registration's ratio-test matches


def overlap(lens: Lens, a: Pose, b: Pose) -> float:
    """Share of view ``a`` that view ``b`` sees, by the rotation model."""
    xs = np.linspace(0, lens.width - 1, 12)
    ys = np.linspace(0, lens.height - 1, 7)
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    world = lens.rays(grid, a.logf) @ a.R  # R^T @ ray, row-wise
    p, ok = lens.project(world @ b.R.T, b.logf)
    return float((ok & lens.inside(p)).mean())


@dataclass
class Registration:
    poses: list[Pose]
    keyframes: list[int]
    f2f: list[dict[str, Any] | None] = field(default_factory=list)


def register(lens: Lens, feats: Sequence[Features], f2f_matches: Sequence[tuple[np.ndarray, np.ndarray] | None],
             candidates: int = 2) -> Registration:
    """Poses of every frame by the rotation model, measured against keyframes.

    ``f2f_matches[t]`` are the inlier matches ``t-1 -> t`` of the frame-to-frame
    homography (None where it failed); they only predict the pose. A frame that
    no keyframe and no predecessor registers is ``lost`` and starts a new
    segment (a cut, or a view the features cannot bridge).
    """
    poses = [Pose(np.eye(3), 0.0, "first", 0)]
    keyframes = [0]
    f2f_fit: list[dict[str, Any] | None] = [None]
    for t in range(1, len(feats)):
        prev = poses[t - 1]
        pred_R, pred_logf, chained = prev.R, prev.logf, None
        pair = f2f_matches[t]
        if pair is not None:
            fit = fit_rotation(lens, pair[0], pair[1], logf_a=prev.logf)
            pred_R, pred_logf = fit.R @ prev.R, prev.logf + fit.s
            chained = fit
            f2f_fit.append({"angle_deg": angle_deg(fit.R), "s": fit.s,
                            "rotation_p50_px": float(np.median(fit.errors)) * TO_1080})
        else:
            f2f_fit.append(None)
        guess = Pose(pred_R, pred_logf, "guess", prev.segment)
        options = sorted(((overlap(lens, guess, poses[k]), k) for k in keyframes
                          if poses[k].segment == prev.segment), reverse=True)
        pose = None
        for share, k in options[:candidates]:
            if share < 0.1:
                break
            pa, pb = match(feats[k], feats[t])
            # A rotation is a homography between undistorted images, not distorted ones.
            H, keep = homography(lens.undistort(pa).astype(np.float32), lens.undistort(pb).astype(np.float32))
            if H is None:
                continue
            ref = poses[k]
            fit = fit_rotation(lens, pa[keep], pb[keep], logf_a=ref.logf,
                               init=(pred_R @ ref.R.T, pred_logf - ref.logf))
            pose = Pose(fit.R @ ref.R, ref.logf + fit.s, "direct", ref.segment, k, int(keep.sum()),
                        float(np.median(fit.errors)) * TO_1080, translation_pair(lens, pa, pb), H, (pa, pb))
            break
        if pose is None and chained is not None and prev.status != "lost":
            pose = Pose(pred_R, pred_logf, "chained", prev.segment, t - 1, len(f2f_matches[t][0]),  # type: ignore[index]
                        float(np.median(chained.errors)) * TO_1080)
        if pose is None:
            pose = Pose(np.eye(3), 0.0, "lost", prev.segment + 1)
            poses.append(pose)
            keyframes.append(t)
            continue
        poses.append(pose)
        best = max((overlap(lens, pose, poses[k]) for k in keyframes if poses[k].segment == pose.segment), default=0.0)
        if best < KEYFRAME_OVERLAP or pose.inliers < KEYFRAME_INLIERS:
            keyframes.append(t)
    return Registration(poses, keyframes, f2f_fit)


# ----------------------------------------------------------------- warps and residuals


@dataclass
class Warp:
    map_x: np.ndarray  # float32 (H, W): source x of each target pixel
    map_y: np.ndarray
    valid: np.ndarray  # bool (H, W)


_GRID: dict[tuple[int, int], np.ndarray] = {}


def pixel_grid(width: int, height: int) -> np.ndarray:
    key = (width, height)
    if key not in _GRID:
        xs, ys = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        _GRID[key] = np.stack([xs, ys], -1)
    return _GRID[key]


def warp_rotation(lens: Lens, target: Pose, source: Pose) -> Warp:
    """Source pixel of each target pixel when both views share one centre."""
    grid = pixel_grid(lens.width, lens.height)
    world = lens.rays(grid, target.logf) @ target.R
    q, ok = lens.project(world @ source.R.T, source.logf)
    valid = ok & lens.inside(q)
    return Warp(q[..., 0].astype(np.float32), q[..., 1].astype(np.float32), valid)


def warp_homography(H: np.ndarray, width: int, height: int) -> Warp:
    """Source pixel of each target pixel for a homography mapping source to target."""
    grid = pixel_grid(width, height).reshape(-1, 2)
    q = apply_h(np.linalg.inv(H), grid).reshape(height, width, 2)
    valid = (q[..., 0] >= 0) & (q[..., 0] <= width - 1) & (q[..., 1] >= 0) & (q[..., 1] <= height - 1)
    return Warp(q[..., 0].astype(np.float32), q[..., 1].astype(np.float32), valid)


_DIS: Any = None


def flow(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """DIS optical flow ``u`` with ``target(p) ~ source(p + u)`` (uint8 grey)."""
    global _DIS
    if _DIS is None:
        _DIS = getattr(cv2, "DISOpticalFlow_create")(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _DIS.calc(target, source, None)


def textured(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8
    return np.hypot(gx, gy) >= TEXTURE


def psnr(mse: float) -> float:
    return 99.0 if mse <= 1e-10 else 10 * math.log10(255.0**2 / mse)


def residual(lens: Lens, target: np.ndarray, source: np.ndarray, warp: Warp, bg_target: np.ndarray,
             bg_source: np.ndarray, *, F: np.ndarray | None = None, attribute: bool = False,
             keep: bool = False) -> dict[str, Any]:
    """Alignment and photometric residual of ``source`` warped into ``target`` (grey uint8).

    With ``attribute``, the squared error is split into exposure (removed by a
    gain and offset), parallax and independent motion (removed by the residual
    flow, split by the epipolar test against ``F``, a source-to-target
    fundamental matrix on undistorted pixels; without ``F`` every misaligned
    pixel counts as independent motion) and the remainder.
    """
    h, w = target.shape
    warped = cv2.remap(source, warp.map_x, warp.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped_bg = cv2.remap(bg_source.astype(np.uint8), warp.map_x, warp.map_y, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT) > 0
    valid = cv2.erode(warp.valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    region = valid & bg_target & warped_bg
    out: dict[str, Any] = {"region": round(float(region.mean()), 4)}
    if region.mean() < MIN_REGION:
        out["measured"] = False
        return out
    t = target.astype(np.float32)
    s = warped.astype(np.float32)
    tv, sv = t[region], s[region]
    e_raw = float(np.mean((tv - sv) ** 2))
    a, b = np.linalg.lstsq(np.stack([sv, np.ones_like(sv)], 1), tv, rcond=None)[0]
    gained = a * s + b
    e_gain = float(np.mean((tv - gained[region]) ** 2))
    u = flow(target, np.clip(gained, 0, 255).astype(np.uint8))
    mag = np.linalg.norm(u, axis=-1) * TO_1080
    tex = region & textured(target)
    m = mag[tex] if tex.any() else mag[region]
    grid = pixel_grid(w, h).astype(np.float32)
    aligned = cv2.remap(gained, grid[..., 0] + u[..., 0], grid[..., 1] + u[..., 1], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    e_flow = float(np.mean((tv - aligned[region]) ** 2))
    out.update({
        "measured": True,
        "textured": round(float(tex.sum() / max(1, region.sum())), 4),
        "psnr": round(psnr(e_raw), 3),
        "psnr_gain": round(psnr(e_gain), 3),
        "psnr_flow": round(psnr(e_flow), 3),
        "gain": round(float(a), 4),
        "offset": round(float(b), 3),
        "flow_p50_px": round(float(np.median(m)), 3),
        "flow_p90_px": round(float(np.percentile(m, 90)), 3),
        **{f"flow_within_{x:g}px": round(float((m <= x).mean()), 4) for x in FLOW_THRESHOLDS},
        "explained": bool(np.percentile(m, 90) <= EXPLAINED_PX),
    })
    if attribute:
        geo = np.maximum((t - gained) ** 2 - (t - aligned) ** 2, 0)
        moved = region & (mag > MOVED_PX)
        parallax = np.zeros_like(moved)
        if F is not None and moved.any():
            ys, xs = np.nonzero(moved)
            px = xs + u[ys, xs, 0]
            py = ys + u[ys, xs, 1]
            q = lens.undistort(np.stack([bilinear(warp.map_x, px, py), bilinear(warp.map_y, px, py)], 1))
            p = lens.undistort(np.stack([xs, ys], 1).astype(float))
            parallax[ys, xs] = sampson(F, q, p) <= EPIPOLAR_PX
        total = e_raw * region.sum()
        shares = {
            "exposure": (e_raw - e_gain) * region.sum() / total,
            "parallax": float(geo[moved & parallax].sum()) / total,
            "independent": float(geo[moved & ~parallax].sum()) / total,
        }
        shares["remainder"] = max(0.0, 1 - sum(shares.values()))
        out["energy_share"] = {k: round(max(0.0, v), 4) for k, v in shares.items()}
        out["moved_share"] = round(float(moved.sum() / region.sum()), 4)
        out["parallax_pixels_share"] = round(float((moved & parallax).sum() / region.sum()), 4)
        radius = np.hypot(grid[..., 0] - w / 2, grid[..., 1] - h / 2) / math.hypot(w / 2, h / 2)
        bins = np.minimum((radius * RADIUS_BINS).astype(int), RADIUS_BINS - 1)
        sel = tex if tex.any() else region
        out["flow_p50_by_radius_px"] = [round(float(np.median(mag[sel & (bins == i)])), 3)
                                        if (sel & (bins == i)).sum() >= 50 else None for i in range(RADIUS_BINS)]
    if keep:
        out["_images"] = {"warped": np.clip(gained, 0, 255).astype(np.uint8), "region": region, "flow_mag": mag}
    return out


def bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``image`` sampled at float positions, clamped to the border."""
    h, w = image.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def sharpness(gray: np.ndarray, background: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sel = background & (cv2.erode(background.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    return float(lap[sel].var()) if sel.sum() > 100 else float("nan")


# ----------------------------------------------------------------- coverage


def _canvas_frame(lens: Lens, poses: Sequence[Pose], frames: Sequence[int]) -> np.ndarray:
    """World-to-canvas rotation: the mean viewing direction looks along +z, mean up is -y."""
    axes = np.array([poses[t].R.T @ np.array([0.0, 0.0, 1.0]) for t in frames])
    ups = np.array([poses[t].R.T @ np.array([0.0, -1.0, 0.0]) for t in frames])
    z = axes.mean(0)
    z = z / np.linalg.norm(z) if np.linalg.norm(z) > 1e-6 else np.array([0.0, 0.0, 1.0])
    up = ups.mean(0)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, -y, z])  # canvas y grows downwards, like image rows


def _angles(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.arctan2(v[..., 0], v[..., 2]), np.arcsin(np.clip(v[..., 1], -1, 1))


def coverage(lens: Lens, poses: Sequence[Pose], times: np.ndarray, background: Callable[[int], np.ndarray],
             warmups: np.ndarray, *, colour: Callable[[int], np.ndarray] | None = None,
             maps: dict[int, float] | None = None, readout_stride: int = 2,
             mosaic_width: int = 1600) -> dict[str, Any]:
    """Warm-up coverage under the rotation model, per segment of registered frames.

    ``coverage_t(w)`` is the share of frame ``t``'s background whose direction
    some frame up to time ``w`` saw as background; ``C(w)`` is its mean over
    frames later than ``w``. ``online_t`` is the share seen by any earlier frame.
    ``maps`` asks for a per-pixel map (at the readout stride) of frame ``t`` for a
    warm-up of ``maps[t]`` seconds: 0 foreground, 1 seen, 2 not seen.
    """
    registered = [t for t, p in enumerate(poses) if p.status != "lost"]
    logfs = np.array([poses[t].logf for t in registered]) if registered else np.zeros(1)
    f_med = lens.f * math.exp(float(np.median(logfs)))
    cell = COVER_CELL / f_med
    width = int(math.ceil(2 * math.pi / cell))
    per_frame: dict[int, np.ndarray] = {}
    online: dict[int, float] = {}
    pixel_maps: dict[int, np.ndarray] = {}
    mosaics = []
    border = np.concatenate([
        np.stack([np.linspace(0, lens.width - 1, 24), np.zeros(24)], 1),
        np.stack([np.linspace(0, lens.width - 1, 24), np.full(24, lens.height - 1)], 1),
        np.stack([np.zeros(14), np.linspace(0, lens.height - 1, 14)], 1),
        np.stack([np.full(14, lens.width - 1), np.linspace(0, lens.height - 1, 14)], 1),
        [[lens.width / 2, lens.height / 2]],
    ])
    for segment in sorted({poses[t].segment for t in registered}):
        frames = [t for t in registered if poses[t].segment == segment]
        Q = _canvas_frame(lens, poses, frames)
        boxes = {}
        lat_lo, lat_hi = math.inf, -math.inf
        for t in frames:
            v = (lens.rays(border, poses[t].logf) @ poses[t].R) @ Q.T
            lon, lat = _angles(v)
            centre = lon[-1]
            lon = centre + (lon - centre + math.pi) % (2 * math.pi) - math.pi
            boxes[t] = (lon.min(), lon.max(), lat.min(), lat.max())
            lat_lo, lat_hi = min(lat_lo, lat.min()), max(lat_hi, lat.max())
        lat0 = lat_lo - 2 * cell
        rows = int(math.ceil((lat_hi - lat_lo) / cell)) + 4
        first = np.full((rows, width), -1, np.int32)
        mosaic = np.zeros((rows, width, 3), np.uint8) if colour is not None else None
        for t in frames:  # mark: inverse-map the frame's footprint cells
            lo_lon, hi_lon, lo_lat, hi_lat = boxes[t]
            c0, c1 = int(math.floor((lo_lon + math.pi) / cell)) - 1, int(math.ceil((hi_lon + math.pi) / cell)) + 1
            r0 = max(0, int(math.floor((lo_lat - lat0) / cell)) - 1)
            r1 = min(rows, int(math.ceil((hi_lat - lat0) / cell)) + 2)
            cols = np.arange(c0, c1)
            rr = np.arange(r0, r1)
            lon_c = (cols + 0.5) * cell - math.pi
            lat_c = lat0 + (rr + 0.5) * cell
            LON, LAT = np.meshgrid(lon_c, lat_c)
            v = np.stack([np.cos(LAT) * np.sin(LON), np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1)
            p, ok = lens.project((v @ Q) @ poses[t].R.T, poses[t].logf)
            ok &= lens.inside(p)
            bg = background(t)
            xi = np.clip(np.rint(p[..., 0]).astype(int), 0, lens.width - 1)
            yi = np.clip(np.rint(p[..., 1]).astype(int), 0, lens.height - 1)
            ok &= bg[yi, xi]
            cc = np.broadcast_to(cols % width, LON.shape)
            rr2 = np.broadcast_to(rr[:, None], LON.shape)
            sel_r, sel_c = rr2[ok], cc[ok]
            new = first[sel_r, sel_c] < 0
            first[sel_r[new], sel_c[new]] = t
            if mosaic is not None and colour is not None and new.any():
                rgb = colour(t)
                mosaic[sel_r[new], sel_c[new]] = rgb[yi[ok][new], xi[ok][new]]
        grid = pixel_grid(lens.width, lens.height)[::readout_stride, ::readout_stride]
        for t in frames:  # read out: each frame's background directions
            bg = background(t)[::readout_stride, ::readout_stride]
            if not bg.any():
                continue
            v = (lens.rays(grid[bg], poses[t].logf) @ poses[t].R) @ Q.T
            lon, lat = _angles(v)
            c = np.floor((lon + math.pi) / cell).astype(int) % width
            r = np.clip(np.floor((lat - lat0) / cell).astype(int), 0, rows - 1)
            seen = first[r, c]
            # A direction this frame sees but no frame marked (rounding at a cell edge) counts as its own.
            seen = np.where(seen < 0, t, seen)
            seen_times = np.sort(times[seen])
            per_frame[t] = np.searchsorted(seen_times, warmups, side="right") / len(seen_times)
            online[t] = float((seen < t).mean())
            if maps and t in maps:
                image = np.zeros(bg.shape, np.uint8)
                image[bg] = np.where(times[seen] <= maps[t], 1, 2)
                pixel_maps[t] = image
        if mosaic is not None:
            filled = first >= 0
            cols_used = np.nonzero(filled.any(0))[0]
            if len(cols_used):
                crop = mosaic[:, cols_used.min():cols_used.max() + 1]
                scale = min(1.0, mosaic_width / crop.shape[1])
                mosaics.append(cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                                          interpolation=cv2.INTER_AREA))
    curve = []
    for j, w in enumerate(warmups):
        later = [per_frame[t][j] for t in per_frame if times[t] > w]
        curve.append(float(np.mean(later)) if later else None)

    def reach(target: float) -> float | None:
        for w, c in zip(warmups, curve):
            if c is not None and c >= target:
                return float(w)
        return None

    return {
        "cell_deg": math.degrees(cell),
        "f_median": f_med,
        "segments": len({poses[t].segment for t in registered}),
        "frames_read": len(per_frame),
        "warmups_s": [float(w) for w in warmups],
        "curve": [None if c is None else round(c, 5) for c in curve],
        "frames_after": [sum(1 for t in per_frame if times[t] > w) for w in warmups],
        "warmup_90_s": reach(0.90),
        "warmup_99_s": reach(0.99),
        "online": {int(t): round(v, 5) for t, v in online.items()},
        "_per_frame": per_frame,
        "_mosaics": mosaics,
        "_maps": pixel_maps,
    }

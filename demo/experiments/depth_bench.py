"""Two-view depth of the sharp bench, reprojected with one pose per frame.

The plate and its depth are the shared file and are not billed. Each frame
pays 6 pose numbers plus an AV1 residual of the color difference. Baseline is
a standalone CRF 63 encode of the same frames.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from demo.pipeline.maps.av1_crf import encode_av1_crf

FFMPEG = os.environ.get("FFMPEG", "/opt/local/bin/ffmpeg")
ROOT = Path("/home/itec/emanuele/Datasets/Egocentric-10K/raw/extracted_new")
A = (ROOT / "factory035_worker001_00001.mp4", 750)
B = (ROOT / "factory035_worker001_00001.mp4", 1530)
C = (ROOT / "factory035_worker001_00000.mp4", 2700)
W, H = 426, 240
FPS = 30.0
SEG = "/home/itec/emanuele/Models/YOLO/yolo11x-seg.pt"


def read_one(path: Path, start: int) -> np.ndarray:
    cmd = [FFMPEG, "-v", "error", "-ss", f"{start / FPS:.3f}", "-i", str(path),
           "-vf", f"scale={W}:{H}", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3).copy()


def read_span(path: Path, start: int, n: int) -> list[np.ndarray]:
    cmd = [FFMPEG, "-v", "error", "-ss", f"{start / FPS:.3f}", "-i", str(path),
           "-vf", f"scale={W}:{H}", "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    fb = W * H * 3
    count = len(raw) // fb
    return [np.frombuffer(raw[i * fb:(i + 1) * fb], dtype=np.uint8).reshape(H, W, 3).copy() for i in range(count)]


def person_mask(model: YOLO, frame: np.ndarray) -> np.ndarray:
    pred = model.predict(frame, verbose=False, classes=[0], conf=0.2)[0]
    mask = np.zeros(frame.shape[:2], np.uint8)
    if pred.masks is None:
        return mask
    for m in pred.masks.data.cpu().numpy():
        resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mask[resized > 0.5] = 255
    return cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1) if mask.any() else mask


def matches(src: np.ndarray, dst: np.ndarray, dst_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray_s = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_d = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray_s, maxCorners=800, qualityLevel=0.01, minDistance=5, blockSize=7)
    if pts is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray_s, gray_d, pts, None, winSize=(21, 21), maxLevel=3)
    if tracked is None or status is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    valid = status.reshape(-1) == 1
    s = pts.reshape(-1, 2)[valid]
    d = tracked.reshape(-1, 2)[valid]
    keep = []
    for i, (x, y) in enumerate(d):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H and dst_mask[yi, xi] == 0:
            keep.append(i)
    if not keep:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return s[keep].astype(np.float64), d[keep].astype(np.float64)


def K_of(focal: float) -> np.ndarray:
    return np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], dtype=np.float64)


def best_pose(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid-search focal length. Returns K, R, t, inlier object points in the source camera."""
    best = None
    for focal in np.linspace(0.45 * W, 1.4 * W, 12):
        K = K_of(float(focal))
        E, inl = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
        if E is None or inl is None or int(inl.sum()) < 20:
            continue
        n_in, R, t, pose_inl = cv2.recoverPose(E, src_pts, dst_pts, K)
        if best is None or n_in > best[0]:
            best = (n_in, K, R, t, pose_inl)
    if best is None:
        raise RuntimeError("no essential matrix")
    _n, K, R, t, pose_inl = best
    print(f"pose inliers={_n} focal={K[0,0]:.1f}", flush=True)
    return K, R, t, pose_inl


def dense_depth(src: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, pose_inl: np.ndarray) -> np.ndarray:
    inl = pose_inl.reshape(-1).astype(bool)
    if inl.shape[0] != src_pts.shape[0]:
        inl = np.ones(src_pts.shape[0], dtype=bool)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    pts4 = cv2.triangulatePoints(P1, P2, src_pts[inl].T, dst_pts[inl].T)
    X = (pts4[:3] / pts4[3]).T
    z = X[:, 2]
    good = np.isfinite(z) & (z > 0.05) & (z < 50)
    depth = np.zeros(src.shape[:2], np.float32)
    xs = src_pts[inl][good, 0].astype(np.int32)
    ys = src_pts[inl][good, 1].astype(np.int32)
    depth[ys, xs] = z[good].astype(np.float32)
    known = (depth > 0).astype(np.uint8) * 255
    # Inpaint depth in the unknown pixels, then keep it only as a smooth field.
    filled = cv2.inpaint(depth, (known == 0).astype(np.uint8) * 255, 5, cv2.INPAINT_NS)
    return filled


def render(plate: np.ndarray, depth: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    ys, xs = np.where(depth > 0.05)
    z = depth[ys, xs].astype(np.float64)
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)
    cam = (R @ pts.T).T + t.reshape(1, 3)
    z2 = cam[:, 2]
    ok = z2 > 0.05
    u = np.rint(K[0, 0] * cam[ok, 0] / z2[ok] + K[0, 2]).astype(np.int32)
    v = np.rint(K[1, 1] * cam[ok, 1] / z2[ok] + K[1, 2]).astype(np.int32)
    src_y = ys[ok]
    src_x = xs[ok]
    z2 = z2[ok]
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, src_y, src_x, z2 = u[inside], v[inside], src_y[inside], src_x[inside], z2[inside]
    order = np.argsort(-z2)  # far first
    canvas = np.full_like(plate, 128)
    zbuf = np.full((H, W), np.inf)
    for i in order:
        if z2[i] < zbuf[v[i], u[i]]:
            zbuf[v[i], u[i]] = z2[i]
            canvas[v[i], u[i]] = plate[src_y[i], src_x[i]]
    # Fill tiny holes from the splat.
    holes = (canvas == 128).all(axis=2).astype(np.uint8) * 255
    if holes.mean() < 0.5:
        canvas = cv2.inpaint(canvas, holes, 3, cv2.INPAINT_TELEA)
    return canvas


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if mask is not None:
        use = mask == 0
        if not np.any(use):
            return float("nan")
        diff = diff[use]
    mse = float(diff.mean())
    if mse < 1e-8:
        return 99.0
    return float(10 * np.log10(255 ** 2 / mse))


def encode(frames: list[np.ndarray], path: Path) -> float:
    raw = path.with_suffix(".src.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()
    encode_av1_crf(raw, path, scale=None, ffmpeg=FFMPEG)
    return path.stat().st_size * 8 / (len(frames) / FPS) / 1000


def main() -> None:
    out = Path("/home/itec/emanuele/tmp/depth-bench")
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(SEG)
    plate = read_one(*A)
    other = read_one(*B)
    plate_m = person_mask(model, plate)
    other_m = person_mask(model, other)
    plate_clean = cv2.inpaint(plate, plate_m, 3, cv2.INPAINT_TELEA) if plate_m.any() else plate
    src_pts, dst_pts = matches(plate_clean, other, other_m)
    print(f"matches={len(src_pts)}", flush=True)
    K, R, t, pose_inl = best_pose(src_pts, dst_pts)
    depth = dense_depth(plate_clean, src_pts, dst_pts, K, R, t, pose_inl)
    cv2.imwrite(str(out / "plate.png"), plate_clean)
    # Depth preview
    vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(out / "depth.png"), vis)

    # How well does this stereo pair explain view B, and a few frames of each clip?
    warped_b = render(plate_clean, depth, K, R, t)
    print(f"viewB bench_psnr={psnr(other, warped_b, other_m):.2f} full={psnr(other, warped_b):.2f}", flush=True)

    for name, path, start in (("near", A[0], A[1]), ("far", B[0], B[1]), ("other", C[0], C[1])):
        frames = read_span(path, start, 60)
        residuals = []
        scores = []
        for frame in frames:
            mask = person_mask(model, frame)
            s_pts, d_pts = matches(plate_clean, frame, mask)
            if len(s_pts) < 12:
                warped = plate_clean.copy()
            else:
                ok, rvec, tvec, _ = cv2.solvePnPRansac(
                    _object_points(s_pts, depth, K),
                    d_pts.reshape(-1, 1, 2),
                    K, None, iterationsCount=100, reprojectionError=2.0,
                ) if False else (False, None, None, None)
                warped = _pnp_render(plate_clean, depth, K, s_pts, d_pts)
            scores.append(psnr(frame, warped, mask))
            delta = np.clip(frame.astype(np.int16) - warped.astype(np.int16) + 128, 0, 255).astype(np.uint8)
            residuals.append(delta)
        base = encode(frames, out / f"{name}_av1.mp4")
        resid = encode(residuals, out / f"{name}_residual.mp4")
        pose = 6 * 2 * 8 / (1 / FPS) / 1000  # 6 int16 per frame
        print(f"{name} bench_psnr={np.nanmean(scores):.2f} av1={base:.2f} residual={resid:.2f} pose={pose:.2f} total={resid+pose:.2f} ratio={(resid+pose)/base:.3f}", flush=True)


def _object_points(pts: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    out = []
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        xi = np.clip(xi, 0, W - 1)
        yi = np.clip(yi, 0, H - 1)
        z = float(depth[yi, xi])
        out.append([(xi - K[0, 2]) * z / K[0, 0], (yi - K[1, 2]) * z / K[1, 1], z])
    return np.asarray(out, dtype=np.float64)


def _pnp_render(plate, depth, K, src_pts, dst_pts) -> np.ndarray:
    obj = _object_points(src_pts, depth, K)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, dst_pts.reshape(-1, 1, 2), K, None, iterationsCount=80, reprojectionError=3.0)
    if not ok:
        return plate.copy()
    R, _ = cv2.Rodrigues(rvec)
    return render(plate, depth, K, R, tvec.reshape(3))


if __name__ == "__main__":
    main()

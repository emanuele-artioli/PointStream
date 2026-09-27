"""Shared plate from the 1080p frames, with the moving foreground removed.

Homographies that scale or shift the frame by more than a small margin are
rejected and replaced with identity. The canvas is cropped to pixels that
were actually observed. The plate is not billed. Each clip pays an AV1 CRF 63
residual at 240p, same recipe as the camera file.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from demo.pipeline.maps.av1_crf import encode_av1_crf

FFMPEG = os.environ.get("FFMPEG", "/opt/local/bin/ffmpeg")
ROOT = Path("/home/itec/emanuele/Datasets/Egocentric-10K/raw/extracted_new")
CLIPS = [
    ("cand_01", ROOT / "factory035_worker001_00001.mp4", 1530),
    ("cand_03", ROOT / "factory035_worker001_00000.mp4", 2700),
    ("third", ROOT / "factory035_worker001_00001.mp4", 750),
]
W, H = 1920, 1080
OUT_W, OUT_H = 426, 240
FPS = 30.0
# A fit that scales past this, or slides by more than this fraction of the
# frame, is the spurious zoom that blew the last canvas out with grey.
MAX_SCALE = 1.08
MIN_SCALE = 0.92
MAX_SHIFT = 0.12


def _read(path: Path, start: int, n: int, size: tuple[int, int], step: int = 1) -> list[np.ndarray]:
    w, h = size
    vf = f"scale={w}:{h}"
    if step > 1:
        vf += f",select='not(mod(n\\,{step}))'"
    cmd = [
        FFMPEG, "-v", "error",
        "-ss", f"{start / FPS:.3f}",
        "-i", str(path),
        "-vf", vf,
        "-vsync", "vfr",
        "-frames:v", str(n if step == 1 else n),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    fb = w * h * 3
    count = len(raw) // fb
    return [np.frombuffer(raw[i * fb:(i + 1) * fb], dtype=np.uint8).reshape(h, w, 3).copy() for i in range(count)]


def _sane(matrix: np.ndarray, width: int, height: int) -> bool:
    block = matrix[:2, :2]
    scale = float(np.sqrt(abs(np.linalg.det(block))))
    tx, ty = float(matrix[0, 2]), float(matrix[1, 2])
    return MIN_SCALE <= scale <= MAX_SCALE and abs(tx) < MAX_SHIFT * width and abs(ty) < MAX_SHIFT * height


def _homographies(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Map each frame into frame 0. Reject a fit that is not a small nudge."""
    identity = np.eye(3, dtype=np.float64)
    h, w = frames[0].shape[:2]
    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(gray0, maxCorners=500, qualityLevel=0.01, minDistance=8, blockSize=7)
    out = [identity.copy()]
    if points is None:
        return [identity.copy() for _ in frames]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    kept = 0
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray0, gray, points, None, winSize=(21, 21), maxLevel=3, criteria=criteria)
        if tracked is None or status is None:
            out.append(identity.copy())
            continue
        valid = status.reshape(-1) == 1
        src = tracked.reshape(-1, 2)[valid]
        dst = points.reshape(-1, 2)[valid]
        if len(src) < 8:
            out.append(identity.copy())
            continue
        mapped, _ = cv2.findHomography(src, dst, cv2.RANSAC, 1.0)
        if mapped is not None and _sane(mapped, w, h):
            out.append(np.asarray(mapped, dtype=np.float64))
            kept += 1
        else:
            out.append(identity.copy())
    print(f"  homographies kept {kept}/{len(frames) - 1}", flush=True)
    return out


def _align_to_first(frames: list[np.ndarray], homographies: list[np.ndarray]) -> list[np.ndarray]:
    h, w = frames[0].shape[:2]
    return [cv2.warpPerspective(frame, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128)) for frame, matrix in zip(frames, homographies)]


def _plate_from_aligned(frames: list[np.ndarray]) -> np.ndarray:
    """After alignment, the hands are what still moves. Remove those and inpaint."""
    stack = np.stack(frames, axis=0).astype(np.float32)
    motion = stack.std(axis=0).mean(axis=2)
    # Head motion is already taken out. The top of the remaining motion is the
    # worker. A fixed threshold marked the whole frame when alignment failed.
    thresh = max(float(np.percentile(motion, 88)), 6.0)
    mask = motion >= thresh
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1).astype(bool)
    print(f"  foreground mask {mask.mean():.3f} thresh {thresh:.1f}", flush=True)
    masked = stack.copy()
    masked[:, mask] = np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(masked, axis=0)
    hole = np.isnan(med).any(axis=2)
    med = np.nan_to_num(med, nan=128.0).astype(np.uint8)
    if hole.any():
        med = cv2.inpaint(med, hole.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    return med


def _composite(frames: list[np.ndarray], homographies: list[np.ndarray]) -> np.ndarray:
    h, w = frames[0].shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    warped_corners = []
    for matrix in homographies:
        pts = cv2.perspectiveTransform(corners.reshape(1, -1, 2), matrix).reshape(-1, 2)
        warped_corners.append(pts)
    all_pts = np.concatenate(warped_corners, axis=0)
    min_xy = np.floor(all_pts.min(axis=0)).astype(np.int32)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(np.int32)
    origin = min_xy.astype(np.float64)
    cw = int(max_xy[0] - min_xy[0])
    ch = int(max_xy[1] - min_xy[1])
    acc = np.zeros((ch, cw, 3), dtype=np.float64)
    weight = np.zeros((ch, cw), dtype=np.float64)
    shift = np.array([[1, 0, -origin[0]], [0, 1, -origin[1]], [0, 0, 1]], dtype=np.float64)
    for frame, matrix in zip(frames, homographies):
        warped = cv2.warpPerspective(frame, shift @ matrix, (cw, ch), flags=cv2.INTER_LINEAR)
        seen = cv2.warpPerspective(np.full((h, w), 255, np.uint8), shift @ matrix, (cw, ch), flags=cv2.INTER_NEAREST) > 0
        acc[seen] += warped[seen]
        weight[seen] += 1
    plate = np.full((ch, cw, 3), 128, np.uint8)
    ok = weight > 0
    plate[ok] = (acc[ok] / weight[ok, None]).astype(np.uint8)
    ys, xs = np.where(ok)
    plate = plate[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return plate


def _warp_to_frame(plate: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Place the plate into the frame. Reject a fit that is not a small nudge."""
    h, w = frame.shape[:2]
    ph, pw = plate.shape[:2]
    # Match in a shared 426x240 space, then lift the homography back.
    small_p = cv2.resize(plate, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    small_f = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    gray_p = cv2.cvtColor(small_p, cv2.COLOR_BGR2GRAY)
    gray_f = cv2.cvtColor(small_f, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray_p, maxCorners=400, qualityLevel=0.01, minDistance=8, blockSize=7)
    identity = np.eye(3, dtype=np.float64)
    mapped = None
    if pts is not None:
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray_p, gray_f, pts, None, winSize=(21, 21), maxLevel=3)
        if tracked is not None and status is not None and int((status.reshape(-1) == 1).sum()) >= 8:
            valid = status.reshape(-1) == 1
            src = pts.reshape(-1, 2)[valid]
            dst = tracked.reshape(-1, 2)[valid]
            mapped, _ = cv2.findHomography(src, dst, cv2.RANSAC, 1.0)
    if mapped is None or not _sane(mapped, OUT_W, OUT_H):
        mapped = identity
    # small_p pixels -> frame pixels. Plate may differ in size from the frame.
    sx = pw / OUT_W
    sy = ph / OUT_H
    fx = w / OUT_W
    fy = h / OUT_H
    to_small = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1]], dtype=np.float64)
    to_frame = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]], dtype=np.float64)
    lifted = to_frame @ mapped @ to_small
    return cv2.warpPerspective(plate, lifted, (w, h), flags=cv2.INTER_AREA, borderValue=(128, 128, 128))


def _psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if mask is not None:
        if not np.any(mask):
            return float("nan")
        diff = diff[mask]
    mse = float(np.mean(diff)) if diff.size else 1e9
    if mse <= 1e-8:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def _encode(frames: list[np.ndarray], path: Path) -> float:
    raw = path.with_suffix(".src.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT_W, OUT_H))
    for frame in frames:
        writer.write(frame)
    writer.release()
    encode_av1_crf(raw, path, scale=None, ffmpeg=FFMPEG)
    return path.stat().st_size * 8 / (len(frames) / FPS) / 1000.0


def main() -> None:
    out = Path("/home/itec/emanuele/tmp/clean-plate")
    out.mkdir(parents=True, exist_ok=True)
    donors = []
    for name, path, start in CLIPS:
        print(name, flush=True)
        frames = _read(path, start, 30, (W, H), step=10)
        print(f"  donor frames {len(frames)}", flush=True)
        homographies = _homographies(frames)
        aligned = _align_to_first(frames, homographies)
        donors.append(_plate_from_aligned(aligned))
    scenes = donors
    # Median of the three cropped plates on a shared top-left. Pad with grey
    # only where one plate is shorter, then crop grey away again.
    mh = max(p.shape[0] for p in scenes)
    mw = max(p.shape[1] for p in scenes)
    acc = np.zeros((mh, mw, 3), dtype=np.float64)
    weight = np.zeros((mh, mw), dtype=np.float64)
    for plate in scenes:
        h, w = plate.shape[:2]
        acc[:h, :w] += plate
        weight[:h, :w] += 1
    shared = np.full((mh, mw, 3), 128, np.uint8)
    ok = weight > 0
    shared[ok] = (acc[ok] / weight[ok, None]).astype(np.uint8)
    ys, xs = np.where(ok & ~((shared == 128).all(axis=2)))
    if len(xs):
        shared = shared[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    cv2.imwrite(str(out / "plate.png"), shared)
    grey = (shared == 128).all(axis=2).mean()
    print(f"plate {shared.shape} grey_frac={grey:.3f} png={(out / 'plate.png').stat().st_size}", flush=True)

    for name, path, start in CLIPS:
        native = _read(path, start, 300, (W, H), step=1)
        small = [cv2.resize(f, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA) for f in native]
        warped_small = []
        full_psnr = []
        # Background proxy: pixels that barely change across this clip.
        stack = np.stack(small, 0).astype(np.float32)
        still = stack.std(axis=0).mean(axis=2) < 6.0
        bg_psnr = []
        for frame, src in zip(native, small):
            warped = _warp_to_frame(shared, frame)
            small_w = cv2.resize(warped, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
            warped_small.append(small_w)
            full_psnr.append(_psnr(src, small_w))
            bg_psnr.append(_psnr(src, small_w, still))
        residual = [np.clip(s.astype(np.int16) - w.astype(np.int16) + 128, 0, 255).astype(np.uint8) for s, w in zip(small, warped_small)]
        base = _encode(small, out / f"{name}_untouched.mp4")
        resid = _encode(residual, out / f"{name}_residual.mp4")
        print(
            f"{name} warp_psnr={np.mean(full_psnr):.2f} bg_psnr={np.mean(bg_psnr):.2f} "
            f"untouched={base:.2f} residual={resid:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

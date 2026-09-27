"""Shared bench from one sharp frame, person removed only inside its mask.

The plate is not billed. Each clip sends one homography per frame, fit on
features outside the person, plus an AV1 CRF 63 residual that is neutral
gray wherever the warp already matches. Standalone AV1 of the same frames
is the baseline.
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
# Two views of the same take, plus one clip from the other recording.
CLIPS = [
    ("kf750", ROOT / "factory035_worker001_00001.mp4", 750),
    ("kf1530", ROOT / "factory035_worker001_00001.mp4", 1530),
    ("cand03", ROOT / "factory035_worker001_00000.mp4", 2700),
]
W, H = 426, 240
FPS = 30.0
SEG = "/home/itec/emanuele/Models/YOLO/yolo11x-seg.pt"
# COCO person.
PERSON = 0


def read_span(path: Path, start: int, n: int) -> list[np.ndarray]:
    cmd = [
        FFMPEG, "-v", "error", "-ss", f"{start / FPS:.3f}", "-i", str(path),
        "-vf", f"scale={W}:{H}", "-frames:v", str(n),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    fb = W * H * 3
    count = len(raw) // fb
    return [np.frombuffer(raw[i * fb:(i + 1) * fb], dtype=np.uint8).reshape(H, W, 3).copy() for i in range(count)]


def person_mask(model: YOLO, frame: np.ndarray) -> np.ndarray:
    pred = model.predict(frame, verbose=False, classes=[PERSON], conf=0.25)[0]
    mask = np.zeros(frame.shape[:2], np.uint8)
    if pred.masks is None:
        return mask
    for m in pred.masks.data.cpu().numpy():
        resized = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask[resized > 0.5] = 255
    if mask.any():
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    return mask


def fill_person(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return frame.copy()
    return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)


def fit_homography(src: np.ndarray, dst: np.ndarray, dst_mask: np.ndarray) -> np.ndarray | None:
    """Map src -> dst using corners outside the person in dst."""
    gray_s = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_d = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray_s, maxCorners=400, qualityLevel=0.01, minDistance=6, blockSize=7)
    if pts is None:
        return None
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray_s, gray_d, pts, None, winSize=(21, 21), maxLevel=3)
    if tracked is None or status is None:
        return None
    valid = status.reshape(-1) == 1
    src_pts = pts.reshape(-1, 2)[valid]
    dst_pts = tracked.reshape(-1, 2)[valid]
    keep = []
    for i, (x, y) in enumerate(dst_pts):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < dst_mask.shape[1] and 0 <= yi < dst_mask.shape[0] and dst_mask[yi, xi] == 0:
            keep.append(i)
    if len(keep) < 8:
        return None
    Hmat, _ = cv2.findHomography(src_pts[keep], dst_pts[keep], cv2.RANSAC, 2.0)
    return None if Hmat is None else np.asarray(Hmat, dtype=np.float64)


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
    out = Path("/home/itec/emanuele/tmp/shared-bench")
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(SEG)
    # Reference is the first keyframe with only the person hole filled.
    ref_frames = read_span(CLIPS[0][1], CLIPS[0][2], 1)
    ref = ref_frames[0]
    ref_mask = person_mask(model, ref)
    plate = fill_person(ref, ref_mask)
    cv2.imwrite(str(out / "plate.png"), plate)
    cv2.imwrite(str(out / "plate_mask.png"), ref_mask)
    print(f"plate person_frac={ref_mask.mean()/255:.3f}", flush=True)

    for name, path, start in CLIPS:
        frames = read_span(path, start, 150)  # 5 seconds, enough to compare
        residuals = []
        full_psnr = []
        bench_psnr = []
        pose_ok = 0
        for frame in frames:
            mask = person_mask(model, frame)
            Hmat = fit_homography(plate, frame, mask)
            if Hmat is None:
                warped = plate.copy()
            else:
                pose_ok += 1
                warped = cv2.warpPerspective(plate, Hmat, (W, H), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
            err = np.abs(frame.astype(np.int16) - warped.astype(np.int16)).mean(axis=2)
            # Keep a residual only on the person and where the bench warp missed.
            need = (mask > 0) | (err > 18)
            residual = np.full_like(frame, 128)
            residual[need] = frame[need]
            residuals.append(residual)
            full_psnr.append(psnr(frame, warped))
            bench_psnr.append(psnr(frame, warped, mask))
        base = encode(frames, out / f"{name}_av1.mp4")
        resid = encode(residuals, out / f"{name}_residual.mp4")
        pose_kbps = pose_ok * 8 * 4 * 8 / (len(frames) / FPS) / 1000  # 8 float32, upper bound
        print(
            f"{name} pose_ok={pose_ok}/{len(frames)} full_psnr={np.nanmean(full_psnr):.2f} "
            f"bench_psnr={np.nanmean(bench_psnr):.2f} av1={base:.2f} residual={resid:.2f} "
            f"pose_upper={pose_kbps:.2f} total={resid + pose_kbps:.2f} "
            f"vs_av1={(resid + pose_kbps) / base:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

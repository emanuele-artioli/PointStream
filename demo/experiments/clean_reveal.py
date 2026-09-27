"""Clean bench plate: fill hand holes with background revealed in other frames.

Frame 750 is the canvas. Other frames are warped onto it with a homography
fit only on non-person features. A pixel inside the person is replaced only
when some other frame sees the bench there. Unobserved holes stay magenta
so a smear cannot pretend to be the scale.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO, YOLOE

FFMPEG = "/opt/local/bin/ffmpeg"
SRC = Path("/home/itec/emanuele/Datasets/Egocentric-10K/raw/extracted_new/factory035_worker001_00001.mp4")
PERSON_SEG = "/home/itec/emanuele/Models/YOLO/yolo11x-seg.pt"
HAND_SEG = "/home/itec/emanuele/Models/YOLO/yoloe-26x-seg.pt"
W, H = 426, 240
FPS = 30.0
START = 750
N = 300
STEP = 3


def read_span() -> list[np.ndarray]:
    cmd = [
        FFMPEG, "-v", "error", "-ss", f"{START / FPS:.3f}", "-i", str(SRC),
        "-vf", f"scale={W}:{H},select='not(mod(n\\,{STEP}))'",
        "-vsync", "vfr", "-frames:v", str(N // STEP),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    fb = W * H * 3
    count = len(raw) // fb
    return [np.frombuffer(raw[i * fb:(i + 1) * fb], dtype=np.uint8).reshape(H, W, 3).copy() for i in range(count)]


def _paint(mask: np.ndarray, pred, conf: float) -> None:
    if pred.masks is None:
        return
    boxes = pred.boxes
    parts = pred.masks.data.cpu().numpy()
    for index, part in enumerate(parts):
        if boxes is not None and float(boxes.conf[index]) < conf:
            continue
        resized = cv2.resize(part, (W, H), interpolation=cv2.INTER_NEAREST)
        mask[resized > 0.5] = 255


def foreground_mask(person: YOLO, hands: YOLOE, frame: np.ndarray) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], np.uint8)
    _paint(mask, person.predict(frame, verbose=False, classes=[0], conf=0.2)[0], 0.2)
    _paint(mask, hands.predict(frame, verbose=False, conf=0.05)[0], 0.05)
    if mask.any():
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    return mask


def fit(src: np.ndarray, dst: np.ndarray, src_mask: np.ndarray, dst_mask: np.ndarray) -> np.ndarray | None:
    """Homography mapping src pixels into dst, using non-person features."""
    gs = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gd = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gs, maxCorners=500, qualityLevel=0.01, minDistance=6, blockSize=7, mask=(src_mask == 0).astype(np.uint8) * 255)
    if pts is None:
        return None
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(gs, gd, pts, None, winSize=(21, 21), maxLevel=3)
    if tracked is None or status is None:
        return None
    valid = status.reshape(-1) == 1
    s = pts.reshape(-1, 2)[valid]
    d = tracked.reshape(-1, 2)[valid]
    keep = []
    for i, (x, y) in enumerate(d):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H and dst_mask[yi, xi] == 0:
            keep.append(i)
    if len(keep) < 12:
        return None
    Hmat, inl = cv2.findHomography(s[keep], d[keep], cv2.RANSAC, 2.0)
    if Hmat is None or inl is None or int(inl.sum()) < 12:
        return None
    return np.asarray(Hmat, dtype=np.float64)


def main() -> None:
    out = Path("/home/itec/emanuele/tmp/clean-reveal")
    out.mkdir(parents=True, exist_ok=True)
    frames = read_span()
    print(f"frames={len(frames)}", flush=True)
    person = YOLO(PERSON_SEG)
    hands = YOLOE(HAND_SEG)
    hands.set_classes(["hand"], hands.get_text_pe(["hand"]))
    masks = [foreground_mask(person, hands, f) for f in frames]
    ref, ref_m = frames[0], masks[0]
    samples: list[np.ndarray] = []
    used = 0
    for frame, mask in zip(frames[1:], masks[1:]):
        Hmat = fit(frame, ref, mask, ref_m)
        if Hmat is None:
            continue
        warped = cv2.warpPerspective(frame, Hmat, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        warped_m = cv2.warpPerspective(mask, Hmat, (W, H), flags=cv2.INTER_NEAREST, borderValue=255)
        sample = warped.astype(np.float32)
        sample[warped_m > 0] = np.nan
        samples.append(sample)
        used += 1
    print(f"donors={used}", flush=True)
    stack = np.stack(samples, axis=0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0)
    known_count = np.sum(~np.isnan(stack[..., 0]), axis=0)
    plate = ref.copy()
    hole = ref_m > 0
    known = hole & (known_count >= 3) & ~np.isnan(med).any(axis=2)
    plate[known] = med[known].astype(np.uint8)
    missing = hole & ~known
    plate[missing] = (255, 0, 255)
    cv2.imwrite(str(out / "plate.png"), plate)
    cv2.imwrite(str(out / "ref.png"), ref)
    cover = float(known.sum() / max(hole.sum(), 1))
    print(f"hole={hole.mean():.3f} covered={cover:.3f} missing={missing.mean():.3f}", flush=True)
    # Side-by-side: camera, plate, error outside the original person (bench must stay).
    err = np.abs(ref.astype(np.int16) - plate.astype(np.int16)).mean(axis=2)
    err[hole] = 0
    err_u8 = np.clip(err * 4, 0, 255).astype(np.uint8)
    err_bgr = cv2.applyColorMap(err_u8, cv2.COLORMAP_HOT)
    trio = np.concatenate([ref, plate, err_bgr], axis=1)
    cv2.imwrite(str(out / "trio.png"), trio)
    bench = err[ref_m == 0]
    mse = float(np.mean(bench.astype(np.float32) ** 2)) if bench.size else 1e9
    psnr = 99.0 if mse < 1e-8 else 10 * np.log10((255 ** 2) / mse)
    print(f"bench_psnr_vs_ref={psnr:.2f}", flush=True)


if __name__ == "__main__":
    main()

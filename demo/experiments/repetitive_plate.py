"""One unpaid plate across scenes of a repetitive 20-minute take.

Scenes come from PointStream's HSV-histogram cutter
(``src.components.scene.hsv``, cut when adjacent-frame correlation < 0.85).
The plate is built from the first long scene and is not billed. Each later
scene pays a homography per frame plus an AV1 CRF 63 residual.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.pipeline.maps.av1_crf import encode_av1_crf
from src.components.background.plate import build_plate
from src.components.scene.hsv import CUT_CORRELATION, hsv_correlation

FFMPEG = os.environ.get("FFMPEG", "/opt/local/bin/ffmpeg")
SRC = Path(
    "/home/itec/emanuele/Datasets/Egocentric-10K/curated/clip_01_factory001_worker001_00001.mp4"
)
W, H = 426, 240
SW, SH = 160, 90
FPS = 30.0
MIN_SCENE_S = 4.0
N_TEST = 4


def _kbps(nbytes: int, n_frames: int) -> float:
    return (nbytes * 8) / (max(n_frames, 1) / FPS) / 1000.0


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-8:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def detect_scenes() -> list[tuple[int, int]]:
    """Stream a tiny proxy and cut where HSV correlation drops below 0.85."""
    cmd = [
        FFMPEG, "-v", "error", "-i", str(SRC),
        "-vf", f"scale={SW}:{SH}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    frame_bytes = SW * SH * 3
    prev = None
    correlations: list[float] = []
    n = 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(SH, SW, 3)
        if prev is not None:
            correlations.append(hsv_correlation(prev, frame))
        prev = frame.copy()
        n += 1
        if n % 3000 == 0:
            print(f"scanned {n}", flush=True)
    proc.wait()
    cuts = [0]
    for index, corr in enumerate(correlations):
        if corr < CUT_CORRELATION:
            cuts.append(index + 1)
    cuts.append(n)
    spans = []
    for start, end in zip(cuts, cuts[1:]):
        if end - start >= int(MIN_SCENE_S * FPS):
            spans.append((start, end))
    print(f"frames={n} cuts={len(cuts) - 2} long_scenes={len(spans)}", flush=True)
    return spans


def read_span(start: int, count: int) -> list[np.ndarray]:
    cmd = [
        FFMPEG, "-v", "error",
        "-ss", f"{start / FPS:.3f}",
        "-i", str(SRC),
        "-vf", f"scale={W}:{H}",
        "-frames:v", str(count),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    fb = W * H * 3
    n = len(raw) // fb
    return [np.frombuffer(raw[i * fb:(i + 1) * fb], dtype=np.uint8).reshape(H, W, 3).copy() for i in range(n)]


def warp_plate(plate: np.ndarray, frame: np.ndarray) -> np.ndarray:
    orb = cv2.ORB_create(nfeatures=800)
    g1 = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return np.full_like(frame, 128)
    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2), key=lambda m: m.distance)[:60]
    if len(matches) < 8:
        return np.full_like(frame, 128)
    pts_src = np.float32([k1[m.queryIdx].pt for m in matches])
    pts_dst = np.float32([k2[m.trainIdx].pt for m in matches])
    Hmat, _mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 3.0)
    if Hmat is None:
        return np.full_like(frame, 128)
    return cv2.warpPerspective(plate, Hmat, (W, H), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))


def encode_frames(frames: list[np.ndarray], path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_suffix(".src.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()
    encode_av1_crf(raw, path, scale=None, ffmpeg=FFMPEG)
    return _kbps(path.stat().st_size, len(frames))


def main() -> None:
    out = Path("/home/itec/emanuele/tmp/repetitive-plate")
    out.mkdir(parents=True, exist_ok=True)
    spans = detect_scenes()
    (out / "scenes.json").write_text(json.dumps([{"start": s, "end": e, "sec": round((e - s) / FPS, 2)} for s, e in spans], indent=2))
    if len(spans) < 2:
        print("not enough scenes", flush=True)
        return
    donor_s, donor_e = spans[0]
    donor_count = min(donor_e - donor_s, 90)
    donor_frames = read_span(donor_s, donor_count)
    # Every 3rd frame keeps the plate build small and still sees the motion.
    stacked = np.stack(donor_frames[::3][:30], axis=0)
    plate, _packed = build_plate(stacked, register=True)
    cv2.imwrite(str(out / "plate.png"), plate)
    print(f"donor {donor_s}:{donor_e} plate {plate.shape}", flush=True)

    rows = []
    tests = spans[1:1 + N_TEST]
    for start, end in tests:
        count = min(end - start, 300)
        frames = read_span(start, count)
        residuals = []
        psnrs = []
        for frame in frames:
            warped = warp_plate(plate, frame)
            residuals.append(np.clip(frame.astype(np.int16) - warped.astype(np.int16) + 128, 0, 255).astype(np.uint8))
            psnrs.append(_psnr(frame, warped))
        tag = f"{start}_{start + count}"
        base = encode_frames(frames, out / f"untouched_{tag}.mp4")
        resid = encode_frames(residuals, out / f"residual_{tag}.mp4")
        meta = _kbps(count * 18, count)  # 9 float16 homography coeffs
        row = {
            "start": start,
            "frames": count,
            "sec": round(count / FPS, 2),
            "warp_psnr": round(float(np.mean(psnrs)), 2),
            "untouched_kbps": round(base, 2),
            "residual_kbps": round(resid, 2),
            "residual_plus_warp_kbps": round(resid + meta, 2),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {
        "n_long_scenes": len(spans),
        "plate_shape": list(plate.shape),
        "plate_billed": False,
        "scenes": rows,
        "mean_untouched_kbps": round(float(np.mean([r["untouched_kbps"] for r in rows])), 2),
        "mean_residual_plus_warp_kbps": round(float(np.mean([r["residual_plus_warp_kbps"] for r in rows])), 2),
    }
    summary["win"] = summary["mean_residual_plus_warp_kbps"] < summary["mean_untouched_kbps"]
    (out / "report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

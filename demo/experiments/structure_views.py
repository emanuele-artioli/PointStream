"""Group distant frames that share the same view, using a fast structure code.

Each frame is a spatial grid of gradient vectors (no color). Frames far apart
in time with a very high cosine on that code are treated as the same view.
The first visit builds a plate that is not billed. Later visits pay only an
AV1 CRF 63 residual against that plate, with no homography.
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

FFMPEG = os.environ.get("FFMPEG", "/opt/local/bin/ffmpeg")
SRC = Path(
    "/home/itec/emanuele/Datasets/Egocentric-10K/curated/clip_01_factory001_worker001_00001.mp4"
)
SW, SH = 80, 48
W, H = 426, 240
FPS = 30.0
STEP = 3  # descriptor every 3 frames
MIN_GAP = 90  # 3 seconds: ignore the current shot
# Same-view bar. Set after we see adjacent-frame cosine; distant matches
# must clear the top of the adjacent distribution.
CELL = 8


def load_gray() -> np.ndarray:
    cmd = [
        FFMPEG, "-v", "error", "-i", str(SRC),
        "-vf", f"scale={SW}:{SH},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    n = len(raw) // (SW * SH)
    return np.frombuffer(raw, dtype=np.uint8).reshape(n, SH, SW)


def structure_codes(gray: np.ndarray) -> np.ndarray:
    """(N, D) L2-normalized cell gradients. One code per STEP frames."""
    sampled = gray[::STEP].astype(np.float32)
    n, h, w = sampled.shape
    gx = np.zeros_like(sampled)
    gy = np.zeros_like(sampled)
    gx[:, :, 1:-1] = sampled[:, :, 2:] - sampled[:, :, :-2]
    gy[:, 1:-1, :] = sampled[:, 2:, :] - sampled[:, :-2, :]
    ch, cw = h // CELL, w // CELL
    gx = gx[:, : ch * CELL, : cw * CELL].reshape(n, ch, CELL, cw, CELL).mean(axis=(2, 4))
    gy = gy[:, : ch * CELL, : cw * CELL].reshape(n, ch, CELL, cw, CELL).mean(axis=(2, 4))
    feat = np.concatenate([gx.reshape(n, -1), gy.reshape(n, -1)], axis=1)
    norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-6
    return feat / norm


def distant_matches(codes: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Pairs of descriptor indices at least MIN_GAP/STEP apart and above threshold."""
    gap = max(MIN_GAP // STEP, 1)
    hits: list[tuple[int, int, float]] = []
    block = 512
    n = codes.shape[0]
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        sim = codes[i0:i1] @ codes.T
        for local, row in enumerate(sim):
            i = i0 + local
            row[: i + gap] = -1
            js = np.flatnonzero(row >= threshold)
            for j in js[:8]:
                hits.append((i, int(j), float(row[j])))
        print(f"matched {i1}/{n} hits={len(hits)}", flush=True)
    return hits


def read_span(start: int, count: int) -> list[np.ndarray]:
    cmd = [
        FFMPEG, "-v", "error", "-ss", f"{start / FPS:.3f}", "-i", str(SRC),
        "-vf", f"scale={W}:{H}", "-frames:v", str(count),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    fb = W * H * 3
    n = len(raw) // fb
    return [np.frombuffer(raw[i * fb:(i + 1) * fb], dtype=np.uint8).reshape(H, W, 3).copy() for i in range(n)]


def encode_frames(frames: list[np.ndarray], path: Path) -> float:
    raw = path.with_suffix(".src.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()
    encode_av1_crf(raw, path, scale=None, ffmpeg=FFMPEG)
    return (path.stat().st_size * 8) / (len(frames) / FPS) / 1000.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse < 1e-8:
        return 99.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def main() -> None:
    out = Path("/home/itec/emanuele/tmp/structure-views")
    out.mkdir(parents=True, exist_ok=True)
    print("decoding structure proxy", flush=True)
    gray = load_gray()
    codes = structure_codes(gray)
    n = codes.shape[0]
    adj = float(np.mean(np.sum(codes[:-1] * codes[1:], axis=1)))
    lag = 103 // STEP
    cyc = float(np.mean(np.sum(codes[:-lag] * codes[lag:], axis=1)))
    # Same view: as similar as the next frame, not merely above a cycle repeat.
    threshold = adj - 0.02
    print(f"codes={n} dim={codes.shape[1]} cosine_adjacent={adj:.3f} cosine_cycle={cyc:.3f} threshold={threshold:.3f}", flush=True)
    hits = distant_matches(codes, threshold)
    # Cluster descriptor indices that matched, then split runs into visits.
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, _s in hits:
        parent[find(i)] = find(j)
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        if find(i) == i and not any(a == i or b == i for a, b, _ in hits):
            continue
        clusters.setdefault(find(i), []).append(i)
    views = []
    for members in clusters.values():
        members.sort()
        visits: list[list[int]] = []
        for idx in members:
            frame = idx * STEP
            if not visits or frame - visits[-1][-1] * STEP > MIN_GAP:
                visits.append([idx])
            else:
                visits[-1].append(idx)
        visits = [v for v in visits if len(v) >= 8]
        if len(visits) >= 2:
            views.append(visits)
    views.sort(key=lambda visits: sum(len(v) for v in visits), reverse=True)
    print(f"distant_hits={len(hits)} multi_visit_views={len(views)}", flush=True)

    rows = []
    for visits in views[:3]:
        a = visits[0]
        b = visits[1]
        start_a = a[0] * STEP
        start_b = b[0] * STEP
        count = min(len(a), len(b), 20) * STEP
        count = min(count, 150)
        frames_a = read_span(start_a, count)
        frames_b = read_span(start_b, count)
        plate = np.median(np.stack(frames_a, axis=0), axis=0).astype(np.uint8)
        residuals = []
        scores = []
        prev_mad = []
        for i, frame in enumerate(frames_b):
            delta = frame.astype(np.int16) - plate.astype(np.int16)
            residuals.append(np.clip(delta + 128, 0, 255).astype(np.uint8))
            scores.append(psnr(frame, plate))
            if i:
                prev_mad.append(float(np.mean(np.abs(frame.astype(np.int16) - frames_b[i - 1].astype(np.int16)))))
        tag = f"{start_b}"
        base = encode_frames(frames_b, out / f"untouched_{tag}.mp4")
        resid = encode_frames(residuals, out / f"residual_{tag}.mp4")
        row = {
            "visit_a_frame": start_a,
            "visit_b_frame": start_b,
            "frames": count,
            "plate_psnr": round(float(np.mean(scores)), 2),
            "prev_frame_mad": round(float(np.mean(prev_mad)), 2),
            "plate_mad": round(float(np.mean(np.abs(frames_b[0].astype(np.int16) - plate.astype(np.int16)))), 2),
            "untouched_kbps": round(base, 2),
            "residual_kbps": round(resid, 2),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        cv2.imwrite(str(out / f"plate_{tag}.png"), plate)

    report = {
        "frames": int(gray.shape[0]),
        "descriptor": f"cell-pooled gx,gy on {SW}x{SH}, cell {CELL}, every {STEP} frames",
        "cosine_adjacent": round(adj, 3),
        "cosine_one_cycle": round(cyc, 3),
        "threshold": round(threshold, 3),
        "distant_hits": len(hits),
        "views_with_two_visits": len(views),
        "tests": rows,
        "win": bool(rows) and all(r["residual_kbps"] < r["untouched_kbps"] for r in rows),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

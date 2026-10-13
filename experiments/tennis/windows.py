"""Tennis windows shared by K4 and K5 (docs/experiments.md, 2026-10-13 K4 and K5).

A window (``tools/datasets/k4_inputs.py`` writes ``windows.json``) is converted once by ffmpeg to 8-bit 4:2:0,
limited range, at its coded size; masks, poses, the codecs and the scores all read that file. TrackNet's ball labels
travel in ``windows.json``. Also here: the camera measure on frames 0.5 s apart, the ball's trajectory symbols and
the bit accounting of H1 generalized to any set of parts.
"""

from __future__ import annotations

import math
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import file_sha256
from experiments.visor.h1 import MI, bit_density, block_dims, parse_inspect, unwrap_order_hints

#: Camera measure (K4 method 4).
CAMERA: dict[str, Any] = {"step_s": 0.5, "analysis": (960, 540), "border": 0.12, "min_inliers": 40, "still_px": 2.0, "magsac_px": 3.0,
          "ratio": 0.75, "sift_features": 2000, "wide_person_height": 0.30}
#: TrackNet visibility codes that put the ball in the frame (1 easy, 2 hard; 0 absent, 3 occluded).
VISIBLE = (1, 2)
#: Raw position bound of the ball's trajectory: 11 + 10 bits at 1280x720 (K5).
RAW_POSITION_BITS = 21


def convert_command(ffmpeg: str, window: dict[str, Any], source: Path, out: Path, frames: int) -> list[str]:
    """ffmpeg to raw 8-bit 4:2:0 limited range at the window's coded size (area filter when it scales)."""
    width, height = window["size"]
    scale = f"scale={width}:{height}:flags=area:out_range=tv,format=yuv420p"
    if window["source"]["kind"] == "jpeg_dir":
        inputs = ["-framerate", str(window["fps"]), "-start_number", "0", "-i", str(source / window["source"]["pattern"])]
    else:
        inputs = ["-i", str(source)]
    return [ffmpeg, "-v", "error", "-nostdin", *inputs, "-frames:v", str(frames), "-vf", scale, "-f", "rawvideo",
            "-pix_fmt", "yuv420p", "-y", str(out)]


def convert(ffmpeg: str, window: dict[str, Any], source: Path, out: Path, frames: int) -> dict[str, Any]:
    began = time.time()
    command = convert_command(ffmpeg, window, source, out, frames)
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
    width, height = window["size"]
    got = out.stat().st_size // (width * height * 3 // 2)
    if got != frames:
        raise RuntimeError(f"{window['id']}: converted {got} of {frames} frames")
    return {"command": command, "frames": got, "sha256": file_sha256(out), "seconds": round(time.time() - began, 2)}


def yuv_to_rgb(yuv: np.ndarray, width: int, height: int) -> np.ndarray:
    """One (h*3/2, w) 4:2:0 limited-range frame -> (h, w, 3) uint8 RGB (src.codecs.quality's matrix)."""
    from src.codecs.quality import KB, KR

    y = (yuv[:height].astype(np.float32) - 16.0) / 219.0
    chroma = yuv[height:].reshape(2, height // 2, width // 2).astype(np.float32)
    u = (np.repeat(np.repeat(chroma[0], 2, 0), 2, 1) - 128.0) / 224.0
    v = (np.repeat(np.repeat(chroma[1], 2, 0), 2, 1) - 128.0) / 224.0
    r = y + 2.0 * (1.0 - KR) * v
    b = y + 2.0 * (1.0 - KB) * u
    g = (y - KR * r - KB * b) / (1.0 - KR - KB)
    return np.clip(np.round(np.stack([r, g, b], -1) * 255.0), 0, 255).astype(np.uint8)


def rgb_frames(yuv_path: Path, width: int, height: int, frames: int, out: Path) -> np.ndarray:
    size = width * height * 3 // 2
    raw = np.memmap(yuv_path, dtype=np.uint8, mode="r", shape=(frames, height * 3 // 2, width))
    array = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(frames, height, width, 3))
    for t in range(frames):
        array[t] = yuv_to_rgb(np.asarray(raw[t]), width, height)
    array.flush()
    del raw
    assert size * frames == yuv_path.stat().st_size
    return np.load(out, mmap_mode="r")


# ----------------------------------------------------------------- ball labels and trajectory


def labels(window: dict[str, Any], frames: int) -> list[tuple[int, int, int]]:
    """(visibility, x, y) per frame at the window's coded size (TrackNet windows only)."""
    rows = window["labels"][:frames]
    if len(rows) != frames:
        raise RuntimeError(f"{window['id']}: {len(rows)} labels for {frames} frames")
    return [(int(v), int(x), int(y)) for v, x, y in rows]


def trajectory_symbols(rows: list[tuple[int, int, int]]) -> dict[str, list[Any]]:
    """The ball's position at 1 px: a visibility symbol per frame; when visible, the residual of a constant-velocity
    prediction from the two previous visible frames (one previous: zero velocity), or an absolute position after a
    gap (counted at `RAW_POSITION_BITS`)."""
    visibility: list[int] = []
    residual: list[tuple[int, int]] = []
    absolute = 0
    prev: list[tuple[int, int]] = []
    for vis, x, y in rows:
        seen = vis in VISIBLE
        visibility.append(int(seen))
        if not seen:
            prev = []
            continue
        if not prev:
            absolute += 1
        elif len(prev) == 1:
            residual.append((x - prev[-1][0], y - prev[-1][1]))
        else:
            px, py = 2 * prev[-1][0] - prev[-2][0], 2 * prev[-1][1] - prev[-2][1]
            residual.append((x - px, y - py))
        prev = (prev + [(x, y)])[-2:]
    return {"visibility": visibility, "residual": residual, "absolute": [absolute]}


def entropy(symbols: list[Any]) -> float:
    """Empirical entropy (bits per symbol); a symbol read back from JSON as a list counts as its tuple."""
    if not symbols:
        return 0.0
    counts = Counter(tuple(s) if isinstance(s, list) else s for s in symbols)
    n = len(symbols)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def trajectory_rate(per_window: list[dict[str, list[Any]]], fps: float) -> dict[str, Any]:
    """Pooled empirical entropy of the trajectory symbols, as bits per frame and kbps; and the raw bound."""
    vis = [s for w in per_window for s in w["visibility"]]
    res = [s for w in per_window for s in w["residual"]]
    absolute = sum(w["absolute"][0] for w in per_window)
    frames = len(vis)
    bits = entropy(vis) * len(vis) + entropy(res) * len(res) + absolute * RAW_POSITION_BITS
    visible = sum(vis)
    raw = frames + visible * RAW_POSITION_BITS
    return {"frames": frames, "visible": visible, "residuals": len(res), "absolute": absolute,
            "visibility_bits_per_symbol": entropy(vis), "residual_bits_per_symbol": entropy(res),
            "bits_per_frame": bits / max(frames, 1), "kbps": bits / max(frames, 1) * fps / 1000.0,
            "raw_bits_per_frame": raw / max(frames, 1), "raw_kbps": raw / max(frames, 1) * fps / 1000.0}


def disk(shape: tuple[int, int], x: float, y: float, radius: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius


# ----------------------------------------------------------------- camera


def _inner(shape: tuple[int, int], border: float) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    by, bx = int(round(h * border)), int(round(w * border))
    mask[by:h - by, bx:w - bx] = 255
    return mask


def camera_pair(a: np.ndarray, b: np.ndarray, sift: Any, scale_to_1080: float) -> dict[str, Any]:
    """Still/moving/cut for two grey analysis frames: SIFT + MAGSAC homography on the inner frame; the largest
    displacement of the inner frame's corners, at 1080p."""
    import cv2

    shape = (int(a.shape[0]), int(a.shape[1]))
    mask = _inner(shape, CAMERA["border"])
    ka, da = sift.detectAndCompute(a, mask)
    kb, db = sift.detectAndCompute(b, mask)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return {"kind": "cut", "inliers": 0, "corner_px": None}
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < CAMERA["ratio"] * m[1].distance]
    if len(good) < CAMERA["min_inliers"]:
        return {"kind": "cut", "inliers": len(good), "corner_px": None}
    src = np.array([ka[m.queryIdx].pt for m in good], np.float32)
    dst = np.array([kb[m.trainIdx].pt for m in good], np.float32)
    try:
        hom, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, CAMERA["magsac_px"] / scale_to_1080)
    except cv2.error:
        hom, inl = None, None
    inliers = int(inl.sum()) if inl is not None else 0
    if hom is None or inliers < CAMERA["min_inliers"]:
        return {"kind": "cut", "inliers": inliers, "corner_px": None}
    h, w = shape
    by, bx = h * CAMERA["border"], w * CAMERA["border"]
    corners = np.array([[bx, by], [w - bx, by], [w - bx, h - by], [bx, h - by]], np.float32).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, hom).reshape(-1, 2) - corners.reshape(-1, 2)
    px = float(np.linalg.norm(moved, axis=1).max() * scale_to_1080)
    return {"kind": "still" if px <= CAMERA["still_px"] else "moving", "inliers": inliers, "corner_px": round(px, 3)}


def camera_series(gray: list[np.ndarray], scale_to_1080: float) -> list[dict[str, Any]]:
    import cv2

    sift = cv2.SIFT.create(nfeatures=CAMERA["sift_features"])
    return [camera_pair(a, b, sift, scale_to_1080) for a, b in zip(gray, gray[1:])]


def camera_summary(pairs: list[dict[str, Any]], step_s: float) -> dict[str, Any]:
    kinds = Counter(p["kind"] for p in pairs)
    n = max(len(pairs), 1)
    runs, run = [], 0
    for p in pairs:
        if p["kind"] == "still":
            run += 1
        elif run:
            runs.append(run * step_s)
            run = 0
    if run:
        runs.append(run * step_s)
    return {"pairs": len(pairs), "still": kinds["still"] / n, "moving": kinds["moving"] / n, "cut": kinds["cut"] / n,
            "still_run_s_median": float(np.median(runs)) if runs else None, "still_runs": len(runs)}


# ----------------------------------------------------------------- bits by part (H1's accounting, any parts)


def mi_counts(labels_map: np.ndarray, parts: int) -> np.ndarray:
    """Per part, its pixels in each MI x MI cell (uint8, 0..16); the map's padding beyond whole cells is dropped."""
    h, w = labels_map.shape
    rows, cols = h // MI, w // MI
    lab = labels_map[: rows * MI, : cols * MI].reshape(rows, MI, cols, MI)
    return np.stack([(lab == p).sum((1, 3)) for p in range(parts)]).astype(np.uint8)


def stream_bits(inspect_bin: str, stream: Path, counts: dict[str, np.ndarray], frames: int) -> dict[str, Any]:
    """Bits per part of one stream for each named partition ``counts[name]`` (frames, parts, rows, cols)."""
    began = time.time()
    proc = subprocess.run([inspect_bin, "-a", "-bs", str(stream)], capture_output=True, text=True, check=True)
    decoded = parse_inspect(proc.stdout)
    del proc
    order = unwrap_order_hints([int(f["orderHint"]) for f in decoded])
    dims = block_dims(decoded[0]["blockSizeMap"])
    totals = {name: np.zeros(c.shape[1]) for name, c in counts.items()}
    per_frame = {name: np.zeros((frames, c.shape[1])) for name, c in counts.items()}
    accounted = outside = 0.0
    seen: Counter[int] = Counter()
    for frame, display in zip(decoded, order):
        density, total = bit_density(frame, dims)
        accounted += total
        seen[display] += 1
        if not 0 <= display < frames:
            outside += total
            continue
        for name, c in counts.items():
            rows, cols = min(density.shape[0], c.shape[2]), min(density.shape[1], c.shape[3])
            split = (c[display, :, :rows, :cols].astype(np.float64) / (MI * MI) * density[None, :rows, :cols]).sum((1, 2))
            totals[name] += split
            per_frame[name][display] += split
    return {"frames_decoded": len(decoded), "display_frames": sorted(seen), "displayed_twice": sorted(d for d, n in seen.items() if n > 1),
            "accounted_bits": round(accounted, 1), "outside_window_bits": round(outside, 1),
            "bits": {name: [round(float(v), 1) for v in t] for name, t in totals.items()},
            "bits_by_frame": {name: np.round(p, 1).tolist() for name, p in per_frame.items()},
            "seconds": round(time.time() - began, 2)}

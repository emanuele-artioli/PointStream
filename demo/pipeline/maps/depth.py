"""Depth map CLI: YOLO26-depth native gray stream + turbo preview.

Default backend is yolo26 via ``YOLO(require("yolo26s_depth"))``. Missing
weights exit 2 and list ``MISSING``; nothing is auto-downloaded.

``--backend dinov3`` is accepted only when ``MODELS["dinov3_vits"]`` exists.
The raw ViT-S pth has no DPT head, so metric depth is not produced (skip).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from demo.evaluation.profile_map import gpu_name, profile_map
from demo.pipeline.background_codec import read_video_frames_robust
from demo.pipeline.maps.contract import MapStream, write_sidecar
from demo.pipeline.maps.encode import try_encode_gray_av1, write_u8_stack
from demo.pipeline.maps.model_paths import MISSING, MODELS, MODELS_ROOT, require

DINOV3_SKIP_REASON = (
    "MODELS['dinov3_vits'] is a ViT-S backbone checkpoint "
    "(dinov3_vits16_pretrain_*.pth) with no DPT depth head. "
    "Metric depth cannot be produced from the raw pth. "
    "Skip — do not train a DPT head."
)


def quantize_depth_u8(
    depth: np.ndarray,
    *,
    lo: float | None = None,
    hi: float | None = None,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> tuple[np.ndarray, float, float]:
    """Map metric depth to uint8 using 1–99 percentile (or supplied lo/hi)."""
    arr = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8), 0.0, 1.0
    values = arr[finite]
    if lo is None or hi is None:
        lo_v, hi_v = np.percentile(values, [p_low, p_high])
        lo = float(lo_v) if lo is None else lo
        hi = float(hi_v) if hi is None else hi
    span = max(float(hi) - float(lo), 1e-6)
    scaled = (arr - float(lo)) / span * 255.0
    scaled = np.where(finite, scaled, 0.0)
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8), float(lo), float(hi)


def colorize_turbo(gray: np.ndarray) -> np.ndarray:
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(np.ascontiguousarray(gray, dtype=np.uint8), cmap)


def _predict_depth(model, frame: np.ndarray) -> np.ndarray:
    results = model.predict(frame, verbose=False)
    result = results[0] if isinstance(results, (list, tuple)) else results
    depth = result.depth.data
    if hasattr(depth, "detach"):
        depth = depth.detach()
    if hasattr(depth, "cpu"):
        depth = depth.cpu()
    if hasattr(depth, "numpy"):
        depth = depth.numpy()
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze(0) if arr.shape[0] == 1 else arr[0]
    if arr.ndim != 2:
        raise RuntimeError(f"unexpected depth shape {arr.shape}")
    if arr.shape[:2] != frame.shape[:2]:
        arr = cv2.resize(arr, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return arr


def _clip_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 1e-3 else 30.0


def _write_preview(u8_frames: list[np.ndarray], out_dir: Path, fps: float) -> tuple[Path, int]:
    vis = [colorize_turbo(g) for g in u8_frames]
    png = out_dir / "preview_depth.png"
    cv2.imwrite(str(png), vis[0])
    for i, frame in enumerate(vis[1:8], start=1):
        cv2.imwrite(str(out_dir / f"preview_depth.{i:02d}.png"), frame)
    mp4 = out_dir / "preview_depth.mp4"
    h, w = vis[0].shape[:2]
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    preview_path = png
    if writer.isOpened():
        for frame in vis:
            writer.write(frame)
        writer.release()
        if mp4.is_file() and mp4.stat().st_size > 0:
            preview_path = mp4
    else:
        writer.release()
    return preview_path, preview_path.stat().st_size if preview_path.is_file() else 0


def _missing_message(key: str) -> str:
    return (
        f"MODELS[{key!r}] is missing under {MODELS_ROOT}. "
        f"Known missing keys: {list(MISSING)}. Do not auto-download."
    )


def run_yolo26_depth(clip: Path, out_dir: Path, max_frames: int | None) -> MapStream:
    weights = require("yolo26s_depth")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed") from exc

    model = YOLO(str(weights))
    frames = read_video_frames_robust(clip, max_frames=max_frames)
    if not frames:
        raise RuntimeError(f"no frames decoded from {clip}")
    fps = _clip_fps(clip)
    depths = [_predict_depth(model, frame) for frame in frames]
    stacked = np.stack(depths, axis=0)
    _, lo, hi = quantize_depth_u8(stacked)
    u8_frames = [quantize_depth_u8(d, lo=lo, hi=hi)[0] for d in depths]

    av1_path = out_dir / "payload_depth.av1.mp4"
    payload = try_encode_gray_av1(u8_frames, av1_path, fps=fps)
    extra: dict = {
        "quantize": "percentile_1_99",
        "depth_p1": lo,
        "depth_p99": hi,
        "units": "meters_then_u8",
    }
    if payload is None:
        npy_path = out_dir / "depth_u8.npy"
        bin_path = out_dir / "depth_u8.bin"
        payload = write_u8_stack(u8_frames, npy_path, bin_path)
        extra["payload_kind"] = "u8_npy"
        extra["bin_path"] = str(bin_path)
        extra["shape"] = list(np.stack(u8_frames, axis=0).shape)
        extra["fallback"] = "ffmpeg_or_libsvtav1_missing"
    else:
        extra["payload_kind"] = "gray_av1"

    preview_path, preview_bytes = _write_preview(u8_frames, out_dir, fps)

    sample = frames[0]

    def extract_fn(frame: np.ndarray) -> np.ndarray:
        return _predict_depth(model, frame)

    def pack_fn(depth: np.ndarray) -> bytes:
        return quantize_depth_u8(depth, lo=lo, hi=hi)[0].tobytes()

    def decode_fn(blob: bytes) -> np.ndarray:
        h, w = sample.shape[:2]
        return np.frombuffer(blob, dtype=np.uint8).reshape(h, w)

    stats = profile_map(extract_fn, sample, pack_fn=pack_fn, decode_fn=decode_fn, n_warmup=1, n_runs=3)
    duration_s = len(frames) / fps
    stream = MapStream(
        map="depth",
        backend=weights.name,
        payload_path=str(payload),
        payload_bytes=payload.stat().st_size,
        preview_path=str(preview_path),
        preview_bytes=preview_bytes,
        duration_s=duration_s,
        n_frames=len(frames),
        fps=fps,
        extract_ms_p50=float(stats["extract_ms_p50"]),
        extract_ms_p95=float(stats["extract_ms_p95"]),
        pack_ms_p50=float(stats["pack_ms_p50"]),
        codec_ms_p50=float(stats["codec_ms_p50"]),
        decode_ms_p50=float(stats["decode_ms_p50"]),
        gpu=str(stats.get("gpu") or gpu_name()),
        kind="native",
        extra=extra,
    )
    write_sidecar(stream, out_dir / "depth.json")
    return stream


def run_dinov3_skip(out_dir: Path) -> int:
    path = MODELS.get("dinov3_vits")
    if path is None:
        print(_missing_message("dinov3_vits"), file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "map": "depth",
        "backend": path.name,
        "skipped": True,
        "reason": DINOV3_SKIP_REASON,
    }
    (out_dir / "skipped.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(DINOV3_SKIP_REASON)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract a native depth map stream.")
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--backend", choices=("yolo26", "dinov3"), default="yolo26")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.backend == "dinov3":
        return run_dinov3_skip(args.out)

    try:
        require("yolo26s_depth")
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    if not args.clip.is_file():
        print(f"clip not found: {args.clip}", file=sys.stderr)
        return 2

    try:
        run_yolo26_depth(args.clip, args.out, args.max_frames)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

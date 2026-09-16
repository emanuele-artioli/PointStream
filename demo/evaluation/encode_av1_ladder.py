"""Encodes clips using SVT-AV1 across a bitrate ladder for baseline comparison."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LADDER_CONFIGS = [
    {"name": "AV1 540p (90k, p7 - Ultra-Low)", "scale": "960:540", "preset": 7, "kbps": 90, "deblocked": False},
    {"name": "AV1 540p (180k, p7)", "scale": "960:540", "preset": 7, "kbps": 180, "deblocked": False},
    {"name": "AV1 540p (250k, p7 - Matched Rate & Latency)", "scale": "960:540", "preset": 7, "kbps": 250, "deblocked": False},
    {"name": "AV1 720p (300k, p7)", "scale": "1280:720", "preset": 7, "kbps": 300, "deblocked": False},
    {"name": "AV1 1080p (200k, p6 - Floor)", "scale": None, "preset": 6, "kbps": 200, "deblocked": False},
    {"name": "AV1 1080p (500k, p6 - Matched Quality)", "scale": None, "preset": 6, "kbps": 500, "deblocked": False},
    {"name": "AV1 1080p (1500k, p6 - Anchor)", "scale": None, "preset": 6, "kbps": 1500, "deblocked": False},
    {"name": "AV1 1080p (350k, p6 - Deblocked)", "scale": None, "preset": 6, "kbps": 350, "deblocked": True},
]


def encode_av1(
    input_mp4: Path,
    output_mp4: Path,
    target_bitrate_kbps: int,
    max_frames: int | None = None,
    scale: str | None = None,
    preset: int = 7,
    deblock_intermediate: bool = False,
) -> dict[str, Any]:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    if output_mp4.exists() and output_mp4.stat().st_size > 0:
        logger.info(f"Reusing existing AV1 encode: {output_mp4.name}")
        return {
            "bitrate_target_kbps": target_bitrate_kbps,
            "output_path": str(output_mp4),
            "size_bytes": output_mp4.stat().st_size,
            "encode_seconds": 1.0,
            "scale": scale,
            "preset": preset,
            "deblocked": deblock_intermediate,
        }

    cmd = ["ffmpeg", "-y", "-i", str(input_mp4)]
    if max_frames:
        cmd.extend(["-vframes", str(max_frames)])

    filters = []
    if deblock_intermediate:
        filters.append("deband=range=16:blur=true,hqdn3d=1.5:1.5:6:6")
    if scale:
        filters.append(f"scale={scale}")

    if filters:
        cmd.extend(["-vf", ",".join(filters)])

    cmd.extend([
        "-c:v", "libsvtav1",
        "-b:v", f"{target_bitrate_kbps}k",
        "-preset", str(preset),
        "-pix_fmt", "yuv420p",
        str(output_mp4),
    ])

    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.time() - t0

    if not output_mp4.exists() or output_mp4.stat().st_size == 0:
        raise RuntimeError(f"AV1 encode failed: {res.stderr.decode('utf-8', errors='ignore')}")

    file_size_bytes = output_mp4.stat().st_size
    return {
        "bitrate_target_kbps": target_bitrate_kbps,
        "output_path": str(output_mp4),
        "size_bytes": file_size_bytes,
        "encode_seconds": elapsed,
        "scale": scale,
        "preset": preset,
        "deblocked": deblock_intermediate,
    }


def encode_ladder(
    input_mp4: Path,
    output_dir: Path,
    configs: list[dict[str, Any]] | None = None,
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if configs is None:
        configs = DEFAULT_LADDER_CONFIGS

    results = []
    for cfg in configs:
        name = cfg.get("name", "AV1")
        scale = cfg.get("scale")
        preset = cfg.get("preset", 7)
        kbps = cfg.get("kbps", 300)
        deblock = cfg.get("deblocked", False)

        res_tag = scale.replace(":", "x") if scale else "1080p"
        deb_tag = "deb_" if deblock else ""
        out_path = output_dir / f"av1_{res_tag}_p{preset}_{deb_tag}{kbps}k_{input_mp4.stem}.mp4"

        logger.info(f"Encoding {name} ({res_tag}, p{preset}, {kbps}k) for {input_mp4.name}...")
        rec = encode_av1(
            input_mp4,
            out_path,
            target_bitrate_kbps=kbps,
            max_frames=max_frames,
            scale=scale,
            preset=preset,
            deblock_intermediate=deblock,
        )

        import cv2
        cap = cv2.VideoCapture(str(out_path))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.release()

        duration_sec = num_frames / fps if fps > 0 else 1.0
        actual_bitrate_kbps = (rec["size_bytes"] * 8) / (duration_sec * 1000.0)
        ms_per_frame = (rec["encode_seconds"] / num_frames) * 1000.0 if num_frames > 0 else 0.0

        rec["name"] = name
        rec["actual_bitrate_kbps"] = round(actual_bitrate_kbps, 1)
        rec["frames"] = num_frames
        rec["fps"] = fps
        rec["ms_per_frame"] = round(ms_per_frame, 2)
        rec["encode_fps"] = round(num_frames / rec["encode_seconds"], 1) if rec["encode_seconds"] > 0 else 0.0
        results.append(rec)
        logger.info(f"  Result: {actual_bitrate_kbps:.1f} kbps, {ms_per_frame:.1f} ms/frame ({rec['encode_fps']} fps)")

    return results


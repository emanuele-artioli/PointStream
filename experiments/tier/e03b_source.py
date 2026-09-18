"""E03B display_low source recipe with native PTS, never writing BP46/BP21 caches.

The 24 fps helper only checks file count and may overwrite historical extracts.
This module always materializes a four-second window in a new directory via
``extract_24fps_pngs`` and maps selected 12 fps positions to native packet PTS.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np

from experiments.headroom.real import extract_24fps_pngs, load_rgb_stack
from experiments.tier.e03b_persist import sha256_path
from experiments.tier.resolution_adaptive import rescale_frames
from src.components.codec.tools import resolve_ffmpeg

BP46_EXTRACT = Path(
    "/home/itec/emanuele/pointstream-data/outputs/bp46-long-scenes/clips/"
    "federer_djokovic/scene_007/extract_24"
)
BP21_EXTRACT = Path(
    "/home/itec/emanuele/pointstream-data/outputs/bp21-headroom/clips/"
    "federer_djokovic/scene_007/extract_24"
)
FORBIDDEN_CACHE_ROOTS = (BP46_EXTRACT, BP21_EXTRACT)

T_START_S = 80.7807
DURATION_S = 4.0
WORKING_FPS = 24.0
TARGET_FPS = 12.0
SHORT_EDGE = 360
SCALE = 1.0 / 6.0
SELECTED_COUNT = 48
EXPECTED_SHAPE = (48, 360, 640, 3)


def extraction_argv(video_path: Path, t_start: float, duration: float, out_dir: Path, ffmpeg: str) -> list[str]:
    """Documented ``extract_24fps_pngs`` argv: ``-ss`` before ``-i``, then ``-r 24``."""
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{t_start:.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.6f}",
        "-r",
        "24",
        str(out_dir / "frame_%06d.png"),
    ]


def _assert_new_extract_dir(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    for forbidden in FORBIDDEN_CACHE_ROOTS:
        if resolved == forbidden.resolve() or forbidden.resolve() in resolved.parents:
            raise ValueError(f"refusing to write historical extraction cache {forbidden}")


def probe_native_frames(
    video_path: Path,
    *,
    t_start: float,
    t_end: float,
    ffprobe: str,
) -> list[dict[str, Any]]:
    """Native video frames in ``[t_start, t_end]`` with packet PTS."""
    argv = [
        ffprobe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-select_streams",
        "v:0",
        "-read_intervals",
        f"{max(0.0, t_start - 0.25)}%{t_end + 0.25}",
        "-show_frames",
        "-show_entries",
        "frame=pkt_pts_time,pkt_dts_time,best_effort_timestamp_time,pict_type,key_frame",
        "-of",
        "json",
        str(video_path),
    ]
    payload = json.loads(subprocess.check_output(argv, text=True))
    frames: list[dict[str, Any]] = []
    for item in payload.get("frames") or []:
        pts_raw = None
        for key in ("pkt_pts_time", "best_effort_timestamp_time", "pkt_dts_time"):
            value = item.get(key)
            if value not in {None, "", "N/A"}:
                pts_raw = value
                break
        if pts_raw is None:
            continue
        try:
            pts = float(pts_raw)
        except (TypeError, ValueError):
            continue
        if pts > 1e6:
            # Integer tick leftover; this stream stores seconds in *_time fields.
            continue
        frames.append(
            {
                "native_pts_s": pts,
                "pkt_dts_time_s": float(item["pkt_dts_time"]) if item.get("pkt_dts_time") not in {None, "N/A"} else None,
                "pict_type": item.get("pict_type"),
                "key_frame": bool(int(item.get("key_frame") or 0)),
            }
        )
    frames.sort(key=lambda row: row["native_pts_s"])
    return frames


def nearest_native(target_s: float, natives: list[dict[str, Any]]) -> dict[str, Any]:
    if not natives:
        raise ValueError("no native frames to map")
    best = min(natives, key=lambda row: abs(float(row["native_pts_s"]) - target_s))
    return {
        "target_time_s": target_s,
        "native_pts_s": best["native_pts_s"],
        "native_offset_s": float(best["native_pts_s"]) - target_s,
        "pict_type": best.get("pict_type"),
        "key_frame": best.get("key_frame"),
    }


def stack_sha256(frames: np.ndarray) -> str:
    contig = np.ascontiguousarray(frames)
    return hashlib.sha256(contig.tobytes()).hexdigest()


@dataclass(frozen=True)
class SourceRecipe:
    payload: dict[str, Any]
    frames: np.ndarray


def materialize_display_low(
    *,
    video_path: Path,
    run_dir: Path,
    ffmpeg: str | None = None,
    t_start: float = T_START_S,
    duration: float = DURATION_S,
) -> SourceRecipe:
    """Extract a fresh 24 fps window, decimate to 12 fps, rescale to 360-short-edge."""
    run_dir = Path(run_dir)
    if (run_dir / "source_recipe.json").is_file() or (run_dir / "prepared_rgb.npy").is_file():
        raise FileExistsError(
            f"refusing to overwrite prepared source in {run_dir}; verify in a new directory"
        )
    extract_dir = run_dir / "extract_24"
    _assert_new_extract_dir(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = ffmpeg or resolve_ffmpeg().path
    argv = extraction_argv(video_path, t_start, duration, extract_dir, ffmpeg_path)
    existing = sorted(extract_dir.glob("frame_*.png"))
    if len(existing) >= 96:
        pngs = existing
    else:
        pngs = extract_24fps_pngs(video_path, t_start, duration, extract_dir, ffmpeg=ffmpeg_path)
    if len(pngs) < 96:
        raise RuntimeError(f"expected at least 96 extracted 24 fps frames, got {len(pngs)}")
    selected_positions = list(range(0, 96, 2))
    selected_pngs = [pngs[index] for index in selected_positions]
    raw = load_rgb_stack(selected_pngs)
    prepared, rescale_s = rescale_frames(raw, SCALE, interpolation=cv2.INTER_LANCZOS4)
    if tuple(prepared.shape) != EXPECTED_SHAPE:
        raise RuntimeError(f"prepared shape {prepared.shape} != {EXPECTED_SHAPE}")

    ffprobe = str(Path(ffmpeg_path).with_name("ffprobe"))
    natives = probe_native_frames(
        video_path, t_start=t_start, t_end=t_start + duration, ffprobe=ffprobe
    )
    if not natives:
        raise RuntimeError("ffprobe returned no native timestamps in the scene window")
    mapping: list[dict[str, Any]] = []
    native_pts: list[float] = []
    for position, png in zip(selected_positions, selected_pngs):
        target = t_start + position / WORKING_FPS
        row = nearest_native(target, natives)
        if abs(float(row["native_offset_s"])) > 0.05:
            raise RuntimeError(
                f"native PTS {row['native_pts_s']} is {row['native_offset_s']}s from target {target}"
            )
        row.update(
            {
                "extracted_frame_index": position,
                "extracted_path": str(png),
                "extracted_sha256": sha256_path(png),
            }
        )
        mapping.append(row)
        native_pts.append(float(row["native_pts_s"]))

    colour = json.loads(
        subprocess.check_output(
            [
                ffprobe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,pix_fmt,color_space,color_primaries,color_transfer,color_range,r_frame_rate,avg_frame_rate",
                "-of",
                "json",
                str(video_path),
            ],
            text=True,
        )
    )
    stream = (colour.get("streams") or [{}])[0]
    prepared_path = run_dir / "prepared_rgb.npy"
    np.save(prepared_path, prepared)
    payload = {
        "schema": "pointstream.e03b_source_recipe.v1",
        "doc_role": "e03b_native_pts_and_transform_identity",
        "campaign": "evaluation-20260914",
        "video": "federer_djokovic",
        "scene": "scene_007",
        "operating_point_id": "display_low",
        "raw_input": {
            "path": str(video_path),
            "sha256": sha256_path(video_path),
            "bytes": video_path.stat().st_size,
            "t_start_s": t_start,
            "duration_s": duration,
        },
        "extraction": {
            "callable": "experiments.headroom.real.extract_24fps_pngs",
            "not_called": "experiments.long_scenes.extract.extract_or_reuse_24fps_frames",
            "argv": argv,
            "out_dir": str(extract_dir),
            "historical_caches_untouched": [str(path) for path in FORBIDDEN_CACHE_ROOTS],
            "extracted_count": len(pngs),
            "selected_positions": selected_positions,
            "interpolation": False,
            "working_fps": WORKING_FPS,
            "target_fps": TARGET_FPS,
        },
        "colour": stream,
        "transform": {
            "spatial_callable": "experiments.tier.resolution_adaptive.rescale_frames",
            "scale": SCALE,
            "interpolation": "cv2.INTER_LANCZOS4",
            "expected_shape": list(EXPECTED_SHAPE),
            "prepared_shape": list(prepared.shape),
            "rescale_seconds": round(float(rescale_s), 4),
        },
        "frame_mapping": mapping,
        "frame_ids": {
            "count": SELECTED_COUNT,
            "fps": TARGET_FPS,
            "native_timestamps": native_pts,
        },
        "prepared_npy": str(prepared_path),
        "prepared_sha256": stack_sha256(prepared),
        "raw_selected_sha256": stack_sha256(raw),
    }
    (run_dir / "source_recipe.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return SourceRecipe(payload=payload, frames=prepared)

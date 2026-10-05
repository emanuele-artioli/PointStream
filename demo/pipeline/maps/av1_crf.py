"""Shared SVT-AV1 recipe for the segmentation demo.

Camera baselines, mask videos, and background arms use this command so the
billed kbps is comparable. Rate control is CRF, not a bitrate cap.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Ladder the site already labels. Native 1080 keeps the source size.
AV1_LADDER: tuple[tuple[str, str | None], ...] = (
    ("180p", "320:180"),
    ("240p", "426:240"),
    ("360p", "640:360"),
    ("540p", "960:540"),
    ("720p", "1280:720"),
    ("1080p", None),
)

AV1_SCALE = "426:240"  # default rung when callers omit scale
AV1_PRESET = "7"
AV1_CRF = "63"
AV1_PIX_FMT = "yuv420p"

# BGR per segmentation class; the hand stays red for older matte readers.
CLASS_COLORS_BGR = {
    "workbench": (255, 140, 40),
    "tool": (40, 210, 70),
    "arm": (40, 180, 255),
    "hand": (40, 40, 255),
}


def av1_output_args(scale: str | None = AV1_SCALE) -> list[str]:
    """FFmpeg output args for one CRF 63 / preset 7 rung. ``scale=None`` = native."""
    args = ["-an"]
    if scale:
        args.extend(["-vf", f"scale={scale}"])
    args.extend(
        [
            "-c:v",
            "libsvtav1",
            "-preset",
            AV1_PRESET,
            "-crf",
            AV1_CRF,
            "-pix_fmt",
            AV1_PIX_FMT,
        ]
    )
    return args


def encode_av1_crf(
    src: Path,
    dest: Path,
    *,
    scale: str | None = AV1_SCALE,
    ffmpeg: str = "ffmpeg",
    max_frames: int | None = None,
) -> Path:
    """Encode an existing video with the shared CRF recipe."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", str(src)]
    if max_frames is not None:
        cmd.extend(["-frames:v", str(max_frames)])
    cmd.extend([*av1_output_args(scale), str(dest)])
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        tail = res.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"AV1 encode failed for {src}: {tail}")
    return dest


def encode_av1_crf_ladder(
    src: Path,
    dest_dir: Path,
    *,
    stem: str,
    ffmpeg: str = "ffmpeg",
    max_frames: int | None = None,
) -> dict[str, Path]:
    """Encode every rung. Returns ``{rung_name: path}``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, scale in AV1_LADDER:
        dest = dest_dir / f"{stem}_{name}_crf{AV1_CRF}.mp4"
        encode_av1_crf(src, dest, scale=scale, ffmpeg=ffmpeg, max_frames=max_frames)
        out[name] = dest
    return out


def pipe_bgr_av1(
    width: int,
    height: int,
    fps: float,
    dest: Path,
    *,
    scale: str | None = AV1_SCALE,
    ffmpeg: str = "ffmpeg",
) -> subprocess.Popen[bytes]:
    """Raw bgr24 frames on stdin, shared AV1 recipe on stdout."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        *av1_output_args(scale),
        str(dest),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

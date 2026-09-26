"""Shared SVT-AV1 recipe for the segmentation demo.

Camera baselines and mask videos use this command so the billed kbps is
comparable. Rate control is CRF, not a bitrate cap.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

AV1_SCALE = "426:240"
AV1_PRESET = "7"
AV1_CRF = "63"
AV1_PIX_FMT = "yuv420p"

# BGR, painted workbench then tool then hand so the hand stays on top.
CLASS_COLORS_BGR = {
    "workbench": (255, 140, 40),
    "tool": (40, 210, 70),
    "hand": (40, 40, 255),
}
PAINT_ORDER = ("workbench", "tool", "hand")


def av1_output_args() -> list[str]:
    return [
        "-an",
        "-vf",
        f"scale={AV1_SCALE}",
        "-c:v",
        "libsvtav1",
        "-preset",
        AV1_PRESET,
        "-crf",
        AV1_CRF,
        "-pix_fmt",
        AV1_PIX_FMT,
    ]


def encode_av1_crf(src: Path, dest: Path, *, ffmpeg: str = "ffmpeg") -> Path:
    """Encode an existing video with the shared recipe."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", str(src), *av1_output_args(), str(dest)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        tail = res.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"AV1 encode failed for {src}: {tail}")
    return dest


def pipe_bgr_av1(
    width: int,
    height: int,
    fps: float,
    dest: Path,
    *,
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
        *av1_output_args(),
        str(dest),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

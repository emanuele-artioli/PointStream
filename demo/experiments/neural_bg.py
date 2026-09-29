"""Feather a foreground mask and cut a workstation hold-out.

The feather is a dilated Gaussian on the mask edge so an inpainter blends
lighting instead of painting a hard seam. The hold-out is a middle window
with a gap on both sides so neighboring frames stay out of training.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def feather_mask(mask: np.ndarray, *, dilate_px: int = 8, blur_px: int = 21) -> np.ndarray:
    """Return a soft mask in [0, 1]. Ones are the hole the inpainter may fill."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim == 3:
        binary = binary.max(axis=2)
    if dilate_px > 0 and binary.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        binary = cv2.dilate(binary, kernel)
    soft = binary.astype(np.float32)
    if blur_px > 1:
        radius = blur_px if blur_px % 2 == 1 else blur_px + 1
        soft = cv2.GaussianBlur(soft, (radius, radius), 0)
    return np.clip(soft, 0.0, 1.0)


def cut_segment(src: Path, dest: Path, *, start_s: float, duration_s: float, ffmpeg: str = "ffmpeg") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start_s:.3f}",
        "-i", str(src),
        "-t", f"{duration_s:.3f}",
        "-c", "copy",
        str(dest),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        tail = res.stderr.decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"cut failed for {dest}: {tail}")


def holdout_windows(duration_s: float, *, holdout_s: float = 10.0, gap_s: float = 2.0) -> dict[str, tuple[float, float]]:
    """Middle hold-out plus the train span on either side, excluding the gaps."""
    if duration_s <= holdout_s + 2 * gap_s + 1.0:
        raise ValueError(f"recording is {duration_s:.1f}s, too short for a {holdout_s:.0f}s hold-out with gaps")
    start = (duration_s - holdout_s) / 2.0
    return {
        "holdout": (start, holdout_s),
        "train_left": (0.0, max(0.0, start - gap_s)),
        "train_right": (start + holdout_s + gap_s, duration_s - (start + holdout_s + gap_s)),
    }

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_dummy_video(path: Path, num_frames: int, width: int, height: int, fps: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        getattr(cv2, "VideoWriter_fourcc")(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create test video at: {path}")

    for frame_idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x0 = (frame_idx * 5) % max(1, width - 40)
        y0 = (frame_idx * 3) % max(1, height - 60)
        cv2.rectangle(frame, (x0, y0), (x0 + 30, y0 + 50), (20, 190, 80), thickness=-1)
        writer.write(frame)

    writer.release()
    return path

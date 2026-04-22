from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.encoder.video_io import encode_video_frames_ffmpeg


def create_dummy_video(path: Path, num_frames: int, width: int, height: int, fps: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    for frame_idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x0 = (frame_idx * 5) % max(1, width - 40)
        y0 = (frame_idx * 3) % max(1, height - 60)
        cv2.rectangle(frame, (x0, y0), (x0 + 30, y0 + 50), (20, 190, 80), thickness=-1)
        frames.append(frame)

    encode_video_frames_ffmpeg(
        output_path=path,
        frames_bgr=frames,
        fps=fps,
        width=width,
        height=height,
        codec="libx264",
        pix_fmt="yuv420p",
        crf=18,
        preset="veryfast",
    )
    return path

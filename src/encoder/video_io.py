from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import torch


@dataclass(frozen=True)
class DecodedVideo:
    tensor: torch.Tensor
    fps: float
    num_frames: int
    width: int
    height: int


def decode_video_to_tensor(video_path: str | Path) -> DecodedVideo:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_tensors: list[torch.Tensor] = []

    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
            frame_tensors.append(frame_tensor)
    finally:
        capture.release()

    if not frame_tensors:
        raise ValueError(f"Video contains no decodable frames: {video_path}")

    # Shape: [Frames, Channels, Height, Width]
    stacked = torch.stack(frame_tensors, dim=0).to(torch.float32) / 255.0
    _, _, height, width = stacked.shape

    return DecodedVideo(
        tensor=stacked,
        fps=fps,
        num_frames=int(stacked.shape[0]),
        width=int(width),
        height=int(height),
    )

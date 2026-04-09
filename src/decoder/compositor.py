from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.encoder.video_io import iter_video_frames_ffmpeg


class ResidualCompositor:
    """Client-side signed residual compositor for reconstruction."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

    def composite(
        self,
        predicted_frames: torch.Tensor,
        residual_video_uri: str | Path,
        width: int,
        height: int,
    ) -> torch.Tensor:
        residual_path = Path(residual_video_uri)
        if not residual_path.exists() or not residual_path.is_file():
            return predicted_frames

        decoded_frames: list[np.ndarray] = []
        for frame in iter_video_frames_ffmpeg(residual_path, width=width, height=height):
            decoded_frames.append(frame)

        if not decoded_frames:
            return predicted_frames

        residual_tensor = (
            torch.from_numpy(np.stack(decoded_frames, axis=0))
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(self._device, dtype=torch.float32)
        )
        predicted_tensor = predicted_frames.to(self._device, dtype=torch.float32)

        valid_frames = min(int(predicted_tensor.shape[0]), int(residual_tensor.shape[0]))
        decoded_diff = residual_tensor[:valid_frames] - 128.0
        final_tensor = torch.clamp(predicted_tensor[:valid_frames] + decoded_diff, 0.0, 255.0).to(torch.uint8)

        if valid_frames == int(predicted_tensor.shape[0]):
            return final_tensor

        tail = predicted_tensor[valid_frames:].to(torch.uint8)
        return torch.cat([final_tensor, tail], dim=0)

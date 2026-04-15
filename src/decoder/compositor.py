from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.encoder.video_io import iter_video_frames_ffmpeg
from src.shared.torch_dtype import is_cuda_device_usable


class ResidualCompositor:
    """Client-side signed residual compositor for reconstruction."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        if self._device.type == "cuda" and not is_cuda_device_usable(self._device):
            self._device = torch.device("cpu")

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

        composited = predicted_frames.to(self._device, dtype=torch.uint8)
        frame_limit = int(composited.shape[0])

        for frame_idx, residual_frame in enumerate(iter_video_frames_ffmpeg(residual_path, width=width, height=height)):
            if frame_idx >= frame_limit:
                break

            residual_tensor = (
                torch.from_numpy(np.asarray(residual_frame, dtype=np.uint8))
                .permute(2, 0, 1)
                .contiguous()
                .to(self._device, dtype=torch.float32)
            )
            predicted_tensor = composited[frame_idx].to(torch.float32)
            composited[frame_idx] = torch.clamp(predicted_tensor + (residual_tensor - 128.0), 0.0, 255.0).to(torch.uint8)

        return composited

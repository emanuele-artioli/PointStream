from __future__ import annotations

import torch

from src.shared.tags import gpu_bound


class SynthesisEngine:
    """Shared mock synthesis engine used symmetrically by encoder and decoder."""

    @gpu_bound
    def synthesize(self, batch: int, frames: int, height: int, width: int) -> torch.Tensor:
        # Shape: [Batch, Frames, Channels, Height, Width]
        return torch.zeros(batch, frames, 3, height, width, dtype=torch.float32)

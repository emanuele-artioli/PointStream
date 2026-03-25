from __future__ import annotations

import torch

from src.shared.schemas import ResidualPacket, VideoChunk
from src.shared.synthesis_engine import SynthesisEngine
from src.shared.tags import gpu_bound


class ResidualCalculator:
    def __init__(self, synthesis_engine: SynthesisEngine) -> None:
        self._synthesis_engine = synthesis_engine

    @gpu_bound
    def process(self, chunk: VideoChunk) -> ResidualPacket:
        # Shape: [Batch, Frames, Channels, Height, Width]
        predicted = self._synthesis_engine.synthesize(
            batch=1,
            frames=chunk.num_frames,
            height=chunk.height,
            width=chunk.width,
        )
        # Shape: [Batch, Frames, Channels, Height, Width]
        ground_truth = torch.zeros_like(predicted)
        # Shape: [Batch, Frames, Channels, Height, Width]
        _residual = ground_truth - predicted

        return ResidualPacket(
            chunk_id=chunk.chunk_id,
            codec="hevc-placeholder",
            residual_video_uri=f"memory://residual/{chunk.chunk_id}.mp4",
        )

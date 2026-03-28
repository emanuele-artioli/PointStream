from __future__ import annotations

import torch

from src.shared.schemas import EncodedChunkPayload, ResidualPacket
from src.shared.synthesis_engine import SynthesisEngine
from src.shared.tags import gpu_bound


class ResidualCalculator:
    def __init__(self, synthesis_engine: SynthesisEngine) -> None:
        self._synthesis_engine = synthesis_engine

    @gpu_bound
    def process(self, payload: EncodedChunkPayload) -> ResidualPacket:
        chunk = payload.chunk
        predicted = self._synthesis_engine.synthesize(payload).frames_bgr.to(torch.float32)
        # Shape: [Frames, Channels, Height, Width]
        ground_truth = torch.zeros_like(predicted)
        # Shape: [Frames, Channels, Height, Width]
        _residual = ground_truth - predicted

        return ResidualPacket(
            chunk_id=chunk.chunk_id,
            codec="hevc-placeholder",
            residual_video_uri=f"memory://residual/{chunk.chunk_id}.mp4",
        )

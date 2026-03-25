from __future__ import annotations

from src.shared.schemas import DecodedChunkResult, EncodedChunkPayload
from src.shared.synthesis_engine import SynthesisEngine
from src.shared.tags import gpu_bound


class DecoderRenderer:
    def __init__(self) -> None:
        self._synthesis_engine = SynthesisEngine()

    @gpu_bound
    def process(self, payload: EncodedChunkPayload) -> DecodedChunkResult:
        chunk = payload.chunk

        # Shape: [Batch, Frames, Channels, Height, Width]
        _base_video = self._synthesis_engine.synthesize(
            batch=1,
            frames=chunk.num_frames,
            height=chunk.height,
            width=chunk.width,
        )

        return DecodedChunkResult(
            chunk_id=chunk.chunk_id,
            output_uri=f"memory://decoded/{chunk.chunk_id}.mp4",
            num_frames=chunk.num_frames,
            width=chunk.width,
            height=chunk.height,
        )

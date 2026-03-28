from __future__ import annotations

from pathlib import Path

import numpy as np

from src.encoder.video_io import encode_video_frames_ffmpeg
from src.shared.schemas import DecodedChunkResult, EncodedChunkPayload
from src.shared.synthesis_engine import SynthesisEngine
from src.shared.tags import gpu_bound


class DecoderRenderer:
    def __init__(self, output_root: str | Path | None = None, deterministic_seed: int = 1337) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self._output_root = Path(output_root) if output_root is not None else project_root / "assets" / "decoded"
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._synthesis_engine = SynthesisEngine(seed=deterministic_seed)

    @gpu_bound
    def process(self, payload: EncodedChunkPayload) -> DecodedChunkResult:
        chunk = payload.chunk
        synthesis = self._synthesis_engine.synthesize(payload)

        frame_tensor = synthesis.frames_bgr
        frames_bgr = [
            np.asarray(frame.permute(1, 2, 0).cpu().numpy(), dtype=np.uint8)
            for frame in frame_tensor
        ]
        output_path = self._output_root / f"{chunk.chunk_id}.mp4"
        encode_video_frames_ffmpeg(
            output_path=output_path,
            frames_bgr=frames_bgr,
            fps=float(chunk.fps),
            width=int(chunk.width),
            height=int(chunk.height),
            codec="libx264",
            pix_fmt="yuv420p",
            crf=18,
            preset="veryfast",
        )

        return DecodedChunkResult(
            chunk_id=chunk.chunk_id,
            output_uri=str(output_path),
            num_frames=chunk.num_frames,
            width=chunk.width,
            height=chunk.height,
        )

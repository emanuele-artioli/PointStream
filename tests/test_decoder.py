from __future__ import annotations

from pathlib import Path

import numpy as np

from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.video_io import encode_video_frames_ffmpeg, probe_video_metadata
from src.shared.synthesis_engine import SynthesisEngine
from tests.video_utils import create_dummy_video


def test_decoder_output_matches_chunk_dimensions(mock_encoder_pipeline) -> None:
    project_root = Path(__file__).resolve().parents[1]
    reconstruction_path = project_root / "assets" / "mock_reconstruction.mp4"
    reconstruction_path.unlink(missing_ok=True)

    video_path = create_dummy_video(
        path=project_root / "assets" / "test_chunks" / "dec001.mp4",
        num_frames=6,
        width=640,
        height=360,
        fps=30.0,
    )

    payload, _decoded = mock_encoder_pipeline.encode_video_file(
        video_path=video_path,
        chunk_id="dec001",
        start_frame_id=0,
    )

    engine = SynthesisEngine(seed=2026)
    synthesis = engine.synthesize(payload)
    synthesis_repeat = engine.synthesize(payload)
    assert np.array_equal(
        synthesis.frames_bgr.cpu().numpy(),
        synthesis_repeat.frames_bgr.cpu().numpy(),
    )
    frames = [
        np.asarray(frame.permute(1, 2, 0).cpu().numpy(), dtype=np.uint8)
        for frame in synthesis.frames_bgr
    ]
    encode_video_frames_ffmpeg(
        output_path=reconstruction_path,
        frames_bgr=frames,
        fps=payload.chunk.fps,
        width=payload.chunk.width,
        height=payload.chunk.height,
        codec="libx264",
        pix_fmt="yuv420p",
        crf=18,
        preset="veryfast",
    )

    rendered_meta = probe_video_metadata(reconstruction_path)
    assert rendered_meta.num_frames == payload.chunk.num_frames
    assert rendered_meta.width == payload.chunk.width
    assert rendered_meta.height == payload.chunk.height

    decoded = DecoderRenderer().process(payload)

    assert decoded.chunk_id == "dec001"
    assert decoded.num_frames == 6
    assert decoded.width == 640
    assert decoded.height == 360
    assert decoded.output_uri.endswith("dec001.mp4")
    assert reconstruction_path.exists()

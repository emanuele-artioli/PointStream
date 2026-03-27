from __future__ import annotations

from pathlib import Path

from src.decoder.mock_renderer import DecoderRenderer
from tests.video_utils import create_dummy_video


def test_decoder_output_matches_chunk_dimensions(mock_encoder_pipeline) -> None:
    project_root = Path(__file__).resolve().parents[1]
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
    decoded = DecoderRenderer().process(payload)

    assert decoded.chunk_id == "dec001"
    assert decoded.num_frames == 6
    assert decoded.width == 640
    assert decoded.height == 360
    assert decoded.output_uri.endswith("dec001.mp4")

from __future__ import annotations

import shutil
from pathlib import Path

from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.video_io import iter_video_frames_ffmpeg, probe_video_metadata
from src.transport.disk import DiskTransport
from tests.video_utils import create_dummy_video


def test_end_to_end_mp4_encode_transport_decode(mock_encoder_pipeline, test_run_artifacts_dir: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]

    video_path = create_dummy_video(
        path=test_run_artifacts_dir / "test_video.mp4",
        num_frames=30,
        width=96,
        height=64,
        fps=30.0,
    )

    metadata = probe_video_metadata(video_path)
    streamed_frames = list(
        iter_video_frames_ffmpeg(
            video_path,
            width=metadata.width,
            height=metadata.height,
        )
    )
    assert len(streamed_frames) == 30
    assert streamed_frames[0].shape == (64, 96, 3)

    transport_root = project_root / ".pointstream"
    chunk_id = "e2e_mock_0001"
    chunk_dir = transport_root / f"chunk_{chunk_id}"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)

    payload, decoded_video_tensor = mock_encoder_pipeline.encode_video_file(
        video_path=video_path,
        chunk_id=chunk_id,
    )

    assert tuple(decoded_video_tensor.shape) == (30, 3, 64, 96)

    transport = DiskTransport(root_dir=transport_root)
    transport.send(payload)

    assert (chunk_dir / "metadata.msgpack").exists()
    assert (chunk_dir / "residual.mp4").exists()

    recovered_payload = transport.receive(chunk_id)
    decoded_result = DecoderRenderer().process(recovered_payload)

    assert recovered_payload.chunk.chunk_id == chunk_id
    assert recovered_payload.chunk.num_frames == 30
    assert recovered_payload.chunk.width == 96
    assert recovered_payload.chunk.height == 64
    assert len(recovered_payload.actors) == 2
    assert len(recovered_payload.actor_references) == 2
    assert len(recovered_payload.rigid_objects) == 1
    assert recovered_payload.ball.object_id == "ball_0"
    assert len(recovered_payload.ball.states) == 30

    assert decoded_result.chunk_id == chunk_id
    assert decoded_result.num_frames == 30
    assert decoded_result.width == 96
    assert decoded_result.height == 64

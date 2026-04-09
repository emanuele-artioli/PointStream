from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.transport.disk import DiskTransport
from tests.video_utils import create_dummy_video


def test_roundtrip_payload(mock_encoder_pipeline) -> None:
    project_root = Path(__file__).resolve().parents[1]
    video_path = create_dummy_video(
        path=project_root / "assets" / "test_chunks" / "rt001.mp4",
        num_frames=8,
        width=320,
        height=180,
        fps=30.0,
    )

    payload, _decoded = mock_encoder_pipeline.encode_video_file(
        video_path=video_path,
        chunk_id="rt001",
        start_frame_id=0,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        transport = DiskTransport(root_dir=tmp_dir)
        transport.send(payload)
        recovered = transport.receive("rt001")

        assert recovered.chunk.chunk_id == "rt001"
        assert len(recovered.actors) == 2

        chunk_dir = Path(tmp_dir) / "chunk_rt001"
        assert (chunk_dir / "metadata.msgpack").exists()
        residual_path = chunk_dir / "residual.mp4"
        assert residual_path.exists()
        assert residual_path.stat().st_size > 0
        assert recovered.residual.residual_video_uri == str(residual_path)


def test_receive_missing_payload_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        transport = DiskTransport(root_dir=tmp_dir)
        with pytest.raises(FileNotFoundError):
            transport.receive("missing_chunk")

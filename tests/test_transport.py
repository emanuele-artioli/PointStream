from __future__ import annotations

import tempfile
from pathlib import Path

import msgpack
import pytest

from src.transport.disk import DiskTransport
from tests.video_utils import create_dummy_video


def test_roundtrip_payload(mock_encoder_pipeline, test_run_artifacts_dir: Path) -> None:
    video_path = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "rt001.mp4",
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
        assert len(recovered.actor_references) == 2

        chunk_dir = Path(tmp_dir) / "chunk_rt001"
        assert (chunk_dir / "metadata.msgpack").exists()
        residual_path = chunk_dir / "residual.mp4"
        assert residual_path.exists()
        assert residual_path.stat().st_size > 0
        assert recovered.residual.residual_video_uri == str(residual_path)

        panorama_path = Path(recovered.panorama.panorama_uri)
        assert panorama_path.exists()
        assert panorama_path.parent == chunk_dir
        assert recovered.panorama.panorama_image is None

        metadata_raw = msgpack.unpackb((chunk_dir / "metadata.msgpack").read_bytes(), raw=False)
        assert metadata_raw["panorama"]["panorama_image"] is None


def test_receive_missing_payload_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        transport = DiskTransport(root_dir=tmp_dir)
        with pytest.raises(FileNotFoundError):
            transport.receive("missing_chunk")


def test_send_materializes_panorama_when_uri_is_not_file(
    mock_encoder_pipeline,
    test_run_artifacts_dir: Path,
) -> None:
    video_path = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "rt002.mp4",
        num_frames=6,
        width=320,
        height=180,
        fps=30.0,
    )

    payload, _decoded = mock_encoder_pipeline.encode_video_file(
        video_path=video_path,
        chunk_id="rt002",
        start_frame_id=0,
    )
    payload = payload.model_copy(
        update={
            "panorama": payload.panorama.model_copy(
                update={
                    "panorama_uri": "memory://panorama/in-memory",
                }
            )
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        transport = DiskTransport(root_dir=tmp_dir)
        transport.send(payload)
        recovered = transport.receive("rt002")

        panorama_path = Path(recovered.panorama.panorama_uri)
        assert panorama_path.exists()
        assert panorama_path.parent == Path(tmp_dir) / "chunk_rt002"
        assert recovered.panorama.panorama_image is None


def test_send_reencodes_panorama_sidecar_with_selected_codec(
    mock_encoder_pipeline,
    test_run_artifacts_dir: Path,
) -> None:
    video_path = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "rt003.mp4",
        num_frames=6,
        width=320,
        height=180,
        fps=30.0,
    )

    payload, _decoded = mock_encoder_pipeline.encode_video_file(
        video_path=video_path,
        chunk_id="rt003",
        start_frame_id=0,
    )
    payload = payload.model_copy(
        update={
            "panorama": payload.panorama.model_copy(
                update={
                    # Force transport to resolve from URI and then re-encode with selected codec.
                    "panorama_image": None,
                }
            )
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        transport = DiskTransport(root_dir=tmp_dir, panorama_encoder="png")
        transport.send(payload)
        recovered = transport.receive("rt003")

        panorama_path = Path(recovered.panorama.panorama_uri)
        assert panorama_path.exists()
        assert panorama_path.parent == Path(tmp_dir) / "chunk_rt003"
        assert panorama_path.suffix == ".png"
        assert recovered.panorama.panorama_image is None

        metadata_raw = msgpack.unpackb((Path(tmp_dir) / "chunk_rt003" / "metadata.msgpack").read_bytes(), raw=False)
        assert metadata_raw["panorama"]["panorama_image"] is None

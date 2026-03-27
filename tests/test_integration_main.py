from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.encoder.execution_pool import TaggedMultiprocessPool, WorkerConfig
from src.main import run_mock_pipeline


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_run_mock_pipeline_emits_summary_and_artifacts(real_actor_extractor, real_tennis_10f_video: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        summary = run_mock_pipeline(
            transport_root=tmp_dir,
            source_uri=str(real_tennis_10f_video),
            num_frames=10,
            actor_extractor=real_actor_extractor,
        )
        assert isinstance(summary["num_actor_packets"], int)
        num_actor_packets = summary["num_actor_packets"]

        assert summary["chunk_id"] == "0001"
        assert num_actor_packets >= 0
        assert num_actor_packets <= 2
        assert summary["num_rigid_object_packets"] == 1
        assert summary["ball_object_id"] == "ball_0"

        chunk_dir = Path(tmp_dir) / "chunk_0001"
        assert (chunk_dir / "metadata.msgpack").exists()
        assert (chunk_dir / "residual.mp4").exists()


def test_run_mock_pipeline_with_tagged_execution_pool(real_actor_extractor, real_tennis_10f_video: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        pool = TaggedMultiprocessPool(config=WorkerConfig(cpu_workers=1, gpu_workers=1))
        summary = run_mock_pipeline(
            transport_root=tmp_dir,
            execution_pool=pool,
            source_uri=str(real_tennis_10f_video),
            num_frames=10,
            actor_extractor=real_actor_extractor,
        )

        assert summary["chunk_id"] == "0001"
        assert pool.cpu_dispatch_count >= 1
        assert pool.gpu_dispatch_count >= 1

        chunk_dir = Path(tmp_dir) / "chunk_0001"
        assert (chunk_dir / "metadata.msgpack").exists()

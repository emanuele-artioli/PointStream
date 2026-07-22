from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import src.main as main_module
from src.encoder.execution_pool import TaggedMultiprocessPool, WorkerConfig
from src.main import run_pipeline
from src.shared.config import PointstreamConfig


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_run_pipeline_emits_summary_and_artifacts(monkeypatch, real_actor_extractor, real_tennis_10f_video: Path) -> None:
    monkeypatch.setattr(main_module, "build_actor_extractor", lambda config: real_actor_extractor)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = PointstreamConfig(source_uri=str(real_tennis_10f_video), num_frames=10)
        summary = run_pipeline(config=config, transport_root=tmp_dir)
        assert isinstance(summary["num_actor_packets"], int)
        num_actor_packets = summary["num_actor_packets"]

        assert summary["chunk_id"] == "0001"
        assert num_actor_packets >= 0
        assert num_actor_packets <= 2
        assert summary["num_rigid_object_packets"] == 0
        assert summary["ball_object_id"] == "ball_0"

        chunk_dir = Path(tmp_dir) / "chunk_0001"
        assert (chunk_dir / "metadata.msgpack").exists()
        assert (chunk_dir / "residual.mp4").exists()


def test_run_pipeline_with_tagged_execution_pool(monkeypatch, real_actor_extractor, real_tennis_10f_video: Path) -> None:
    pool = TaggedMultiprocessPool(config=WorkerConfig(cpu_workers=1, gpu_workers=1))
    monkeypatch.setattr(main_module, "build_actor_extractor", lambda config: real_actor_extractor)
    monkeypatch.setattr(main_module, "build_execution_pool", lambda config: pool)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = PointstreamConfig(source_uri=str(real_tennis_10f_video), num_frames=10, execution_pool="tagged")
        summary = run_pipeline(config=config, transport_root=tmp_dir)

        assert summary["chunk_id"] == "0001"
        assert pool.cpu_dispatch_count >= 1
        assert pool.gpu_dispatch_count >= 1

        chunk_dir = Path(tmp_dir) / "chunk_0001"
        assert (chunk_dir / "metadata.msgpack").exists()

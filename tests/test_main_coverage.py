from __future__ import annotations

import json
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.main as main_module
from src.encoder import residual as residual_module
from src.shared.config import PointstreamConfig


class _FakeEncoderPipeline:
    instances: list["_FakeEncoderPipeline"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.shutdown_called = False
        self.encoded_chunk = None
        _FakeEncoderPipeline.instances.append(self)

    def encode_chunk(self, chunk):
        self.encoded_chunk = chunk
        return SimpleNamespace(
            chunk=SimpleNamespace(chunk_id=chunk.chunk_id),
            actors=[object(), object()],
            rigid_objects=[object()],
            ball=SimpleNamespace(object_id="ball_0"),
            residual=SimpleNamespace(residual_video_uri="memory://residual/chunk.mp4"),
        )

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeDiskTransport:
    instances: list["_FakeDiskTransport"] = []

    def __init__(self, root_dir: str, config: Any = None) -> None:
        self.root_dir = root_dir
        self._payload = None
        _FakeDiskTransport.instances.append(self)

    def send(self, payload) -> None:
        self._payload = payload

    def receive(self, chunk_id: str):
        assert self._payload is not None
        assert chunk_id == self._payload.chunk.chunk_id
        return self._payload


class _FakeDecoderRenderer:
    instances: list["_FakeDecoderRenderer"] = []

    def __init__(self, output_root=None, deterministic_seed=1337, config=None) -> None:
        self.output_root = output_root
        self.deterministic_seed = deterministic_seed
        _FakeDecoderRenderer.instances.append(self)

    def process(self, payload):
        return SimpleNamespace(output_uri=f"/tmp/{payload.chunk.chunk_id}_decoded")


def _patch_pipeline_dependencies(monkeypatch, metadata_num_frames: int = 12):
    _FakeEncoderPipeline.instances.clear()
    _FakeDiskTransport.instances.clear()
    _FakeDecoderRenderer.instances.clear()

    monkeypatch.setattr(
        main_module,
        "probe_video_metadata",
        lambda _uri: SimpleNamespace(
            num_frames=metadata_num_frames,
            fps=30.0,
            width=320,
            height=180,
        ),
    )
    monkeypatch.setattr(main_module, "EncoderPipeline", _FakeEncoderPipeline)
    monkeypatch.setattr(main_module, "DiskTransport", _FakeDiskTransport)
    monkeypatch.setattr(main_module, "DecoderRenderer", _FakeDecoderRenderer)
    
    # Mock builders
    monkeypatch.setattr(main_module, "_build_execution_pool", lambda config: object())
    monkeypatch.setattr(main_module, "_build_actor_extractor", lambda config: object())
    monkeypatch.setattr(main_module, "_build_ball_extractor", lambda config: object())
    monkeypatch.setattr(main_module, "_build_reference_extractor", lambda config: object())
    monkeypatch.setattr(main_module, "_build_residual_calculator", lambda config: object())


def test_run_pipeline_builds_summary_with_provided_source(monkeypatch) -> None:
    _patch_pipeline_dependencies(monkeypatch, metadata_num_frames=12)

    config = PointstreamConfig(source_uri="/tmp/input.mp4", num_frames=5)
    summary = main_module.run_pipeline(
        config=config,
        transport_root="/tmp/pointstream_tests",
    )

    assert summary["chunk_id"] == "0001"
    assert summary["num_actor_packets"] == 2
    assert summary["num_rigid_object_packets"] == 1
    assert summary["ball_object_id"] == "ball_0"
    assert str(summary["residual_uri"]).endswith("chunk.mp4")
    assert str(summary["decoded_uri"]).endswith("0001_decoded")
    assert summary["transport_backend"] == "disk"

    encoder = _FakeEncoderPipeline.instances[0]
    assert encoder.encoded_chunk is not None
    assert encoder.encoded_chunk.num_frames == 5
    assert encoder.shutdown_called is True


def test_run_pipeline_uses_generated_source_when_missing_uri(monkeypatch) -> None:
    _patch_pipeline_dependencies(monkeypatch, metadata_num_frames=9)

    captured: dict[str, str | None] = {"source_uri": None}

    def _probe(uri: str):
        captured["source_uri"] = uri
        return SimpleNamespace(num_frames=9, fps=25.0, width=160, height=90)

    monkeypatch.setattr(main_module, "probe_video_metadata", _probe)
    monkeypatch.setattr(main_module, "_ensure_mock_source_video", lambda *args, **kwargs: "/tmp/generated.mp4")

    config = PointstreamConfig(source_uri=None, num_frames=None)
    summary = main_module.run_pipeline(
        config=config,
        transport_root="/tmp/pointstream_tests",
    )

    assert captured["source_uri"] == "/tmp/generated.mp4"
    assert summary["chunk_id"] == "0001"

    encoder = _FakeEncoderPipeline.instances[0]
    assert encoder.encoded_chunk is not None
    assert encoder.encoded_chunk.num_frames == 9


def test_script_run_pipeline_executes_main_block(monkeypatch, capsys) -> None:
    fake_main_module = types.ModuleType("src.main")
    def fake_run_cli(argv=None):
        print('{"chunk_id": "script_0001", "ok": True}')
        return 0

    setattr(fake_main_module, "run_cli", fake_run_cli)
    setattr(fake_main_module, "run_pipeline", lambda config: {"chunk_id": "script_0001", "ok": True})
    setattr(fake_main_module, "load_config", lambda *args, **kwargs: PointstreamConfig())

    monkeypatch.setitem(sys.modules, "src.main", fake_main_module)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("scripts.run_pipeline", run_name="__main__")
    assert exc_info.value.code == 0

    out = capsys.readouterr().out
    assert "script_0001" in out
    assert "ok" in out


def test_residual_reexport_contract() -> None:
    assert "BaseImportanceMapper" in residual_module.__all__
    assert "BinaryActorImportanceMapper" in residual_module.__all__
    assert "ResidualCalculator" in residual_module.__all__

    assert residual_module.BaseImportanceMapper is not None
    assert residual_module.BinaryActorImportanceMapper is not None
    assert residual_module.ResidualCalculator is not None


def test_ensure_mock_source_video_returns_existing_asset(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "tests_run"
    test_video = run_root / "runtime_sources" / "tennis_chunk_0001.mp4"
    test_video.parent.mkdir(parents=True, exist_ok=True)
    test_video.write_bytes(b"existing")

    resolved = main_module._ensure_mock_source_video(PointstreamConfig(), runtime_output_root=run_root)
    assert resolved == str(test_video)


def test_run_cli_accepts_input_and_config(monkeypatch, tmp_path: Path) -> None:
    source_video = tmp_path / "input.mp4"
    source_video.write_bytes(b"x")
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = tmp_path / "config.yaml"
    config_file.write_text("source_uri: /tmp/ignore.mp4")

    captured: dict[str, Any] = {}

    def _fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return {
            "chunk_id": "0001",
            "num_actor_packets": 1,
            "num_rigid_object_packets": 1,
            "ball_object_id": "ball_0",
            "residual_uri": "memory://residual/chunk.mp4",
            "decoded_uri": "memory://decoded/chunk.mp4",
            "run_output_root": str(output_dir)
        }

    monkeypatch.setattr(main_module, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(main_module, "_create_timestamped_output_dir", lambda base_root=None: output_dir)
    try:
        import importlib
        importlib.import_module("yaml")
        monkeypatch.setattr(main_module, "load_config", lambda path, cli_overrides=None: PointstreamConfig(**(cli_overrides or {"source_uri": "/tmp/ignore.mp4"})))
    except ImportError:
        monkeypatch.setattr(main_module, "load_config", lambda path, cli_overrides=None: PointstreamConfig(**(cli_overrides or {"source_uri": "/tmp/ignore.mp4"})))

    exit_code = main_module.run_cli(
        [
            "--input",
            str(source_video),
            "--config",
            str(config_file),
        ]
    )
    assert exit_code == 0

    assert "config" in captured
    config = captured["config"]
    assert config.source_uri == str(source_video)

    summary_file = output_dir / "run_summary.json"
    assert summary_file.exists()
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary["chunk_id"] == "0001"

from __future__ import annotations

import json
import os
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.main as main_module
from src.encoder import residual as residual_module


class _FakeEncoderPipeline:
    instances: list["_FakeEncoderPipeline"] = []

    def __init__(
        self,
        execution_pool=None,
        actor_extractor=None,
        ball_extractor=None,
        reference_extractor=None,
        residual_calculator=None,
    ) -> None:
        self.execution_pool = execution_pool
        self.actor_extractor = actor_extractor
        self.ball_extractor = ball_extractor
        self.reference_extractor = reference_extractor
        self.residual_calculator = residual_calculator
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

    def __init__(self, root_dir: str) -> None:
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

    def __init__(self, output_root=None, deterministic_seed=1337) -> None:
        self.output_root = output_root
        self.deterministic_seed = deterministic_seed
        _FakeDecoderRenderer.instances.append(self)

    def process(self, payload):
        return SimpleNamespace(output_uri=f"/tmp/{payload.chunk.chunk_id}_decoded.mp4")


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


def test_run_mock_pipeline_builds_summary_with_provided_source(monkeypatch) -> None:
    _patch_pipeline_dependencies(monkeypatch, metadata_num_frames=12)

    summary = main_module.run_mock_pipeline(
        transport_root="/tmp/pointstream_tests",
        source_uri="/tmp/input.mp4",
        num_frames=5,
    )

    assert summary["chunk_id"] == "0001"
    assert summary["num_actor_packets"] == 2
    assert summary["num_rigid_object_packets"] == 1
    assert summary["ball_object_id"] == "ball_0"
    assert str(summary["residual_uri"]).endswith("chunk.mp4")
    assert str(summary["decoded_uri"]).endswith("0001_decoded.mp4")
    assert summary["transport_backend"] == "disk"

    encoder = _FakeEncoderPipeline.instances[0]
    assert encoder.encoded_chunk is not None
    assert encoder.encoded_chunk.num_frames == 5
    assert encoder.shutdown_called is True


def test_run_mock_pipeline_uses_generated_source_when_missing_uri(monkeypatch) -> None:
    _patch_pipeline_dependencies(monkeypatch, metadata_num_frames=9)

    captured: dict[str, str | None] = {"source_uri": None}

    def _probe(uri: str):
        captured["source_uri"] = uri
        return SimpleNamespace(num_frames=9, fps=25.0, width=160, height=90)

    monkeypatch.setattr(main_module, "probe_video_metadata", _probe)
    monkeypatch.setattr(main_module, "_ensure_mock_source_video", lambda: "/tmp/generated.mp4")

    summary = main_module.run_mock_pipeline(
        transport_root="/tmp/pointstream_tests",
        source_uri=None,
        num_frames=None,
    )

    assert captured["source_uri"] == "/tmp/generated.mp4"
    assert summary["chunk_id"] == "0001"

    encoder = _FakeEncoderPipeline.instances[0]
    assert encoder.encoded_chunk is not None
    assert encoder.encoded_chunk.num_frames == 9


def test_script_run_mock_pipeline_executes_main_block(monkeypatch, capsys) -> None:
    fake_main_module = types.ModuleType("src.main")
    setattr(fake_main_module, "run_mock_pipeline", lambda: {"chunk_id": "script_0001", "ok": True})

    monkeypatch.setitem(sys.modules, "src.main", fake_main_module)
    runpy.run_module("scripts.run_mock_pipeline", run_name="__main__")

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


def test_ensure_mock_source_video_returns_existing_asset(monkeypatch, tmp_path: Path) -> None:
    test_video = tmp_path / "assets" / "test_chunks" / "tennis_chunk_0001.mp4"
    test_video.parent.mkdir(parents=True, exist_ok=True)
    test_video.write_bytes(b"existing")

    class _FakePath:
        def __init__(self, path: Path):
            self._path = path

        def resolve(self):
            return self

        @property
        def parents(self):
            return [self._path.parent, self._path.parent.parent]

    monkeypatch.setattr(main_module, "Path", lambda _p: _FakePath(tmp_path / "src" / "main.py"))

    resolved = main_module._ensure_mock_source_video()
    assert resolved == str(test_video)


def test_run_cli_accepts_input_and_writes_default_summary(monkeypatch, tmp_path: Path) -> None:
    source_video = tmp_path / "input.mp4"
    source_video.write_bytes(b"x")
    output_dir = tmp_path / "artifacts"

    captured: dict[str, object | None] = {
        "transport_root": None,
        "source_uri": None,
        "num_frames": None,
    }

    def _fake_run_mock_pipeline(**kwargs):
        captured["transport_root"] = kwargs.get("transport_root")
        captured["source_uri"] = kwargs.get("source_uri")
        captured["num_frames"] = kwargs.get("num_frames")
        return {
            "chunk_id": "0001",
            "num_actor_packets": 1,
            "num_rigid_object_packets": 1,
            "ball_object_id": "ball_0",
            "residual_uri": "memory://residual/chunk.mp4",
            "decoded_uri": "memory://decoded/chunk.mp4",
        }

    monkeypatch.setattr(main_module, "run_mock_pipeline", _fake_run_mock_pipeline)

    exit_code = main_module.run_cli(
        [
            "--input",
            str(source_video),
            "--output-dir",
            str(output_dir),
            "--num-frames",
            "9",
        ]
    )
    assert exit_code == 0

    assert captured["transport_root"] == str(output_dir)
    assert captured["source_uri"] == str(source_video.resolve())
    assert captured["num_frames"] == 9

    summary_file = output_dir / "run_summary.json"
    assert summary_file.exists()
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary["chunk_id"] == "0001"


def test_run_cli_supports_no_summary_file(monkeypatch, tmp_path: Path) -> None:
    source_video = tmp_path / "input.mp4"
    source_video.write_bytes(b"x")
    output_dir = tmp_path / "artifacts"

    monkeypatch.setattr(
        main_module,
        "run_mock_pipeline",
        lambda **kwargs: {
            "chunk_id": "0002",
            "num_actor_packets": 0,
            "num_rigid_object_packets": 0,
            "ball_object_id": "ball_0",
            "residual_uri": "memory://residual/chunk.mp4",
            "decoded_uri": "memory://decoded/chunk.mp4",
        },
    )

    exit_code = main_module.run_cli(
        [
            "--input",
            str(source_video),
            "--output-dir",
            str(output_dir),
            "--no-summary-file",
        ]
    )
    assert exit_code == 0
    assert not (output_dir / "run_summary.json").exists()


def test_run_cli_rejects_invalid_num_frames(monkeypatch) -> None:
    monkeypatch.setattr(
        main_module,
        "run_mock_pipeline",
        lambda **kwargs: {},
    )

    with pytest.raises(ValueError, match="positive integer"):
        main_module.run_cli(["--num-frames", "0"])


def test_run_cli_rejects_missing_input_video(monkeypatch, tmp_path: Path) -> None:
    missing_video = tmp_path / "missing.mp4"

    monkeypatch.setattr(
        main_module,
        "run_mock_pipeline",
        lambda **kwargs: {},
    )

    with pytest.raises(FileNotFoundError):
        main_module.run_cli(["--input", str(missing_video)])


def test_run_cli_passes_module_switches_and_env_overrides(monkeypatch, tmp_path: Path) -> None:
    source_video = tmp_path / "input.mp4"
    source_video.write_bytes(b"x")
    output_dir = tmp_path / "results"

    captured: dict[str, object] = {}

    def _fake_run_mock_pipeline(**kwargs):
        captured.update(kwargs)
        return {
            "chunk_id": "ablation_01",
            "num_actor_packets": 0,
            "num_rigid_object_packets": 0,
            "ball_object_id": "ball_0",
            "residual_uri": "memory://residual/chunk.mp4",
            "decoded_uri": "memory://decoded/chunk.mp4",
            "transport_backend": "disk",
        }

    monkeypatch.setattr(main_module, "run_mock_pipeline", _fake_run_mock_pipeline)
    monkeypatch.delenv("POINTSTREAM_ENABLE_GENAI", raising=False)
    monkeypatch.delenv("POINTSTREAM_GENAI_BACKEND", raising=False)

    exit_code = main_module.run_cli(
        [
            "--input",
            str(source_video),
            "--output-dir",
            str(output_dir),
            "--chunk-id",
            "ablation_01",
            "--execution-pool",
            "tagged",
            "--cpu-workers",
            "2",
            "--gpu-workers",
            "3",
            "--actor-extractor",
            "mock",
            "--ball-extractor",
            "mock",
            "--reference-jpeg-quality",
            "80",
            "--importance-mapper",
            "uniform",
            "--seed",
            "7",
            "--disable-genai",
            "--genai-backend",
            "controlnet",
            "--no-summary-file",
        ]
    )

    assert exit_code == 0
    assert captured["transport_root"] == str(output_dir)
    assert captured["chunk_id"] == "ablation_01"
    assert captured["execution_pool"] is not None
    assert captured["actor_extractor"] is not None
    assert captured["ball_extractor"] is not None
    assert captured["reference_extractor"] is not None
    assert captured["residual_calculator"] is not None
    assert os.environ.get("POINTSTREAM_ENABLE_GENAI") == "0"
    assert os.environ.get("POINTSTREAM_GENAI_BACKEND") == "controlnet"

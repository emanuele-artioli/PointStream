from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import src.main as main_module
from src.encoder import residual as residual_module


class _FakeEncoderPipeline:
    instances: list["_FakeEncoderPipeline"] = []

    def __init__(self, execution_pool=None, actor_extractor=None) -> None:
        self.execution_pool = execution_pool
        self.actor_extractor = actor_extractor
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

    def __init__(self) -> None:
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

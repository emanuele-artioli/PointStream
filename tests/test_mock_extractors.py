from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import pytest

from src.encoder import mock_extractors as me
from src.shared.schemas import VideoChunk


def _install_fake_actor_components(monkeypatch: pytest.MonkeyPatch) -> None:
    class BasePoseEstimator:
        pass

    class BaseSegmenter:
        pass

    class DwposeEstimator(BasePoseEstimator):
        def __init__(self, torchscript_device: str) -> None:
            self.torchscript_device = torchscript_device

    class NoOpSegmenter(BaseSegmenter):
        pass

    class PayloadEncoder:
        def __init__(self, pose_delta_threshold: float) -> None:
            self.pose_delta_threshold = pose_delta_threshold

    class PipelineBuilder:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.render_out_dirs: list[Path] = []

        def run(self, chunk: VideoChunk, frames_bgr: list[np.ndarray]):
            _ = chunk
            _ = frames_bgr
            return [], []

        def render_debug_keyframes(
            self,
            chunk: VideoChunk,
            frames_bgr: list[np.ndarray],
            frame_states,
            actor_packets,
            out_dir: Path,
        ) -> None:
            _ = chunk
            _ = frames_bgr
            _ = frame_states
            _ = actor_packets
            self.render_out_dirs.append(out_dir)

    class StandardTennisHeuristic:
        pass

    class Yolo26Detector:
        def __init__(self, model_name: str, model=None) -> None:
            self.model_name = model_name
            self.model = model

    class YoloPoseEstimator(BasePoseEstimator):
        def __init__(self, model_name: str, model=None) -> None:
            self.model_name = model_name
            self.model = model

    class YoloSegmenter(BaseSegmenter):
        def __init__(self, model_name: str, model=None) -> None:
            self.model_name = model_name
            self.model = model

    module: Any = types.ModuleType("src.encoder.actor_components")
    setattr(module, "BasePoseEstimator", BasePoseEstimator)
    setattr(module, "BaseSegmenter", BaseSegmenter)
    setattr(module, "DwposeEstimator", DwposeEstimator)
    setattr(module, "NoOpSegmenter", NoOpSegmenter)
    setattr(module, "PayloadEncoder", PayloadEncoder)
    setattr(module, "PipelineBuilder", PipelineBuilder)
    setattr(module, "StandardTennisHeuristic", StandardTennisHeuristic)
    setattr(module, "Yolo26Detector", Yolo26Detector)
    setattr(module, "YoloPoseEstimator", YoloPoseEstimator)
    setattr(module, "YoloSegmenter", YoloSegmenter)
    monkeypatch.setitem(sys.modules, "src.encoder.actor_components", module)


def _make_chunk(path: Path, *, frames: int = 5) -> VideoChunk:
    return VideoChunk(
        chunk_id="mock001",
        source_uri=str(path),
        start_frame_id=10,
        fps=25.0,
        num_frames=frames,
        width=64,
        height=48,
    )


def test_actor_extractor_rejects_invalid_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_actor_components(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported pose backend"):
        me.ActorExtractor(pose_backend="unsupported", render_debug_keyframes=False)

    with pytest.raises(ValueError, match="Unsupported segmenter backend"):
        me.ActorExtractor(segmenter_backend="unsupported", render_debug_keyframes=False)


def test_actor_extractor_load_frames_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_actor_components(monkeypatch)
    extractor = me.ActorExtractor(render_debug_keyframes=False)

    missing_chunk = _make_chunk(tmp_path / "missing.mp4")
    with pytest.raises(FileNotFoundError):
        extractor._load_frames(missing_chunk)

    source = tmp_path / "empty.mp4"
    source.write_bytes(b"not-a-real-video")

    monkeypatch.setattr(me, "probe_video_metadata", lambda source: types.SimpleNamespace(width=64, height=48))
    monkeypatch.setattr(me, "iter_video_frames_ffmpeg", lambda source, width, height: iter(()))

    with pytest.raises(ValueError, match="decoded zero frames"):
        extractor._load_frames(_make_chunk(source))


def test_actor_extractor_process_with_states_renders_debug(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_actor_components(monkeypatch)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"x")

    monkeypatch.setattr(me, "probe_video_metadata", lambda source: types.SimpleNamespace(width=64, height=48))

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(me, "iter_video_frames_ffmpeg", lambda source, width, height: iter((frame, frame)))

    monkeypatch.setenv("POINTSTREAM_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))

    extractor = me.ActorExtractor(render_debug_keyframes=True)
    chunk = _make_chunk(source, frames=2)
    result = extractor.process_with_states(chunk)

    assert result.frame_states == []
    assert result.actor_packets == []
    render_out_dirs = getattr(extractor._pipeline, "render_out_dirs")
    assert render_out_dirs
    assert render_out_dirs[0].name == "debug_actors"


def test_mock_trackers_emit_contract_packets(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"x")
    chunk = _make_chunk(source, frames=6)

    rigid_packets = me.ObjectTracker().process(chunk)
    assert len(rigid_packets) == 1
    rigid = rigid_packets[0]
    assert rigid.object_id == "racket_0"
    assert rigid.trajectory_spec.shape == [1, 6, 32, 2]
    assert rigid.events[0].event_type == "keyframe"
    assert rigid.events[1].event_type == "interpolate"

    ball = me.BallTracker().process(chunk)
    assert ball.object_id == "ball_0"
    assert ball.trajectory_spec.shape == [1, 6, 4]
    assert ball.events[0].event_type == "keyframe"
    assert ball.events[1].event_type == "interpolate"

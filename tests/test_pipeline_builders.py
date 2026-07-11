"""Fast, mocked coverage of src/encoder/pipeline_builders.py's branch logic
(which backend/mapper/pool a given config selects), without loading real
YOLO weights or building real heavy objects."""

from __future__ import annotations

from typing import Any

import pytest

from src.encoder import pipeline_builders as pb
from src.shared.config import PointstreamConfig


class TestBuildExecutionPool:
    def test_inline_mode_returns_none(self) -> None:
        config = PointstreamConfig(execution_pool="inline")
        assert pb.build_execution_pool(config) is None

    # NOTE: `build_execution_pool`'s "tagged" branch has a pre-existing
    # constructor-signature mismatch against the real `WorkerConfig`/
    # `TaggedMultiprocessPool` (flagged separately, task_50eedbcf) — both are
    # faked here so these tests cover this function's own mode-dispatch
    # logic without tripping over that unrelated, already-broken call.
    def test_tagged_mode_builds_tagged_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _FakeWorkerConfig:
            def __init__(self, **kwargs: Any) -> None:
                captured["worker_config_kwargs"] = kwargs

        class _FakeTaggedPool:
            def __init__(self, **kwargs: Any) -> None:
                captured["pool_kwargs"] = kwargs

        monkeypatch.setattr(pb, "WorkerConfig", _FakeWorkerConfig)
        monkeypatch.setattr(pb, "TaggedMultiprocessPool", _FakeTaggedPool)
        config = PointstreamConfig(execution_pool="tagged", cpu_workers=3, gpu_workers=2, gpu_dtype="fp32")
        pool = pb.build_execution_pool(config)
        assert isinstance(pool, _FakeTaggedPool)
        assert captured["pool_kwargs"]["cpu_workers"] == 3
        assert captured["pool_kwargs"]["gpu_workers"] == 2

    def test_tagged_mode_defaults_workers_to_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _FakeWorkerConfig:
            def __init__(self, **kwargs: Any) -> None:
                pass

        class _FakeTaggedPool:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(pb, "WorkerConfig", _FakeWorkerConfig)
        monkeypatch.setattr(pb, "TaggedMultiprocessPool", _FakeTaggedPool)
        config = PointstreamConfig(execution_pool="tagged", cpu_workers=None, gpu_workers=None)
        pb.build_execution_pool(config)
        assert captured["cpu_workers"] == 1
        assert captured["gpu_workers"] == 1

    def test_unknown_mode_raises(self) -> None:
        config = PointstreamConfig(execution_pool="bogus")
        with pytest.raises(ValueError, match="Unknown execution pool mode"):
            pb.build_execution_pool(config)


class TestBuildActorExtractor:
    def test_passes_backends_and_metadata_mask_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _FakeActorExtractor:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(pb, "ActorExtractor", _FakeActorExtractor)
        config = PointstreamConfig(
            detector="yolo26x.pt",
            pose_estimator="yolo26x-pose.pt",
            segmenter="yolo26x-seg.pt",
            compositing_mask_mode="metadata-source-mask",
        )
        extractor = pb.build_actor_extractor(config)
        assert isinstance(extractor, _FakeActorExtractor)
        assert captured["detector_backend"] == "yolo26x.pt"
        assert captured["pose_backend"] == "yolo26x-pose.pt"
        assert captured["segmenter_backend"] == "yolo26x-seg.pt"
        assert captured["include_mask_metadata"] is True

    def test_metadata_mask_flag_false_for_other_modes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        monkeypatch.setattr(pb, "ActorExtractor", lambda **kwargs: captured.update(kwargs) or object())
        config = PointstreamConfig(compositing_mask_mode="postgen-seg-client")
        pb.build_actor_extractor(config)
        assert captured["include_mask_metadata"] is False


class TestBuildBallExtractor:
    def test_segmentation_mode_builds_segmentation_extractor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _FakeSegExtractor:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(pb, "SegmentationBallExtractor", _FakeSegExtractor)
        config = PointstreamConfig(ball_extractor="segmentation", ball_det_conf=0.5, ball_det_model="yolo26n.pt")
        extractor = pb.build_ball_extractor(config)
        assert isinstance(extractor, _FakeSegExtractor)
        assert captured["confidence"] == 0.5
        assert captured["model_name"] == "yolo26n.pt"

    def test_segmentation_mode_defaults_confidence_and_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        monkeypatch.setattr(pb, "SegmentationBallExtractor", lambda **kwargs: captured.update(kwargs) or object())
        config = PointstreamConfig(ball_extractor="segmentation", ball_det_conf=None, ball_det_model=None)
        pb.build_ball_extractor(config)
        assert captured["confidence"] == 0.25
        assert captured["model_name"] == "yolo26n.pt"

    def test_difference_mode_builds_difference_extractor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _FakeDiffExtractor:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(pb, "BallExtractor", _FakeDiffExtractor)
        config = PointstreamConfig(ball_extractor="difference", ball_difference_threshold=12.0, ball_min_blob_area=9, ball_max_side=100)
        extractor = pb.build_ball_extractor(config)
        assert isinstance(extractor, _FakeDiffExtractor)
        assert captured["difference_threshold"] == 12.0
        assert captured["min_blob_area"] == 9
        assert captured["detection_max_side"] == 100


class TestBuildReferenceExtractor:
    def test_passes_jpeg_quality_and_padding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _FakeRefExtractor:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(pb, "ReferenceExtractor", _FakeRefExtractor)
        config = PointstreamConfig(reference_jpeg_quality=80, reference_padding_ratio=0.2)
        extractor = pb.build_reference_extractor(config)
        assert isinstance(extractor, _FakeRefExtractor)
        assert captured["jpeg_quality"] == 80
        assert captured["bbox_padding_ratio"] == 0.2


class TestBuildResidualCalculator:
    def test_uniform_mapper_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class _FakeResidualCalculator:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(pb, "ResidualCalculator", _FakeResidualCalculator)
        config = PointstreamConfig(importance_mapper="uniform", seed=42)
        calc = pb.build_residual_calculator(config)
        assert isinstance(calc, _FakeResidualCalculator)
        assert isinstance(captured["importance_mapper"], pb.UniformImportanceMapper)
        assert captured["seed"] == 42

    def test_non_uniform_mode_uses_binary_actor_mapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        monkeypatch.setattr(pb, "ResidualCalculator", lambda **kwargs: captured.update(kwargs) or object())
        config = PointstreamConfig(importance_mapper="binary")
        pb.build_residual_calculator(config)
        assert isinstance(captured["importance_mapper"], pb.BinaryActorImportanceMapper)

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

    def test_tagged_mode_builds_tagged_pool(self) -> None:
        config = PointstreamConfig(execution_pool="tagged", cpu_workers=3, gpu_workers=2, gpu_dtype="fp32")
        pool = pb.build_execution_pool(config)
        try:
            assert isinstance(pool, pb.TaggedMultiprocessPool)
            assert pool._config.cpu_workers == 3
            assert pool._config.gpu_workers == 2
        finally:
            assert pool is not None
            pool.shutdown()

    def test_tagged_mode_defaults_workers_to_one(self) -> None:
        config = PointstreamConfig(execution_pool="tagged", cpu_workers=None, gpu_workers=None)
        pool = pb.build_execution_pool(config)
        try:
            assert isinstance(pool, pb.TaggedMultiprocessPool)
            assert pool._config.cpu_workers == 1
            assert pool._config.gpu_workers == 1
        finally:
            assert pool is not None
            pool.shutdown()

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

    def test_residual_tuning_knobs_are_forwarded_from_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Regression: these five config keys were previously read nowhere —
        # editing them in a YAML config silently had zero effect (report 10
        # Phase 5.0 / report 8 2026-07-11 entry).
        captured: dict[str, Any] = {}
        monkeypatch.setattr(pb, "ResidualCalculator", lambda **kwargs: captured.update(kwargs) or object())
        config = PointstreamConfig(
            residual_background_downscale=4,
            residual_batch_size=16,
            downscale_interpolation="nearest",
            residual_block_size=4,
            residual_block_threshold=2.0,
        )
        pb.build_residual_calculator(config)
        assert captured["background_block_downscale_factor"] == 4
        assert captured["residual_batch_size"] == 16
        assert captured["downscale_interpolation"] == "nearest"
        assert captured["residual_block_size"] == 4
        assert captured["block_information_threshold"] == 2.0

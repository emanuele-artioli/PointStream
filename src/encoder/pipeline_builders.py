"""Shared construction of encoder-side pipeline components.

Extracted from `src/main.py` (report 10 Phase 2) so a caller that needs to
encode *many* chunks in one process — e.g. the full-match orchestrator,
which must not reload YOLO/pose/segmentation models per scene sub-chunk —
can build each component once and reuse it, instead of going through
`run_pipeline`'s single-chunk, build-everything-per-call path.
"""

from __future__ import annotations

from typing import Any

from src.encoder.actor_pipeline import ActorExtractor
from src.encoder.ball_extractor import BallExtractor
from src.encoder.execution_pool import BaseExecutionPool, TaggedMultiprocessPool, WorkerConfig
from src.encoder.reference_extractor import ReferenceExtractor
from src.encoder.residual_calculator import BaseImportanceMapper, BinaryActorImportanceMapper, ResidualCalculator, UniformImportanceMapper
from src.encoder.segmentation_ball_extractor import SegmentationBallExtractor
from src.shared.config import PointstreamConfig


def build_execution_pool(config: PointstreamConfig) -> BaseExecutionPool | None:
    mode = config.execution_pool.strip().lower()
    if mode == "inline":
        return None
    if mode == "tagged":
        return TaggedMultiprocessPool(
            config=WorkerConfig(
                cpu_workers=config.cpu_workers or 1,
                gpu_workers=config.gpu_workers or 1,
            ),
        )
    raise ValueError(f"Unknown execution pool mode: {mode}")


def build_actor_extractor(config: PointstreamConfig) -> ActorExtractor | None:
    normalized_mask_mode = config.compositing_mask_mode.strip().lower()
    include_mask_metadata = normalized_mask_mode == "metadata-source-mask"

    return ActorExtractor(
        config=config,
        render_debug_keyframes=False,
        detector_backend=config.detector,
        detector_caption=config.target_class_caption,
        pose_backend=config.pose_estimator,
        segmenter_backend=config.segmenter,
        segmenter_caption=config.target_class_caption,
        pose_delta_threshold=config.payload_pose_delta_threshold,
        include_mask_metadata=include_mask_metadata,
        metadata_mask_codec=config.metadata_mask_codec,
    )


def build_ball_extractor(config: PointstreamConfig) -> Any | None:
    mode = config.ball_extractor.strip().lower()
    if mode == "segmentation":
        return SegmentationBallExtractor(
            confidence=config.ball_det_conf or 0.25,
            model_name=config.ball_det_model or "yolo26n.pt",
            config=config,
        )
    return BallExtractor(
        difference_threshold=config.ball_difference_threshold,
        min_blob_area=config.ball_min_blob_area,
        detection_max_side=config.ball_max_side,
    )


def build_reference_extractor(config: PointstreamConfig) -> ReferenceExtractor:
    return ReferenceExtractor(
        jpeg_quality=config.reference_jpeg_quality,
        bbox_padding_ratio=config.reference_padding_ratio,
    )


def build_residual_calculator(config: PointstreamConfig) -> ResidualCalculator:
    mapper_mode = config.importance_mapper.strip().lower()
    mapper: BaseImportanceMapper
    if mapper_mode == "uniform":
        mapper = UniformImportanceMapper()
    else:
        mapper = BinaryActorImportanceMapper()

    return ResidualCalculator(
        config=config,
        seed=config.seed,
        importance_mapper=mapper,
    )

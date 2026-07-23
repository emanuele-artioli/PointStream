"""Backwards-compatible shim for the old single-module actor components.

The implementation moved to the `src.encoder.actors` package on 2026-07-22.
This module stays for two reasons: `actor_pipeline.py` imports from this path
lazily, and `tests/test_actor_pipeline.py` substitutes a fake module at exactly
this name in `sys.modules` — so the path is part of a contract, not merely a
convenience.

Prefer importing from `src.encoder.actors` (or one of its submodules) in new
code.
"""

from __future__ import annotations

from src.encoder.actors import (  # noqa: F401
    BaseDetector,
    BaseHeuristic,
    BasePoseEstimator,
    BaseSegmenter,
    CannySegmenter,
    DwposeEstimator,
    NoOpSegmenter,
    PayloadEncoder,
    PipelineBuilder,
    SamSegmenter,
    StandardTennisHeuristic,
    Yolo26Detector,
    YoloEDetector,
    YoloPoseEstimator,
    YoloSegmenter,
    YoloeSegmenter,
    _assets_weights_dir,
    _bbox_area,
    _bbox_center,
    _clip_bbox,
    _configure_ultralytics_weights_dir,
    _require_local_or_optin_weight,
    _resolve_local_weight_path,
)

__all__ = [
    "BaseDetector",
    "BaseHeuristic",
    "BasePoseEstimator",
    "BaseSegmenter",
    "CannySegmenter",
    "DwposeEstimator",
    "NoOpSegmenter",
    "PayloadEncoder",
    "PipelineBuilder",
    "SamSegmenter",
    "StandardTennisHeuristic",
    "Yolo26Detector",
    "YoloEDetector",
    "YoloPoseEstimator",
    "YoloSegmenter",
    "YoloeSegmenter",
]

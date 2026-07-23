"""Actor extraction: finding the players and packing them into the payload.

Split out of the single 1399-line `src.encoder.actor_components` module. The
grouping follows the pipeline stage each class serves:

    weights       weight resolution and bbox geometry, used by all of the below
    detection     YOLO backends that find people and rackets
    heuristics    deciding which of those detections are the players
    segmentation  masks that drive compositing
    pose          keypoints used as generative conditioning
    payload       packing actors into the transmitted payload
    builder       assembling a pipeline from config

Names are re-exported here, and `src.encoder.actor_components` remains as a
shim, so existing imports keep working.
"""

from src.encoder.actors.builder import PipelineBuilder
from src.encoder.actors.detection import BaseDetector, Yolo26Detector, YoloEDetector
from src.encoder.actors.heuristics import BaseHeuristic, StandardTennisHeuristic
from src.encoder.actors.payload import PayloadEncoder
from src.encoder.actors.pose import BasePoseEstimator, DwposeEstimator, YoloPoseEstimator
from src.encoder.actors.segmentation import (
    BaseSegmenter,
    CannySegmenter,
    NoOpSegmenter,
    SamSegmenter,
    YoloeSegmenter,
    YoloSegmenter,
)
from src.encoder.actors.weights import (  # noqa: F401
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

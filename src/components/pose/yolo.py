"""YOLO26 pose estimator. Emits COCO-17, stored as canonical WholeBody-133."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.components.detection.geometry import Box
from src.components.detection.parsing import first_keypoints
from src.components.detection.types import Detection, is_person
from src.components.detection.weights import resolve_weight
from src.components.pose.wire import Pose, from_coco17
from src.contracts.keypoints import COCO_17, KeypointSchema


class YoloPoseEstimator:
    """Per-crop YOLO pose. Classes without a skeleton are skipped."""

    def __init__(
        self,
        model_name: str = "yolo26n-pose.pt",
        model: Any | None = None,
        conf: float = 0.25,
    ) -> None:
        self.model_name = model_name
        self.conf = conf
        self.emits = COCO_17
        self._model = model if model is not None else self._load_model()

    def _load_model(self) -> Any:
        from ultralytics import YOLO

        path = resolve_weight(self.model_name)
        try:
            return YOLO(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize YOLO pose from {path}.") from exc

    def estimate(
        self,
        frame: np.ndarray,
        detection: Detection,
        *,
        bbox: Box | None = None,
    ) -> Pose | None:
        if not is_person(detection.class_name):
            return None
        box = bbox or detection.bbox
        height, width = frame.shape[:2]
        crop_box = box.clip(width, height)
        x1, y1, x2, y2 = (
            int(np.floor(crop_box.x1)),
            int(np.floor(crop_box.y1)),
            int(np.ceil(crop_box.x2)),
            int(np.ceil(crop_box.y2)),
        )
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        results = self._model.predict(source=crop, verbose=False, conf=self.conf)
        keypoints = first_keypoints(results)
        if keypoints is None or keypoints.shape[0] != len(self.emits):
            return None
        keypoints[:, 0] += float(x1)
        keypoints[:, 1] += float(y1)
        return from_coco17(keypoints)

    def estimate_to_schema(
        self,
        frame: np.ndarray,
        detection: Detection,
        consumer: KeypointSchema | str,
    ) -> Pose | None:
        """Estimate, then project onto the schema the generator actually reads."""
        from src.components.pose.wire import to_wire

        pose = self.estimate(frame, detection)
        if pose is None:
            return None
        return to_wire(pose, consumer)

"""YOLO26 detector. Default, and the fallback comparator."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.components.detection.geometry import Box
from src.components.detection.parsing import parse_boxes
from src.components.detection.types import (
    COCO_PERSON_ID,
    COCO_SPORTS_BALL_ID,
    COCO_TENNIS_RACKET_ID,
    Detection,
    is_ball,
    is_person,
    is_racket,
)
from src.components.detection.weights import resolve_weight

_ALLOWED = frozenset({COCO_PERSON_ID, COCO_TENNIS_RACKET_ID, COCO_SPORTS_BALL_ID})


class YoloDetector:
    """Closed-vocabulary YOLO26 detector.

    Construction accepts an already-built ``model`` so tests can mock inference.
    Loading from disk always goes through :func:`resolve_weight`, never a bare
    filename that could trigger an ultralytics download.
    """

    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        model: Any | None = None,
        conf: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.conf = conf
        self._model = model if model is not None else self._load_model()

    def _load_model(self) -> Any:
        from ultralytics import YOLO

        path = resolve_weight(self.model_name)
        try:
            return YOLO(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize YOLO detector from {path}."
            ) from exc

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        results = self._model.predict(source=frame, verbose=False, conf=self.conf)
        return parse_boxes(
            results,
            frame_width=width,
            frame_height=height,
            allowed_class_ids=_ALLOWED,
        )

    def predict_roi(self, frame: np.ndarray, bbox: Box, class_name: str) -> Box | None:
        height, width = frame.shape[:2]
        crop_box = bbox.padded(0.20, width, height)
        x1, y1, x2, y2 = (int(crop_box.x1), int(crop_box.y1), int(crop_box.x2), int(crop_box.y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        class_id = _class_id_for(class_name)
        kwargs: dict[str, Any] = {"source": crop, "verbose": False, "conf": 0.05}
        if class_id is not None:
            kwargs["classes"] = [class_id]
        results = self._model.predict(**kwargs)
        detections = parse_boxes(
            results,
            frame_width=crop.shape[1],
            frame_height=crop.shape[0],
            allowed_class_ids=None if class_id is None else frozenset({class_id}),
        )
        if not detections:
            return None
        best = max(detections, key=lambda item: item.score)
        mapped = Box(
            best.bbox.x1 + x1,
            best.bbox.y1 + y1,
            best.bbox.x2 + x1,
            best.bbox.y2 + y1,
        )
        return mapped.clip(width, height)


def _class_id_for(class_name: str) -> int | None:
    if is_person(class_name):
        return COCO_PERSON_ID
    if is_racket(class_name):
        return COCO_TENNIS_RACKET_ID
    if is_ball(class_name):
        return COCO_SPORTS_BALL_ID
    return None

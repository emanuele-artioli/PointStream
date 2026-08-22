"""YOLO26 instance segmenter."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.components.detection.geometry import Box
from src.components.detection.parsing import first_mask
from src.components.detection.types import Detection
from src.components.detection.weights import resolve_weight


class YoloSegmenter:
    """Per-crop YOLO-seg. Returns a binary mask the size of the crop."""

    def __init__(
        self,
        model_name: str = "yolo26n-seg.pt",
        model: Any | None = None,
        conf: float = 0.2,
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
            raise RuntimeError(f"Failed to initialize YOLO segmenter from {path}.") from exc

    def segment(self, frame: np.ndarray, detection: Detection) -> np.ndarray | None:
        height, width = frame.shape[:2]
        box = detection.bbox.clip(width, height)
        return self._segment_box(frame, box)

    def _segment_box(self, frame: np.ndarray, box: Box) -> np.ndarray | None:
        x1, y1, x2, y2 = (
            int(np.floor(box.x1)),
            int(np.floor(box.y1)),
            int(np.ceil(box.x2)),
            int(np.ceil(box.y2)),
        )
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        results = self._model.predict(source=crop, verbose=False, conf=self.conf)
        mask = first_mask(results)
        if mask is None:
            return None
        if mask.shape != crop.shape[:2]:
            mask = _resize_nearest(mask, crop.shape[0], crop.shape[1])
        return mask


def _resize_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

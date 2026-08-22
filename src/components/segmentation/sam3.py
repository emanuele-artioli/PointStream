"""SAM3 segmenter, prompted by the subject's box."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.components.detection.parsing import first_mask
from src.components.detection.types import Detection
from src.components.detection.weights import resolve_weight


class Sam3Segmenter:
    """Box-prompted SAM3. Returns a crop-local binary mask."""

    def __init__(
        self,
        model_name: str = "sam3.pt",
        model: Any | None = None,
        conf: float = 0.25,
    ) -> None:
        self.model_name = model_name
        self.conf = conf
        self._model = model if model is not None else self._load_model()

    def _load_model(self) -> Any:
        from ultralytics import SAM

        path = resolve_weight(self.model_name)
        try:
            return SAM(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize SAM3 segmenter from {path}.") from exc

    def segment(self, frame: np.ndarray, detection: Detection) -> np.ndarray | None:
        height, width = frame.shape[:2]
        box = detection.bbox.clip(width, height)
        x1 = int(np.floor(box.x1))
        y1 = int(np.floor(box.y1))
        x2 = int(np.ceil(box.x2))
        y2 = int(np.ceil(box.y2))
        if x2 <= x1 or y2 <= y1:
            return None
        results = self._model.predict(
            source=frame,
            bboxes=[[float(x1), float(y1), float(x2), float(y2)]],
            verbose=False,
            conf=self.conf,
        )
        mask = first_mask(results)
        if mask is None:
            return None
        if mask.shape[:2] != (height, width):
            import cv2

            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        local = mask[y1:y2, x1:x2]
        if local.size == 0:
            return None
        return local

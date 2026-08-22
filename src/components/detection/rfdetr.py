"""RF-DETR detector.

The ``rfdetr`` package is not installed, and it cannot be added to this env:
current PyPI ``rfdetr`` requires ``transformers>=5.1,<6``, while this env pins
``transformers==4.46.3`` (diffusers 0.30.3, controlnet-aux, Moore-AnimateAnyone).
Installing it would upgrade transformers and silently break those backends.

A second conda env was not created. Construction without an injected ``model``
fails with that explanation. Tests inject a duck-typed model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.components.detection.geometry import Box
from src.components.detection.parsing import parse_boxes, parse_supervision
from src.components.detection.types import (
    COCO_ID_TO_NAME,
    COCO_PERSON_ID,
    COCO_SPORTS_BALL_ID,
    COCO_TENNIS_RACKET_ID,
    Detection,
)

_ALLOWED = frozenset({COCO_PERSON_ID, COCO_TENNIS_RACKET_ID, COCO_SPORTS_BALL_ID})

MISSING_MESSAGE = (
    "RF-DETR is not installed in the pointstream environment, and it cannot be "
    "added without a second env: rfdetr currently requires transformers>=5.1, "
    "but this env pins transformers==4.46.3 (required by diffusers 0.30.3, "
    "controlnet-aux and Moore-AnimateAnyone). Pass an already-built model= to "
    "construct the wrapper in tests; do not pip-install rfdetr here."
)


class RfDetrDetector:
    """RF-DETR wrapper. Fails clearly when the package is not importable."""

    def __init__(
        self,
        model_name: str | None = None,
        model: Any | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        if model is not None:
            self._model = model
            return
        try:
            from rfdetr import RFDETRBase
        except ImportError as exc:
            raise RuntimeError(MISSING_MESSAGE) from exc
        # Package imported, so it now coexists; construct the stock model.
        # model_name is reserved for a future checkpoint override the package
        # does not currently take at construction.
        _ = model_name
        self._model = RFDETRBase()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        raw = self._call_predict(frame)
        parsed = self._parse(raw, width=width, height=height)
        return [item for item in parsed if item.class_id in _ALLOWED or item.class_id is None]

    def predict_roi(self, frame: np.ndarray, bbox: Box, class_name: str) -> Box | None:
        height, width = frame.shape[:2]
        crop_box = bbox.padded(0.20, width, height)
        x1, y1, x2, y2 = (int(crop_box.x1), int(crop_box.y1), int(crop_box.x2), int(crop_box.y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        raw = self._call_predict(crop)
        detections = self._parse(raw, width=crop.shape[1], height=crop.shape[0])
        if not detections:
            return None
        best = max(detections, key=lambda item: item.score)
        return Box(
            best.bbox.x1 + x1,
            best.bbox.y1 + y1,
            best.bbox.x2 + x1,
            best.bbox.y2 + y1,
        ).clip(width, height)

    def _call_predict(self, image: np.ndarray) -> Any:
        predict = getattr(self._model, "predict")
        try:
            return predict(image, threshold=self.threshold)
        except TypeError:
            return predict(source=image, verbose=False, conf=self.threshold)

    def _parse(self, raw: Any, *, width: int, height: int) -> list[Detection]:
        if raw is None:
            return []
        if hasattr(raw, "xyxy") and not hasattr(raw, "boxes"):
            return parse_supervision(
                raw,
                frame_width=width,
                frame_height=height,
                names=COCO_ID_TO_NAME,
                allowed_class_ids=_ALLOWED,
            )
        return parse_boxes(
            raw,
            frame_width=width,
            frame_height=height,
            allowed_class_ids=_ALLOWED,
            names=COCO_ID_TO_NAME,
        )

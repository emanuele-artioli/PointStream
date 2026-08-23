"""SAM3 detector. Open-vocabulary; supersedes SAM2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.components.detection.geometry import Box
from src.components.detection.parsing import parse_boxes
from src.components.detection.types import Detection
from src.components.detection.weights import resolve_weight


class Sam3Detector:
    """SAM3 as a detector, prompted by class text when a prompt is given.

    Ultralytics' ``SAM.predict`` forwards geometric prompts but not text. The
    real-weights path therefore uses ``SAM3SemanticPredictor`` so a class prompt
    actually reaches the model. Tests inject a duck-typed ``model`` with
    ``predict(...)`` and never load weights.
    """

    def __init__(
        self,
        model_name: str = "sam3.pt",
        model: Any | None = None,
        prompt: str | None = None,
        prompts: Sequence[str] | None = None,
        conf: float = 0.25,
    ) -> None:
        self.model_name = model_name
        self.conf = conf
        self._prompts = _split_prompts(prompt, prompts)
        self._model = model
        self._predictor: Any | None = None
        if model is None:
            self._predictor = self._load_semantic_predictor()

    def _load_semantic_predictor(self) -> Any:
        from ultralytics.models.sam.predict import SAM3SemanticPredictor

        path = resolve_weight(self.model_name)
        overrides = {
            "conf": self.conf,
            "task": "segment",
            "mode": "predict",
            "imgsz": 1008,
            "model": str(path),
            "verbose": False,
        }
        predictor = SAM3SemanticPredictor(overrides=overrides)
        predictor.setup_model()
        return predictor

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        results = self._run(frame, bboxes=None, texts=self._prompts or None)
        return parse_boxes(results, frame_width=width, frame_height=height)

    def predict_roi(self, frame: np.ndarray, bbox: Box, class_name: str) -> Box | None:
        height, width = frame.shape[:2]
        crop_box = bbox.padded(0.20, width, height)
        texts = [class_name, *self._prompts] if class_name else self._prompts
        results = self._run(
            frame,
            bboxes=[[crop_box.x1, crop_box.y1, crop_box.x2, crop_box.y2]],
            texts=texts or None,
        )
        detections = parse_boxes(results, frame_width=width, frame_height=height)
        if not detections:
            return None
        return max(detections, key=lambda item: item.score).bbox

    def _run(
        self,
        frame: np.ndarray,
        *,
        bboxes: list[list[float]] | None,
        texts: Sequence[str] | None,
    ) -> Any:
        if self._model is not None:
            return self._model.predict(
                source=frame,
                bboxes=bboxes,
                text=list(texts) if texts else None,
                verbose=False,
                conf=self.conf,
            )
        assert self._predictor is not None
        kwargs: dict[str, Any] = {"conf": self.conf}
        if bboxes is not None:
            kwargs["bboxes"] = bboxes
        if texts:
            kwargs["text"] = list(texts)
        return self._predictor(frame, **kwargs)


def _split_prompts(prompt: str | None, prompts: Sequence[str] | None) -> list[str]:
    values: list[str] = []
    if prompt:
        values.extend(part.strip() for part in prompt.split(",") if part.strip())
    if prompts:
        values.extend(part.strip() for part in prompts if part.strip())
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for item in values:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique

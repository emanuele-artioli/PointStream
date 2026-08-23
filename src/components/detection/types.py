"""Detections and the detector protocol.

These types are the subject-stream currency: selection, tracking, pose and
segmentation all consume them. They live here rather than in `contracts` because
they describe a runtime value, not a config-time check.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np

from src.components.detection.geometry import Box

#: Closed-vocabulary names YOLO/RF-DETR emit for a person, plus the tennis
#: domain's name for the same class. Selectors accept any of them.
PERSON_CLASSES: frozenset[str] = frozenset({"person", "player", "tennis player"})
RACKET_CLASSES: frozenset[str] = frozenset({"racket", "tennis racket"})
BALL_CLASSES: frozenset[str] = frozenset({"ball", "sports ball", "tennis ball"})

COCO_PERSON_ID = 0
COCO_SPORTS_BALL_ID = 32
COCO_TENNIS_RACKET_ID = 38

COCO_ID_TO_NAME: dict[int, str] = {
    COCO_PERSON_ID: "person",
    COCO_SPORTS_BALL_ID: "sports ball",
    COCO_TENNIS_RACKET_ID: "tennis racket",
}


def is_person(class_name: str) -> bool:
    return class_name.strip().lower() in PERSON_CLASSES


def is_racket(class_name: str) -> bool:
    return class_name.strip().lower() in RACKET_CLASSES


def is_ball(class_name: str) -> bool:
    return class_name.strip().lower() in BALL_CLASSES


@dataclass(frozen=True)
class Detection:
    """One candidate object in one frame."""

    class_name: str
    bbox: Box
    score: float = 1.0
    track_id: str | None = None
    class_id: int | None = None

    def with_track_id(self, track_id: str) -> Detection:
        return replace(self, track_id=track_id)

    def with_class_name(self, class_name: str) -> Detection:
        return replace(self, class_name=class_name)

    def with_bbox(self, bbox: Box) -> Detection:
        return replace(self, bbox=bbox)


@runtime_checkable
class Detector(Protocol):
    """Finds candidate objects in a single frame."""

    def detect(self, frame: np.ndarray) -> Sequence[Detection]:
        """Return detections for one BGR frame of shape (H, W, 3)."""
        ...


@runtime_checkable
class RoiPredictor(Protocol):
    """Re-detects one class inside a previously known box.

    Track recovery is a composed policy, not a detector base-class method.
    Any backend that can search a crop implements this, so a non-YOLO detector
    can reuse the same recovery logic.
    """

    def predict_roi(self, frame: np.ndarray, bbox: Box, class_name: str) -> Box | None:
        """Best box of `class_name` inside a padded crop of `bbox`, or None."""
        ...

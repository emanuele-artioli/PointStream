"""Identity tracker: IoU matching plus an optional recovery policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.components.detection.types import Detection, RoiPredictor
from src.components.tracking.recovery import RecoveryPolicy


class IdentityTracker:
    """Carry object identity across frames.

    Recovery is composed, not inherited: pass any :class:`RoiPredictor` (a
    YOLO detector, a SAM3 detector, or a test double).
    """

    def __init__(
        self,
        recovery: RecoveryPolicy | None = None,
        iou_threshold: float = 0.3,
        quotas: Mapping[str, int] | None = None,
    ) -> None:
        self.recovery = recovery if recovery is not None else RecoveryPolicy(quotas=quotas)
        self.iou_threshold = iou_threshold
        self._previous: list[Detection] = []
        self._next_id = 1

    def reset(self) -> None:
        self._previous = []
        self._next_id = 1

    def update(
        self,
        frame: object,
        detections: Sequence[Detection],
        predictor: RoiPredictor | None = None,
    ) -> list[Detection]:
        filled = self.recovery.recover(
            frame=frame,
            detections=detections,
            previous=self._previous,
            predictor=predictor,
        )
        matched = self._assign_ids(filled)
        self._previous = matched
        return matched

    def _assign_ids(self, detections: Sequence[Detection]) -> list[Detection]:
        assigned: list[Detection] = []
        used_previous: set[int] = set()
        for detection in detections:
            if detection.track_id:
                assigned.append(detection)
                continue
            best_index = None
            best_iou = self.iou_threshold
            for index, previous in enumerate(self._previous):
                if index in used_previous:
                    continue
                if previous.class_name != detection.class_name:
                    continue
                overlap = detection.bbox.iou(previous.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index is not None:
                previous_id = self._previous[best_index].track_id
                if previous_id:
                    used_previous.add(best_index)
                    assigned.append(detection.with_track_id(previous_id))
                    continue
            track_id = f"{detection.class_name}_{self._next_id}"
            self._next_id += 1
            assigned.append(detection.with_track_id(track_id))
        return assigned

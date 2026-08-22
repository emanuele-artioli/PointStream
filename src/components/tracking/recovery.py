"""Track-recovery as a composed policy, reusable by any detector.

The arrangement this replaces lived on ``Yolo26Detector`` and reached
``YoloEDetector`` only by inheritance, so a non-YOLO backend could not reuse
it. Here the policy asks a :class:`~src.components.detection.types.RoiPredictor`
to search a crop; YOLO, SAM3 and RF-DETR (or a test double) all qualify.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.components.detection.types import Detection, RoiPredictor

#: Tennis broadcast: two players, two rackets. Overridable per call.
DEFAULT_QUOTAS: dict[str, int] = {"player": 2, "racket": 2}


class RecoveryPolicy:
    """Fill per-class quotas from the previous frame when this frame is short.

    Order of attempts for each missing slot:

    1. Re-detect in a padded crop of the previous box, via ``predictor``.
    2. Hold the previous box (identity continuity).
    3. Leave the slot empty — do **not** synthesise a box. Inventing a player
       at a canned position is a silent wrong answer.
    """

    def __init__(
        self,
        quotas: Mapping[str, int] | None = None,
        max_retries: int = 1,
    ) -> None:
        self.quotas = dict(quotas) if quotas is not None else dict(DEFAULT_QUOTAS)
        self.max_retries = max(1, max_retries)

    def recover(
        self,
        *,
        frame: object,
        detections: Sequence[Detection],
        previous: Sequence[Detection] | None,
        predictor: RoiPredictor | None = None,
    ) -> list[Detection]:
        recovered = list(detections)
        if not self.quotas:
            return recovered
        for class_name, minimum in self.quotas.items():
            have = [item for item in recovered if item.class_name == class_name]
            missing = minimum - len(have)
            if missing <= 0:
                continue
            seeds = [
                item for item in (previous or []) if item.class_name == class_name
            ]
            recovered.extend(
                self._recover_class(
                    frame=frame,
                    class_name=class_name,
                    missing=missing,
                    seeds=seeds,
                    predictor=predictor,
                )
            )
        return recovered

    def _recover_class(
        self,
        *,
        frame: object,
        class_name: str,
        missing: int,
        seeds: Sequence[Detection],
        predictor: RoiPredictor | None,
    ) -> list[Detection]:
        found: list[Detection] = []
        for miss_index in range(missing):
            if miss_index >= len(seeds):
                break
            seed = seeds[miss_index]
            box = None
            if predictor is not None:
                for _ in range(self.max_retries):
                    box = predictor.predict_roi(frame, seed.bbox, class_name)  # type: ignore[arg-type]
                    if box is not None:
                        break
            if box is not None:
                track_id = seed.track_id or f"{class_name}_recovered_{miss_index}"
                found.append(
                    Detection(
                        class_name=class_name,
                        bbox=box,
                        score=seed.score,
                        track_id=track_id,
                    )
                )
                continue
            found.append(seed)
        return found

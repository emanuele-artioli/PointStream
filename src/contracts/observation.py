"""Versioned observation records shared by offline preparation and runtime.

An observation names the exact source frame, object identity, coordinates,
transform history, and estimator provenance that produced it. Missing
observations are records too; callers never need to infer them from absent
array entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

OBSERVATION_SCHEMA = "pointstream.observation.v1"
PIXEL_COORDINATES = "pixel_xy_top_left_full_frame"


class ObservationStatus(str, Enum):
    OBSERVED = "observed"
    MISSING = "missing"
    INVALID = "invalid"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class Pts:
    """Exact presentation timestamp in stream ticks and its rational timebase."""

    value: int
    timebase_num: int
    timebase_den: int

    def __post_init__(self) -> None:
        if type(self.value) is not int:
            raise ValueError("PTS value must be an integer stream tick")
        if self.timebase_num <= 0 or self.timebase_den <= 0:
            raise ValueError("PTS timebase must be a positive rational")

    @property
    def seconds(self) -> float:
        return self.value * self.timebase_num / self.timebase_den

    def to_record(self) -> dict[str, int]:
        return {
            "value": self.value,
            "timebase_num": self.timebase_num,
            "timebase_den": self.timebase_den,
        }


@dataclass(frozen=True)
class EstimatorProvenance:
    name: str
    model_revision: str
    checkpoint_sha256: str
    config_sha256: str
    policy: str

    def __post_init__(self) -> None:
        if not all((self.name, self.model_revision, self.checkpoint_sha256, self.config_sha256)):
            raise ValueError("estimator provenance requires model, revision, checkpoint, and config hashes")
        if self.policy not in {"offline_bidirectional", "offline_causal", "runtime_causal"}:
            raise ValueError(f"unsupported perception policy {self.policy!r}")


@dataclass(frozen=True)
class Observation:
    """A class-specific mask or an explicit missing/invalid observation."""

    source_id: str
    frame_index: int
    pts: Pts
    object_id: str
    object_class: str
    coordinate_system: str
    frame_width: int
    frame_height: int
    provenance: EstimatorProvenance
    status: ObservationStatus
    mask: np.ndarray | None = None
    transforms: tuple[dict[str, Any], ...] = ()
    associated_player_id: str | None = None
    associated_wrist: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id or not self.object_id or not self.object_class:
            raise ValueError("source_id, object_id, and object_class are required")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if self.coordinate_system != PIXEL_COORDINATES:
            raise ValueError(f"unsupported coordinate system {self.coordinate_system!r}")
        if self.status is ObservationStatus.OBSERVED:
            if self.mask is None:
                raise ValueError("observed records require a mask")
            mask = np.asarray(self.mask)
            if mask.ndim != 2 or mask.shape != (self.frame_height, self.frame_width):
                raise ValueError("observed mask must have the full source-frame shape (H, W)")
            if not np.all((mask == 0) | (mask == 1) | (mask == 255)):
                raise ValueError("observed mask must be binary")
            if self.reason is not None:
                raise ValueError("observed records cannot carry a failure reason")
        elif self.mask is not None:
            raise ValueError("non-observed records cannot carry a mask")
        elif not self.reason:
            raise ValueError("missing, invalid, and quarantined records need a reason")
        if self.object_class != "racket" and (
            self.associated_player_id is not None or self.associated_wrist is not None
        ):
            raise ValueError("only racket observations can carry player/wrist associations")
        if self.object_class == "racket" and self.associated_player_id is not None:
            if self.associated_wrist not in {"left_wrist", "right_wrist"}:
                raise ValueError("associated racket observations must name the associated wrist")
        elif self.associated_wrist is not None:
            raise ValueError("an associated wrist requires a racket/player association")

    @classmethod
    def missing(
        cls,
        *,
        source_id: str,
        frame_index: int,
        pts: Pts,
        object_id: str,
        object_class: str,
        frame_width: int,
        frame_height: int,
        provenance: EstimatorProvenance,
        reason: str,
        associated_player_id: str | None = None,
        associated_wrist: str | None = None,
    ) -> "Observation":
        return cls(
            source_id=source_id,
            frame_index=frame_index,
            pts=pts,
            object_id=object_id,
            object_class=object_class,
            coordinate_system=PIXEL_COORDINATES,
            frame_width=frame_width,
            frame_height=frame_height,
            provenance=provenance,
            status=ObservationStatus.MISSING,
            associated_player_id=associated_player_id,
            associated_wrist=associated_wrist,
            reason=reason,
        )

    def to_record(self, *, include_mask: bool = False) -> dict[str, Any]:
        """Return deterministic JSON metadata, optionally embedding the mask."""
        result: dict[str, Any] = {
            "schema": OBSERVATION_SCHEMA,
            "source_id": self.source_id,
            "frame_index": self.frame_index,
            "pts": self.pts.to_record(),
            "object_id": self.object_id,
            "object_class": self.object_class,
            "coordinate_system": self.coordinate_system,
            "frame_size": [self.frame_width, self.frame_height],
            "status": self.status.value,
            "transforms": list(self.transforms),
            "provenance": {
                "name": self.provenance.name,
                "model_revision": self.provenance.model_revision,
                "checkpoint_sha256": self.provenance.checkpoint_sha256,
                "config_sha256": self.provenance.config_sha256,
                "policy": self.provenance.policy,
            },
            "associated_player_id": self.associated_player_id,
            "associated_wrist": self.associated_wrist,
            "reason": self.reason,
        }
        if include_mask and self.mask is not None:
            result["mask"] = np.asarray(self.mask, dtype=np.uint8).tolist()
        return result


__all__ = [
    "EstimatorProvenance",
    "OBSERVATION_SCHEMA",
    "Observation",
    "ObservationStatus",
    "PIXEL_COORDINATES",
    "Pts",
]

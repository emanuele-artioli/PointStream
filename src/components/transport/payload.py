"""The document transport moves: metadata (including the schedule) plus blobs.

Serialization and the medium are separate modules. This one is only the
in-memory shape and the schedule's on-the-wire encoding — the artefact Phase C
must honour rather than recompute.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from src.contracts.objectstream import (
    FrameAction,
    FrameDecision,
    TemporalSchedule,
)

#: Payload key Phase C reads. Encoder writes it; reconstruction must not
#: replace it by running the policy again.
SCHEDULE_KEY: Final = "temporal_schedule"

#: Schema id so a later revision can be detected instead of silently misread.
SCHEDULE_SCHEMA: Final = "pointstream.temporal-schedule.v1"


@dataclass(frozen=True)
class PlannedSchedule:
    """A ``TemporalSchedule`` plus the perception mask the contract type lacks.

    ``FrameAction`` records what is transmitted and generated. Pipeline
    sparsity — which frames run detection, pose and segmentation — is a
    separate bit: an interpolate frame still runs perception when pipeline
    sparsity is off. That bit has to travel next to the decisions, or Phase C
    would have to infer it from config and the two sides would drift.
    """

    schedule: TemporalSchedule
    perception: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    scene_motion: float | None = None
    adapted_threshold: float | None = None

    def perception_frames(self, object_id: str) -> tuple[int, ...]:
        """Frames on which perception runs for ``object_id``, in order."""
        return tuple(self.perception.get(object_id, ()))

    def perception_count(self, object_id: str | None = None) -> int:
        """How many perception-stage runs this schedule asks for.

        That count is the encode-time cost pipeline sparsity actually saves.
        """
        if object_id is not None:
            return len(self.perception_frames(object_id))
        return sum(len(frames) for frames in self.perception.values())


@dataclass(frozen=True)
class ChunkPayload:
    """One chunk as transport sees it: schedule-bearing metadata plus sidecars.

    Args:
        chunk_id: Identity the medium uses as a key.
        schedule: Per-frame decisions, already planned. Required so the
            decoder never has to re-plan.
        blobs: Named sidecars — JPEG appearance/background, residual file —
            keyed by a plain filename. The serializer does not re-encode them.
        extra: Any other metadata the encoder wants on the wire.
    """

    chunk_id: str
    schedule: PlannedSchedule
    blobs: Mapping[str, bytes] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)


def dump_schedule(planned: PlannedSchedule) -> dict[str, Any]:
    """Plain mapping that msgpack (and Phase C) can store."""
    return {
        "schema": SCHEDULE_SCHEMA,
        "discontinuities": sorted(int(cut) for cut in planned.schedule.discontinuities),
        "decisions": [
            {
                "frame_index": int(item.frame_index),
                "object_id": str(item.object_id),
                "action": item.action.value,
                "anchor": item.anchor,
                "target": item.target,
            }
            for item in planned.schedule.decisions
        ],
        "perception": {
            str(object_id): [int(index) for index in frames]
            for object_id, frames in sorted(planned.perception.items())
        },
        "scene_motion": planned.scene_motion,
        "adapted_threshold": planned.adapted_threshold,
    }


def load_schedule(data: Mapping[str, Any]) -> PlannedSchedule:
    """Rebuild the planned schedule from ``dump_schedule`` output.

    Raises:
        ValueError: If the mapping is not a schedule of this schema.
    """
    schema = data.get("schema")
    if schema != SCHEDULE_SCHEMA:
        raise ValueError(
            f"Unsupported temporal-schedule schema {schema!r}; "
            f"expected {SCHEDULE_SCHEMA!r}."
        )
    discontinuities = frozenset(int(cut) for cut in data.get("discontinuities", ()))
    decisions = tuple(
        FrameDecision(
            frame_index=int(item["frame_index"]),
            object_id=str(item["object_id"]),
            action=FrameAction(item["action"]),
            anchor=item.get("anchor"),
            target=item.get("target"),
        )
        for item in data.get("decisions", ())
    )
    perception = {
        str(object_id): tuple(int(index) for index in frames)
        for object_id, frames in dict(data.get("perception", {})).items()
    }
    schedule = TemporalSchedule(decisions=decisions, discontinuities=discontinuities)
    return PlannedSchedule(
        schedule=schedule,
        perception=perception,
        scene_motion=_optional_float(data.get("scene_motion")),
        adapted_threshold=_optional_float(data.get("adapted_threshold")),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)

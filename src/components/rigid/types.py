"""Rigid-object artifacts and payload accounting.

A rigid object has no skeleton. Keypoints are not a motion representation for
a racket or a ball; those classes carry a shape (convex hull, blob) plus,
for a racket, a wrist anchor borrowed from a *player* pose. The component
refuses an explicit keypoints field on the rigid object itself rather than
inventing a pose.

Phase C residual reads ``deferred_to_residual``: those classes were switched
off and must land in the residual. ``cost().byte_count`` is the measured
shape payload. Detection misses (strategy on, nothing found) are *not*
deferred — the residual still contains them, which is a miss, not an ablation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import struct

from src.contracts.objectstream import WireCost


@dataclass(frozen=True)
class ObservedObject:
    """One detection the rigid stage may turn into a shape.

    ``keypoints`` on a rigid class is a config error, not an input. Player
    wrists arrive separately as ``PlayerPose``.
    """

    object_id: str
    object_class: str
    frame_index: int
    bbox: tuple[float, float, float, float] | None = None
    mask: object | None = None
    keypoints: object | None = None


@dataclass(frozen=True)
class PlayerPose:
    """A player's skeleton, used only to anchor a racket to a wrist."""

    object_id: str
    frame_index: int
    keypoints: object
    schema_name: str = "coco-wholebody-133"


@dataclass(frozen=True)
class RigidShape:
    """One object's shape on one frame. No pose vector."""

    object_id: str
    object_class: str
    kind: str
    frame_index: int
    points: tuple[tuple[float, float], ...]
    wrist_anchor: tuple[float, float] | None = None
    radius: float | None = None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "object_id": self.object_id,
            "object_class": self.object_class,
            "kind": self.kind,
            "frame_index": self.frame_index,
            "points": [list(point) for point in self.points],
        }
        if self.wrist_anchor is not None:
            record["wrist_anchor"] = list(self.wrist_anchor)
        if self.radius is not None:
            record["radius"] = self.radius
        return record


def encode_shapes(shapes: Sequence[RigidShape]) -> bytes:
    """Deterministic measured payload for a set of shapes.

    JSON of the records, then a length-prefixed binary packing of the floats
    so two semantically equal payloads compare equal and a missing object
    changes the byte count.
    """
    body = json.dumps(
        [shape.to_record() for shape in shapes],
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    floats: list[float] = []
    for shape in shapes:
        for x, y in shape.points:
            floats.extend((x, y))
        if shape.wrist_anchor is not None:
            floats.extend(shape.wrist_anchor)
        if shape.radius is not None:
            floats.append(shape.radius)
    packed = struct.pack(f"<{len(floats)}d", *floats) if floats else b""
    return body + packed


@dataclass(frozen=True)
class RigidPayload:
    """What the rigid stage emits for one span, including the off-corner."""

    shapes: tuple[RigidShape, ...] = ()
    enabled_classes: frozenset[str] = field(default_factory=frozenset)
    deferred_to_residual: frozenset[str] = field(default_factory=frozenset)
    payload: bytes = b""
    backend: str = "none"

    @property
    def artifact_counts(self) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for shape in self.shapes:
            counts[shape.object_class] = counts.get(shape.object_class, 0) + 1
        return counts

    def cost(self) -> WireCost:
        if not self.enabled_classes:
            return WireCost(
                values=0,
                byte_count=0,
                exact=True,
                basis="rigid objects off; residual carries racket and ball",
            )
        return WireCost(
            values=sum(len(shape.points) * 2 for shape in self.shapes),
            byte_count=len(self.payload),
            exact=True,
            basis=(
                f"rigid {self.backend}: {len(self.shapes)} shapes, "
                f"deferred={sorted(self.deferred_to_residual) or 'none'}, measured"
            ),
        )


def payload_from_shapes(
    shapes: Sequence[RigidShape],
    *,
    enabled_classes: frozenset[str],
    deferred_to_residual: frozenset[str],
    backend: str,
) -> RigidPayload:
    encoded = encode_shapes(shapes) if shapes else b""
    return RigidPayload(
        shapes=tuple(shapes),
        enabled_classes=enabled_classes,
        deferred_to_residual=deferred_to_residual,
        payload=encoded,
        backend=backend,
    )

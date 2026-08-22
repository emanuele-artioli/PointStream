"""Keypoints on the wire: canonical internally, consumer schema transmitted.

Canonical for humans is COCO-WholeBody-133. Sending those 133 joints to a
conditioner that reads OpenPose-18 is wasted payload. :func:`to_wire` projects
down and marks joints the source cannot supply as not-present rather than
filling them with zeros that look like a detection at the origin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.contracts.keypoints import (
    CANONICAL_HUMAN,
    COCO_17,
    KeypointSchema,
    Projection,
    project,
    schema as lookup_schema,
)

#: Confidence at or below this is treated as absent, never as a joint at origin.
ABSENT_CONFIDENCE = 0.0


@dataclass(frozen=True)
class Pose:
    """Keypoints under a declared schema.

    ``values`` is ``(N, 3)`` — x, y, confidence — in schema index order.
    ``present`` is ``(N,)`` bool; a False entry must not be read as a coordinate.
    """

    schema: KeypointSchema
    values: np.ndarray
    present: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        present = np.asarray(self.present, dtype=bool)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "present", present)
        n = len(self.schema)
        if values.shape != (n, 3):
            raise ValueError(
                f"Pose under {self.schema.name!r} needs values shape {(n, 3)}, "
                f"got {tuple(values.shape)}."
            )
        if present.shape != (n,):
            raise ValueError(
                f"Pose under {self.schema.name!r} needs present shape {(n,)}, "
                f"got {tuple(present.shape)}."
            )


def to_canonical(values: np.ndarray, source: KeypointSchema | str) -> Pose:
    """Lift an estimator's output into the canonical human schema.

    Joints the estimator does not produce are marked absent. Zeros in those
    slots are the storage default, not a detection.
    """
    source_schema = lookup_schema(source) if isinstance(source, str) else source
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"expected (N, 3) keypoints, got {array.shape}")
    if array.shape[0] != len(source_schema):
        raise ValueError(
            f"source schema {source_schema.name!r} has {len(source_schema)} joints, "
            f"got {array.shape[0]}."
        )
    source_present = array[:, 2] > ABSENT_CONFIDENCE
    source_pose = Pose(schema=source_schema, values=array[:, :3], present=source_present)
    return apply_projection(source_pose, project(source_schema, CANONICAL_HUMAN))


def to_wire(pose: Pose, consumer: KeypointSchema | str) -> Pose:
    """Project canonical (or any) keypoints into what the generator consumes."""
    target = lookup_schema(consumer) if isinstance(consumer, str) else consumer
    if pose.schema.name == target.name:
        return pose
    return apply_projection(pose, project(pose.schema, target))


def wire_schema(consumer: KeypointSchema | str) -> KeypointSchema:
    """The schema that goes on the wire for a given consumer.

    Distinct from the canonical internal schema on purpose.
    """
    return lookup_schema(consumer) if isinstance(consumer, str) else consumer


def apply_projection(pose: Pose, projection: Projection) -> Pose:
    """Rewrite `pose` into ``projection.target``.

    Direct joints copy when present. Derived joints (OpenPose ``neck``) are the
    midpoint of their parents only when every parent is present. Absent target
    joints stay at the origin with ``present=False`` and confidence 0.
    """
    n = len(projection.target)
    values = np.zeros((n, 3), dtype=np.float32)
    present = np.zeros(n, dtype=bool)
    source = pose.values
    source_present = pose.present

    for target_idx, source_idx in projection.direct.items():
        if source_present[source_idx]:
            values[target_idx] = source[source_idx]
            present[target_idx] = True

    for target_idx, parents in projection.derived.items():
        if all(source_present[parent] for parent in parents):
            xy = source[list(parents), :2].mean(axis=0)
            conf = float(source[list(parents), 2].min())
            values[target_idx] = np.array([xy[0], xy[1], conf], dtype=np.float32)
            present[target_idx] = True

    return Pose(schema=projection.target, values=values, present=present)


def from_coco17(values: np.ndarray) -> Pose:
    """YOLO-pose emits COCO-17; store it as canonical WholeBody with feet/face/hands absent."""
    return to_canonical(values, COCO_17)

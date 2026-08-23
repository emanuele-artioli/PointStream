"""Keypoint motion: a per-frame pose vector under a declared schema."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.contracts.capabilities import MOTION_KEYPOINTS
from src.contracts.keypoints import OPENPOSE_18, KeypointSchema
from src.contracts.objectstream import KeypointMotion


class KeypointMotionEncoder:
    """Pack ``(N, values)`` keypoints. ``N`` must match the wire schema."""

    kind = MOTION_KEYPOINTS

    def __init__(
        self,
        schema: KeypointSchema | None = None,
        values_per_joint: int = 3,
        bytes_per_value: int = 2,
    ) -> None:
        self.schema = schema or OPENPOSE_18
        self.values_per_joint = values_per_joint
        self.bytes_per_value = bytes_per_value

    def encode(self, keypoints: Any) -> tuple[KeypointMotion, bytes]:
        points = np.asarray(keypoints)
        if points.ndim != 2:
            raise ValueError(
                f"keypoints encode expected (N, {self.values_per_joint}), got {tuple(points.shape)}."
            )
        if points.shape[0] != len(self.schema):
            raise ValueError(
                f"keypoints encode got {points.shape[0]} joints, schema {self.schema.name} "
                f"declares {len(self.schema)}."
            )
        if points.shape[1] < self.values_per_joint:
            raise ValueError(
                f"keypoints encode needs {self.values_per_joint} values per joint, "
                f"got {points.shape[1]}."
            )
        packed = np.ascontiguousarray(
            points[:, : self.values_per_joint].astype(np.float16)
        )
        descriptor = KeypointMotion(
            schema=self.schema,
            values_per_joint=self.values_per_joint,
            bytes_per_value=self.bytes_per_value,
        )
        return descriptor, packed.tobytes()

    def decode(self, payload: bytes) -> np.ndarray:
        values = np.frombuffer(payload, dtype=np.float16)
        return values.reshape(len(self.schema), self.values_per_joint).copy()

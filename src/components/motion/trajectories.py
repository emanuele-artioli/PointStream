"""Sparse trajectories. Dense per-pixel flow is refused by name."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.contracts.capabilities import MOTION_ENCODED_VIDEO, MOTION_SPARSE_TRAJECTORIES
from src.contracts.objectstream import MAX_SPARSE_POINTS, SparseTrajectories


class SparseTrajectoryEncoder:
    """A handful of tracked points. ``SparseTrajectories`` enforces the ceiling."""

    kind = MOTION_SPARSE_TRAJECTORIES

    def __init__(self, values_per_point: int = 2, bytes_per_value: int = 2) -> None:
        self.values_per_point = values_per_point
        self.bytes_per_value = bytes_per_value

    def encode(self, points: Any) -> tuple[SparseTrajectories, bytes]:
        array = np.asarray(points)
        if array.ndim == 3 and array.shape[-1] == 2:
            # (H, W, 2) or (T, N, 2) — a dense field is (H, W, 2) with H, W
            # image-sized. A trajectory clip is (T, N, 2) with N small.
            # Anything whose last two spatial extents look like an image is
            # dense flow wearing a sparse name.
            if array.shape[0] * array.shape[1] > MAX_SPARSE_POINTS:
                raise ValueError(
                    f"sparse-trajectories was given a {tuple(array.shape)} array, which "
                    f"is dense flow ({array.shape[0] * array.shape[1]} samples), not a "
                    f"handful of tracked points. Use motion representation "
                    f"{MOTION_ENCODED_VIDEO!r} if a dense per-pixel answer is wanted, "
                    f"or pass (N, {self.values_per_point}) with N <= {MAX_SPARSE_POINTS}."
                )
            # (T, N, 2) with N small: take the last frame's points? No — that
            # silently drops time. Refuse; the caller should pass (N, 2).
            raise ValueError(
                f"sparse-trajectories encode expected (N, {self.values_per_point}), "
                f"got {tuple(array.shape)}. A temporal stack belongs in the payload "
                f"as one (N, V) set per transmitted frame, not as a dense volume."
            )
        if array.ndim != 2:
            raise ValueError(
                f"sparse-trajectories encode expected (N, {self.values_per_point}), "
                f"got {tuple(array.shape)}."
            )
        # Construction raises above MAX_SPARSE_POINTS — that is the contract.
        descriptor = SparseTrajectories(
            point_count=int(array.shape[0]),
            values_per_point=self.values_per_point,
            bytes_per_value=self.bytes_per_value,
        )
        packed = np.ascontiguousarray(array[:, : self.values_per_point].astype(np.float16))
        return descriptor, packed.tobytes()

    def decode(self, payload: bytes, descriptor: SparseTrajectories) -> np.ndarray:
        values = np.frombuffer(payload, dtype=np.float16)
        return values.reshape(descriptor.point_count, descriptor.values_per_point).copy()

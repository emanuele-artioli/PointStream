"""Pass-through selector: every detection is salient.

The general domain names this. Tennis uses the heuristic instead. It is
domain-agnostic on purpose — "every detection treated as salient" is a
lattice choice, not a tennis rule.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.components.detection.types import Detection


class IdentitySelector:
    """Keep every detection, in the order they arrived."""

    def select(
        self,
        detections: Sequence[Detection],
        frame_shape: tuple[int, int] | None = None,
    ) -> list[Detection]:
        _ = frame_shape
        return list(detections)

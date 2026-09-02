"""Deliver an explicitly ineligible scene through the conventional codec.

This is separate from SOURCE_PASSTHROUGH, which deliberately sends raw pixels.
The one-byte route tag is part of the delivered payload, not a free decision.
Automatic eligibility prediction is not implemented here: the caller supplies
the recorded scene route.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.components.codec.measure import TimedRoundtrip, timed_roundtrip
from src.contracts.config import FallbackConfig


@dataclass(frozen=True)
class FallbackDelivery:
    trip: TimedRoundtrip
    routing_header: bytes = b"\x01"

    @property
    def transport_bytes(self) -> int:
        return int(self.trip.size_bytes) + len(self.routing_header)


def deliver_fallback(
    frames: np.ndarray, config: FallbackConfig, *, route: str, fps: float = 24.0
) -> FallbackDelivery:
    if route != "conventional_fallback":
        raise ValueError("fallback delivery requires an explicit conventional_fallback route")
    request = config.encode_request()
    request.validate()
    trip = timed_roundtrip(frames, request=request, fps=fps)
    if trip.size_bytes <= 0 or trip.frames.shape != frames.shape:
        raise ValueError("fallback codec produced an empty or wrong-shape delivery")
    return FallbackDelivery(trip)

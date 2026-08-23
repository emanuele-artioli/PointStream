"""Foreground and background bitrate headroom — the motivating measurement.

Not a PointStream result. These numbers say what a conventional codec spends on
players and on the background, which is the prize *if* a generator works.
"""

from experiments.headroom.measure import (
    BG_ORDERS_OF_MAGNITUDE,
    FG_MODEST,
    FG_STRONG,
    bg_headroom,
    declared_bounds,
    fg_headroom,
)
from experiments.headroom.remove import flat_fill, plate_fill
from experiments.headroom.synthetic import handheld_clip, tennis_clip

__all__ = [
    "BG_ORDERS_OF_MAGNITUDE",
    "FG_MODEST",
    "FG_STRONG",
    "bg_headroom",
    "declared_bounds",
    "fg_headroom",
    "flat_fill",
    "handheld_clip",
    "plate_fill",
    "tennis_clip",
]

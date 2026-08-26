"""Task-facing training helpers.

The stop rule lives here so a diffusion-loss drop cannot keep a hopeless run
alive. Imports from this package are allowed to be legacy: ``src.shared`` is
not yet on the rewrite layer list.
"""

from src.shared.training.stop import (
    DEFAULT_FLOOR_LPIPS,
    DEFAULT_FLOOR_PSNR,
    DEFAULT_NULL_LPIPS,
    MIN_EPOCHS,
    PATIENCE,
    StopBounds,
    StopDecision,
    TaskStopRule,
    write_bounds,
)

__all__ = [
    "DEFAULT_FLOOR_LPIPS",
    "DEFAULT_FLOOR_PSNR",
    "DEFAULT_NULL_LPIPS",
    "MIN_EPOCHS",
    "PATIENCE",
    "StopBounds",
    "StopDecision",
    "TaskStopRule",
    "write_bounds",
]

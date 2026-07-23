"""Backwards-compatible shim for the old single-module GenAI compositor.

The implementation moved to the `src.decoder.compositing` package on
2026-07-22. This path stays because several engines and tests bind to it
directly — `controlnet_engine`, `pix2pix_engine` and `spade4tennis_engine` all
import their base strategy and the pose-rendering helpers from here, and
`synthesis_engine` and `residual_calculator` import `DiffusersCompositor`.

Prefer importing from `src.decoder.compositing` (or one of its submodules) in
new code.
"""

from __future__ import annotations

from src.decoder.compositing import (  # noqa: F401
    AnimateAnyoneStrategy,
    BaseCompositor,
    BaseGenAIStrategy,
    BaselineControlNetStrategy,
    DiffusersCompositor,
    GenAICompositor,
    _render_pose_condition,
    _render_pose_with_racket,
    _require_local_or_optin_weight,
    _resolve_local_weight_path,
    _to_numpy_bgr,
)

__all__ = [
    "AnimateAnyoneStrategy",
    "BaseCompositor",
    "BaseGenAIStrategy",
    "BaselineControlNetStrategy",
    "DiffusersCompositor",
    "GenAICompositor",
]

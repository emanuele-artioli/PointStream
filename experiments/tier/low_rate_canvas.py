"""Activate BP44's canonical canvas. Merging the PR is not enough.

The geometry lives in BP44. This module is the E1 call site: set
``background.canvas='canonical'`` and pass each clip's ``context_id`` into
``run(..., context_ids=)``. If those APIs are missing, refuse rather than
silently keep independent local canvases.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import replace
from typing import Any


_BP44_MISSING = (
    "E1 needs BP44: BackgroundConfig.canvas / background.context_id and "
    "run(..., context_ids=). Merging BP44 does not activate the fix; this "
    "sweep must set canvas='canonical' and pass the loaded context ids."
)


def clip_context_ids(clips: list[Any]) -> tuple[str, ...]:
    """Per-chunk background context, aligned with ``clips``."""
    ids: list[str] = []
    for clip in clips:
        video = getattr(clip, "video", "?")
        scene = getattr(clip, "scene", "?")
        cid = getattr(clip, "context_id", None)
        if cid is None or str(cid).strip() == "":
            raise SystemExit(
                f"{video}/{scene} has no context_id. Long-scene clips must "
                "carry the BP46 context so a canvas reset is not invented."
            )
        ids.append(str(cid))
    return tuple(ids)


def with_canonical_background(
    background: Any,
    *,
    method: str,
    stream_codec: str,
    stream_crf: int,
    context_id: str,
) -> Any:
    """Copy ``background`` onto the offline canonical canvas."""
    if not hasattr(background, "canvas") or not hasattr(background, "context_id"):
        raise SystemExit(_BP44_MISSING)
    return replace(
        background,
        method=method,
        stream_codec=stream_codec,
        stream_crf=int(stream_crf),
        keyframe_interval=0,
        reference_mode="last",
        canvas="canonical",
        context_id=str(context_id),
    )


def require_run_accepts_context_ids(run_fn: Callable[..., Any] | None = None) -> Callable[..., Any]:
    """The runner must take ``context_ids``. A missing kwarg is a silent canvas skip."""
    target: Callable[..., Any]
    if run_fn is None:
        from src.runner import run as target
    else:
        target = run_fn
    if "context_ids" not in inspect.signature(target).parameters:
        raise SystemExit(_BP44_MISSING)
    return target


__all__ = [
    "clip_context_ids",
    "require_run_accepts_context_ids",
    "with_canonical_background",
]

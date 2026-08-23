"""Array layout helpers. Numpy only — generators import this, the registry does not.

Also hosts the shared letterbox prepare used by ControlNet, pix2pix and SPADE
so the three backends cannot drift on how a crop lands on the canvas.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def as_numpy(value: Any) -> np.ndarray:
    """Detach a tensor-like value to a numpy array, or pass an array through."""
    if isinstance(value, np.ndarray):
        return value
    cpu = getattr(value, "detach", None)
    if callable(cpu):
        value = value.detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = value.cpu()
    as_np = getattr(value, "numpy", None)
    if callable(as_np):
        return np.asarray(as_np())
    return np.asarray(value)


def as_hwc(value: Any) -> np.ndarray:
    """uint8 HWC image. Accepts CHW or HWC."""
    array = as_numpy(value)
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError(f"Expected HW or HWC/CHW image, got shape {tuple(array.shape)}.")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def as_chw(value: Any) -> np.ndarray:
    """uint8 CHW image. Accepts CHW or HWC."""
    array = as_hwc(value)
    if array.ndim == 2:
        return array[np.newaxis, ...]
    return np.transpose(array, (2, 0, 1))


def prepare_letterboxed(
    appearance: Any,
    bbox: tuple[int, int, int, int] | None,
    canvas_width: int,
    canvas_height: int,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Letterbox appearance and named condition images onto one canvas.

    The pre-rewrite ControlNet classes each copied ~40 lines that scaled with
    ``max(canvas) / max(bbox)`` and overflowed whenever the aspects disagreed.
    Every backend that puts a crop on a canvas goes through this helper, which
    uses ``fit_to_canvas`` (min-scale, always inside).
    """
    import cv2

    from src.components.generation.pose import letterbox_from_bbox, letterbox_image

    appearance_hwc = as_hwc(appearance)
    src_h, src_w = appearance_hwc.shape[:2]
    box = letterbox_from_bbox(bbox, src_w, src_h, canvas_width, canvas_height)
    prepared: dict[str, Any] = {
        "appearance": letterbox_image(appearance_hwc, box),
        "letterbox": box,
    }
    if not extras:
        return prepared
    for name, value in extras.items():
        if value is None:
            continue
        image = as_hwc(value)
        if image.shape[:2] != (src_h, src_w):
            interp = cv2.INTER_NEAREST if image.ndim == 2 else cv2.INTER_LINEAR
            image = cv2.resize(image, (src_w, src_h), interpolation=interp)
        prepared[name] = letterbox_image(image, box)
    return prepared

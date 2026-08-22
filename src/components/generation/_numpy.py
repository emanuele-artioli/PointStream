"""Array layout helpers. Numpy only — generators import this, the registry does not."""

from __future__ import annotations

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

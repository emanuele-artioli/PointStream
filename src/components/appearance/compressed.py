"""Compressed-image appearance: JPEG quality and downscale are independent knobs."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.contracts.capabilities import APPEARANCE_COMPRESSED_IMAGE
from src.contracts.objectstream import CompressedImage

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


def resolve_downscale(value: int | float) -> float:
    """Map a config divisor or a linear factor onto CompressedImage's (0, 1] scale.

    ``AppearanceConfig.downscale`` is an int divisor (1 = full, 2 = half).
    ``CompressedImage.downscale`` is a linear factor in (0, 1]. Both spellings
    are accepted here so the two knobs stay independently settable.
    """
    if isinstance(value, bool):
        raise ValueError("downscale cannot be a boolean.")
    if isinstance(value, int) and value >= 1:
        return 1.0 / float(value)
    factor = float(value)
    if not 0.0 < factor <= 1.0:
        raise ValueError(
            f"downscale must be an int divisor >= 1 or a factor in (0, 1], got {value!r}."
        )
    return factor


def as_hwc(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(f"compressed-image encode expected HWC or CHW, got {tuple(array.shape)}.")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


class CompressedImageAppearance:
    """JPEG-encode a crop. Quality quantises; downscale drops resolution."""

    kind = APPEARANCE_COMPRESSED_IMAGE

    def __init__(self, quality: int = 90, downscale: int | float = 1) -> None:
        self.quality = quality
        self.downscale = resolve_downscale(downscale)

    def encode(self, image: Any, *, quality: int | None = None, downscale: int | float | None = None) -> tuple[CompressedImage, bytes]:
        if cv2 is None:
            raise RuntimeError("opencv is required to JPEG-encode an appearance crop.")
        crop = as_hwc(image)
        src_h, src_w = crop.shape[:2]
        factor = self.downscale if downscale is None else resolve_downscale(downscale)
        q = self.quality if quality is None else int(quality)
        sent_w = max(1, round(src_w * factor))
        sent_h = max(1, round(src_h * factor))
        if (sent_w, sent_h) != (src_w, src_h):
            crop = cv2.resize(crop, (sent_w, sent_h), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("cv2.imencode failed to produce a JPEG appearance.")
        payload = encoded.tobytes()
        descriptor = CompressedImage(
            width=src_w,
            height=src_h,
            quality=q,
            downscale=factor,
            measured_bytes=len(payload),
        )
        return descriptor, payload

    def decode(self, payload: bytes) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("opencv is required to decode a JPEG appearance.")
        array = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("JPEG appearance payload did not decode.")
        return image

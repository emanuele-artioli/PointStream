"""Reversible crop, resize, padding, and coordinate conversion primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CropTransform:
    """A source-frame crop placed into a fixed-size, aspect-preserving canvas."""

    source_width: int
    source_height: int
    crop_x0: int
    crop_y0: int
    crop_x1: int
    crop_y1: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int

    def __post_init__(self) -> None:
        if min(self.source_width, self.source_height, self.target_width, self.target_height) <= 0:
            raise ValueError("source and target dimensions must be positive")
        if not (0 <= self.crop_x0 < self.crop_x1 <= self.source_width):
            raise ValueError("crop x bounds are outside the source frame")
        if not (0 <= self.crop_y0 < self.crop_y1 <= self.source_height):
            raise ValueError("crop y bounds are outside the source frame")
        if not (0 < self.resized_width <= self.target_width):
            raise ValueError("resized width must fit the target canvas")
        if not (0 < self.resized_height <= self.target_height):
            raise ValueError("resized height must fit the target canvas")
        if self.pad_left < 0 or self.pad_top < 0:
            raise ValueError("padding offsets must be non-negative")
        if self.pad_left + self.resized_width > self.target_width:
            raise ValueError("horizontal padding and image exceed the target canvas")
        if self.pad_top + self.resized_height > self.target_height:
            raise ValueError("vertical padding and image exceed the target canvas")

    @property
    def scale_x(self) -> float:
        return self.resized_width / (self.crop_x1 - self.crop_x0)

    @property
    def scale_y(self) -> float:
        return self.resized_height / (self.crop_y1 - self.crop_y0)

    def source_to_canvas(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1] != 2:
            raise ValueError("points must end in an x,y pair")
        out = values.copy()
        out[..., 0] = (out[..., 0] - self.crop_x0) * self.scale_x + self.pad_left
        out[..., 1] = (out[..., 1] - self.crop_y0) * self.scale_y + self.pad_top
        return out

    def canvas_to_source(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1] != 2:
            raise ValueError("points must end in an x,y pair")
        out = values.copy()
        out[..., 0] = (out[..., 0] - self.pad_left) / self.scale_x + self.crop_x0
        out[..., 1] = (out[..., 1] - self.pad_top) / self.scale_y + self.crop_y0
        return out

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": "crop_resize_pad_v1",
            "source_size": [self.source_width, self.source_height],
            "crop_xyxy_half_open": [self.crop_x0, self.crop_y0, self.crop_x1, self.crop_y1],
            "target_size": [self.target_width, self.target_height],
            "resized_size": [self.resized_width, self.resized_height],
            "pad_left_top": [self.pad_left, self.pad_top],
            "scale_xy": [self.scale_x, self.scale_y],
        }

    def resize_pad(
        self,
        image: np.ndarray,
        *,
        interpolation: int = cv2.INTER_LINEAR,
        value: int | tuple[int, ...] = 0,
    ) -> np.ndarray:
        array = np.asarray(image)
        if array.shape[:2] != (self.source_height, self.source_width):
            raise ValueError("image shape does not match transform source dimensions")
        crop = array[self.crop_y0 : self.crop_y1, self.crop_x0 : self.crop_x1]
        resized = cv2.resize(crop, (self.resized_width, self.resized_height), interpolation=interpolation)
        canvas_shape = (self.target_height, self.target_width, *array.shape[2:])
        canvas = np.full(canvas_shape, value, dtype=array.dtype)
        y1 = self.pad_top + self.resized_height
        x1 = self.pad_left + self.resized_width
        canvas[self.pad_top : y1, self.pad_left : x1] = resized
        return canvas

    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        array = np.asarray(mask)
        if array.shape != (self.source_height, self.source_width):
            raise ValueError("mask shape does not match transform source dimensions")
        return self.resize_pad((array != 0).astype(np.uint8), interpolation=cv2.INTER_NEAREST)

    def restore_mask(self, canvas_mask: np.ndarray) -> np.ndarray:
        array = np.asarray(canvas_mask)
        if array.shape != (self.target_height, self.target_width):
            raise ValueError("canvas mask shape does not match transform target dimensions")
        out = np.zeros((self.source_height, self.source_width), dtype=np.uint8)
        x0, y0 = self.pad_left, self.pad_top
        x1, y1 = x0 + self.resized_width, y0 + self.resized_height
        resized = cv2.resize(
            (array[y0:y1, x0:x1] != 0).astype(np.uint8),
            (self.crop_x1 - self.crop_x0, self.crop_y1 - self.crop_y0),
            interpolation=cv2.INTER_NEAREST,
        )
        out[self.crop_y0 : self.crop_y1, self.crop_x0 : self.crop_x1] = resized
        return out


def make_crop_transform(
    bbox: tuple[float, float, float, float],
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> CropTransform:
    """Build a reversible crop transform from XYXY boxes and (W,H) sizes."""
    source_width, source_height = source_size
    target_width, target_height = target_size
    x0 = max(0, min(source_width, int(np.floor(bbox[0]))))
    y0 = max(0, min(source_height, int(np.floor(bbox[1]))))
    x1 = max(0, min(source_width, int(np.ceil(bbox[2]))))
    y1 = max(0, min(source_height, int(np.ceil(bbox[3]))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"degenerate crop bounds {bbox!r} in {source_size!r}")
    scale = min(target_width / (x1 - x0), target_height / (y1 - y0))
    resized_width = max(1, min(target_width, int(round((x1 - x0) * scale))))
    resized_height = max(1, min(target_height, int(round((y1 - y0) * scale))))
    return CropTransform(
        source_width=source_width,
        source_height=source_height,
        crop_x0=x0,
        crop_y0=y0,
        crop_x1=x1,
        crop_y1=y1,
        target_width=target_width,
        target_height=target_height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=(target_width - resized_width) // 2,
        pad_top=(target_height - resized_height) // 2,
    )


def crop_and_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    transform: CropTransform,
    *,
    value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Blacken background pixels, then apply the shared crop/resize transform."""
    image = np.asarray(frame)
    binary = np.asarray(mask) != 0
    if image.shape[:2] != (transform.source_height, transform.source_width):
        raise ValueError("frame shape does not match transform source dimensions")
    if binary.shape != image.shape[:2]:
        raise ValueError("mask shape must match frame height and width")
    masked = np.where(binary[..., None], image, value).astype(image.dtype, copy=False)
    crop = transform.resize_pad(masked, interpolation=cv2.INTER_LINEAR, value=value)
    resized_mask = transform.resize_mask(binary.astype(np.uint8))
    crop = np.where(resized_mask[..., None] != 0, crop, value).astype(crop.dtype, copy=False)
    return crop, resized_mask


def render_object_view(
    frame: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    target_size: tuple[int, int] = (512, 512),
) -> tuple[np.ndarray, np.ndarray, CropTransform]:
    """Render a black-background object crop and mask using the shared transform."""
    image = np.asarray(frame)
    if image.ndim != 3:
        raise ValueError("object view rendering requires an HWC color frame")
    height, width = image.shape[:2]
    transform = make_crop_transform(
        bbox,
        source_size=(width, height),
        target_size=target_size,
    )
    appearance, condition_mask = crop_and_mask(image, mask, transform)
    return appearance, condition_mask, transform


__all__ = ["CropTransform", "crop_and_mask", "make_crop_transform", "render_object_view"]

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path

import cv2
import numpy as np


class BasePanoramaEncoder(ABC):
    """Strategy interface for panorama sidecar image encoding."""

    name: str
    extension: str

    @abstractmethod
    def encode(self, image_bgr: np.ndarray, output_stem: Path) -> Path:
        """Encode a BGR panorama image to disk and return the encoded path."""
        raise NotImplementedError


def _ensure_bgr_uint8(image_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Invalid panorama image shape: "
            f"expected [H, W, 3], got {tuple(image.shape)}"
        )
    return image


def _parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got '{value}'") from exc


class JpegPanoramaEncoder(BasePanoramaEncoder):
    name = "jpeg"
    extension = ".jpg"

    def __init__(self, quality: int = 90) -> None:
        self._quality = int(np.clip(int(quality), 1, 100))

    def encode(self, image_bgr: np.ndarray, output_stem: Path) -> Path:
        image = _ensure_bgr_uint8(image_bgr)
        output_path = output_stem.with_suffix(self.extension)
        ok = cv2.imwrite(
            str(output_path),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self._quality)],
        )
        if not ok:
            raise RuntimeError(f"Failed to encode JPEG panorama sidecar: {output_path}")
        return output_path


class PngPanoramaEncoder(BasePanoramaEncoder):
    name = "png"
    extension = ".png"

    def __init__(self, compression: int = 3) -> None:
        self._compression = int(np.clip(int(compression), 0, 9))

    def encode(self, image_bgr: np.ndarray, output_stem: Path) -> Path:
        image = _ensure_bgr_uint8(image_bgr)
        output_path = output_stem.with_suffix(self.extension)
        ok = cv2.imwrite(
            str(output_path),
            image,
            [int(cv2.IMWRITE_PNG_COMPRESSION), int(self._compression)],
        )
        if not ok:
            raise RuntimeError(f"Failed to encode PNG panorama sidecar: {output_path}")
        return output_path


def build_panorama_encoder(panorama_encoder: str | BasePanoramaEncoder | None = None) -> BasePanoramaEncoder:
    if isinstance(panorama_encoder, BasePanoramaEncoder):
        return panorama_encoder

    codec = panorama_encoder or os.environ.get("POINTSTREAM_PANORAMA_CODEC", "jpeg")
    normalized = str(codec).strip().lower()

    if normalized in {"jpeg", "jpg"}:
        quality = _parse_int_env("POINTSTREAM_PANORAMA_JPEG_QUALITY", default=90)
        return JpegPanoramaEncoder(quality=quality)

    if normalized == "png":
        compression = _parse_int_env("POINTSTREAM_PANORAMA_PNG_COMPRESSION", default=3)
        return PngPanoramaEncoder(compression=compression)

    raise ValueError(
        f"Unsupported panorama encoder '{codec}'. "
        "Supported values: jpeg, png."
    )

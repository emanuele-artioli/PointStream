from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class BasePanoramaEncoder(ABC):
    """Strategy interface for panorama sidecar image encoding."""

    name: str
    extension: str

    @property
    @abstractmethod
    def codec_id(self) -> str:
        """Identifier capturing codec name and quality/compression settings.

        Two encoders with the same `codec_id` are guaranteed to produce byte-identical
        output for the same input pixels; matching only `name` would miss a quality
        mismatch (e.g. jpeg q50 vs jpeg q90).
        """
        raise NotImplementedError

    @abstractmethod
    def encode_bytes(self, image_bgr: np.ndarray) -> bytes:
        """Encode a BGR panorama image to codec bytes, without touching disk."""
        raise NotImplementedError

    def encode(self, image_bgr: np.ndarray, output_stem: Path) -> Path:
        """Encode a BGR panorama image to disk and return the encoded path."""
        output_path = output_stem.with_suffix(self.extension)
        output_path.write_bytes(self.encode_bytes(image_bgr))
        return output_path


def _ensure_bgr_uint8(image_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Invalid panorama image shape: "
            f"expected [H, W, 3], got {tuple(image.shape)}"
        )
    return image


class JpegPanoramaEncoder(BasePanoramaEncoder):
    name = "jpeg"
    extension = ".jpg"

    def __init__(self, quality: int = 90) -> None:
        self._quality = int(np.clip(int(quality), 1, 100))

    @property
    def codec_id(self) -> str:
        return f"jpeg:{self._quality}"

    def encode_bytes(self, image_bgr: np.ndarray) -> bytes:
        image = _ensure_bgr_uint8(image_bgr)
        ok, buffer = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self._quality)],
        )
        if not ok:
            raise RuntimeError("Failed to encode JPEG panorama bytes")
        return bytes(buffer)


class PngPanoramaEncoder(BasePanoramaEncoder):
    name = "png"
    extension = ".png"

    def __init__(self, compression: int = 3) -> None:
        self._compression = int(np.clip(int(compression), 0, 9))

    @property
    def codec_id(self) -> str:
        return f"png:{self._compression}"

    def encode_bytes(self, image_bgr: np.ndarray) -> bytes:
        image = _ensure_bgr_uint8(image_bgr)
        ok, buffer = cv2.imencode(
            ".png",
            image,
            [int(cv2.IMWRITE_PNG_COMPRESSION), int(self._compression)],
        )
        if not ok:
            raise RuntimeError("Failed to encode PNG panorama bytes")
        return bytes(buffer)


def round_trip_panorama(image_bgr: np.ndarray, encoder: BasePanoramaEncoder) -> tuple[bytes, np.ndarray]:
    """Encode then decode `image_bgr` through `encoder`, returning the sidecar bytes and resulting pixels.

    The encoder pipeline calls this so residuals are computed against the same lossy
    reconstruction the client will decode from the transmitted panorama sidecar, instead
    of against pre-codec pixels the client never sees (Residual Guarantee, CLAUDE.md).
    """
    encoded_bytes = encoder.encode_bytes(image_bgr)
    decoded = cv2.imdecode(np.frombuffer(encoded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise RuntimeError(f"Failed to decode round-tripped panorama bytes for codec '{encoder.name}'")
    return encoded_bytes, np.asarray(decoded, dtype=np.uint8)


def build_panorama_encoder(panorama_encoder: str | BasePanoramaEncoder | None = None, config: Any = None) -> BasePanoramaEncoder:
    if isinstance(panorama_encoder, BasePanoramaEncoder):
        return panorama_encoder

    codec = panorama_encoder or (config.panorama_codec if config else "jpeg")
    normalized = str(codec).strip().lower()

    if normalized in {"jpeg", "jpg"}:
        quality = config.panorama_jpeg_quality if config and config.panorama_jpeg_quality else 90
        return JpegPanoramaEncoder(quality=int(quality))

    if normalized == "png":
        compression = config.panorama_png_compression if config and config.panorama_png_compression is not None else 3
        return PngPanoramaEncoder(compression=int(compression))

    raise ValueError(
        f"Unsupported panorama encoder '{codec}'. "
        "Supported values: jpeg, png."
    )

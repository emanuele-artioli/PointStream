"""Presley-style compact background plate encoding and restoration.

Grounds PointStream's background model in the findings of the Presley project
(/home/itec/emanuele/presley):
1. Background compression is an intentional rate-distortion trade: transmitting
   at 0.5x resolution with an edge-preserving pre-filter frees ~80% of background
   wire while preserving key visual structures (court lines, net posts).
2. Uses GeometryHeader for zero-inference, bit-exact geometric restoration
   on the client via bilinear interpolation.
3. Decouples plate compression from residual calculation: residuals do not
   re-code smoothed background court texture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from src.components.background.scale import (
    TransportScaleError,
    downsample_plate,
    restore_plate,
    unpack_header,
)
from src.components.background.types import BackgroundArtifact

SUPPORTED_FILTERS: Final[frozenset[str]] = frozenset({"none", "bilateral", "gaussian"})
SUPPORTED_CODECS: Final[frozenset[str]] = frozenset({"webp", "jpeg", "av1", "vvc"})


@dataclass(frozen=True)
class PresleyPlateConfig:
    """Configuration for Presley-style compact plate encoding.

    Args:
        scale: Resolution scale factor (must be 0.5 or 1.0).
        filter_type: Edge-preserving filter ('bilateral', 'gaussian', 'none').
        filter_radius: Neighborhood diameter for filtering.
        filter_sigma_color: Filter sigma in the color space.
        filter_sigma_space: Filter sigma in the coordinate space.
        codec: Image/intra codec ('webp', 'jpeg', 'av1', 'vvc').
        quality: Quality factor (1-100 for webp/jpeg) or QP for av1/vvc.
    """

    scale: float = 0.5
    filter_type: str = "bilateral"
    filter_radius: int = 5
    filter_sigma_color: float = 25.0
    filter_sigma_space: float = 25.0
    codec: str = "webp"
    quality: int = 40


class PresleyPlateEncoder:
    """Encodes a background canvas into a compact, pre-filtered sidecar artifact."""

    def __init__(self, config: PresleyPlateConfig | None = None) -> None:
        """Initialize encoder with configuration."""
        self._config = config if config is not None else PresleyPlateConfig()
        if self._config.scale not in (0.5, 1.0):
            raise ValueError(
                f"unsupported scale={self._config.scale}; must be 0.5 or 1.0"
            )
        if self._config.filter_type not in SUPPORTED_FILTERS:
            raise ValueError(
                f"unsupported filter={self._config.filter_type!r}; "
                f"allowed filters: {sorted(SUPPORTED_FILTERS)}"
            )
        if self._config.codec not in SUPPORTED_CODECS:
            raise ValueError(
                f"unsupported codec={self._config.codec!r}; "
                f"allowed codecs: {sorted(SUPPORTED_CODECS)}"
            )

    @property
    def config(self) -> PresleyPlateConfig:
        """Active plate configuration."""
        return self._config

    def pre_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply configured spatial pre-filtering to suppress noise while preserving edges.

        Args:
            image: uint8 BGR array of shape (H, W, 3).

        Returns:
            Pre-filtered uint8 BGR array of identical shape.
        """
        filter_type = self._config.filter_type
        if filter_type == "none":
            return image.copy()
        if filter_type == "bilateral":
            return cv2.bilateralFilter(
                image,
                d=int(self._config.filter_radius),
                sigmaColor=float(self._config.filter_sigma_color),
                sigmaSpace=float(self._config.filter_sigma_space),
            )
        if filter_type == "gaussian":
            ksize = int(self._config.filter_radius)
            if ksize % 2 == 0:
                ksize += 1
            return cv2.GaussianBlur(
                image,
                (ksize, ksize),
                sigmaX=float(self._config.filter_sigma_space),
            )
        raise ValueError(f"unsupported filter: {filter_type}")

    def encode(
        self,
        plate: np.ndarray,
        scene_id: str = "",
        chunk_id: str = "",
    ) -> BackgroundArtifact:
        """Downsample, pre-filter, and encode the background canvas.

        Args:
            plate: Full-resolution canvas (H, W, 3) uint8 BGR.
            scene_id: Identifying string for the video scene.
            chunk_id: Identifying string for the GOP or chunk.

        Returns:
            BackgroundArtifact with packed GeometryHeader and compressed payload.
            Wire budget invariant: len(artifact.payload) <= 6.0 kB at scale=0.5.
        """
        downsampled, header = downsample_plate(plate, self._config.scale)
        filtered = self.pre_filter(downsampled)

        codec = self._config.codec
        quality = int(self._config.quality)
        if codec == "webp":
            ext = ".webp"
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]
        elif codec == "jpeg":
            ext = ".jpg"
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        else:
            raise ValueError(f"unsupported codec for encoding: {codec}")

        success, encoded = cv2.imencode(ext, filtered, params)
        if not success:
            raise RuntimeError(f"failed to encode plate with codec {codec}")

        payload = encoded.tobytes()

        return BackgroundArtifact(
            method="presley-compact",
            codec=codec,
            codec_id=f"{codec}:q{quality}",
            mode="full",
            payload=payload,
            width=header.original_width,
            height=header.original_height,
            scene_id=scene_id if scene_id else None,
            chunk_id=chunk_id,
            geometry_header=header.pack(),
        )


class PresleyPlateDecoder:
    """Decodes a compact sidecar artifact and restores the full-resolution canvas."""

    def __init__(self) -> None:
        """Initialize decoder."""

    def decode(self, artifact: BackgroundArtifact) -> np.ndarray:
        """Decode sidecar bytes and restore full original geometry.

        Args:
            artifact: BackgroundArtifact containing payload and geometry_header.

        Returns:
            Restored full-resolution canvas (original_H, original_W, 3) uint8 BGR.
            Raises ValueError if payload is corrupted or geometry_header is invalid.
        """
        if not artifact.payload:
            raise ValueError("empty artifact payload")

        buf = np.frombuffer(artifact.payload, dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("failed to decode image payload: corrupted bitstream")

        if not artifact.geometry_header:
            raise ValueError("missing geometry_header")

        try:
            header = unpack_header(artifact.geometry_header)
        except TransportScaleError as exc:
            raise ValueError("invalid geometry_header") from exc

        return restore_plate(decoded, header)

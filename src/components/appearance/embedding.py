"""Image-embedding appearance: a compact colour/texture vector, not CLIP until wired."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.contracts.capabilities import APPEARANCE_IMAGE_EMBEDDING
from src.contracts.objectstream import ImageEmbedding

_GRID = 3
_STATS = 6  # mean RGB + std RGB
_HIST_BINS = 16


def embedding_dimensions() -> int:
    """Grid colour stats plus a coarse histogram. Stated exactly, not guessed."""
    return _GRID * _GRID * _STATS + _HIST_BINS * 3


class ImageEmbeddingAppearance:
    """Deterministic appearance vector. Different crops produce different vectors.

    CLIP/IP-Adapter weights are not loaded here: the pairing and the cost are
    what this representation has to get right first. A CLIP encode is an
    integration-test swap once a checkpoint is present.
    """

    kind = APPEARANCE_IMAGE_EMBEDDING

    def __init__(self, bytes_per_value: int = 2, tokens: int = 1) -> None:
        self.bytes_per_value = bytes_per_value
        self.tokens = tokens
        self.dimensions = embedding_dimensions()

    def encode(self, image: Any) -> tuple[ImageEmbedding, bytes]:
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.ndim != 3:
            raise ValueError(f"image-embedding encode expected an image, got {tuple(array.shape)}.")
        rgb = array[:, :, :3].astype(np.float32)
        height, width, _ = rgb.shape
        cell_h = max(1, height // _GRID)
        cell_w = max(1, width // _GRID)
        stats: list[float] = []
        for row in range(_GRID):
            for col in range(_GRID):
                cell = rgb[
                    row * cell_h : (row + 1) * cell_h,
                    col * cell_w : (col + 1) * cell_w,
                ]
                if cell.size == 0:
                    stats.extend([0.0] * _STATS)
                    continue
                mean = cell.reshape(-1, 3).mean(axis=0)
                std = cell.reshape(-1, 3).std(axis=0)
                stats.extend(mean.tolist())
                stats.extend(std.tolist())
        hist: list[float] = []
        for channel in range(3):
            counts, _ = np.histogram(rgb[:, :, channel], bins=_HIST_BINS, range=(0.0, 255.0))
            total = max(1, int(counts.sum()))
            hist.extend((counts.astype(np.float32) / total).tolist())
        vector = np.asarray(stats + hist, dtype=np.float16)
        if vector.size != self.dimensions:
            raise RuntimeError(
                f"image-embedding packed {vector.size} values, declared {self.dimensions}."
            )
        payload = np.ascontiguousarray(vector).tobytes()
        descriptor = ImageEmbedding(
            dimensions=self.dimensions,
            tokens=self.tokens,
            bytes_per_value=self.bytes_per_value,
        )
        return descriptor, payload

    def decode(self, payload: bytes) -> np.ndarray:
        """Embeddings are not invertible. Return the vector, not an image."""
        return np.frombuffer(payload, dtype=np.float16).copy()

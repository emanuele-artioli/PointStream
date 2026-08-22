"""Diffusion-latent appearance. Spatial pack, not a VAE, until one is wired."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.contracts.capabilities import APPEARANCE_DIFFUSION_LATENT
from src.contracts.objectstream import DiffusionLatent

_LATENT_STRIDE = 8
_LATENT_CHANNELS = 4


def _as_hwc(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3:
        raise ValueError(f"diffusion-latent encode expected an image, got {tuple(array.shape)}.")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array[:, :, :3]


class DiffusionLatentAppearance:
    """Pack a crop into a 4 x H/8 x W/8 float16 grid.

    This is a wire representation with an exact, stated cost. It is **not** a
    Stable Diffusion VAE encode — loading one is an integration-test concern
    once a licensed checkpoint is present. The layout matches what a VAE
    consumer expects so swapping the encoder later does not change the pairing.
    """

    kind = APPEARANCE_DIFFUSION_LATENT

    def __init__(self, bytes_per_value: int = 2) -> None:
        self.bytes_per_value = bytes_per_value

    def encode(self, image: Any) -> tuple[DiffusionLatent, bytes]:
        crop = _as_hwc(image)
        height, width = crop.shape[:2]
        grid_h = max(1, height // _LATENT_STRIDE)
        grid_w = max(1, width // _LATENT_STRIDE)
        # Area-average each stride cell into RGB, plus a luma channel.
        crop_f = crop.astype(np.float32) / 255.0
        trimmed = crop_f[: grid_h * _LATENT_STRIDE, : grid_w * _LATENT_STRIDE]
        blocks = trimmed.reshape(grid_h, _LATENT_STRIDE, grid_w, _LATENT_STRIDE, 3)
        rgb = blocks.mean(axis=(1, 3))
        luma = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])[..., np.newaxis]
        latent = np.concatenate([rgb, luma], axis=2)  # H W 4
        latent_chw = np.transpose(latent, (2, 0, 1)).astype(np.float16)
        payload = np.ascontiguousarray(latent_chw).tobytes()
        descriptor = DiffusionLatent(
            channels=_LATENT_CHANNELS,
            height=grid_h,
            width=grid_w,
            bytes_per_value=self.bytes_per_value,
            measured_bytes=len(payload),
        )
        return descriptor, payload

    def decode(self, payload: bytes, descriptor: DiffusionLatent) -> np.ndarray:
        latent = np.frombuffer(payload, dtype=np.float16).reshape(
            descriptor.channels, descriptor.height, descriptor.width
        )
        rgb = np.transpose(latent[:3], (1, 2, 0)).astype(np.float32)
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        scale_h = descriptor.height * _LATENT_STRIDE
        scale_w = descriptor.width * _LATENT_STRIDE
        # Nearest upsample; a VAE decoder would replace this.
        return np.repeat(np.repeat(rgb, _LATENT_STRIDE, axis=0), _LATENT_STRIDE, axis=1)[
            :scale_h, :scale_w
        ]

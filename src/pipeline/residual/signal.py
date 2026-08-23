"""Residual signal: source minus reconstruction, coarsened, applied.

The residual absorbs whatever disabled stages would have handled. If turning
a stage off makes this signal *smaller*, the reconstruction is still carrying
work that stage was supposed to stop doing.

Lossless stores signed int16 so apply-after-compute is bit-identity with the
source. Lossy biases into uint8 with a +128 offset (the representation a
video codec actually encodes) and may drop low-activity blocks and downscale
the background. Absent stores nothing: the reconstruction is unaided.

This module does not invoke a codec. Byte counts here are the pixel payload
handed to one — sparse (nonzero) counts for lossy, dense int16 for lossless —
so a test can see absent vs lossless change the payload without standing up
ffmpeg.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from src.contracts.config import ResidualConfig
from src.contracts.lattice import STAGE_RESIDUAL, StageLattice
from src.contracts.objectstream import WireCost
from src.pipeline.reconstruction.clips import as_clip, require_same_shape
from src.pipeline.residual.spectrum import (
    Coarseness,
    ResidualPoint,
    ResidualVariant,
    point_for,
)

OFFSET = 128


@dataclass(frozen=True)
class ResidualPayload:
    """What the residual stage produced for one clip.

    ``frames`` is None when absent. ``lossy_uint8`` is the +128 representation
    a codec would encode; ``lossless_int16`` is the exact signed difference.
    ``active_blocks`` and ``nonzero_bytes`` are the information content — a
    dense array's ``nbytes`` does not shrink when blocks are zeroed, and
    comparing that would make coarseness look free.
    """

    variant: ResidualVariant
    coarseness: Coarseness
    frames: np.ndarray | None
    byte_count: int
    nonzero_bytes: int
    active_blocks: int
    l1_energy: float
    cost: WireCost

    @property
    def is_absent(self) -> bool:
        return self.variant is ResidualVariant.NONE


@dataclass(frozen=True)
class ResidualResult:
    """Payload plus the clip after the residual is applied."""

    payload: ResidualPayload
    reconstructed: np.ndarray
    """Reconstruction after adding the residual (absent → unchanged)."""


def signed_residual(source: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    """``source - reconstruction`` as int16. Exact; no clipping."""
    src = as_clip(source, path="source").astype(np.int16)
    recon = as_clip(reconstruction, path="reconstruction").astype(np.int16)
    require_same_shape(src, recon, path="residual")
    return src - recon


def l1_energy(residual: np.ndarray) -> float:
    """Sum of absolute residual values. The absorption invariant is on this."""
    return float(np.abs(residual).sum())


def encode_lossy(signed: np.ndarray) -> np.ndarray:
    """Bias into uint8. Differences outside [-128, 127] clip — that is lossy."""
    return np.clip(signed.astype(np.int16) + OFFSET, 0, 255).astype(np.uint8)


def decode_lossy(encoded: np.ndarray) -> np.ndarray:
    return encoded.astype(np.int16) - OFFSET


def apply_signed(reconstruction: np.ndarray, signed: np.ndarray) -> np.ndarray:
    recon = as_clip(reconstruction, path="reconstruction").astype(np.int16)
    require_same_shape(recon, signed, path="apply-residual")
    return np.clip(recon + signed, 0, 255).astype(np.uint8)


def block_activity_gate(
    residual: np.ndarray,
    *,
    block_size: int,
    threshold: float,
) -> np.ndarray:
    """Zero blocks whose mean absolute residual is below ``threshold``.

    Threshold is in pixel units: 2.0 drops blocks whose mean error is below
    two grey levels. ``block_size <= 1`` or ``threshold <= 0`` is a no-op.
    """
    if block_size <= 1 or threshold <= 0.0:
        return residual
    if residual.ndim != 4:
        raise ValueError(f"residual must be (T, H, W, C); got {residual.shape}.")
    frames, height, width, channels = residual.shape
    pad_h = (block_size - (height % block_size)) % block_size
    pad_w = (block_size - (width % block_size)) % block_size
    padded = np.pad(residual, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    padded_h, padded_w = padded.shape[1], padded.shape[2]
    n_h, n_w = padded_h // block_size, padded_w // block_size
    blocks = padded.reshape(frames, n_h, block_size, n_w, block_size, channels)
    activity = np.abs(blocks).mean(axis=(2, 4, 5))
    keep = activity >= float(threshold)
    mask = np.repeat(np.repeat(keep[:, :, None, :, None], block_size, axis=2), block_size, axis=4)
    mask = mask.reshape(frames, padded_h, padded_w)
    gated = padded * mask[..., None]
    return gated[:, :height, :width, :]


def downscale_background(
    residual: np.ndarray,
    actor_mask: np.ndarray | None,
    *,
    factor: int,
) -> np.ndarray:
    """Keep object residual at full resolution; coarsen the background.

    ``factor <= 1`` is a no-op. A mask covering every pixel leaves the
    residual untouched — there is no background to coarsen.
    """
    if factor <= 1:
        return residual
    if residual.ndim != 4:
        raise ValueError(f"residual must be (T, H, W, C); got {residual.shape}.")
    frames, height, width, _channels = residual.shape
    if actor_mask is None:
        object_pixels = np.zeros((frames, height, width), dtype=bool)
    else:
        object_pixels = _align_mask(actor_mask, frames, height, width)
    if bool(np.all(object_pixels)):
        return residual

    down_h = max(1, int(math.ceil(height / float(factor))))
    down_w = max(1, int(math.ceil(width / float(factor))))
    coarsened = np.empty_like(residual)
    for index in range(frames):
        small = cv2.resize(
            residual[index].astype(np.float32),
            (down_w, down_h),
            interpolation=cv2.INTER_AREA,
        )
        coarsened[index] = cv2.resize(
            small, (width, height), interpolation=cv2.INTER_NEAREST
        )
    keep = object_pixels[..., None]
    return np.where(keep, residual, coarsened)


def compute_residual(
    source: np.ndarray,
    reconstruction: np.ndarray,
    *,
    lattice: StageLattice,
    residual: ResidualConfig | None = None,
    actor_mask: np.ndarray | None = None,
    coarseness: Coarseness | None = None,
) -> ResidualResult:
    """Build the residual payload and the clip after applying it.

    Bounds: absent → 0 bytes, reconstruction unchanged. Lossless → apply
    restores the source bit-for-bit when the signed difference fits in int16
    (it always does for uint8 pairs). Lossy → payload information drops as
    the rung coarsens; it must never drop because a *stage* was disabled.
    """
    src = as_clip(source, path="source")
    recon = as_clip(reconstruction, path="reconstruction")
    require_same_shape(src, recon, path="residual")
    point = point_for(lattice, residual, coarseness=coarseness)

    if point.variant is ResidualVariant.NONE or STAGE_RESIDUAL not in lattice.enabled:
        empty = ResidualPayload(
            variant=ResidualVariant.NONE,
            coarseness=Coarseness.ABSENT,
            frames=None,
            byte_count=0,
            nonzero_bytes=0,
            active_blocks=0,
            l1_energy=l1_energy(signed_residual(src, recon)),
            cost=WireCost(
                values=0,
                byte_count=0,
                exact=True,
                basis="residual absent; unaided reconstruction",
            ),
        )
        return ResidualResult(payload=empty, reconstructed=recon.copy())

    signed = signed_residual(src, recon)
    if point.variant is ResidualVariant.LOSSLESS:
        payload = _lossless_payload(signed, point)
        restored = apply_signed(recon, signed)
        return ResidualResult(payload=payload, reconstructed=restored)

    cfg = point.config if point.config is not None else ResidualConfig()
    working = signed.astype(np.float32)
    working = block_activity_gate(
        working, block_size=cfg.block_size, threshold=cfg.block_threshold
    )
    working = downscale_background(working, actor_mask, factor=cfg.background_downscale)
    encoded = encode_lossy(np.rint(working).astype(np.int16))
    payload = _lossy_payload(encoded, working, point, cfg)
    restored = apply_signed(recon, decode_lossy(encoded))
    return ResidualResult(payload=payload, reconstructed=restored)


def apply_residual(reconstruction: np.ndarray, payload: ResidualPayload) -> np.ndarray:
    """Decoder-side: add the residual onto the reconstruction."""
    recon = as_clip(reconstruction, path="reconstruction")
    if payload.is_absent or payload.frames is None:
        return recon.copy()
    if payload.variant is ResidualVariant.LOSSLESS:
        return apply_signed(recon, payload.frames)
    return apply_signed(recon, decode_lossy(payload.frames))


def _lossless_payload(signed: np.ndarray, point: ResidualPoint) -> ResidualPayload:
    stored = np.asarray(signed, dtype=np.int16)
    nonzero = int(np.count_nonzero(stored))
    return ResidualPayload(
        variant=ResidualVariant.LOSSLESS,
        coarseness=Coarseness.LOSSLESS,
        frames=stored,
        byte_count=int(stored.nbytes),
        nonzero_bytes=nonzero * int(stored.dtype.itemsize),
        active_blocks=int(np.count_nonzero(np.any(stored != 0, axis=-1))),
        l1_energy=l1_energy(stored),
        cost=WireCost(
            values=int(stored.size),
            byte_count=int(stored.nbytes),
            exact=True,
            basis="lossless int16 residual, dense",
        ),
    )


def _lossy_payload(
    encoded: np.ndarray,
    working: np.ndarray,
    point: ResidualPoint,
    config: ResidualConfig,
) -> ResidualPayload:
    nonzero = int(np.count_nonzero(encoded != OFFSET))
    block = max(1, config.block_size)
    frames, height, width, _ = encoded.shape
    n_h = int(math.ceil(height / block))
    n_w = int(math.ceil(width / block))
    active = 0
    for t in range(frames):
        for by in range(n_h):
            for bx in range(n_w):
                y1, y2 = by * block, min(height, (by + 1) * block)
                x1, x2 = bx * block, min(width, (bx + 1) * block)
                if np.any(encoded[t, y1:y2, x1:x2] != OFFSET):
                    active += 1
    return ResidualPayload(
        variant=ResidualVariant.LOSSY,
        coarseness=point.coarseness,
        frames=encoded,
        byte_count=int(encoded.nbytes),
        nonzero_bytes=nonzero,
        active_blocks=active,
        l1_energy=l1_energy(working),
        cost=WireCost(
            values=nonzero,
            byte_count=nonzero,
            exact=True,
            basis=(
                f"lossy uint8 residual, {nonzero} nonzero bytes, "
                f"{active} active blocks of {block}"
            ),
        ),
    )


def _align_mask(mask: np.ndarray, frames: int, height: int, width: int) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim == 2:
        if array.shape != (height, width):
            raise ValueError(
                f"actor mask shape {array.shape} does not match frame {(height, width)}."
            )
        return np.broadcast_to(array, (frames, height, width))
    if array.shape != (frames, height, width):
        raise ValueError(
            f"actor mask shape {array.shape} does not match clip {(frames, height, width)}."
        )
    return array

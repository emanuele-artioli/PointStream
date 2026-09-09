"""Residual signal: source minus reconstruction, coarsened, applied.

The residual absorbs whatever disabled stages would have handled. If turning
a stage off makes this signal *smaller*, the reconstruction is still carrying
work that stage was supposed to stop doing.

Lossless stores signed int16 so apply-after-compute is bit-identity with the
source. Lossy biases into uint8 (clipped or full_range representation) and may
drop low-activity blocks and downscale the background. Absent stores nothing:
the reconstruction is unaided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.contracts.config import ResidualConfig
from src.contracts.lattice import STAGE_RESIDUAL, StageLattice
from src.contracts.objectstream import WireCost
from src.pipeline.reconstruction.clips import as_clip, require_same_shape
from src.pipeline.residual.lossy import (
    OFFSET,
    block_activity_gate,
    decode_lossy,
    downscale_background,
    encode_lossy,
    subsample_chroma,
)
from src.pipeline.residual.spectrum import (
    Coarseness,
    ResidualPoint,
    ResidualVariant,
    point_for,
)


@dataclass(frozen=True)
class ResidualPayload:
    """What the residual stage produced for one clip.

    ``frames`` is None when absent. ``lossy_uint8`` is the representation a
    codec would encode; ``lossless_int16`` is the exact signed difference.
    ``active_blocks`` and ``nonzero_bytes`` are the information content.
    """

    variant: ResidualVariant
    coarseness: Coarseness
    frames: np.ndarray | None
    byte_count: int
    nonzero_bytes: int
    active_blocks: int
    l1_energy: float
    cost: WireCost
    mode: str = "clipped"
    scale: float = 1.0
    offset: float = 128.0

    @property
    def is_absent(self) -> bool:
        return self.variant is ResidualVariant.NONE


@dataclass(frozen=True)
class ResidualResult:
    """Payload plus the clip after the residual is applied."""

    payload: ResidualPayload
    reconstructed: np.ndarray
    """Reconstruction after adding the residual (absent → unchanged)."""
    base: np.ndarray | None = None
    """The server predictor P_s before residual was added."""
    transmitted: Any = None
    """The transmitted wire representation (e.g. TransmittedResidual) if coded."""


def signed_residual(source: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    """``source - reconstruction`` as int16. Exact; no clipping."""
    src = as_clip(source, path="source").astype(np.int16)
    recon = as_clip(reconstruction, path="reconstruction").astype(np.int16)
    require_same_shape(src, recon, path="residual")
    return src - recon


def l1_energy(residual: np.ndarray) -> float:
    """Sum of absolute residual values. The absorption invariant is on this."""
    return float(np.abs(residual).sum())


def apply_signed(reconstruction: np.ndarray, signed: np.ndarray) -> np.ndarray:
    recon = as_clip(reconstruction, path="reconstruction").astype(np.int16)
    require_same_shape(recon, signed, path="apply-residual")
    return np.clip(recon + signed, 0, 255).astype(np.uint8)


def compute_residual(
    source: np.ndarray,
    reconstruction: np.ndarray,
    *,
    lattice: StageLattice,
    residual: ResidualConfig | None = None,
    actor_mask: np.ndarray | None = None,
    coarseness: Coarseness | None = None,
    representation: str | None = None,
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
        return ResidualResult(payload=empty, reconstructed=recon.copy(), base=recon.copy())

    signed = signed_residual(src, recon)
    if point.variant is ResidualVariant.LOSSLESS:
        payload = _lossless_payload(signed, point)
        restored = apply_signed(recon, signed)
        return ResidualResult(payload=payload, reconstructed=restored, base=recon.copy())

    cfg = point.config if point.config is not None else ResidualConfig()
    mode = representation or getattr(cfg, "representation", "clipped")
    subsample = getattr(cfg, "subsample_chroma", False)

    working = signed.astype(np.float32)
    working = block_activity_gate(
        working, block_size=cfg.block_size, threshold=cfg.block_threshold
    )
    working = downscale_background(working, actor_mask, factor=cfg.background_downscale)
    if subsample:
        working = subsample_chroma(working)

    encoded = encode_lossy(np.rint(working).astype(np.int16), mode=mode)
    payload = _lossy_payload(encoded, working, point, cfg, mode=mode)
    restored = apply_signed(recon, decode_lossy(encoded, mode=mode))
    return ResidualResult(payload=payload, reconstructed=restored, base=recon.copy())


def apply_residual(reconstruction: np.ndarray, payload: Any) -> np.ndarray:
    """Decoder-side: add the residual onto the reconstruction."""
    recon = as_clip(reconstruction, path="reconstruction")
    if payload is None:
        return recon.copy()

    if isinstance(payload, np.ndarray):
        if payload.dtype == np.int16:
            return apply_signed(recon, payload)
        return apply_signed(recon, decode_lossy(payload))

    if getattr(payload, "is_absent", False):
        return recon.copy()

    # TransmittedResidual from bitstream or unencoded array
    if hasattr(payload, "bitstream") and getattr(payload, "is_coded", False):
        from src.pipeline.residual.codec import decode_residual_stream

        signed_diff = decode_residual_stream(payload)
        return apply_signed(recon, signed_diff)

    if getattr(payload, "variant", None) is ResidualVariant.LOSSLESS:
        if payload.frames is not None:
            return apply_signed(recon, payload.frames)
        return recon.copy()

    frames = getattr(payload, "frames", None)
    if frames is None:
        frames = getattr(payload, "raw_frames", None)
    if frames is None:
        return recon.copy()

    mode = getattr(payload, "mode", "clipped")
    scale = getattr(payload, "scale", 1.0)
    offset = getattr(payload, "offset", 128.0)
    return apply_signed(recon, decode_lossy(frames, mode=mode, scale=scale, offset=offset))


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
            exact=False,
            basis="lossless int16 residual, dense array — pre-codec, not a bitstream",
        ),
        mode="raw_int16",
        scale=1.0,
        offset=0.0,
    )


def _lossy_payload(
    encoded: np.ndarray,
    working: np.ndarray,
    point: ResidualPoint,
    config: ResidualConfig,
    *,
    mode: str = "clipped",
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
            exact=False,
            basis=(
                f"lossy uint8 residual ({mode}), {nonzero} nonzero bytes, "
                f"{active} active blocks of {block} — pre-codec, not a bitstream"
            ),
        ),
        mode=mode,
        scale=1.0,
        offset=float(OFFSET),
    )

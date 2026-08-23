"""Region-of-interest maps, encoder-native formatting, and the pixel-domain arm.

Native delta-QP maps exist for AV1 (``--roi-map-file``, 64x64, q_index units)
and HEVC (kvazaar ``--roi``, signed 8-bit CTU offsets). VVC has no region map;
AVC's ffmpeg ``addroi`` is unverified. Both of those use the in-house arm:
degrade non-salient blocks in the pixel domain, then encode without a map.

AV1 offsets are q_index (0–255) against ``--qp`` (0–63), roughly four per QP
step. Offsets past about −120 / +60 make the regions *converge* and both lose
quality — that is not "stronger ROI", and a table built on it is not a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Final

import numpy as np

from src.contracts.errors import CodecConstraintError

AV1_BLOCK: Final = 64
"""SVT-AV1 applies ROI offsets per 64x64 superblock, row-major."""

HEVC_CTU: Final = 64
"""Kvazaar default CTU size. The file may be any size; kvazaar scales it."""

AV1_MAX_SEGMENTS: Final = 8
"""AV1 spec cap. SvtAv1EncApp rejects a map with more distinct offsets."""

# Inclusive bounds. Past these the two regions converge (measured on SVT-AV1 1.8).
AV1_OFFSET_MIN: Final = -119
AV1_OFFSET_MAX: Final = 59

# qoffset for addroi is in [-1, 1]; x264 QP runs 0–51.
ADDROI_QP_SCALE: Final = 51.0


@dataclass(frozen=True)
class BlockRoiMap:
    """Per-block QP / q_index offsets covering one frame (repeated across time).

    Args:
        block_size: Side of one block in luma pixels.
        blocks_wide: Blocks along x.
        blocks_high: Blocks along y.
        offsets: Row-major per-block offsets. Length is ``blocks_wide * blocks_high``.
        inside_offset: Offset used inside the labelled region, if it is a rectangle.
        outside_offset: Offset used elsewhere, if the map is two-valued.
        col0, col1, row0, row1: Inclusive-exclusive rectangle in block coordinates,
            when the map was built as a centred region. Used by ``addroi``.
    """

    block_size: int
    blocks_wide: int
    blocks_high: int
    offsets: tuple[int, ...]
    inside_offset: int = 0
    outside_offset: int = 0
    col0: int = 0
    col1: int = 0
    row0: int = 0
    row1: int = 0

    def __post_init__(self) -> None:
        expected = self.blocks_wide * self.blocks_high
        if len(self.offsets) != expected:
            raise ValueError(
                f"ROI map has {len(self.offsets)} offsets, expected "
                f"{self.blocks_wide}x{self.blocks_high} = {expected}."
            )
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}.")

    @classmethod
    def centred(
        cls,
        width: int,
        height: int,
        *,
        inside_offset: int,
        outside_offset: int = 0,
        fraction: float = 0.4,
        block_size: int = AV1_BLOCK,
    ) -> BlockRoiMap:
        """A centred rectangle covering roughly ``fraction`` of each axis."""
        blocks_wide = -(-width // block_size)
        blocks_high = -(-height // block_size)
        span_w = max(1, round(blocks_wide * fraction))
        span_h = max(1, round(blocks_high * fraction))
        col0 = (blocks_wide - span_w) // 2
        row0 = (blocks_high - span_h) // 2
        col1 = col0 + span_w
        row1 = row0 + span_h
        values: list[int] = []
        for row in range(blocks_high):
            for col in range(blocks_wide):
                inside = row0 <= row < row1 and col0 <= col < col1
                values.append(inside_offset if inside else outside_offset)
        return cls(
            block_size=block_size,
            blocks_wide=blocks_wide,
            blocks_high=blocks_high,
            offsets=tuple(values),
            inside_offset=inside_offset,
            outside_offset=outside_offset,
            col0=col0,
            col1=col1,
            row0=row0,
            row1=row1,
        )

    def pixel_mask(self, width: int, height: int) -> np.ndarray:
        """Boolean mask, True inside the labelled rectangle, at pixel resolution."""
        mask = np.zeros((height, width), dtype=bool)
        y0 = self.row0 * self.block_size
        y1 = min(self.row1 * self.block_size, height)
        x0 = self.col0 * self.block_size
        x1 = min(self.col1 * self.block_size, width)
        if y1 > y0 and x1 > x0:
            mask[y0:y1, x0:x1] = True
        return mask

    def offset_at(self, row: int, col: int) -> int:
        return self.offsets[row * self.blocks_wide + col]


def write_svtav1(map_: BlockRoiMap, path: Path, frames: int) -> None:
    """SVT-AV1 ``--roi-map-file``: one line per frame, 64x64 blocks, q_index units.

    Raises:
        CodecConstraintError: Offsets outside the range that actually differentiates
            regions, or more than eight distinct values after palette quantisation
            cannot be represented.
    """
    if frames <= 0:
        raise ValueError(f"frames must be positive, got {frames}.")
    offsets = _av1_offsets(map_)
    _assert_av1_offsets_safe(offsets)
    palette = _av1_palette(offsets)
    line = " ".join(str(value) for value in palette)
    path.write_text("".join(f"{index} {line}\n" for index in range(frames)), encoding="ascii")


def write_kvazaar(map_: BlockRoiMap, path: Path, frames: int) -> None:
    """Kvazaar ``--roi`` binary: per frame, two int32 dimensions then int8 deltas."""
    if frames <= 0:
        raise ValueError(f"frames must be positive, got {frames}.")
    cols, rows = map_.blocks_wide, map_.blocks_high
    deltas = np.asarray(map_.offsets, dtype=np.int32)
    if np.any(deltas < -128) or np.any(deltas > 127):
        raise CodecConstraintError(
            "hevc",
            "roi_map offset",
            f"[{int(deltas.min())}, {int(deltas.max())}]",
            ["signed 8-bit values in [-128, 127]"],
        )
    packed = np.round(deltas).astype(np.int8).tobytes()
    parts = [struct.pack("ii", cols, rows) + packed for _ in range(frames)]
    path.write_bytes(b"".join(parts))


def addroi_filter(map_: BlockRoiMap, width: int, height: int) -> str:
    """ffmpeg ``addroi`` filter string for the labelled rectangle, or empty.

    ``qoffset`` is in [-1, 1], negative = better quality. Integer map offsets
    are scaled by ``ADDROI_QP_SCALE``. A zero-offset rectangle is omitted —
    that is the sharper localisation test, not a missing filter.
    """
    if map_.inside_offset == 0 or map_.col1 <= map_.col0 or map_.row1 <= map_.row0:
        return ""
    x = map_.col0 * map_.block_size
    y = map_.row0 * map_.block_size
    w = min(map_.col1 * map_.block_size, width) - x
    h = min(map_.row1 * map_.block_size, height) - y
    if w <= 0 or h <= 0:
        return ""
    qoffset = float(np.clip(map_.inside_offset / ADDROI_QP_SCALE, -1.0, 1.0))
    return f"addroi={x}:{y}:{w}:{h}:{qoffset}"


def degrade_luma(luma: np.ndarray, map_: BlockRoiMap) -> np.ndarray:
    """Quantise blocks whose offset is positive (non-salient / fewer bits).

    Negative offsets mean "spend more" — those blocks are left intact so the
    encoder can. Strength grows with the offset: a +1 step halves the number
    of reconstructible luma levels in that block. This is the VVC (and
    unverified-AVC) ROI arm; it is a pre-process, not an encoder flag.
    """
    if luma.ndim != 2:
        raise ValueError(f"degrade_luma expects (H, W), got {luma.shape}")
    height, width = luma.shape
    out = np.array(luma, dtype=np.uint8, copy=True)
    for row in range(map_.blocks_high):
        for col in range(map_.blocks_wide):
            offset = map_.offset_at(row, col)
            if offset <= 0:
                continue
            y0 = row * map_.block_size
            x0 = col * map_.block_size
            y1 = min(y0 + map_.block_size, height)
            x1 = min(x0 + map_.block_size, width)
            block = out[y0:y1, x0:x1].astype(np.float64)
            mean = float(block.mean())
            # offset 1 is a nudge toward the mean; 16 and above flatten the block.
            alpha = min(1.0, int(offset) / 16.0)
            mixed = (1.0 - alpha) * block + alpha * mean
            out[y0:y1, x0:x1] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)
    return out


def degrade_video(luma: np.ndarray, map_: BlockRoiMap) -> np.ndarray:
    """Apply ``degrade_luma`` to every frame in ``(T, H, W)``."""
    if luma.ndim != 3:
        raise ValueError(f"degrade_video expects (T, H, W), got {luma.shape}")
    return np.stack([degrade_luma(frame, map_) for frame in luma])


def _av1_offsets(map_: BlockRoiMap) -> tuple[int, ...]:
    """Offsets at 64x64. Non-64 maps are nearest-neighbour resampled."""
    if map_.block_size == AV1_BLOCK:
        return map_.offsets
    # Repeat / coarsen so each 64x64 superblock gets the offset of its origin block.
    src = np.asarray(map_.offsets, dtype=np.int32).reshape(map_.blocks_high, map_.blocks_wide)
    pixel_h = map_.blocks_high * map_.block_size
    pixel_w = map_.blocks_wide * map_.block_size
    new_h = -(-pixel_h // AV1_BLOCK)
    new_w = -(-pixel_w // AV1_BLOCK)
    out: list[int] = []
    for row in range(new_h):
        for col in range(new_w):
            y = min((row * AV1_BLOCK) // map_.block_size, map_.blocks_high - 1)
            x = min((col * AV1_BLOCK) // map_.block_size, map_.blocks_wide - 1)
            out.append(int(src[y, x]))
    return tuple(out)


def _assert_av1_offsets_safe(offsets: tuple[int, ...]) -> None:
    lo, hi = min(offsets), max(offsets)
    if lo < AV1_OFFSET_MIN or hi > AV1_OFFSET_MAX:
        raise CodecConstraintError(
            "av1",
            "roi_map offset (q_index)",
            f"[{lo}, {hi}]",
            [
                f"[{AV1_OFFSET_MIN}, {AV1_OFFSET_MAX}] — larger offsets make "
                "the regions converge and both lose quality, they are not stronger"
            ],
        )


def _av1_palette(offsets: tuple[int, ...]) -> tuple[int, ...]:
    """At most eight distinct values. Prefer a palette that still contains 0."""
    unique = tuple(sorted(set(offsets)))
    if len(unique) <= AV1_MAX_SEGMENTS:
        return offsets
    lo, hi = min(offsets), max(offsets)
    # 7 levels, not 8, so 0 survives when the range straddles it — 8 levels
    # push a near-zero offset to the smallest nonzero and shift everything.
    n_levels = AV1_MAX_SEGMENTS - 1
    levels = np.unique(np.round(np.linspace(lo, hi, n_levels)).astype(int))
    if lo < 0 < hi and 0 not in levels:
        levels = np.unique(np.append(levels, 0))
        if len(levels) > AV1_MAX_SEGMENTS:
            # Drop the level closest to 0 that is not 0 itself.
            others = [int(v) for v in levels if v != 0]
            drop = min(others, key=abs)
            levels = np.array([int(v) for v in levels if v != drop])
    snapped = [int(levels[np.abs(levels - value).argmin()]) for value in offsets]
    if len(set(snapped)) > AV1_MAX_SEGMENTS:
        raise CodecConstraintError(
            "av1",
            "roi_map distinct offsets",
            str(len(set(snapped))),
            [f"at most {AV1_MAX_SEGMENTS} (AV1 segment cap)"],
        )
    return tuple(snapped)

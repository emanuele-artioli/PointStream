"""Coded-raster size and restore policy for background transport scaling.

This changes the resolution sent to the stream codec. Scene registration,
canonical geometry, output resolution and object coordinates stay in the
original plate's coordinates. Homographies are never rewritten into coded
space.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

MAGIC: bytes = b"PSBG"
VERSION: int = 1
RESTORE_NONE: int = 0
RESTORE_LINEAR: int = 1
RESTORE_NAMES: dict[int, str] = {RESTORE_NONE: "none", RESTORE_LINEAR: "linear"}

#: magic(4) version(u8) restore(u8) pad(u16) orig_w/h coded_w/h (u32) scale n/d (u16)
HEADER_STRUCT: struct.Struct = struct.Struct("<4sBBHIIIIHH")
HEADER_BYTES: int = HEADER_STRUCT.size

SUPPORTED_SCALES: tuple[float, ...] = (1.0, 0.5)


class TransportScaleError(ValueError):
    """A scale, raster or charged header that must not be used."""


def require_supported_scale(scale: float) -> float:
    """Accept only the first-experiment scales. Anything else is a hard refuse."""
    value = float(scale)
    if value not in SUPPORTED_SCALES:
        raise TransportScaleError(
            f"background.transport_scale={value!r} is not supported; "
            f"this implementation allows only {list(SUPPORTED_SCALES)}"
        )
    return value


def coded_dimensions(width: int, height: int, scale: float) -> tuple[int, int]:
    """Even coded width/height. Scale 1.0 is the original raster, unresampled."""
    scale = require_supported_scale(scale)
    if width < 1 or height < 1:
        raise TransportScaleError(
            f"original raster {width}x{height} is too small to code"
        )
    if scale == 1.0:
        return int(width), int(height)

    def even_floor(extent: int) -> int:
        scaled = math.floor(extent * scale)
        return int(scaled - (scaled % 2))

    coded_w = even_floor(width)
    coded_h = even_floor(height)
    if coded_w < 2 or coded_h < 2:
        raise TransportScaleError(
            f"coded raster {coded_w}x{coded_h} from {width}x{height} at scale "
            f"{scale} is too small"
        )
    return coded_w, coded_h


@dataclass(frozen=True)
class GeometryHeader:
    """Charged reconstruction metadata. Pack this; do not infer it from encoder state."""

    original_width: int
    original_height: int
    coded_width: int
    coded_height: int
    scale_num: int
    scale_den: int
    restore: int

    @property
    def scale(self) -> float:
        return float(self.scale_num) / float(self.scale_den)

    @property
    def restore_name(self) -> str:
        try:
            return RESTORE_NAMES[self.restore]
        except KeyError as exc:
            raise TransportScaleError(f"unknown restore policy id {self.restore}") from exc

    def pack(self) -> bytes:
        return HEADER_STRUCT.pack(
            MAGIC,
            VERSION,
            self.restore,
            0,
            self.original_width,
            self.original_height,
            self.coded_width,
            self.coded_height,
            self.scale_num,
            self.scale_den,
        )


def header_for(original_width: int, original_height: int, scale: float) -> GeometryHeader:
    scale = require_supported_scale(scale)
    coded_w, coded_h = coded_dimensions(original_width, original_height, scale)
    if scale == 1.0:
        num, den, restore = 1, 1, RESTORE_NONE
    else:
        num, den, restore = 1, 2, RESTORE_LINEAR
    return GeometryHeader(
        original_width=int(original_width),
        original_height=int(original_height),
        coded_width=coded_w,
        coded_height=coded_h,
        scale_num=num,
        scale_den=den,
        restore=restore,
    )


def unpack_header(blob: bytes) -> GeometryHeader:
    packed = bytes(blob)
    if len(packed) != HEADER_BYTES:
        raise TransportScaleError(
            f"geometry header is {len(packed)} bytes, expected {HEADER_BYTES}"
        )
    magic, version, restore, _pad, orig_w, orig_h, coded_w, coded_h, num, den = (
        HEADER_STRUCT.unpack(packed)
    )
    if magic != MAGIC or int(version) != VERSION:
        raise TransportScaleError(
            f"geometry header magic/version {magic!r}/{version} is not {MAGIC!r}/{VERSION}"
        )
    if int(den) < 1:
        raise TransportScaleError("geometry header scale denominator must be >= 1")
    header = GeometryHeader(
        original_width=int(orig_w),
        original_height=int(orig_h),
        coded_width=int(coded_w),
        coded_height=int(coded_h),
        scale_num=int(num),
        scale_den=int(den),
        restore=int(restore),
    )
    _ = header.restore_name
    expected = header_for(header.original_width, header.original_height, header.scale)
    if (
        header.coded_width != expected.coded_width
        or header.coded_height != expected.coded_height
        or header.restore != expected.restore
    ):
        raise TransportScaleError(
            "geometry header does not match the supported reconstruction policy"
        )
    return header


def downsample_plate(plate: np.ndarray, scale: float) -> tuple[np.ndarray, GeometryHeader]:
    """INTER_AREA into the charged coded raster. Scale 1.0 copies, it does not resize."""
    array = np.ascontiguousarray(np.asarray(plate, dtype=np.uint8))
    height, width = int(array.shape[0]), int(array.shape[1])
    header = header_for(width, height, scale)
    if header.restore == RESTORE_NONE:
        return array, header
    import cv2

    coded = cv2.resize(
        array,
        (header.coded_width, header.coded_height),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(coded), header


def restore_plate(coded: np.ndarray, header: GeometryHeader) -> np.ndarray:
    """Restore to the original canonical size using only the charged header."""
    array = np.ascontiguousarray(np.asarray(coded, dtype=np.uint8))
    if int(array.shape[0]) != header.coded_height or int(array.shape[1]) != header.coded_width:
        raise TransportScaleError(
            f"decoded plate is {array.shape[1]}x{array.shape[0]}, header coded size is "
            f"{header.coded_width}x{header.coded_height}"
        )
    if header.restore == RESTORE_NONE:
        if (
            header.coded_width != header.original_width
            or header.coded_height != header.original_height
        ):
            raise TransportScaleError(
                "restore=none requires coded size to equal the original canonical size"
            )
        return array
    import cv2

    restored = cv2.resize(
        array,
        (header.original_width, header.original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.ascontiguousarray(restored)

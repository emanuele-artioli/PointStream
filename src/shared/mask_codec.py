from __future__ import annotations

from typing import Literal
from typing import cast
import zlib

import cv2
import numpy as np


MaskCodecName = Literal["rle-v1", "bitpack-z1", "png", "poly-v1"]
MaskCodecChoice = Literal["rle-v1", "bitpack-z1", "png", "poly-v1", "auto"]


def normalize_mask_codec(codec: str | None) -> MaskCodecChoice:
    if codec is None:
        return "auto"

    normalized = codec.strip().lower().replace("_", "-")
    aliases = {
        "auto": "auto",
        "rle": "rle-v1",
        "rle-v1": "rle-v1",
        "rlev1": "rle-v1",
        "bitpack": "bitpack-z1",
        "bitpack-z1": "bitpack-z1",
        "poly": "poly-v1",
        "polygon": "poly-v1",
        "poly-v1": "poly-v1",
        "png": "png",
    }
    return cast(MaskCodecChoice, aliases.get(normalized, "auto"))


def encode_binary_mask(mask: np.ndarray, codec: str | None = None) -> tuple[MaskCodecName, bytes, int, int]:
    binary = _to_binary_mask(mask)
    height, width = binary.shape

    selected = normalize_mask_codec(codec)
    if selected == "auto":
        rle_payload = _encode_rle_v1(binary)
        bitpack_payload = _encode_bitpack_z1(binary)
        if len(bitpack_payload) <= len(rle_payload):
            return "bitpack-z1", bitpack_payload, int(height), int(width)
        return "rle-v1", rle_payload, int(height), int(width)

    if selected == "png":
        ok, encoded = cv2.imencode(".png", binary * 255)
        if not ok:
            raise RuntimeError("Failed to encode binary mask as PNG")
        return "png", encoded.tobytes(), int(height), int(width)

    if selected == "bitpack-z1":
        return "bitpack-z1", _encode_bitpack_z1(binary), int(height), int(width)

    return "rle-v1", _encode_rle_v1(binary), int(height), int(width)


def encode_polygon_mask(
    polygons: list[np.ndarray] | list[list[list[float]]],
    *,
    height: int,
    width: int,
    codec: str | None = None,
) -> tuple[MaskCodecName, bytes, int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"Mask dimensions must be positive, got {(height, width)}")

    selected = normalize_mask_codec(codec)
    if selected not in {"auto", "poly-v1"}:
        raise ValueError(f"Polygon masks only support 'poly-v1' codec, got '{codec}'")

    payload = _encode_poly_v1(
        polygons=polygons,
        height=int(height),
        width=int(width),
    )
    return "poly-v1", payload, int(height), int(width)


def decode_binary_mask(
    *,
    codec: str,
    payload: bytes,
    height: int,
    width: int,
) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError(f"Mask dimensions must be positive, got {(height, width)}")

    normalized = normalize_mask_codec(codec)
    if normalized == "auto":
        normalized = "rle-v1"

    if normalized == "png":
        encoded_np = np.frombuffer(payload, dtype=np.uint8)
        decoded = cv2.imdecode(encoded_np, cv2.IMREAD_GRAYSCALE)
        if decoded is None or decoded.size == 0:
            raise ValueError("Failed to decode PNG mask payload")
        return _to_binary_mask(decoded) * 255

    if normalized == "poly-v1":
        return _decode_poly_v1(payload=payload, height=height, width=width)

    if normalized == "bitpack-z1":
        return _decode_bitpack_z1(payload=payload, height=height, width=width)

    return _decode_rle_v1(payload=payload, height=height, width=width)


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {tuple(array.shape)}")
    return np.asarray(array > 0, dtype=np.uint8)


def _encode_rle_v1(binary_mask: np.ndarray) -> bytes:
    flat = np.asarray(binary_mask.reshape(-1), dtype=np.uint8)
    if flat.size == 0:
        return bytes([0])

    first_value = int(flat[0])
    transitions = np.flatnonzero(np.diff(flat) != 0) + 1
    boundaries = np.concatenate(([0], transitions, [flat.size]))
    run_lengths = np.diff(boundaries).astype(np.uint32)

    return bytes([first_value]) + run_lengths.tobytes()


def _encode_bitpack_z1(binary_mask: np.ndarray) -> bytes:
    flat = np.asarray(binary_mask.reshape(-1), dtype=np.uint8)
    packed = np.packbits(flat, bitorder="little")
    return zlib.compress(packed.tobytes(), level=1)


def _decode_rle_v1(*, payload: bytes, height: int, width: int) -> np.ndarray:
    if len(payload) < 1:
        raise ValueError("RLE payload is empty")

    first_value = int(payload[0])
    run_blob = payload[1:]
    if len(run_blob) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    if len(run_blob) % 4 != 0:
        raise ValueError("RLE payload length is not aligned to uint32 runs")

    runs = np.frombuffer(run_blob, dtype=np.uint32)
    values = ((np.arange(int(runs.size), dtype=np.uint8) + first_value) % 2).astype(np.uint8)
    flat = np.repeat(values, runs.astype(np.int64))

    expected = int(height) * int(width)
    if int(flat.size) != expected:
        raise ValueError(
            f"Decoded RLE mask has wrong size: expected {expected}, got {int(flat.size)}"
        )

    return (np.asarray(flat.reshape(height, width), dtype=np.uint8) * 255)


def _decode_bitpack_z1(*, payload: bytes, height: int, width: int) -> np.ndarray:
    decompressed = zlib.decompress(payload)
    packed = np.frombuffer(decompressed, dtype=np.uint8)
    unpacked = np.unpackbits(packed, bitorder="little")

    expected = int(height) * int(width)
    if int(unpacked.size) < expected:
        raise ValueError(
            f"Decoded bitpacked mask is truncated: expected at least {expected} bits, got {int(unpacked.size)}"
        )

    flat = np.asarray(unpacked[:expected], dtype=np.uint8)
    return flat.reshape(height, width) * 255


def _encode_poly_v1(
    *,
    polygons: list[np.ndarray] | list[list[list[float]]],
    height: int,
    width: int,
) -> bytes:
    if height <= 0 or width <= 0:
        raise ValueError(f"Mask dimensions must be positive, got {(height, width)}")

    normalized_polygons: list[np.ndarray] = []
    max_x = min(65535, max(0, width - 1))
    max_y = min(65535, max(0, height - 1))

    for polygon in polygons:
        coords = np.asarray(polygon, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 3:
            continue

        quantized = np.empty_like(coords, dtype=np.uint16)
        quantized[:, 0] = np.asarray(np.clip(np.round(coords[:, 0]), 0.0, float(max_x)), dtype=np.uint16)
        quantized[:, 1] = np.asarray(np.clip(np.round(coords[:, 1]), 0.0, float(max_y)), dtype=np.uint16)
        normalized_polygons.append(quantized)

    if len(normalized_polygons) > 65535:
        raise ValueError(f"Too many polygons for poly-v1 payload: {len(normalized_polygons)}")

    payload = bytearray()
    payload += int(len(normalized_polygons)).to_bytes(2, byteorder="little", signed=False)

    for polygon in normalized_polygons:
        num_points = int(polygon.shape[0])
        if num_points > 65535:
            raise ValueError(f"Polygon has too many points for poly-v1 payload: {num_points}")
        payload += int(num_points).to_bytes(2, byteorder="little", signed=False)
        payload += np.asarray(polygon.reshape(-1), dtype=np.uint16).tobytes()

    return bytes(payload)


def _decode_poly_v1(*, payload: bytes, height: int, width: int) -> np.ndarray:
    if len(payload) < 2:
        raise ValueError("poly-v1 payload is too short")

    offset = 0
    num_polygons = int.from_bytes(payload[offset : offset + 2], byteorder="little", signed=False)
    offset += 2

    mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(num_polygons):
        if offset + 2 > len(payload):
            raise ValueError("poly-v1 payload truncated before polygon point count")
        num_points = int.from_bytes(payload[offset : offset + 2], byteorder="little", signed=False)
        offset += 2

        coords_blob_size = int(num_points) * 2 * 2
        if offset + coords_blob_size > len(payload):
            raise ValueError("poly-v1 payload truncated before polygon coordinates")

        coords_u16 = np.frombuffer(payload[offset : offset + coords_blob_size], dtype=np.uint16)
        offset += coords_blob_size

        if coords_u16.size == 0:
            continue

        coords = np.asarray(coords_u16, dtype=np.int32).reshape(-1, 2)
        if coords.shape[0] < 3:
            continue

        coords[:, 0] = np.clip(coords[:, 0], 0, max(0, width - 1))
        coords[:, 1] = np.clip(coords[:, 1], 0, max(0, height - 1))
        cv2.fillPoly(mask, [coords], color=(255.0,))

    if offset != len(payload):
        raise ValueError("poly-v1 payload contains trailing bytes")

    return mask

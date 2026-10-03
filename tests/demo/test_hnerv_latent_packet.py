from __future__ import annotations

import base64
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from demo.experiments.hnerv_latent_packet import (
    MAGIC,
    METHODS,
    decode_packet,
    encode_packet,
    write_segment_packets,
)


def _metadata(length: int, *, frame_start: int = 120) -> dict:
    return {
        "checkpoint_sha256": "a" * 64,
        "frame_start": frame_start,
        "fps": {"numerator": 30, "denominator": 1},
        "frame_ids": [f"frame-{frame_start + i}" for i in range(length)],
        "quantizer": {
            "min": np.asarray([0.0, -1.5, 3.25], dtype="<f4").reshape(1, 1, 1, 3),
            "scale": np.asarray([1.0, 0.5, 2.0], dtype="<f4").reshape(1, 1, 1, 3),
        },
        "decoder_setup_bytes": 123456,
    }


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("method", METHODS)
def test_random_and_constant_codes_round_trip_exactly(channels: int, method: str) -> None:
    rng = np.random.default_rng(17 + channels)
    random_codes = rng.integers(0, 64, size=(8, 9, 16, channels), dtype=np.uint8)
    random_codes[0, 0, 0, 0] = 0
    random_codes[-1, -1, -1, -1] = 63
    packet = encode_packet(random_codes, _metadata(8), method=method)
    decoded, header = decode_packet(packet, expected_checkpoint_sha256="a" * 64)
    assert np.array_equal(decoded, random_codes)
    assert header["shape"] == list(random_codes.shape)
    assert header["frame_start"] == 120
    constant = np.full((1, 9, 16, channels), 63, dtype=np.uint8)
    single, _ = decode_packet(encode_packet(constant, _metadata(1), method=method))
    assert np.array_equal(single, constant)


def test_modular_temporal_wraparound_has_exact_independent_decode() -> None:
    codes = np.zeros((8, 9, 16, 3), dtype=np.uint8)
    codes[:, 0, 0, 0] = np.asarray([63, 0, 1, 1, 63, 0, 63, 1], dtype=np.uint8)
    packet = encode_packet(codes, _metadata(8), method="delta-zlib6")
    decoded, _ = decode_packet(packet)
    assert np.array_equal(decoded, codes)


def test_quantizer_metadata_preserves_dtype_shape_byte_order_and_bytes() -> None:
    metadata = _metadata(1)
    metadata["quantizer"]["min"] = np.asarray([0x01020304], dtype=">u4")
    codes = np.zeros((1, 9, 16, 3), dtype=np.uint8)
    decoded, header = decode_packet(encode_packet(codes, metadata, method="packed6"))
    assert decoded[0, 0, 0, 0] == 0
    record = header["quantizer"]["min"]
    assert record["dtype"] == ">u4"
    assert record["shape"] == [1]
    assert np.frombuffer(base64.b64decode(record["data_b64"]), dtype=">u4").tolist() == [0x01020304]


def test_corruption_truncation_and_checkpoint_mismatch_fail_closed() -> None:
    codes = np.zeros((1, 9, 16, 3), dtype=np.uint8)
    packet = encode_packet(codes, _metadata(1), method="zlib6")
    with pytest.raises(ValueError, match="truncated"):
        decode_packet(packet[:8])
    with pytest.raises(ValueError, match="checkpoint identity"):
        decode_packet(packet, expected_checkpoint_sha256="b" * 64)
    damaged = bytearray(packet)
    damaged[-1] ^= 0x20
    with pytest.raises(ValueError, match="CRC"):
        decode_packet(bytes(damaged))
    header_start = len(MAGIC) + 4
    header_length = struct.unpack(">I", packet[len(MAGIC):header_start])[0]
    header = json.loads(packet[header_start:header_start + header_length])
    header["frame_start"] += 1
    changed_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    bad_header = MAGIC + struct.pack(">I", len(changed_header)) + changed_header + packet[header_start + header_length:]
    with pytest.raises(ValueError, match="header CRC"):
        decode_packet(bad_header)


def test_invalid_shapes_codes_precision_and_metadata_are_rejected() -> None:
    metadata = _metadata(2)
    with pytest.raises(ValueError, match="shape"):
        encode_packet(np.zeros((2, 2, 3), dtype=np.uint8), metadata, method="packed6")
    with pytest.raises(ValueError, match="six-bit"):
        encode_packet(np.full((1, 9, 16, 3), 64, dtype=np.uint8), _metadata(1), method="packed6")
    metadata["fps"] = {"numerator": 60, "denominator": 1}
    with pytest.raises(ValueError, match="30/1"):
        encode_packet(np.zeros((1, 9, 16, 3), dtype=np.uint8), metadata, method="packed6")


def test_packaging_counts_full_files_and_resets_each_segment(tmp_path: Path) -> None:
    codes = np.zeros((32, 9, 16, 3), dtype=np.uint8)
    codes[:, :, :, :] = np.arange(32, dtype=np.uint8)[:, None, None, None] % 64
    metadata = _metadata(32)
    result = write_segment_packets(codes, metadata, tmp_path)
    assert result["record_count"] == 111  # (32 one-frame + 4 eight-frame + 1 32-frame) * 3 methods
    paths = [Path(row["path"]) for row in result["packets"]]
    assert sum(path.stat().st_size for path in paths) == sum(row["stream_bytes"] for row in result["packets"])
    first8 = next(Path(row["path"]) for row in result["packets"] if row["segment_length"] == 8 and row["frame_start"] == 120 and row["method"] == "delta-zlib6")
    codes8, header8 = decode_packet(first8.read_bytes())
    assert header8["frame_ids"] == [f"frame-{i}" for i in range(120, 128)]
    assert np.array_equal(codes8, codes[:8])
    # Decoder setup is an explicit separate field, never silently folded into
    # latent packet bytes or mistaken for the encoded packet length.
    assert header8["decoder_setup_bytes"] == 123456
    assert first8.stat().st_size != header8["decoder_setup_bytes"]
    with pytest.raises(FileExistsError, match="nonempty"):
        write_segment_packets(codes, metadata, tmp_path)


def test_future_codes_cannot_change_an_earlier_reset_packet() -> None:
    first = np.zeros((8, 9, 16, 3), dtype=np.uint8)
    second = first.copy()
    second[4:] = 42
    meta = _metadata(8)
    packet_meta = meta | {"frame_ids": ["frame-120"]}
    before = encode_packet(first[:1].copy(), packet_meta, method="delta-zlib6")
    after = encode_packet(second[:1].copy(), packet_meta, method="delta-zlib6")
    assert before == after


def test_decoder_does_not_accept_wrong_packet_magic() -> None:
    with pytest.raises(ValueError, match="magic"):
        decode_packet(b"not a packet")

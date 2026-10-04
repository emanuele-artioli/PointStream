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
    dequantize,
    encode_packet,
    summarize_packets,
    write_segment_packets,
)


def _metadata(length: int, *, frame_start: int = 120, quantizer: dict | None = None) -> dict:
    return {
        "checkpoint_sha256": "a" * 64,
        "frame_start": frame_start,
        "fps": {"numerator": 30, "denominator": 1},
        "frame_ids": [f"frame-{frame_start + i}" for i in range(length)],
        "quantizer": quantizer or {"min": np.asarray(-1.25, dtype="<f4"), "scale": np.asarray(0.03125, dtype="<f4")},
        "decoder_setup_bytes": 123456,
    }


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("bits", [4, 6])
@pytest.mark.parametrize("method", METHODS)
def test_random_constant_and_extreme_codes_round_trip(channels: int, bits: int, method: str) -> None:
    top = (1 << bits) - 1
    rng = np.random.default_rng(17 + channels + bits)
    codes = rng.integers(0, top + 1, size=(8, channels, 9, 16), dtype=np.uint8)
    codes[0, 0, 0, 0], codes[-1, -1, -1, -1] = 0, top
    decoded, header = decode_packet(encode_packet(codes, _metadata(8), method=method, bit_depth=bits), expected_checkpoint_sha256="a" * 64)
    assert np.array_equal(decoded, codes) and header["layout"] == "NCHW" and header["bit_depth"] == bits
    constant = np.full((1, channels, 9, 16), top, dtype=np.uint8)
    assert np.array_equal(decode_packet(encode_packet(constant, _metadata(1), method=method, bit_depth=bits))[0], constant)


@pytest.mark.parametrize("bits", [4, 6])
def test_modular_temporal_wraparound(bits: int) -> None:
    top = (1 << bits) - 1
    codes = np.zeros((8, 3, 9, 16), dtype=np.uint8)
    codes[:, 0, 0, 0] = np.asarray([top, 0, 1, 1, top, 0, top, 1], dtype=np.uint8)
    decoded, _ = decode_packet(encode_packet(codes, _metadata(8), method="delta-zlib", bit_depth=bits))
    assert np.array_equal(decoded, codes)


def test_quantizer_arrays_keep_dtype_shape_endianness_and_invert_exactly() -> None:
    quantizer = {"min": np.asarray([[[[-2.0]], [[0.5]], [[1.0]]]], dtype=">f2"), "scale": np.asarray(0.25, dtype="<f4")}
    codes = np.arange(3 * 9 * 16, dtype=np.uint8).reshape(1, 3, 9, 16) % 64
    decoded, header = decode_packet(encode_packet(codes, _metadata(1, quantizer=quantizer), method="packed", bit_depth=6))
    record = header["quantizer"]["min"]
    assert record["dtype"] == ">f2" and record["shape"] == [1, 3, 1, 1]
    assert np.frombuffer(base64.b64decode(record["data_b64"]), dtype=">f2").tolist() == [-2.0, 0.5, 1.0]
    expected = np.asarray([-2.0, 0.5, 1.0], np.float32)[None, :, None, None] + np.float32(0.25) * codes.astype(np.float32)
    assert np.array_equal(dequantize(decoded, header["quantizer"]), expected)
    zero_range = {"min": np.asarray(3.0, np.float32), "scale": np.asarray(0.0, np.float32)}
    assert np.isfinite(dequantize(np.zeros((1, 3, 9, 16), np.uint8), zero_range)).all()
    with pytest.raises(ValueError, match="floating-point"):
        encode_packet(codes, _metadata(1, quantizer={"min": np.asarray(1, ">u4"), "scale": np.asarray(1.0)}), method="packed", bit_depth=6)
    with pytest.raises(ValueError, match="broadcast"):
        encode_packet(codes, _metadata(1, quantizer={"min": np.zeros(5, np.float32), "scale": np.asarray(1.0)}), method="packed", bit_depth=6)


def test_corrupt_truncated_wrong_checkpoint_and_wrong_shape_fail() -> None:
    codes = np.zeros((1, 3, 9, 16), dtype=np.uint8)
    packet = encode_packet(codes, _metadata(1), method="zlib", bit_depth=6)
    with pytest.raises(ValueError, match="truncated"):
        decode_packet(packet[:8])
    with pytest.raises(ValueError, match="checkpoint identity"):
        decode_packet(packet, expected_checkpoint_sha256="b" * 64)
    damaged = bytearray(packet)
    damaged[-1] ^= 0x20
    with pytest.raises(ValueError, match="CRC"):
        decode_packet(bytes(damaged))
    start = len(MAGIC) + 4
    length = struct.unpack(">I", packet[len(MAGIC):start])[0]
    header = json.loads(packet[start:start + length])
    header["frame_start"] += 1
    changed = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="header CRC"):
        decode_packet(MAGIC + struct.pack(">I", len(changed)) + changed + packet[start + length:])
    with pytest.raises(ValueError, match="NCHW"):
        encode_packet(np.zeros((1, 9, 16, 3), np.uint8), _metadata(1), method="packed", bit_depth=6)
    with pytest.raises(ValueError, match="6-bit"):
        encode_packet(np.full((1, 3, 9, 16), 64, np.uint8), _metadata(1), method="packed", bit_depth=6)
    with pytest.raises(ValueError, match="4-bit"):
        encode_packet(np.full((1, 3, 9, 16), 16, np.uint8), _metadata(1), method="packed", bit_depth=4)
    with pytest.raises(ValueError, match="magic"):
        decode_packet(b"not a packet")


def _segments(codes: np.ndarray, length: int) -> list[dict]:
    return [{"codes": codes[s:s + length], "metadata": _metadata(length, frame_start=120 + s)} for s in range(0, 32, length)]


def test_independent_segments_count_whole_files_and_exclude_setup(tmp_path: Path) -> None:
    codes = (np.arange(32, dtype=np.uint8)[:, None, None, None] * np.ones((1, 3, 9, 16), np.uint8)) % 64
    records = []
    for length in (1, 8, 32):
        records += write_segment_packets(_segments(codes, length), tmp_path / f"L{length}", bit_depth=6)["packets"]
    assert len(records) == (32 + 4 + 1) * 3
    assert all(Path(r["path"]).stat().st_size == r["file_bytes"] == r["payload_bytes"] + r["header_bytes"] for r in records)
    rows = summarize_packets(records, frames=32, setup_bytes=123456)
    for row in rows:
        assert row["setup_inclusive_bytes"] == row["latent_only_bytes"] + 123456
        assert row["latent_only_kbps"] == pytest.approx(8 * row["latent_only_bytes"] * 30 / (1000 * 32))
    second = next(r for r in records if r["segment_length"] == 8 and r["frame_start"] == 128 and r["method"] == "delta-zlib")
    decoded, header = decode_packet(Path(second["path"]).read_bytes())
    assert np.array_equal(decoded, codes[8:16]) and header["frame_ids"][0] == "frame-128"
    with pytest.raises(FileExistsError, match="nonempty"):
        write_segment_packets(_segments(codes, 8), tmp_path / "L8", bit_depth=6)
    with pytest.raises(ValueError, match="cover"):
        summarize_packets([r for r in records if r["frame_start"] != 120], frames=32, setup_bytes=0)


def test_future_codes_cannot_change_an_earlier_packet() -> None:
    first = np.zeros((32, 3, 9, 16), dtype=np.uint8)
    second = first.copy()
    second[8:] = 42
    meta = _metadata(8)
    assert encode_packet(first[:8], meta, method="delta-zlib", bit_depth=6) == encode_packet(second[:8], meta, method="delta-zlib", bit_depth=6)


def test_lossless_packaging_changes_rate_not_codes(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    codes = np.repeat(rng.integers(0, 64, size=(1, 4, 9, 16), dtype=np.uint8), 32, axis=0)
    records = write_segment_packets(_segments(codes, 32), tmp_path, bit_depth=6)["packets"]
    hashes = {r["codes_sha256"] for r in records}
    assert len(hashes) == 1
    sizes = {r["method"]: r["file_bytes"] for r in records}
    assert sizes["delta-zlib"] < sizes["zlib"] <= sizes["packed"] + 64


def test_decoder_output_ignores_source_tensors() -> None:
    torch = pytest.importorskip("torch")
    from demo.experiments import hnerv_frozen as frozen

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 1)

        def forward(self, embed):
            return torch.sigmoid(self.conv(embed))

    class Model(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, source, input_embed=None):
            return self.decoder(input_embed), [input_embed], 0.0

    decoder = Decoder()
    embed = torch.rand(2, 3, 9, 16)
    sources = [np.zeros((9, 16, 3), np.uint8)] * 2
    report = frozen.source_independence(Model(decoder), decoder, embed, sources, device="cpu")
    assert report["independent"] and report["repeat_max_abs_diff"] == 0.0

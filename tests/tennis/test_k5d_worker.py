import hashlib
import struct
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from experiments.tennis.k5d_worker import resource, validate_native_header
from src.pointstream import ball, ball_stream


def stream() -> ball_stream.Stream:
    # Synthetic container bytes test parsing only; no codec evidence is implied.
    header = struct.pack("<4sHH4sHHIIII", b"DKIF", 0, 32, b"AV01", 16, 12, 30, 1, 2, 0)
    frames = b"".join(struct.pack("<IQ", 1, t) + b"x" for t in range(2))
    return ball_stream.Stream(16, 12, Fraction(30), 2, "svt-av1-ivf", "svt-test",
                              header + frames, ball.encode_frames([None, None], [None, None], 16, 12))


def test_ivf_header_is_bound_to_wire_geometry_and_count() -> None:
    sample = stream()
    validate_native_header(sample)
    for bad in (replace(sample, width=18), replace(sample, fps=Fraction(25)),
                replace(sample, frames=1), replace(sample, native=sample.native[:-1])):
        with pytest.raises(ValueError):
            validate_native_header(bad)


def test_resource_identifier_and_file_hash(tmp_path: Path) -> None:
    decoder = tmp_path / "decoder"
    decoder.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    mapping = {"resources": {"svt-test": {"codec": "svt-av1-ivf", "decoder": str(decoder), "decoder_sha256": digest}}}
    assert resource(stream(), mapping)["decoder"] == str(decoder)
    with pytest.raises(ValueError):
        resource(replace(stream(), resource_id="different"), mapping)
    decoder.write_bytes(b"different")
    with pytest.raises(ValueError, match="invalid installed"):
        resource(stream(), mapping)

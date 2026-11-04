from dataclasses import replace
from fractions import Fraction
import hashlib

import numpy as np
import pytest

from experiments.tennis.k5c_independent_ball import render_frames
from src.pointstream import ball, ball_stream


def example() -> tuple[ball_stream.Stream, ball.Track, list[ball.Appearance | None]]:
    track: ball.Track = [(5, 6), (6, 6), None, (8, 7), None]
    a = ball.Appearance((220, 210, 190), 0, 0, 0)
    apps = [a, a, None, a, None]
    stream = ball_stream.Stream(16, 12, Fraction(30000, 1001), len(track), "svt-av1-ivf",
                                "svt-2.0+decoder@sha256:abc", b"DKIF\x00native\x00",
                                ball.encode_frames(track, apps, 16, 12))
    return stream, track, apps


def test_roundtrip_render_and_actual_wire_length() -> None:
    stream, track, apps = example()
    wire = ball_stream.pack(stream)
    restored = ball_stream.unpack(wire)
    assert restored == stream
    assert restored.native == b"DKIF\x00native\x00"
    assert restored.duration == Fraction(1001, 6000)
    frames = np.full((5, 12, 16, 3), 70, dtype=np.uint8)
    output = ball_stream.render_from_native_decoded_frames(restored, frames)
    np.testing.assert_array_equal(output, render_frames(frames, track, apps))
    np.testing.assert_array_equal(output[2], frames[2])
    assert len(wire) > len(stream.native) + len(stream.ball_payload)


def test_absent_ball_roundtrip() -> None:
    stream, _, _ = example()
    stream = replace(stream, ball_payload=ball.encode_frames([None] * 5, [None] * 5, 16, 12))
    frames = np.zeros((5, 12, 16, 3), dtype=np.uint8)
    np.testing.assert_array_equal(ball_stream.render_from_native_decoded_frames(ball_stream.unpack(ball_stream.pack(stream)), frames), frames)


def test_reject_bad_wire_and_payload() -> None:
    stream, _, _ = example()
    wire = ball_stream.pack(stream)
    for corrupted in (wire[:-1], wire + b"x", wire[:20] + bytes([wire[20] ^ 1]) + wire[21:]):
        with pytest.raises(ValueError):
            ball_stream.unpack(corrupted)
    # Even a correctly checksummed noncanonical ball payload is rejected.
    bad = replace(stream, ball_payload=stream.ball_payload + b"\x00")
    with pytest.raises(ValueError, match="noncanonical"):
        ball_stream.pack(bad)
    # A changed version with a fresh checksum cannot be interpreted as v1.
    changed = wire[:4] + b"\x02" + wire[5:-32]
    with pytest.raises(ValueError, match="version"):
        ball_stream.unpack(changed + hashlib.sha256(changed).digest())


def test_reject_invalid_metadata_and_frame_shape() -> None:
    stream, _, _ = example()
    for invalid in (replace(stream, frames=0), replace(stream, fps=Fraction(0)),
                    replace(stream, codec="unknown"), replace(stream, resource_id="x" * 129),
                    replace(stream, width=0), replace(stream, native=b"")):
        with pytest.raises(ValueError):
            ball_stream.pack(invalid)
    with pytest.raises(ValueError, match="geometry"):
        ball_stream.render_from_native_decoded_frames(stream, np.zeros((4, 12, 16, 3), dtype=np.uint8))

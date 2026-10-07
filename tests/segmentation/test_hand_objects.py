"""The EPIC-KITCHENS hand-object reader: protobuf wire format and the no-globals unpickler."""

from __future__ import annotations

import pickle
import struct

import pytest

from src.segmentation import hand_objects


def varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def field(number: int, wire: int, payload: bytes | int) -> bytes:
    key = varint(number << 3 | wire)
    if wire == 0:
        assert isinstance(payload, int)
        return key + varint(payload)
    if wire == 5:
        assert isinstance(payload, bytes)
        return key + payload
    assert isinstance(payload, bytes)
    return key + varint(len(payload)) + payload


def f32(value: float) -> bytes:
    return struct.pack("<f", value)


def box(left: float, top: float, right: float, bottom: float) -> bytes:
    return b"".join(field(i + 1, 5, f32(v)) for i, v in enumerate((left, top, right, bottom)))


def detections(video: str, number: int, hands: list[bytes], objects: list[bytes]) -> bytes:
    out = field(1, 2, video.encode()) + field(2, 0, number)
    out += b"".join(field(3, 2, h) for h in hands) + b"".join(field(4, 2, o) for o in objects)
    return out


def hand(side: int, score: float, state: int, coords: tuple[float, float, float, float]) -> bytes:
    offset = field(1, 5, f32(0.25)) + field(2, 5, f32(-0.5))
    # proto3 omits zero values: side 0 (left) and state 0 are left out on purpose.
    out = field(1, 2, box(*coords)) + field(2, 5, f32(score)) + field(4, 2, offset)
    if state:
        out += field(3, 0, state)
    if side:
        out += field(5, 0, side)
    return out


def test_reads_frames_hands_and_objects() -> None:
    frames = [
        detections("P01_107", 1, [hand(0, 0.9, 3, (0.1, 0.2, 0.3, 0.4)), hand(1, 0.2, 0, (0.5, 0.5, 0.9, 1.0))],
                   [field(1, 2, box(0.0, 0.0, 0.5, 0.5)) + field(2, 5, f32(0.7))]),
        detections("P01_107", 2, [], []),
    ]
    out = hand_objects.load_detections(pickle.dumps(frames))
    assert [f.frame_number for f in out] == [1, 2]
    left, right = out[0].hands
    assert left.side == "left hand" and left.state == "portable_object"
    assert right.side == "right hand" and right.state == "no_contact"
    assert left.box.right == pytest.approx(0.3) and left.object_offset == pytest.approx((0.25, -0.5))
    assert out[0].objects[0].score == pytest.approx(0.7)
    assert left.box.pixels(1920, 1080) == pytest.approx((192.0, 216.0, 576.0, 432.0))
    assert [h.side for h in hand_objects.hands_at(out, 1, min_score=0.5)] == ["left hand"]
    assert hand_objects.hands_at(out, 2) == []


def test_frame_lookup_checks_the_frame_number() -> None:
    out = hand_objects.load_detections(pickle.dumps([detections("V", 5, [], [])]))
    with pytest.raises(ValueError, match="holds frame 5"):
        hand_objects.hands_at(out, 1)


class Payload:
    pass


def test_refuses_pickled_objects() -> None:
    with pytest.raises(pickle.UnpicklingError):
        hand_objects.load_detections(pickle.dumps([Payload()]))
    with pytest.raises(ValueError, match="list of bytes"):
        hand_objects.load_detections(pickle.dumps({"a": 1}))


def test_rejects_truncated_messages() -> None:
    whole = detections("V", 1, [hand(1, 0.9, 0, (0.1, 0.1, 0.2, 0.2))], [])
    with pytest.raises(ValueError):
        hand_objects.parse_frame(whole[:-3])

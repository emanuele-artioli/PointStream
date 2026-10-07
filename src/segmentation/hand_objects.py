"""EPIC-KITCHENS-100 hand-object detections (Shan et al. 2020 detector output).

The release (doi:10.5523/bris.3l8eci2oqgst92n14w2yqi5ytu, ``hand-objects/<P>/<video>.pkl``)
stores one file per video: a pickled list of serialized protobuf messages, one
per EPIC rgb frame in order, with the schema of
``epic-kitchens-100-hand-object-bboxes`` (``src/public_lib/epic_kitchens/hoa/types.proto``)::

    Detections { string video_id = 1; int32 frame_number = 2;
                 repeated HandDetection hands = 3; repeated ObjectDetection objects = 4; }
    HandDetection { BBox bbox = 1; float score = 2; HandState state = 3;
                    FloatVector object_offset = 4; HandSide side = 5; }
    ObjectDetection { BBox bbox = 1; float score = 2; }
    BBox { float left = 1; float top = 2; float right = 3; float bottom = 4; }
    FloatVector { float x = 1; float y = 2; }

Boxes are normalised to [0, 1] of the frame (the detector ran on 456x256
frames; the release's renderer scales them by the image size). List entry
``i`` is EPIC rgb frame ``i + 1``. Hands were kept down to score 0.1 and objects
down to 0.01; the authors suggest 0.5 for both.

The official reader needs ``protobuf``, which the PointStream environment does
not carry, and unpickles with the full pickle machinery. This module instead
unpickles with no globals allowed (the payload is a list of bytes) and decodes
the five messages from the protobuf wire format directly; it was cross-checked
against the official reader (docs/resources.md#datasets). These boxes are a
detector's output, not ground truth: use them only as an independent check.
"""

from __future__ import annotations

import io
import pickle
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HAND_SIDES = {0: "left hand", 1: "right hand"}
HAND_STATES = {0: "no_contact", 1: "self_contact", 2: "another_person", 3: "portable_object", 4: "stationary_object"}


@dataclass(frozen=True)
class Box:
    """Normalised (left, top, right, bottom) in [0, 1] of the frame."""

    left: float
    top: float
    right: float
    bottom: float

    def pixels(self, width: int, height: int) -> tuple[float, float, float, float]:
        return (self.left * width, self.top * height, self.right * width, self.bottom * height)


@dataclass(frozen=True)
class Hand:
    box: Box
    score: float
    side: str
    state: str
    object_offset: tuple[float, float]


@dataclass(frozen=True)
class Object:
    box: Box
    score: float


@dataclass(frozen=True)
class FrameDetections:
    video_id: str
    frame_number: int
    hands: tuple[Hand, ...]
    objects: tuple[Object, ...]


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"hand-object detections must be a list of bytes, not {module}.{name}")


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("truncated varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _fields(data: bytes) -> list[tuple[int, int, Any]]:
    """``(field number, wire type, value)`` of one message; unknown wire types are an error."""
    out: list[tuple[int, int, Any]] = []
    pos = 0
    while pos < len(data):
        key, pos = _varint(data, pos)
        number, wire = key >> 3, key & 7
        value: Any
        if wire == 0:
            value, pos = _varint(data, pos)
        elif wire == 1:
            value, pos = data[pos:pos + 8], pos + 8
        elif wire == 2:
            length, pos = _varint(data, pos)
            value, pos = data[pos:pos + length], pos + length
        elif wire == 5:
            value, pos = data[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        if pos > len(data):
            raise ValueError("truncated protobuf field")
        out.append((number, wire, value))
    return out


def _float(value: bytes) -> float:
    return float(struct.unpack("<f", value)[0])


def _floats(data: bytes, names: tuple[str, ...]) -> dict[str, float]:
    """A message of float fields 1..n (proto3: absent means 0)."""
    values = dict.fromkeys(names, 0.0)
    for number, wire, value in _fields(data):
        if wire == 5 and 1 <= number <= len(names):
            values[names[number - 1]] = _float(value)
    return values


def _box(data: bytes) -> Box:
    return Box(**_floats(data, ("left", "top", "right", "bottom")))


def _hand(data: bytes) -> Hand:
    box, score, state, offset, side = Box(0, 0, 0, 0), 0.0, 0, (0.0, 0.0), 0
    for number, wire, value in _fields(data):
        if number == 1 and wire == 2:
            box = _box(value)
        elif number == 2 and wire == 5:
            score = _float(value)
        elif number == 3 and wire == 0:
            state = value
        elif number == 4 and wire == 2:
            vector = _floats(value, ("x", "y"))
            offset = (vector["x"], vector["y"])
        elif number == 5 and wire == 0:
            side = value
    if side not in HAND_SIDES or state not in HAND_STATES:
        raise ValueError(f"unknown hand side {side} or state {state}")
    return Hand(box, score, HAND_SIDES[side], HAND_STATES[state], offset)


def _object(data: bytes) -> Object:
    box, score = Box(0, 0, 0, 0), 0.0
    for number, wire, value in _fields(data):
        if number == 1 and wire == 2:
            box = _box(value)
        elif number == 2 and wire == 5:
            score = _float(value)
    return Object(box, score)


def parse_frame(data: bytes) -> FrameDetections:
    """One serialized ``Detections`` message."""
    video, number = "", 0
    hands: list[Hand] = []
    objects: list[Object] = []
    for field, wire, value in _fields(data):
        if field == 1 and wire == 2:
            video = value.decode("utf-8")
        elif field == 2 and wire == 0:
            number = value - (1 << 64) if value >= 1 << 63 else value
        elif field == 3 and wire == 2:
            hands.append(_hand(value))
        elif field == 4 and wire == 2:
            objects.append(_object(value))
    return FrameDetections(video, number, tuple(hands), tuple(objects))


def load_detections(source: Path | str | bytes) -> list[FrameDetections]:
    """Every frame of one released ``<video>.pkl``; entry ``i`` is EPIC rgb frame ``i + 1``."""
    raw = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    payload = _NoGlobals(io.BytesIO(raw)).load()
    if not isinstance(payload, list) or not all(isinstance(item, bytes) for item in payload):
        raise ValueError("hand-object detections must be a pickled list of bytes")
    return [parse_frame(item) for item in payload]


def hands_at(detections: list[FrameDetections], epic_frame: int, *, min_score: float = 0.5) -> list[Hand]:
    """Hands detected on EPIC rgb frame ``epic_frame`` (1-indexed) with score >= ``min_score``."""
    frame = detections[epic_frame - 1]
    if frame.frame_number != epic_frame:
        raise ValueError(f"entry {epic_frame - 1} holds frame {frame.frame_number}, not {epic_frame}")
    return [hand for hand in frame.hands if hand.score >= min_score]


__all__ = [
    "HAND_SIDES",
    "HAND_STATES",
    "Box",
    "FrameDetections",
    "Hand",
    "Object",
    "hands_at",
    "load_detections",
    "parse_frame",
]

"""Experimental, lossless hand-track packet for foreground smoke tests.

Version 2 deliberately favors an auditable decoder contract over rate. Boxes
and joints use unsigned 1/16-pixel full-frame coordinates. Packets contain the
track identity, handedness and presence events needed by a decoder; source
poses and boxes are never consulted after encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import zlib
from typing import Iterable, Sequence

import numpy as np

MAGIC = b"PSFG"
VERSION = 2
COORD_SCALE = 16
FPS_NUM = 30
FPS_DEN = 1
MAX_FRAMES = 1800
MAX_DIMENSION = 4095  # 4095 * 16 fits in uint16.

# magic, version, method, width, height, fps numerator/denominator, units,
# segment start, frame count, record count, compressed length, plain CRC32.
_HEADER = struct.Struct("<4sBBHHHHHIIIII")
_RECORD = struct.Struct("<HBB")  # track_id, handedness (0/1/2), delta flag
_METHOD_RAW = 0
_METHOD_ZLIB = 1
_METHOD_DELTA_ZLIB = 2


@dataclass(frozen=True)
class HandDetection:
    """One proposed hand before causal identity assignment."""

    bbox: tuple[float, float, float, float]
    joints: tuple[tuple[float, float], ...]
    handedness: str | None = None
    score: float | None = None
    candidate_id: str = ""


@dataclass(frozen=True)
class TrackedHand:
    track_id: int
    handedness: str | None
    bbox: tuple[float, float, float, float]
    joints: tuple[tuple[float, float], ...]
    presence: bool = True


@dataclass(frozen=True)
class FrameAssignment:
    hands: tuple[TrackedHand, ...]
    supported: bool
    reason: str | None = None
    duplicate_count: int = 0


def _as_bbox(bbox: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain x1,y1,x2,y2")
    values = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox contains nonfinite coordinates")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("bbox is empty or outside the frame")
    return values


def _as_joints(
    joints: Iterable[Sequence[float]], width: int, height: int
) -> tuple[tuple[float, float], ...]:
    values = tuple(tuple(float(value) for value in point[:2]) for point in joints)
    if len(values) != 21 or any(len(point) != 2 for point in values):
        raise ValueError("a hand must contain exactly 21 x/y joints")
    if not all(math.isfinite(v) for point in values for v in point):
        raise ValueError("joints contain nonfinite coordinates")
    if any(x < 0 or x > width or y < 0 or y > height for x, y in values):
        raise ValueError("joint is outside the frame")
    return values


def _quantize(value: float, limit: int) -> int:
    code = int(math.floor(value * COORD_SCALE + 0.5))
    if code < 0 or code > limit * COORD_SCALE:
        raise ValueError("coordinate exceeds uint16 1/16-pixel range")
    return code


def _codes(hand: TrackedHand, width: int, height: int) -> tuple[int, ...]:
    box = _as_bbox(hand.bbox, width, height)
    joints = _as_joints(hand.joints, width, height)
    return tuple(
        _quantize(v, max(width, height)) for v in (*box, *(v for point in joints for v in point))
    )


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _center_distance(a: Sequence[float], b: Sequence[float], width: int, height: int) -> float:
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return math.hypot(ax - bx, ay - by) / math.hypot(width, height)


def _joint_center(joints: Sequence[Sequence[float]]) -> tuple[float, float]:
    return (
        sum(float(point[0]) for point in joints) / len(joints),
        sum(float(point[1]) for point in joints) / len(joints),
    )


def _joint_distance(
    a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], width: int, height: int
) -> float:
    ax, ay = _joint_center(a)
    bx, by = _joint_center(b)
    return math.hypot(ax - bx, ay - by) / math.hypot(width, height)


class CausalTrackAssociator:
    """Greedy deterministic association using box IoU and normalized centers.

    Handedness is metadata only; it never determines identity. A track may be
    recovered for two missing frames. A frame with more than two unique boxes
    is marked unsupported and none of its detections are emitted.
    """

    def __init__(self, width: int, height: int, *, max_gap: int = 2) -> None:
        _validate_dimensions(width, height)
        if max_gap < 0 or max_gap > 2:
            raise ValueError("max_gap must be between 0 and 2")
        self.width, self.height, self.max_gap = width, height, max_gap
        self.next_id = 0
        self.frame_index = -1
        self.events: list[dict] = []
        self._tracks: dict[int, tuple[tuple[float, float, float, float], tuple, int]] = {}

    def assign(self, detections: Sequence[HandDetection]) -> FrameAssignment:
        self.frame_index += 1
        checked: list[HandDetection] = []
        for detection in detections:
            bbox = _as_bbox(detection.bbox, self.width, self.height)
            joints = _as_joints(detection.joints, self.width, self.height)
            if detection.handedness is not None and detection.handedness.lower() not in (
                "left",
                "right",
            ):
                raise ValueError("handedness must be left, right, or None")
            score = detection.score
            if score is not None and not math.isfinite(float(score)):
                raise ValueError("candidate score must be finite")
            checked.append(
                HandDetection(bbox, joints, detection.handedness, score, detection.candidate_id)
            )

        # Suppress near-identical boxes before tracking. Score is only a stable
        # duplicate tie-breaker; it is not treated as a probability or label.
        ranked = sorted(
            checked,
            key=lambda item: (
                -(float(item.score) if item.score is not None else -math.inf),
                -((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])),
                item.candidate_id,
                item.bbox,
            ),
        )
        unique: list[HandDetection] = []
        for detection in ranked:
            if any(_iou(detection.bbox, prior.bbox) >= 0.85 for prior in unique):
                continue
            unique.append(detection)
        duplicate_count = len(checked) - len(unique)
        if len(unique) > 2:
            self._expire_tracks()
            return FrameAssignment((), False, "more_than_two_unresolved_hands", duplicate_count)

        possible: list[tuple[float, int, int]] = []
        for track_id, (prior_box, prior_joints, seen_frame) in self._tracks.items():
            if self.frame_index - seen_frame > self.max_gap + 1:
                continue
            for index, detection in enumerate(unique):
                distance = _center_distance(prior_box, detection.bbox, self.width, self.height)
                joints = _joint_distance(prior_joints, detection.joints, self.width, self.height)
                overlap = _iou(prior_box, detection.bbox)
                if distance <= 0.35 and joints <= 0.35 and (overlap >= 0.02 or distance <= 0.10):
                    cost = 1.0 - overlap + 0.5 * distance + 0.5 * joints
                    possible.append((cost, track_id, index))
                    self.events.append(
                        {
                            "frame_index": self.frame_index,
                            "kind": "association_cost",
                            "track_id": track_id,
                            "candidate_id": detection.candidate_id,
                            "cost": cost,
                            "iou": overlap,
                            "bbox_distance": distance,
                            "joint_distance": joints,
                        }
                    )
        possible.sort()
        assigned_tracks: dict[int, int] = {}
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _cost, track_id, index in possible:
            if track_id not in used_tracks and index not in used_detections:
                assigned_tracks[index] = track_id
                used_tracks.add(track_id)
                used_detections.add(index)

        result: list[TrackedHand] = []
        for index, detection in enumerate(unique):
            track_id = assigned_tracks.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                self.events.append(
                    {
                        "frame_index": self.frame_index,
                        "kind": "reset",
                        "track_id": track_id,
                        "candidate_id": detection.candidate_id,
                    }
                )
            self._tracks[track_id] = (detection.bbox, detection.joints, self.frame_index)
            handedness = detection.handedness
            if handedness is not None:
                handedness = "left" if handedness.lower() == "left" else "right"
            result.append(TrackedHand(track_id, handedness, detection.bbox, detection.joints))
        self._expire_tracks()
        return FrameAssignment(
            tuple(sorted(result, key=lambda hand: hand.track_id)), True, None, duplicate_count
        )

    def _expire_tracks(self) -> None:
        self._tracks = {
            track_id: state
            for track_id, state in self._tracks.items()
            if self.frame_index - state[2] <= self.max_gap + 1
        }


def _validate_dimensions(width: int, height: int) -> None:
    if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION):
        raise ValueError(f"frame dimensions must be between 1 and {MAX_DIMENSION}")


def _hand_record(
    hand: TrackedHand, width: int, height: int, previous: dict[int, tuple[int, ...]], delta: bool
) -> tuple[bytes, tuple[int, ...]]:
    if not 0 <= hand.track_id <= 65535:
        raise ValueError("track_id must fit uint16")
    codes = _codes(hand, width, height)
    side = {None: 0, "left": 1, "right": 2}.get(
        hand.handedness.lower() if hand.handedness else None
    )
    if side is None:
        raise ValueError("handedness must be left, right, or None")
    prior = previous.get(hand.track_id)
    use_delta = bool(delta and prior is not None)
    values = tuple((cur - old) & 0xFFFF for cur, old in zip(codes, prior)) if use_delta else codes
    record = bytearray(_RECORD.pack(hand.track_id, side, int(use_delta)))
    record.extend(struct.pack("<46H", *values))
    previous[hand.track_id] = codes
    return bytes(record), codes


def encode_segment(
    frames: Sequence[Sequence[TrackedHand]],
    *,
    width: int,
    height: int,
    start_frame: int,
    method: str = "raw",
    fps_num: int = FPS_NUM,
    fps_den: int = FPS_DEN,
) -> bytes:
    """Encode a segment using raw, zlib, or exact modular-delta+zlib coding."""
    _validate_dimensions(width, height)
    if not (1 <= len(frames) <= MAX_FRAMES) or start_frame < 0:
        raise ValueError("segment must contain 1..1800 frames and a nonnegative start")
    if not (1 <= fps_num <= 65535 and 1 <= fps_den <= 65535):
        raise ValueError("fps must be a positive uint16 rational")
    methods = {"raw": _METHOD_RAW, "zlib": _METHOD_ZLIB, "delta_zlib": _METHOD_DELTA_ZLIB}
    if method not in methods:
        raise ValueError(f"unsupported packet method {method!r}")
    method_code = methods[method]
    plain = bytearray()
    previous: dict[int, tuple[int, ...]] = {}
    total_records = 0
    for frame in frames:
        if len(frame) > 2:
            raise ValueError("a frame may encode at most two resolved hands")
        ids = [hand.track_id for hand in frame]
        if len(ids) != len(set(ids)):
            raise ValueError("track IDs must be unique within a frame")
        ordered = sorted(frame, key=lambda hand: hand.track_id)
        plain.append(len(ordered))
        seen: set[int] = set()
        for hand in ordered:
            record, _ = _hand_record(
                hand, width, height, previous, method_code == _METHOD_DELTA_ZLIB
            )
            plain.extend(record)
            total_records += 1
            seen.add(hand.track_id)
        # Keep prior coordinates over short absences. They are encoder state
        # inside this segment only; packet decoding remains self-contained.
        previous = {
            key: value
            for key, value in previous.items()
            if key in seen or method_code == _METHOD_DELTA_ZLIB
        }
    raw = bytes(plain)
    payload = raw if method_code == _METHOD_RAW else zlib.compress(raw, level=9)
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        method_code,
        width,
        height,
        fps_num,
        fps_den,
        COORD_SCALE,
        start_frame,
        len(frames),
        total_records,
        len(payload),
        zlib.crc32(raw) & 0xFFFFFFFF,
    )
    return header + payload


def decode_segment(packet: bytes) -> dict:
    """Validate and independently decode one experimental segment."""
    if len(packet) < _HEADER.size:
        raise ValueError("truncated packet header")
    (
        magic,
        version,
        method,
        width,
        height,
        fps_num,
        fps_den,
        units,
        start_frame,
        frame_count,
        expected_records,
        payload_length,
        expected_crc,
    ) = _HEADER.unpack_from(packet)
    if magic != MAGIC or version != VERSION:
        raise ValueError("invalid packet magic or version")
    if method not in (_METHOD_RAW, _METHOD_ZLIB, _METHOD_DELTA_ZLIB):
        raise ValueError("unsupported compression method")
    _validate_dimensions(width, height)
    if units != COORD_SCALE or fps_num == 0 or fps_den == 0:
        raise ValueError("invalid coordinate units or frame rate")
    if not (1 <= frame_count <= MAX_FRAMES) or start_frame + frame_count > 2**32 - 1:
        raise ValueError("invalid segment frame range")
    if payload_length != len(packet) - _HEADER.size:
        raise ValueError("payload length mismatch or trailing bytes")
    payload = packet[_HEADER.size :]
    max_plain = frame_count * (1 + 2 * (_RECORD.size + 92))
    try:
        if method == _METHOD_RAW:
            plain = payload
        else:
            inflater = zlib.decompressobj()
            plain = inflater.decompress(payload, max_plain + 1)
            if inflater.unconsumed_tail or not inflater.eof or inflater.unused_data:
                raise ValueError("invalid or oversized compressed payload")
    except zlib.error as exc:
        raise ValueError("invalid compressed payload") from exc
    if len(plain) > max_plain or zlib.crc32(plain) & 0xFFFFFFFF != expected_crc:
        raise ValueError("payload CRC mismatch or excessive decoded length")

    offset = 0
    frames: list[list[TrackedHand]] = []
    previous: dict[int, tuple[int, ...]] = {}
    actual_records = 0
    for _frame_index in range(frame_count):
        if offset >= len(plain):
            raise ValueError("truncated frame record")
        count = plain[offset]
        offset += 1
        if count > 2:
            raise ValueError("frame has more than two resolved tracks")
        frame: list[TrackedHand] = []
        seen: set[int] = set()
        for _hand_index in range(count):
            if offset + _RECORD.size + 92 > len(plain):
                raise ValueError("truncated hand record")
            track_id, side, flags = _RECORD.unpack_from(plain, offset)
            offset += _RECORD.size
            if track_id in seen or side not in (0, 1, 2) or flags not in (0, 1):
                raise ValueError("invalid or duplicate hand record fields")
            seen.add(track_id)
            encoded = struct.unpack_from("<46H", plain, offset)
            offset += 92
            if flags:
                if method != _METHOD_DELTA_ZLIB or track_id not in previous:
                    raise ValueError("delta record has no segment-local prior")
                codes = tuple(
                    (value + prior) & 0xFFFF for value, prior in zip(encoded, previous[track_id])
                )
            else:
                codes = tuple(encoded)
            previous[track_id] = codes
            values = tuple(code / COORD_SCALE for code in codes)
            bbox = values[:4]
            joints = tuple((values[4 + 2 * i], values[5 + 2 * i]) for i in range(21))
            _as_bbox(bbox, width, height)
            _as_joints(joints, width, height)
            frame.append(TrackedHand(track_id, (None, "left", "right")[side], bbox, joints, True))
            actual_records += 1
        frames.append(frame)
    if offset != len(plain) or actual_records != expected_records:
        raise ValueError("record count or trailing payload mismatch")
    return {
        "width": width,
        "height": height,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "start_frame": start_frame,
        "method": ("raw", "zlib", "delta_zlib")[method],
        "frames": frames,
    }


def consumer_slots(decoded: dict) -> list[tuple[int, int]]:
    """Frame/track pairs a decoder may condition or score. Source ids are absent."""
    start = int(decoded["start_frame"])
    return [
        (start + offset, hand.track_id)
        for offset, frame in enumerate(decoded["frames"])
        for hand in frame
        if hand.presence
    ]


def av1_slots(decoded: dict) -> list[tuple[int, int]]:
    return consumer_slots(decoded)


def generator_slots(decoded: dict) -> list[tuple[int, int]]:
    return consumer_slots(decoded)


def accept_scored_prediction(decoded: dict, frame_index: int, track_id: int) -> None:
    if (int(frame_index), int(track_id)) not in set(consumer_slots(decoded)):
        raise KeyError("candidate is not present in the decoded packet")


def decoded_bbox_int(hand: TrackedHand, width: int, height: int) -> tuple[int, int, int, int]:
    """Apply one explicit nearest-integer rule before using the source crop."""
    x1, y1, x2, y2 = hand.bbox
    values = tuple(int(math.floor(value + 0.5)) for value in (x1, y1, x2, y2))
    bx1 = min(width - 1, max(0, values[0]))
    by1 = min(height - 1, max(0, values[1]))
    bx2 = min(width, max(bx1 + 1, values[2]))
    by2 = min(height, max(by1 + 1, values[3]))
    return bx1, by1, bx2, by2


def map_points_to_letterbox(
    points: Sequence[Sequence[float]], meta: dict[str, float]
) -> np.ndarray:
    """Map full-frame points with the realized integer resize dimensions."""
    x1, y1 = meta["orig_x1"], meta["orig_y1"]
    orig_w, orig_h = meta["orig_w"], meta["orig_h"]
    sx = meta["new_w"] / orig_w
    sy = meta["new_h"] / orig_h
    return np.asarray(
        [
            [(float(x) - x1) * sx + meta["pad_x"], (float(y) - y1) * sy + meta["pad_y"]]
            for x, y in points
        ],
        dtype=np.float64,
    )

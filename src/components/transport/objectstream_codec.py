"""``pointstream.objectstream.v1``: one wire format for per-object geometry.

Boxes and keypoints for every tracked object in a segment travel in one
bitstream. It replaces four formats that grew separately:

- The main codec's raw float16 keypoints and JSON boxes. These had no
  prediction and no entropy coding.
- The demo's PK hand packets: 47 fixed bytes per hand, u8 box-relative.
- DWB2: closed-loop deltas and Exp-Golomb codes, but slots are matched by
  order rather than identity.
- PSFG v2: 1/16-pixel lossless codes, track ids, CRC and independent segments,
  but uint16 deltas under zlib.

Two layers are kept apart on purpose.

**Quantizer** (floats <-> integer codes). Each axis is put on a declared grid:

- the frame (``2**bits - 1`` steps across the frame, as in DWB2/PK boxes);
- the pixel (``2**bits`` codes per pixel; 4 bits is PSFG's 1/16 px);
- the object's own reconstructed box (``2**bits - 1`` steps across it, as in
  DWB2/PK landmarks).

Joints are quantized against the *reconstructed* box. Encoder and decoder
therefore share it, which the PK packer did not.

**Entropy layer** (codes <-> bits). This layer is lossless over integer codes.

- *Track state:* every track opens a segment with an absolute record.
  Later records choose, per record, between absolute fixed-width codes and
  signed Exp-Golomb-k residuals against the track's last reconstructed codes,
  whichever is shorter.
- *Optional box-motion prediction:* pixel-grid joints may also be predicted by
  shifting the track's last joints with the box-centre motion.
- *Absent joints* cost one bit when the present set is unchanged.
- *Exp-Golomb order:* the encoder picks the order (0..7) separately for box
  and joint residuals and declares it in the header.
- *Self-contained segments:* the CRC32 covers the payload, and every segment
  decodes on its own.

Since the entropy layer is lossless, parity with DWB2 and PSFG is checked on
codes: their decoded integers re-encode here and come back identical.
"""

from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

MAGIC: Final = b"PSOS"
VERSION: Final = 1
SCHEMA: Final = "pointstream.objectstream.v1"

GRID_FRAME: Final = "frame"
GRID_PIXEL: Final = "pixel"
GRID_BOX: Final = "box"
_GRID_CODES: Final = {GRID_FRAME: 0, GRID_PIXEL: 1, GRID_BOX: 2}
_GRID_NAMES: Final = {code: name for name, code in _GRID_CODES.items()}

_FLAG_BOX_MOTION: Final = 0x01
_MAX_K: Final = 7
_MAX_FRAMES: Final = 1 << 20
_MAX_DIMENSION: Final = 1 << 15

_HEADER: Final = struct.Struct("<4sBBHHHHBBBBBBIIB")
_CLASS: Final = struct.Struct("<BBBB")
_TRAILER: Final = struct.Struct("<II")


# --------------------------------------------------------------------------
# Declarations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectClass:
    """One class of object the segment carries.

    Args:
        name: Class name, as the domain profile spells it (``hand``, ``player``).
        joints: Keypoints per object; 0 for classes without a skeleton.
        attribute_bits: Raw per-record attribute width, such as hand side.
        confidence_bits: Raw per-record confidence width (DWB2 uses 7).
    """

    name: str
    joints: int = 0
    attribute_bits: int = 0
    confidence_bits: int = 0

    def __post_init__(self) -> None:
        encoded = self.name.encode("utf-8")
        if not encoded or len(encoded) > 255:
            raise ValueError("ObjectClass name must be 1..255 UTF-8 bytes.")
        if not 0 <= self.joints <= 255:
            raise ValueError("ObjectClass joints must fit in a byte.")
        if not 0 <= self.attribute_bits <= 16 or not 0 <= self.confidence_bits <= 16:
            raise ValueError("Attribute and confidence widths must be 0..16 bits.")


@dataclass(frozen=True)
class StreamConfig:
    """Frame geometry and quantization grids for one segment."""

    width: int
    height: int
    fps_num: int = 30
    fps_den: int = 1
    box_grid: str = GRID_PIXEL
    box_bits: int = 4
    joint_grid: str = GRID_PIXEL
    joint_bits: int = 4
    box_motion_prediction: bool = False

    def __post_init__(self) -> None:
        if not (1 <= self.width < _MAX_DIMENSION and 1 <= self.height < _MAX_DIMENSION):
            raise ValueError(f"Frame dimensions must be 1..{_MAX_DIMENSION - 1}.")
        if not (1 <= self.fps_num <= 65535 and 1 <= self.fps_den <= 65535):
            raise ValueError("fps must be a positive uint16 rational.")
        if self.box_grid not in (GRID_FRAME, GRID_PIXEL):
            raise ValueError(f"box_grid must be {GRID_FRAME!r} or {GRID_PIXEL!r}.")
        if self.joint_grid not in (GRID_BOX, GRID_PIXEL):
            raise ValueError(f"joint_grid must be {GRID_BOX!r} or {GRID_PIXEL!r}.")
        for name in ("box_bits", "joint_bits"):
            grid = self.box_grid if name == "box_bits" else self.joint_grid
            bits = getattr(self, name)
            low = 0 if grid == GRID_PIXEL else 1
            high = 6 if grid == GRID_PIXEL else 16
            if not low <= bits <= high:
                raise ValueError(f"{name} must be {low}..{high} on the {grid} grid.")
        if self.box_motion_prediction and self.joint_grid == GRID_PIXEL and self.box_grid != GRID_PIXEL:
            raise ValueError("Pixel-grid box-motion prediction needs pixel-grid boxes.")

    def box_limits(self) -> tuple[int, int]:
        """Largest legal box code on x and on y."""
        return (_axis_limit(self.box_grid, self.box_bits, self.width),
                _axis_limit(self.box_grid, self.box_bits, self.height))

    def joint_limits(self) -> tuple[int, int]:
        """Largest legal joint code on x and on y."""
        if self.joint_grid == GRID_BOX:
            top = (1 << self.joint_bits) - 1
            return (top, top)
        return (_axis_limit(GRID_PIXEL, self.joint_bits, self.width),
                _axis_limit(GRID_PIXEL, self.joint_bits, self.height))


def _axis_limit(grid: str, bits: int, dimension: int) -> int:
    if grid == GRID_FRAME:
        return (1 << bits) - 1
    return dimension << bits


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CodedObject:
    """One object in one frame, as integer codes on the declared grids.

    ``joints`` is flat ``x0, y0, x1, y1, ...``. Entries for absent joints are
    carried as 0 and must be ignored by consumers.
    """

    track_id: int
    class_index: int
    box: tuple[int, int, int, int]
    joints: tuple[int, ...] = ()
    present: tuple[bool, ...] = ()
    attribute: int = 0
    confidence: int = 0


@dataclass(frozen=True)
class ObjectObservation:
    """One object in one frame, in pixels."""

    track_id: int
    class_index: int
    box: tuple[float, float, float, float]
    joints: tuple[tuple[float, float], ...] = ()
    present: tuple[bool, ...] | None = None
    attribute: int = 0
    confidence: int = 0


@dataclass
class QuantizationReport:
    """What quantization did to the input: clamps and the worst error in pixels."""

    clamped_codes: int = 0
    max_box_error_px: float = 0.0
    max_joint_error_px: float = 0.0


@dataclass(frozen=True)
class EncodedSegment:
    """A packed segment and the byte split the ledger charges."""

    data: bytes
    header_bytes: int
    payload_bytes: int
    k_box: int
    k_joint: int
    quantization: QuantizationReport = field(default_factory=QuantizationReport)

    @property
    def total_bytes(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class DecodedSegment:
    """Everything a receiver needs; nothing outside the packet is consulted."""

    config: StreamConfig
    classes: tuple[ObjectClass, ...]
    segment_start: int
    frames: tuple[tuple[CodedObject, ...], ...]

    def observations(self) -> tuple[tuple[ObjectObservation, ...], ...]:
        """Frames dequantized to pixels."""
        return tuple(
            tuple(dequantize(self.config, self.classes, item) for item in frame)
            for frame in self.frames
        )


# --------------------------------------------------------------------------
# Quantizer
# --------------------------------------------------------------------------


def _round(value: float) -> int:
    return int(math.floor(value + 0.5))


def _clamp(code: int, top: int, report: QuantizationReport) -> int:
    if code < 0:
        report.clamped_codes += 1
        return 0
    if code > top:
        report.clamped_codes += 1
        return top
    return code


def _box_to_codes(
    config: StreamConfig, box: Sequence[float], report: QuantizationReport
) -> tuple[int, int, int, int]:
    limit_x, limit_y = config.box_limits()
    # Box-relative joints need the reconstructed box to contain the source box,
    # or joints on its edge would be clamped. Round outward in that case.
    outward = config.joint_grid == GRID_BOX
    codes = []
    for index, value in enumerate(box):
        if not math.isfinite(value):
            raise ValueError("Box coordinates must be finite.")
        dimension = config.width if index % 2 == 0 else config.height
        top = limit_x if index % 2 == 0 else limit_y
        if config.box_grid == GRID_FRAME:
            scaled = value / dimension * top
        else:
            scaled = value * (1 << config.box_bits)
        if not outward:
            raw = _round(scaled)
        elif index < 2:
            raw = math.floor(scaled + 1e-9)
        else:
            raw = math.ceil(scaled - 1e-9)
        codes.append(_clamp(raw, top, report))
    return (codes[0], codes[1], codes[2], codes[3])


def _box_from_codes(config: StreamConfig, codes: Sequence[int]) -> tuple[float, float, float, float]:
    limit_x, limit_y = config.box_limits()
    values = []
    for index, code in enumerate(codes):
        if config.box_grid == GRID_FRAME:
            dimension = config.width if index % 2 == 0 else config.height
            top = limit_x if index % 2 == 0 else limit_y
            values.append(code / top * dimension)
        else:
            values.append(code / (1 << config.box_bits))
    return (values[0], values[1], values[2], values[3])


def _box_span(box: Sequence[float]) -> tuple[float, float]:
    return (max(box[2] - box[0], 1e-6), max(box[3] - box[1], 1e-6))


def quantize(
    config: StreamConfig,
    classes: Sequence[ObjectClass],
    observation: ObjectObservation,
    report: QuantizationReport | None = None,
) -> CodedObject:
    """Codes for one observation; joints are placed in the reconstructed box."""
    report = report if report is not None else QuantizationReport()
    declared = classes[observation.class_index]
    box_codes = _box_to_codes(config, observation.box, report)
    box = _box_from_codes(config, box_codes)
    for raw, rec in zip(observation.box, box):
        report.max_box_error_px = max(report.max_box_error_px, abs(raw - rec))
    if len(observation.joints) != declared.joints:
        raise ValueError(
            f"Class {declared.name!r} declares {declared.joints} joints, "
            f"got {len(observation.joints)}."
        )
    present = (
        tuple(bool(item) for item in observation.present)
        if observation.present is not None
        else (True,) * declared.joints
    )
    if len(present) != declared.joints:
        raise ValueError("present must have one flag per joint.")
    limit_x, limit_y = config.joint_limits()
    span_x, span_y = _box_span(box)
    codes: list[int] = []
    for (x, y), flag in zip(observation.joints, present):
        if not flag:
            codes.extend((0, 0))
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("Present joints must be finite.")
        if config.joint_grid == GRID_BOX:
            cx = _clamp(_round((x - box[0]) / span_x * limit_x), limit_x, report)
            cy = _clamp(_round((y - box[1]) / span_y * limit_y), limit_y, report)
        else:
            scale = 1 << config.joint_bits
            cx = _clamp(_round(x * scale), limit_x, report)
            cy = _clamp(_round(y * scale), limit_y, report)
        codes.extend((cx, cy))
    coded = CodedObject(
        track_id=observation.track_id,
        class_index=observation.class_index,
        box=box_codes,
        joints=tuple(codes),
        present=present,
        attribute=observation.attribute,
        confidence=observation.confidence,
    )
    rebuilt = dequantize(config, classes, coded)
    for (x, y), (rx, ry), flag in zip(observation.joints, rebuilt.joints, present):
        if flag:
            report.max_joint_error_px = max(report.max_joint_error_px, abs(x - rx), abs(y - ry))
    return coded


def dequantize(
    config: StreamConfig, classes: Sequence[ObjectClass], coded: CodedObject
) -> ObjectObservation:
    """Pixels for one coded object."""
    box = _box_from_codes(config, coded.box)
    limit_x, limit_y = config.joint_limits()
    span_x, span_y = _box_span(box)
    joints: list[tuple[float, float]] = []
    for index in range(len(coded.present)):
        cx, cy = coded.joints[2 * index], coded.joints[2 * index + 1]
        if not coded.present[index]:
            joints.append((math.nan, math.nan))
        elif config.joint_grid == GRID_BOX:
            joints.append((box[0] + cx / limit_x * span_x, box[1] + cy / limit_y * span_y))
        else:
            scale = 1 << config.joint_bits
            joints.append((cx / scale, cy / scale))
    return ObjectObservation(
        track_id=coded.track_id,
        class_index=coded.class_index,
        box=box,
        joints=tuple(joints),
        present=tuple(coded.present),
        attribute=coded.attribute,
        confidence=coded.confidence,
    )


# --------------------------------------------------------------------------
# Bits
# --------------------------------------------------------------------------


class _BitWriter:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._accumulator = 0
        self._count = 0

    def write(self, value: int, bits: int) -> None:
        if bits == 0:
            return
        if value < 0 or value >> bits:
            raise ValueError(f"{value} does not fit in {bits} bits.")
        self._accumulator = (self._accumulator << bits) | value
        self._count += bits
        while self._count >= 8:
            self._count -= 8
            self._buffer.append((self._accumulator >> self._count) & 0xFF)
        self._accumulator &= (1 << self._count) - 1

    def ue(self, value: int, k: int = 0) -> None:
        prefix = (value >> k) + 1
        width = prefix.bit_length() - 1
        self.write(0, width)
        self.write(prefix, width + 1)
        self.write(value & ((1 << k) - 1), k)

    def se(self, value: int, k: int = 0) -> None:
        self.ue(_zigzag(value), k)

    def finish(self) -> bytes:
        if self._count:
            self._buffer.append((self._accumulator << (8 - self._count)) & 0xFF)
            self._accumulator = 0
            self._count = 0
        return bytes(self._buffer)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._position = 0

    def read(self, bits: int) -> int:
        value = 0
        for _ in range(bits):
            byte, offset = divmod(self._position, 8)
            if byte >= len(self._data):
                raise ValueError("Truncated objectstream payload.")
            value = (value << 1) | ((self._data[byte] >> (7 - offset)) & 1)
            self._position += 1
        return value

    def ue(self, k: int = 0) -> int:
        width = 0
        while self.read(1) == 0:
            width += 1
            if width > 40:
                raise ValueError("Exp-Golomb prefix is implausibly long.")
        prefix = (1 << width) | self.read(width)
        return ((prefix - 1) << k) | self.read(k)

    def se(self, k: int = 0) -> int:
        return _unzigzag(self.ue(k))

    def finish(self) -> None:
        byte, offset = divmod(self._position, 8)
        if offset:
            if self._data[byte] & ((1 << (8 - offset)) - 1):
                raise ValueError("Nonzero padding after the last record.")
            byte += 1
        if byte != len(self._data):
            raise ValueError("Trailing bytes after the last record.")


def _zigzag(value: int) -> int:
    return value << 1 if value >= 0 else ((-value) << 1) - 1


def _unzigzag(code: int) -> int:
    return -((code + 1) >> 1) if code & 1 else code >> 1


def _ue_bits(value: int, k: int) -> int:
    width = ((value >> k) + 1).bit_length() - 1
    return 2 * width + 1 + k


def _se_bits(value: int, k: int) -> int:
    return _ue_bits(_zigzag(value), k)


# --------------------------------------------------------------------------
# Entropy layer
# --------------------------------------------------------------------------


@dataclass
class _Track:
    box: tuple[int, int, int, int]
    joints: tuple[int, ...]
    present: tuple[bool, ...]


def _fixed_widths(config: StreamConfig) -> tuple[tuple[int, int], tuple[int, int]]:
    box_x, box_y = config.box_limits()
    joint_x, joint_y = config.joint_limits()
    return (
        (box_x.bit_length(), box_y.bit_length()),
        (joint_x.bit_length(), joint_y.bit_length()),
    )


def _joint_prediction(
    config: StreamConfig, state: _Track, box: Sequence[int], index: int
) -> tuple[int, int]:
    px, py = state.joints[2 * index], state.joints[2 * index + 1]
    if not config.box_motion_prediction:
        return px, py
    if config.joint_grid == GRID_BOX:
        # Re-anchor: the joint's last pixel position, expressed in the new box.
        # Without this, box jitter shifts every box-relative code at once.
        limit_x, limit_y = config.joint_limits()
        old = _box_from_codes(config, state.box)
        new = _box_from_codes(config, box)
        old_x, old_y = _box_span(old)
        new_x, new_y = _box_span(new)
        x = old[0] + px / limit_x * old_x
        y = old[1] + py / limit_y * old_y
        return _round((x - new[0]) / new_x * limit_x), _round((y - new[1]) / new_y * limit_y)
    if config.box_grid != GRID_PIXEL:
        return px, py
    scale = (1 << config.joint_bits) / (1 << config.box_bits)
    dx = ((box[0] + box[2]) - (state.box[0] + state.box[2])) / 2 * scale
    dy = ((box[1] + box[3]) - (state.box[1] + state.box[3])) / 2 * scale
    return px + _round(dx), py + _round(dy)


def _residuals(
    config: StreamConfig, item: CodedObject, state: _Track
) -> tuple[list[int], list[int | None]]:
    box = [cur - old for cur, old in zip(item.box, state.box)]
    joints: list[int | None] = []
    for index, flag in enumerate(item.present):
        if not flag:
            continue
        if index < len(state.present) and state.present[index]:
            px, py = _joint_prediction(config, state, item.box, index)
            joints.extend((item.joints[2 * index] - px, item.joints[2 * index + 1] - py))
        else:
            joints.extend((None, None))
    return box, joints


def _validate_codes(config: StreamConfig, classes: Sequence[ObjectClass], item: CodedObject) -> None:
    if not 0 <= item.class_index < len(classes):
        raise ValueError(f"Class index {item.class_index} is not declared.")
    declared = classes[item.class_index]
    if len(item.present) != declared.joints or len(item.joints) != 2 * declared.joints:
        raise ValueError(f"Object of class {declared.name!r} has the wrong joint count.")
    if item.track_id < 0:
        raise ValueError("Track ids must be nonnegative.")
    if item.attribute >> declared.attribute_bits or item.attribute < 0:
        raise ValueError("Attribute does not fit its declared width.")
    if item.confidence >> declared.confidence_bits or item.confidence < 0:
        raise ValueError("Confidence does not fit its declared width.")
    box_x, box_y = config.box_limits()
    joint_x, joint_y = config.joint_limits()
    for index, code in enumerate(item.box):
        if not 0 <= code <= (box_x if index % 2 == 0 else box_y):
            raise ValueError("Box code is outside the declared grid.")
    for index, flag in enumerate(item.present):
        if flag and not (
            0 <= item.joints[2 * index] <= joint_x and 0 <= item.joints[2 * index + 1] <= joint_y
        ):
            raise ValueError("Joint code is outside the declared grid.")


def _best_k(values: Sequence[int]) -> int:
    if not values:
        return 0
    return min(range(_MAX_K + 1), key=lambda k: sum(_se_bits(v, k) for v in values))


def encode_codes(
    config: StreamConfig,
    classes: Sequence[ObjectClass],
    frames: Sequence[Sequence[CodedObject]],
    *,
    segment_start: int = 0,
) -> EncodedSegment:
    """Pack coded objects. Lossless: ``decode`` returns these codes exactly."""
    classes = tuple(classes)
    if not 1 <= len(frames) <= _MAX_FRAMES:
        raise ValueError(f"A segment holds 1..{_MAX_FRAMES} frames.")
    if not 1 <= len(classes) <= 255:
        raise ValueError("A segment declares 1..255 classes.")
    if not 0 <= segment_start < 1 << 32:
        raise ValueError("segment_start must fit uint32.")

    ordered: list[list[CodedObject]] = []
    box_residuals: list[int] = []
    joint_residuals: list[int] = []
    states: dict[int, _Track] = {}
    track_class: dict[int, int] = {}
    for frame in frames:
        items = sorted(frame, key=lambda item: item.track_id)
        ids = [item.track_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("Track ids must be unique within a frame.")
        for item in items:
            _validate_codes(config, classes, item)
            if track_class.setdefault(item.track_id, item.class_index) != item.class_index:
                raise ValueError(f"Track {item.track_id} changes class inside a segment.")
            state = states.get(item.track_id)
            if state is not None:
                box, joints = _residuals(config, item, state)
                box_residuals.extend(box)
                joint_residuals.extend(value for value in joints if value is not None)
            states[item.track_id] = _Track(item.box, item.joints, item.present)
        ordered.append(items)
    k_box = _best_k(box_residuals)
    k_joint = _best_k(joint_residuals)

    (box_wx, box_wy), (joint_wx, joint_wy) = _fixed_widths(config)
    writer = _BitWriter()
    states = {}
    for items in ordered:
        writer.ue(len(items))
        previous_id = -1
        for item in items:
            writer.ue(item.track_id - previous_id - 1)
            previous_id = item.track_id
            declared = classes[item.class_index]
            state = states.get(item.track_id)
            if state is None:
                writer.ue(item.class_index)
            writer.write(item.attribute, declared.attribute_bits)
            writer.write(item.confidence, declared.confidence_bits)
            if declared.joints:
                if state is not None and item.present == state.present:
                    writer.write(1, 1)
                else:
                    writer.write(0, 1)
                    for flag in item.present:
                        writer.write(int(flag), 1)
            absolute = sum(box_wx if i % 2 == 0 else box_wy for i in range(4)) + sum(
                joint_wx + joint_wy for flag in item.present if flag
            )
            use_delta = False
            if state is not None:
                box, joints = _residuals(config, item, state)
                delta = sum(_se_bits(v, k_box) for v in box) + sum(
                    _se_bits(v, k_joint) if v is not None else (joint_wx + joint_wy) / 2
                    for v in joints
                )
                use_delta = delta < absolute
                writer.write(int(use_delta), 1)
            if use_delta:
                assert state is not None
                box, joints = _residuals(config, item, state)
                for value in box:
                    writer.se(value, k_box)
                cursor = 0
                for index, flag in enumerate(item.present):
                    if not flag:
                        continue
                    pair = joints[cursor : cursor + 2]
                    cursor += 2
                    if pair[0] is None:
                        writer.write(item.joints[2 * index], joint_wx)
                        writer.write(item.joints[2 * index + 1], joint_wy)
                    else:
                        writer.se(pair[0], k_joint)
                        writer.se(pair[1], k_joint)  # type: ignore[arg-type]
            else:
                for index, code in enumerate(item.box):
                    writer.write(code, box_wx if index % 2 == 0 else box_wy)
                for index, flag in enumerate(item.present):
                    if flag:
                        writer.write(item.joints[2 * index], joint_wx)
                        writer.write(item.joints[2 * index + 1], joint_wy)
            states[item.track_id] = _Track(item.box, item.joints, item.present)
    payload = writer.finish()

    header = bytearray(
        _HEADER.pack(
            MAGIC,
            VERSION,
            _FLAG_BOX_MOTION if config.box_motion_prediction else 0,
            config.width,
            config.height,
            config.fps_num,
            config.fps_den,
            _GRID_CODES[config.box_grid],
            config.box_bits,
            _GRID_CODES[config.joint_grid],
            config.joint_bits,
            k_box,
            k_joint,
            segment_start,
            len(frames),
            len(classes),
        )
    )
    for declared in classes:
        name = declared.name.encode("utf-8")
        header.extend(
            _CLASS.pack(declared.joints, declared.attribute_bits, declared.confidence_bits, len(name))
        )
        header.extend(name)
    header.extend(_TRAILER.pack(len(payload), zlib.crc32(payload) & 0xFFFFFFFF))
    return EncodedSegment(
        data=bytes(header) + payload,
        header_bytes=len(header),
        payload_bytes=len(payload),
        k_box=k_box,
        k_joint=k_joint,
    )


def encode(
    config: StreamConfig,
    classes: Sequence[ObjectClass],
    frames: Sequence[Sequence[ObjectObservation]],
    *,
    segment_start: int = 0,
) -> EncodedSegment:
    """Quantize pixel observations, then pack them."""
    report = QuantizationReport()
    coded = [[quantize(config, classes, item, report) for item in frame] for frame in frames]
    packed = encode_codes(config, classes, coded, segment_start=segment_start)
    return EncodedSegment(
        data=packed.data,
        header_bytes=packed.header_bytes,
        payload_bytes=packed.payload_bytes,
        k_box=packed.k_box,
        k_joint=packed.k_joint,
        quantization=report,
    )


def decode(data: bytes) -> DecodedSegment:
    """Decode one segment from its bytes alone.

    Raises:
        ValueError: On any malformed, truncated or corrupted segment.
    """
    if len(data) < _HEADER.size:
        raise ValueError("Truncated objectstream header.")
    (
        magic,
        version,
        flags,
        width,
        height,
        fps_num,
        fps_den,
        box_grid,
        box_bits,
        joint_grid,
        joint_bits,
        k_box,
        k_joint,
        segment_start,
        frame_count,
        class_count,
    ) = _HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError("Not an objectstream v1 segment.")
    if flags & ~_FLAG_BOX_MOTION or k_box > _MAX_K or k_joint > _MAX_K:
        raise ValueError("Unsupported objectstream flags or Exp-Golomb order.")
    if box_grid not in _GRID_NAMES or joint_grid not in _GRID_NAMES:
        raise ValueError("Unknown quantization grid.")
    if not 1 <= frame_count <= _MAX_FRAMES or class_count < 1:
        raise ValueError("Invalid frame or class count.")
    config = StreamConfig(
        width=width,
        height=height,
        fps_num=fps_num,
        fps_den=fps_den,
        box_grid=_GRID_NAMES[box_grid],
        box_bits=box_bits,
        joint_grid=_GRID_NAMES[joint_grid],
        joint_bits=joint_bits,
        box_motion_prediction=bool(flags & _FLAG_BOX_MOTION),
    )
    offset = _HEADER.size
    classes: list[ObjectClass] = []
    for _ in range(class_count):
        if offset + _CLASS.size > len(data):
            raise ValueError("Truncated class table.")
        joints, attribute_bits, confidence_bits, name_length = _CLASS.unpack_from(data, offset)
        offset += _CLASS.size
        if offset + name_length > len(data):
            raise ValueError("Truncated class name.")
        name = data[offset : offset + name_length].decode("utf-8")
        offset += name_length
        classes.append(ObjectClass(name, joints, attribute_bits, confidence_bits))
    if offset + _TRAILER.size > len(data):
        raise ValueError("Truncated payload descriptor.")
    payload_length, expected_crc = _TRAILER.unpack_from(data, offset)
    offset += _TRAILER.size
    payload = data[offset:]
    if len(payload) != payload_length:
        raise ValueError("Payload length mismatch or trailing bytes.")
    if zlib.crc32(payload) & 0xFFFFFFFF != expected_crc:
        raise ValueError("Payload CRC mismatch.")

    (box_wx, box_wy), (joint_wx, joint_wy) = _fixed_widths(config)
    reader = _BitReader(payload)
    states: dict[int, _Track] = {}
    track_class: dict[int, int] = {}
    frames: list[tuple[CodedObject, ...]] = []
    for _ in range(frame_count):
        count = reader.ue()
        previous_id = -1
        frame: list[CodedObject] = []
        for _ in range(count):
            track_id = previous_id + 1 + reader.ue()
            previous_id = track_id
            state = states.get(track_id)
            if state is None:
                class_index = reader.ue()
                if class_index >= len(classes):
                    raise ValueError("Record names an undeclared class.")
                track_class[track_id] = class_index
            class_index = track_class[track_id]
            declared = classes[class_index]
            attribute = reader.read(declared.attribute_bits)
            confidence = reader.read(declared.confidence_bits)
            if declared.joints:
                if reader.read(1):
                    if state is None:
                        raise ValueError("Repeated present set without a prior record.")
                    present = state.present
                else:
                    present = tuple(bool(reader.read(1)) for _ in range(declared.joints))
            else:
                present = ()
            use_delta = bool(reader.read(1)) if state is not None else False
            joints = [0] * (2 * declared.joints)
            if use_delta:
                assert state is not None
                box = tuple(old + reader.se(k_box) for old in state.box)
                for index, flag in enumerate(present):
                    if not flag:
                        continue
                    if index < len(state.present) and state.present[index]:
                        px, py = _joint_prediction(config, state, box, index)
                        joints[2 * index] = px + reader.se(k_joint)
                        joints[2 * index + 1] = py + reader.se(k_joint)
                    else:
                        joints[2 * index] = reader.read(joint_wx)
                        joints[2 * index + 1] = reader.read(joint_wy)
            else:
                box = tuple(reader.read(box_wx if i % 2 == 0 else box_wy) for i in range(4))
                for index, flag in enumerate(present):
                    if flag:
                        joints[2 * index] = reader.read(joint_wx)
                        joints[2 * index + 1] = reader.read(joint_wy)
            item = CodedObject(
                track_id=track_id,
                class_index=class_index,
                box=(box[0], box[1], box[2], box[3]),
                joints=tuple(joints),
                present=tuple(present),
                attribute=attribute,
                confidence=confidence,
            )
            _validate_codes(config, classes, item)
            states[track_id] = _Track(item.box, item.joints, item.present)
            frame.append(item)
        frames.append(tuple(frame))
    reader.finish()
    return DecodedSegment(
        config=config,
        classes=tuple(classes),
        segment_start=segment_start,
        frames=tuple(frames),
    )

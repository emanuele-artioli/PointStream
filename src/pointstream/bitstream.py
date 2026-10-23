"""PointStream's bitstream: one container per window, every byte of it counted as rate (PLAN H5).

Layout (big-endian)::

    header   b"PSTM", version u8, width u16, height u16, fps numerator u32, fps denominator u32, frames u16
    segment  tag u8, payload length (LEB128), payload            (repeated, in any order)

Segments (`Tag`):

* ``BACKGROUND``: route u8 (`ROUTES`), codec u8 (`CODECS`), coded width u16, coded height u16, then the
  codec's stream: AV1 as its bare OBU sequence (IVF's framing removed, as B2 counts SVT-AV1's payload;
  `ivf_to_obus` / `obus_to_ivf`), DCVC-UF as its own container (`src.codecs.dcvc_uf_worker`, whole file).
* ``FOREGROUND``: scale u8 (1, 2 or 4: coded at width/scale), codec u8, coded width u16, coded height u16,
  slot count u8 and per slot (class u8, enhancer id u8), then the codec's stream.
* ``MASK``: the foreground's per-frame class map (absent while F is granted to the decoder, H5's first run).
* ``MODEL``: kind u8 (`MODEL_KINDS`), then the first 8 bytes of the model's sha256: a reference to a network
  the decoder already holds (shipped once per venue, or with the decoder), never its weights.
* ``BALL``: the ball mode's positions and per-window appearance (`src.pointstream.ball`, PLAN K5b).
* ``CARRIER``: F carried by the foreground stream itself (no MASK): key R, G, B u8, threshold u8, keep radius u8
  (`src.pointstream.composite.carrier_payload`, PLAN H5p 4c).

The rate of a window is the container's length.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from fractions import Fraction

MAGIC = b"PSTM"
VERSION = 1
HEADER = struct.Struct(">4sBHHIIH")
CODED = struct.Struct(">BBHH")  # route or scale, codec, coded width, coded height
IVF_HEADER = 32
IVF_FRAME_HEADER = 12
OBU_TEMPORAL_DELIMITER = 2
REFERENCE_BYTES = 8


class Tag(IntEnum):
    BACKGROUND = 1
    FOREGROUND = 2
    MASK = 3
    MODEL = 4
    BALL = 5
    CARRIER = 6


ROUTES = {"svt": 0, "enh": 1, "scene": 2, "dcvc": 3}
CODECS = {"av1": 0, "dcvc-uf": 1}
CLASSES = {"hand": 1, "object": 2}
MODEL_KINDS = {"background": 0, "foreground": 1}


def leb128(value: int) -> bytes:
    if value < 0:
        raise ValueError("negative length")
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def read_leb128(data: bytes, offset: int) -> tuple[int, int]:
    """(value, offset after it)."""
    value = shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated LEB128")
        byte = data[offset]
        value |= (byte & 0x7F) << shift
        offset += 1
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 56:
            raise ValueError("LEB128 too long")


# ----------------------------------------------------------------- AV1: IVF <-> bare OBUs


def ivf_to_obus(data: bytes) -> tuple[bytes, int]:
    """The concatenated temporal units of an IVF file and their count."""
    if len(data) < IVF_HEADER or data[:4] != b"DKIF":
        raise ValueError("not an IVF file")
    offset = struct.unpack("<H", data[6:8])[0]
    units = []
    while offset < len(data):
        if offset + IVF_FRAME_HEADER > len(data):
            raise ValueError("truncated IVF frame header")
        size = struct.unpack("<I", data[offset:offset + 4])[0]
        units.append(data[offset + IVF_FRAME_HEADER:offset + IVF_FRAME_HEADER + size])
        offset += IVF_FRAME_HEADER + size
        if offset > len(data):
            raise ValueError("truncated IVF frame")
    return b"".join(units), len(units)


def temporal_units(obus: bytes) -> list[bytes]:
    """Split a low-overhead AV1 OBU sequence (every OBU with its size field) at its temporal delimiters."""
    units: list[bytes] = []
    start = offset = 0
    while offset < len(obus):
        header = obus[offset]
        if header & 0x80:
            raise ValueError("forbidden bit set in an OBU header")
        kind = (header >> 3) & 0x0F
        if not header & 0x02:
            raise ValueError("an OBU without its size field")
        body = offset + 1 + (1 if header & 0x04 else 0)
        size, body = read_leb128(obus, body)
        if kind == OBU_TEMPORAL_DELIMITER and offset > start:
            units.append(obus[start:offset])
            start = offset
        offset = body + size
        if offset > len(obus):
            raise ValueError("truncated OBU")
    if offset > start:
        units.append(obus[start:offset])
    return units


def obus_to_ivf(obus: bytes, width: int, height: int, fps: Fraction) -> bytes:
    """An IVF file (AV01) holding ``obus``, one IVF frame per temporal unit, timestamps 0, 1, 2…."""
    units = temporal_units(obus)
    head = struct.pack("<4sHH4sHHIIII", b"DKIF", 0, IVF_HEADER, b"AV01", width, height, fps.numerator,
                       fps.denominator, len(units), 0)  # time base: rate then scale; 4 bytes reserved
    return head + b"".join(struct.pack("<IQ", len(u), t) + u for t, u in enumerate(units))


# ----------------------------------------------------------------- segments


@dataclass(frozen=True)
class Coded:
    """A background or foreground stream: its route (background) or scale (foreground), codec and coded size,
    the enhancer slots (foreground only) and the codec's own stream."""

    kind: str  # "background" | "foreground"
    selector: int  # ROUTES value (background) or scale (foreground)
    codec: str
    width: int
    height: int
    stream: bytes
    slots: tuple[tuple[int, int], ...] = ()

    def payload(self) -> bytes:
        head = CODED.pack(self.selector, CODECS[self.codec], self.width, self.height)
        if self.kind == "foreground":
            head += bytes([len(self.slots)]) + b"".join(bytes([c, e]) for c, e in self.slots)
        elif self.slots:
            raise ValueError("a background stream has no enhancer slots")
        return head + self.stream

    @staticmethod
    def parse(kind: str, payload: bytes) -> Coded:
        selector, codec_id, width, height = CODED.unpack_from(payload)
        codec = next(name for name, value in CODECS.items() if value == codec_id)
        offset, slots = CODED.size, ()  # type: tuple[int, tuple[tuple[int, int], ...]]
        if kind == "foreground":
            count = payload[offset]
            slots = tuple((payload[offset + 1 + 2 * i], payload[offset + 2 + 2 * i]) for i in range(count))
            offset += 1 + 2 * count
        return Coded(kind, selector, codec, width, height, payload[offset:], slots)


@dataclass(frozen=True)
class ModelRef:
    kind: str  # MODEL_KINDS key
    sha256: str  # hex; the first REFERENCE_BYTES are sent

    def payload(self) -> bytes:
        return bytes([MODEL_KINDS[self.kind]]) + bytes.fromhex(self.sha256)[:REFERENCE_BYTES]

    @staticmethod
    def parse(payload: bytes) -> ModelRef:
        kind = next(name for name, value in MODEL_KINDS.items() if value == payload[0])
        if len(payload) != 1 + REFERENCE_BYTES:
            raise ValueError("a model reference is a kind and an 8-byte sha256 prefix")
        return ModelRef(kind, payload[1:].hex())


@dataclass
class Container:
    width: int
    height: int
    fps: Fraction
    frames: int
    segments: list[tuple[Tag, bytes]] = field(default_factory=list)

    def add(self, tag: Tag, payload: bytes) -> None:
        self.segments.append((tag, payload))

    def add_coded(self, coded: Coded) -> None:
        self.add(Tag.BACKGROUND if coded.kind == "background" else Tag.FOREGROUND, coded.payload())

    def add_model(self, ref: ModelRef) -> None:
        self.add(Tag.MODEL, ref.payload())

    def pack(self) -> bytes:
        out = bytearray(HEADER.pack(MAGIC, VERSION, self.width, self.height, self.fps.numerator,
                                    self.fps.denominator, self.frames))
        for tag, payload in self.segments:
            out += bytes([int(tag)]) + leb128(len(payload)) + payload
        return bytes(out)

    @staticmethod
    def unpack(data: bytes) -> Container:
        magic, version, width, height, num, den, frames = HEADER.unpack_from(data)
        if magic != MAGIC or version != VERSION:
            raise ValueError("not a PointStream container of this version")
        out = Container(width, height, Fraction(num, den), frames)
        offset = HEADER.size
        while offset < len(data):
            tag = Tag(data[offset])
            length, offset = read_leb128(data, offset + 1)
            if offset + length > len(data):
                raise ValueError("truncated segment")
            out.add(tag, data[offset:offset + length])
            offset += length
        return out

    def coded(self, tag: Tag) -> list[Coded]:
        kind = "background" if tag == Tag.BACKGROUND else "foreground"
        return [Coded.parse(kind, p) for t, p in self.segments if t == tag]

    def models(self) -> list[ModelRef]:
        return [ModelRef.parse(p) for t, p in self.segments if t == Tag.MODEL]

    def breakdown(self) -> dict[str, int]:
        """Bytes per part of the packed container, summing to its length: the header and each tag's segments,
        their framing included."""
        parts = {"header": HEADER.size}
        for tag, payload in self.segments:
            key = tag.name.lower()
            parts[key] = parts.get(key, 0) + 1 + len(leb128(len(payload))) + len(payload)
        return parts

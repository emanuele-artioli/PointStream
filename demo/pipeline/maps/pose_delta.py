"""DWB2 pose payload: box-relative u8 landmarks, absent parts omitted, exp-Golomb deltas.

Each present hand, face, or body keeps the demo quantizer: 7-bit part confidence,
an 8-bit box across the frame, and 8-bit landmarks inside that box. A part that
is missing contributes only the one-bit count code for zero. Present parts are
delta-coded against the previous reconstructed box and in-box landmarks, so the
decoder applies landmark deltas in the same box it just reconstructed. The first
frame of a slot, and any later frame where a raw key is shorter, is sent as
absolute u8 codes.
"""

from __future__ import annotations

import struct

MAGIC_DWB1 = b"DWB1"
MAGIC_DWB2 = b"DWB2"

N_HAND = 21
N_FACE = 68
N_BODY = 17

_PARTS = (
    (b"PK", N_HAND, True),
    (b"PF", N_FACE, False),
    (b"PB", N_BODY, False),
)


class _BitsOut:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.acc = 0
        self.n = 0

    def write(self, value: int, nbits: int) -> None:
        if nbits < 0:
            raise ValueError(f"nbits must be >= 0, got {nbits}")
        if nbits == 0:
            return
        if value < 0 or value >= (1 << nbits):
            raise ValueError(f"value {value} does not fit in {nbits} bits")
        for shift in range(nbits - 1, -1, -1):
            self.acc = (self.acc << 1) | ((value >> shift) & 1)
            self.n += 1
            if self.n == 8:
                self.buf.append(self.acc)
                self.acc = 0
                self.n = 0

    def finish(self) -> bytes:
        if self.n:
            self.buf.append(self.acc << (8 - self.n))
            self.acc = 0
            self.n = 0
        return bytes(self.buf)


class _BitsIn:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.i = 0
        self.acc = 0
        self.n = 0

    def read(self, nbits: int) -> int:
        value = 0
        for _ in range(nbits):
            if self.n == 0:
                if self.i >= len(self.data):
                    raise ValueError("truncated DWB2 bitstream")
                self.acc = self.data[self.i]
                self.i += 1
                self.n = 8
            self.n -= 1
            value = (value << 1) | ((self.acc >> self.n) & 1)
        return value


def _zigzag(delta: int) -> int:
    if delta >= 0:
        return delta << 1
    return ((-delta) << 1) - 1


def _unzigzag(code: int) -> int:
    if code & 1:
        return -((code + 1) >> 1)
    return code >> 1


def _ue_len(code: int) -> int:
    width = (code + 1).bit_length() - 1
    return width + width + 1


def _write_ue(out: _BitsOut, code: int) -> None:
    if code < 0:
        raise ValueError(f"exp-Golomb code must be >= 0, got {code}")
    x = code + 1
    width = x.bit_length() - 1
    out.write(0, width)
    out.write(x, width + 1)


def _read_ue(inp: _BitsIn) -> int:
    width = 0
    while inp.read(1) == 0:
        width += 1
        if width > 16:
            raise ValueError("exp-Golomb code is too wide for a u8 delta")
    if width == 0:
        return 0
    return ((1 << width) | inp.read(width)) - 1


def _write_se(out: _BitsOut, delta: int) -> None:
    _write_ue(out, _zigzag(delta))


def _read_se(inp: _BitsIn) -> int:
    return _unzigzag(_read_ue(inp))


def _delta_bits(cur: list[int], prev: list[int]) -> int:
    return sum(_ue_len(_zigzag(a - b)) for a, b in zip(cur, prev))


def _parse_part(packet: bytes, magic: bytes, n_kpts: int, handed: bool) -> list[tuple]:
    if len(packet) < 3 or packet[:2] != magic:
        raise ValueError(f"pose part must start with {magic!r}")
    count = packet[2]
    per = 1 + 4 + n_kpts * 2
    expect = 3 + count * per
    if len(packet) != expect:
        raise ValueError(f"{magic!r} packet is {len(packet)} bytes, expected {expect} for {count} instances")
    offset = 3
    instances: list[tuple] = []
    for _ in range(count):
        flags = packet[offset]
        offset += 1
        box = list(packet[offset : offset + 4])
        offset += 4
        lm = list(packet[offset : offset + n_kpts * 2])
        offset += n_kpts * 2
        conf = flags & 0x7F
        if handed:
            instances.append(((flags >> 7) & 1, conf, box, lm))
        else:
            instances.append((conf, box, lm))
    return instances


def _emit_part(magic: bytes, instances: list[tuple], handed: bool) -> bytes:
    packet = bytearray(magic)
    packet.append(len(instances))
    for inst in instances:
        if handed:
            side, conf, box, lm = inst
            packet.append(((side & 1) << 7) | (conf & 0x7F))
        else:
            conf, box, lm = inst
            packet.append(conf & 0x7F)
        packet.extend(box)
        packet.extend(lm)
    return bytes(packet)


def _coords(inst: tuple, handed: bool) -> list[int]:
    box, lm = (inst[2], inst[3]) if handed else (inst[1], inst[2])
    return list(box) + list(lm)


def _write_instance(out: _BitsOut, inst: tuple, prev: tuple | None, handed: bool) -> None:
    if handed:
        side, conf, _, _ = inst
        out.write(side & 1, 1)
    else:
        conf = inst[0]
    out.write(conf & 0x7F, 7)
    cur = _coords(inst, handed)
    use_delta = prev is not None and _delta_bits(cur, _coords(prev, handed)) <= 8 * len(cur)
    out.write(1 if use_delta else 0, 1)
    if use_delta:
        assert prev is not None
        for a, b in zip(cur, _coords(prev, handed)):
            _write_se(out, a - b)
        return
    for value in cur:
        out.write(value, 8)


def _read_instance(inp: _BitsIn, prev: tuple | None, n_kpts: int, handed: bool) -> tuple:
    side = inp.read(1) if handed else 0
    conf = inp.read(7)
    n_coord = 4 + n_kpts * 2
    if inp.read(1) == 1:
        if prev is None:
            raise ValueError("DWB2 delta has no reconstructed box to apply it to")
        base = _coords(prev, handed)
        cur = [base[i] + _read_se(inp) for i in range(n_coord)]
        if any(v < 0 or v > 255 for v in cur):
            raise ValueError("DWB2 delta left the u8 box")
    else:
        cur = [inp.read(8) for _ in range(n_coord)]
    box, lm = cur[:4], cur[4:]
    if handed:
        return (side, conf, box, lm)
    return (conf, box, lm)


def encode_dwb1(frames: list[tuple[bytes, bytes, bytes]]) -> bytes:
    """The previous mux: three length prefixes and a packet even when the part is empty."""
    blob = bytearray(MAGIC_DWB1)
    blob.extend(struct.pack("<I", len(frames)))
    for hand, face, body in frames:
        blob.extend(struct.pack("<III", len(hand), len(face), len(body)))
        blob.extend(hand)
        blob.extend(face)
        blob.extend(body)
    return bytes(blob)


def parse_dwb1(blob: bytes) -> list[tuple[bytes, bytes, bytes]]:
    if len(blob) < 8 or blob[:4] != MAGIC_DWB1:
        raise ValueError("not a DWB1 pose payload")
    (n_frames,) = struct.unpack_from("<I", blob, 4)
    offset = 8
    frames: list[tuple[bytes, bytes, bytes]] = []
    for _ in range(n_frames):
        if offset + 12 > len(blob):
            raise ValueError("truncated DWB1 frame header")
        hand_n, face_n, body_n = struct.unpack_from("<III", blob, offset)
        offset += 12
        end = offset + hand_n + face_n + body_n
        if end > len(blob):
            raise ValueError("truncated DWB1 frame")
        hand = blob[offset : offset + hand_n]
        face = blob[offset + hand_n : offset + hand_n + face_n]
        body = blob[offset + hand_n + face_n : end]
        frames.append((hand, face, body))
        offset = end
    if offset != len(blob):
        raise ValueError("DWB1 payload has trailing bytes")
    return frames


def encode_pose_stream(frames: list[tuple[bytes, bytes, bytes]]) -> bytes:
    """Pack per-frame PK/PF/PB packets into a DWB2 payload."""
    parsed: list[list[list[tuple]]] = []
    for frame in frames:
        if len(frame) != 3:
            raise ValueError("each pose frame needs hand, face, and body packets")
        parsed.append([_parse_part(packet, magic, n_kpts, handed) for packet, (magic, n_kpts, handed) in zip(frame, _PARTS)])

    out = _BitsOut()
    prev: list[list[tuple | None]] = [[], [], []]
    for parts in parsed:
        for index, instances in enumerate(parts):
            handed = _PARTS[index][2]
            _write_ue(out, len(instances))
            slots = prev[index]
            if len(slots) < len(instances):
                slots.extend([None] * (len(instances) - len(slots)))
            for slot, inst in enumerate(instances):
                _write_instance(out, inst, slots[slot], handed)
                slots[slot] = inst
            del slots[len(instances) :]
    header = bytearray(MAGIC_DWB2)
    header.extend(struct.pack("<IBBB", len(frames), N_HAND, N_FACE, N_BODY))
    header.extend(out.finish())
    return bytes(header)


def decode_pose_stream(blob: bytes) -> list[tuple[bytes, bytes, bytes]]:
    """Inverse of `encode_pose_stream`. Packets match the PK/PF/PB bytes that were packed."""
    if len(blob) < 11 or blob[:4] != MAGIC_DWB2:
        raise ValueError("not a DWB2 pose payload")
    n_frames, n_hand, n_face, n_body = struct.unpack_from("<IBBB", blob, 4)
    counts = (n_hand, n_face, n_body)
    inp = _BitsIn(blob[11:])
    prev: list[list[tuple | None]] = [[], [], []]
    frames: list[tuple[bytes, bytes, bytes]] = []
    for _ in range(n_frames):
        packets: list[bytes] = []
        for index, ((magic, _default_k, handed), n_kpts) in enumerate(zip(_PARTS, counts)):
            count = _read_ue(inp)
            slots = prev[index]
            if len(slots) < count:
                slots.extend([None] * (count - len(slots)))
            instances: list[tuple] = []
            for slot in range(count):
                inst = _read_instance(inp, slots[slot], n_kpts, handed)
                slots[slot] = inst
                instances.append(inst)
            del slots[count:]
            packets.append(_emit_part(magic, instances, handed))
        frames.append((packets[0], packets[1], packets[2]))
    return frames


def transcode_dwb1(blob: bytes) -> bytes:
    """Re-pack an existing DWB1 mux. The quantized landmarks stay the same."""
    return encode_pose_stream(parse_dwb1(blob))

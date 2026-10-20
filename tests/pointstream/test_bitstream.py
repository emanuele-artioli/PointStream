from fractions import Fraction

import pytest

from src.pointstream import bitstream as bs


def obu(kind: int, body: bytes) -> bytes:
    return bytes([(kind << 3) | 0x02]) + bs.leb128(len(body)) + body


def av1_units(n: int) -> list[bytes]:
    """Temporal units as SVT-AV1 writes them: a temporal delimiter, then (on the first) a sequence header,
    then a frame OBU."""
    units = []
    for t in range(n):
        unit = obu(bs.OBU_TEMPORAL_DELIMITER, b"")
        if t == 0:
            unit += obu(1, b"\x00\x01\x02")
        unit += obu(6, bytes(range(t, t + 200)))  # long enough for a two-byte LEB128 size
        units.append(unit)
    return units


@pytest.mark.parametrize("value", [0, 1, 127, 128, 300, 2**21, 2**35 + 7])
def test_leb128_round_trip(value: int) -> None:
    encoded = bs.leb128(value)
    assert bs.read_leb128(encoded + b"tail", 0) == (value, len(encoded))


def test_ivf_and_obus_round_trip() -> None:
    units = av1_units(5)
    obus = b"".join(units)
    assert bs.temporal_units(obus) == units
    ivf = bs.obus_to_ivf(obus, 64, 48, Fraction(60000, 1001))
    assert ivf[:4] == b"DKIF" and len(ivf) == bs.IVF_HEADER + 5 * bs.IVF_FRAME_HEADER + len(obus)
    assert bs.ivf_to_obus(ivf) == (obus, 5)


def test_obus_reject_missing_size_field() -> None:
    with pytest.raises(ValueError):
        bs.temporal_units(bytes([bs.OBU_TEMPORAL_DELIMITER << 3]))


def test_container_round_trip_and_rate() -> None:
    fps = Fraction(50, 1)
    background = bs.Coded("background", bs.ROUTES["scene"], "dcvc-uf", 960, 544, b"PSDC" + bytes(1000))
    foreground = bs.Coded("foreground", 4, "dcvc-uf", 480, 272, b"PSDC" + bytes(range(256)) * 3,
                          ((bs.CLASSES["hand"], 7), (bs.CLASSES["object"], 0)))
    c = bs.Container(1920, 1080, fps, 240)
    c.add_coded(background)
    c.add_coded(foreground)
    c.add_model(bs.ModelRef("background", "ab" * 32))
    c.add_model(bs.ModelRef("foreground", "cd" * 32))
    data = c.pack()
    back = bs.Container.unpack(data)
    assert (back.width, back.height, back.fps, back.frames) == (1920, 1080, fps, 240)
    assert back.segments == c.segments
    assert back.coded(bs.Tag.BACKGROUND) == [background]
    assert back.coded(bs.Tag.FOREGROUND) == [foreground]
    assert back.models() == [bs.ModelRef("background", "ab" * 8), bs.ModelRef("foreground", "cd" * 8)]
    parts = back.breakdown()
    assert sum(parts.values()) == len(data)
    assert parts["background"] == 1 + 2 + bs.CODED.size + 1004  # tag, two-byte length, coded head, stream


def test_container_rejects_truncation_and_wrong_magic() -> None:
    c = bs.Container(64, 64, Fraction(25), 2)
    c.add(bs.Tag.MASK, bytes(50))
    data = c.pack()
    with pytest.raises(ValueError):
        bs.Container.unpack(data[:-1])
    with pytest.raises(ValueError):
        bs.Container.unpack(b"XXXX" + data[4:])


def test_background_has_no_slots() -> None:
    with pytest.raises(ValueError):
        bs.Coded("background", 0, "av1", 64, 64, b"", ((1, 1),)).payload()

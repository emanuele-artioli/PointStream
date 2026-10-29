import struct

import pytest

from experiments.tennis.parallel_audit import ivf, near


def test_audit_rejects_missing_reordered_and_trailing_native_frames():
    header = b'DKIF' + struct.pack('<HH4sHHIIII', 0, 32, b'AV01', 64, 64, 25, 1, 2, 0)
    first = struct.pack('<IQ', 1, 0) + b'a'
    second = struct.pack('<IQ', 1, 1) + b'b'
    ivf(header + first + second, 2, 64, 64, 25)
    for payload in (header + first, header + second + first, header + first + second + b'x'):
        with pytest.raises(ValueError):
            ivf(payload, 2, 64, 64, 25)


def test_audit_does_not_accept_nonfinite_or_disagreeing_accounting():
    near(.125, .125)
    for value in (float('nan'), float('inf'), .126):
        with pytest.raises(ValueError):
            near(value, .125)

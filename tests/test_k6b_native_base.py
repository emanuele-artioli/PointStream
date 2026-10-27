"""Native stream accounting rejects damaged or mistimed IVF payloads."""
import struct

import pytest

from experiments.tennis.k6b_native_base import ivf


def test_ivf_accounting_and_corruption(tmp_path):
    path = tmp_path / 'sample.ivf'
    header = b'DKIF' + struct.pack('<HH4sHHIIII', 0, 32, b'AV01', 240, 136, 25, 1, 2, 0)
    def frame(t):
        return struct.pack('<IQ', 3, t) + b'abc'
    original = header + frame(0) + frame(1)
    path.write_bytes(original)
    result = ivf(path, 2, 240, 136, 25)
    assert result['payload_bytes'] == 6
    assert result['header_bytes'] == 56
    assert result['bytes'] == 62
    path.write_bytes(original[:-1])
    with pytest.raises(ValueError, match='truncated'):
        ivf(path, 2, 240, 136, 25)
    path.write_bytes(header + frame(0) + frame(2))
    with pytest.raises(ValueError, match='timestamp'):
        ivf(path, 2, 240, 136, 25)

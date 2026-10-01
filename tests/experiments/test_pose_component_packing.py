import struct
import pytest
from experiments.packet_study.pose_component import HEADER, pack, unpack, read_pk_records

@pytest.mark.parametrize('level', [1, 6, 9])
def test_framed_roundtrip_includes_header(level):
    data = b'PK repeated pose fields' * 100
    blob = pack(data, level)
    assert unpack(blob) == data
    assert len(blob) > HEADER.size

@pytest.mark.parametrize('mutation', ['header', 'truncate', 'trailing', 'checksum'])
def test_corruption_rejected(mutation):
    blob = bytearray(pack(b'pose' * 100))
    if mutation == 'header': blob[4] = 2
    elif mutation == 'truncate': blob = blob[:-1]
    elif mutation == 'trailing': blob += b'x'
    else: blob[22] ^= 1
    with pytest.raises(ValueError): unpack(bytes(blob))

def test_pk_registered_record_counts_and_truncation():
    records = [b'PK\0', b'PK\1' + b'\0' * 47]
    blob = b''.join(struct.pack('<I', len(r)) + r for r in records)
    assert read_pk_records(blob) == records
    with pytest.raises(ValueError): read_pk_records(blob[:-1])
    with pytest.raises(ValueError): read_pk_records(blob + b'x')

import hashlib
import io
import zipfile
import numpy as np
import pytest
from experiments.packet_study.background_study import physical_inventory, validate_parent, validate_receiver, SHAPE


def archive(parts):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as z:
        for name, blob in parts.items(): z.writestr(name, blob)
    return out.getvalue()


def test_complete_inventory_counts_nested_packet_once():
    arr = io.BytesIO(); np.save(arr, np.frombuffer(b'native-av1', np.uint8))
    inner = archive({'background_payload_0.npy': arr.getvalue()})
    blob = archive({'background_payload_segment_0000.packet': inner, 'reset_manifest.json': b'{}'})
    result = physical_inventory(blob)
    assert result['complete_physical_bytes'] == len(blob)
    assert result['compressed_members_bytes'] + result['archive_headers_bytes'] == len(blob)
    native = result['members'][0]['contained_packet']['members'][0]['contained_native_plate']
    assert native['bytes'] == len(b'native-av1')
    assert native['sha256'] == hashlib.sha256(b'native-av1').hexdigest()


def test_parent_identity_rejects_changed_bytes():
    blob = b'registered parent'
    spec = {'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}
    validate_parent(blob, spec)
    with pytest.raises(ValueError, match='identity changed'): validate_parent(blob + b'x', spec)


def test_receiver_requires_full_48_frame_identity():
    decoded = np.zeros(SHAPE, np.uint8)
    result = {'shape': list(SHAPE), 'rgb_sha256': hashlib.sha256(decoded.tobytes()).hexdigest()}
    validate_receiver(result, decoded)
    with pytest.raises(ValueError, match='omitted'): validate_receiver(result, decoded[:47])
    with pytest.raises(ValueError, match='identity mismatch'): validate_receiver({**result, 'rgb_sha256': 'wrong'}, decoded)

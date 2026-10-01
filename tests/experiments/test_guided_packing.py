import hashlib
import io
import json
import numpy as np
import pytest
from experiments.packet_study.guided_packing import SHAPE, validate_input, validate_receiver


def envelope(frames=16):
    metadata = {'schema': 1, 'frame_count': frames, 'height': 2160, 'width': 3840}
    stream = io.BytesIO()
    np.savez(stream, metadata=np.frombuffer(json.dumps(metadata).encode(), np.uint8))
    data = stream.getvalue()
    return data, {'package_bytes': len(data), 'package_sha256': hashlib.sha256(data).hexdigest()}


def test_original_byte_identity_and_complete_denominator_required():
    blob, spec = envelope()
    assert validate_input(blob, spec)['frame_count'] == 16
    with pytest.raises(ValueError, match='identity changed'):
        validate_input(blob + b'x', spec)
    short, short_spec = envelope(frames=15)
    with pytest.raises(ValueError, match='dimensions differ'):
        validate_input(short, short_spec)


def test_receipt_requires_all_frames_and_exact_full_rgb_hash():
    result = {'shape': SHAPE, 'dtype': 'uint8', 'rgb_frame_sha256': ['x'] * 16, 'decoded_rgb_sha256': 'known'}
    spec = {'decoded_rgb_sha256': 'known'}
    validate_receiver(result, spec)
    with pytest.raises(ValueError, match='parity failed'):
        validate_receiver({**result, 'decoded_rgb_sha256': 'changed'}, spec)
    with pytest.raises(ValueError, match='omitted'):
        validate_receiver({**result, 'rgb_frame_sha256': ['x'] * 15}, spec)

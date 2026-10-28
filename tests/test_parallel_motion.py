import numpy as np
import pytest

from experiments.tennis.parallel_motion import decode_motion, encode_motion, fit_motion, warp_reference


def test_delivered_motion_roundtrip_and_rejected_singular_geometry():
    matrices = np.repeat(np.eye(3)[None], 2, axis=0)
    matrices[1, 0, 2] = 4
    assert np.array_equal(decode_motion(encode_motion(matrices), 2), matrices)
    with pytest.raises(ValueError):
        decode_motion(encode_motion(matrices), 3)
    matrices[1] = 0
    with pytest.raises(ValueError):
        decode_motion(encode_motion(matrices), 2)


def test_warp_uses_luma_chroma_coordinates_and_explicit_borders():
    reference = np.full((816, 960), 128, np.uint8)
    reference[:544] = np.arange(960, dtype=np.uint16)[None] % 256
    reference[544:].reshape(2, 272, 480)[:, :, 20] = 77
    matrix = np.eye(3)
    matrix[0, 2] = 4
    result = warp_reference(reference, matrix)
    assert np.array_equal(result[:544, 4:], reference[:544, :-4])
    assert np.all(result[:544, :4] == 16)
    assert np.all(result[544:].reshape(2, 272, 480)[:, :, 22] == 77)
    assert np.array_equal(warp_reference(reference, np.eye(3)), reference)


def test_textureless_reference_has_paid_identity_fallback():
    reference = np.full((816, 960), 128, np.uint8)
    packet, stats = fit_motion(reference, np.stack([reference, reference]), np.zeros((2, 540, 960), bool))
    assert stats == {'fitted': 0, 'identity_fallbacks': 2}
    assert np.array_equal(decode_motion(packet, 2), np.repeat(np.eye(3)[None], 2, axis=0))

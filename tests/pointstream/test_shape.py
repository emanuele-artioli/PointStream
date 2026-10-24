"""Exercise the actual arithmetic bitstream, motion signs and odd edge blocks."""
import numpy as np
import pytest

from src.pointstream import shape


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('kind', ['empty', 'full', 'moving', 'noise'])
def test_roundtrip(mode, kind):
    masks = np.zeros((4, 35, 41), np.uint8)
    if kind == 'full':
        masks[:] = 1
    elif kind == 'moving':
        for t in range(4):
            masks[t, 3 + t:20 + t, 2 + 2*t:25 + 2*t] = 1
    elif kind == 'noise':
        masks = np.random.default_rng(17).integers(0, 2, masks.shape, dtype=np.uint8)
    payload = shape.encode(masks, mode)
    assert np.array_equal(shape.decode(payload), masks)
    with pytest.raises(ValueError):
        shape.decode(payload[:-1])

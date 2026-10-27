"""Decoded-base fallback and player ordering are receiver-visible behavior."""
import numpy as np

from experiments.tennis.k6b_real_base_compose import compose


def test_base_fallback_and_far_to_near_order():
    base = np.full((6, 6, 3), 17, dtype=np.uint8)
    far = np.full((2, 2, 3), 90, dtype=np.uint8)
    near = np.full((2, 2, 3), 210, dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    masks = {'far': mask, 'near': mask, 'missing': mask}
    box = np.array([1, 1, 3, 3])
    result, support = compose(base, {'far': far, 'near': near}, {'far': box, 'near': box}, masks, ['far', 'near', 'missing'])
    assert (result[1:4, 1:4] == 210).all()
    assert (result[~support] == 17).all()
    assert not support[4, 4]
    assert (base == 17).all()

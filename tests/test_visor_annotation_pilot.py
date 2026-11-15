import numpy as np
import pytest

from src.segmentation.visor import Frame
from tools.datasets.visor_annotation_pilot import select_indices, union_proposal


def test_reserved_selection_preserves_every_manual_anchor_and_endpoints():
    assert select_indices(175, 238, [175, 212]) == [175, 184, 193, 212, 225, 238]
    assert select_indices(217, 280, [217, 238, 260, 271]) == [217, 238, 249, 260, 271, 280]
    assert select_indices(229, 292, [229, 279]) == [229, 241, 254, 266, 279, 292]
    with pytest.raises(ValueError, match='silently omit'):
        select_indices(0, 20, list(range(10)))


def test_unknown_placeholder_is_not_used_as_dense_or_human_annotation():
    assert not union_proposal(None, dense=False).any()
    frame = Frame(1, ({'name': 'right hand', 'segments': [[[0, 0], [10, 0], [10, 10], [0, 10]]]},), ())
    native = union_proposal(frame, dense=False)
    dense = union_proposal(frame, dense=True)
    assert native.shape == dense.shape == (1080, 1920)
    assert native.dtype == dense.dtype == bool
    assert dense.sum() > native.sum()
    np.testing.assert_array_equal(native[:11, :11], True)

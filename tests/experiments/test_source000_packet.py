"""Small-array checks of registered source-only sender input adaptation."""
from dataclasses import dataclass, replace
import hashlib

import numpy as np
import pytest

from experiments.gate_a_confirmation.source000_packet import apply_appearance_policy, slice_clip


@dataclass(frozen=True)
class Object:
    object_id: str
    appearance: np.ndarray
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    frame_index: int
    conditioning: object = None
    supplied_crop: object = None


@dataclass(frozen=True)
class Clip:
    n_frames: int
    frames: np.ndarray
    masks: np.ndarray
    objects: tuple[Object, ...]
    scene: str = 'scene_000'


def example():
    frames = np.arange(96*4*5*3, dtype=np.uint8).reshape(96,4,5,3)
    masks = np.zeros((96,4,5), dtype=bool)
    masks[:,1:3,1:4] = True
    first = Object('first', np.full((2,3,3),127,np.uint8), (1,1,4,3), masks.copy(), 1)
    later = replace(first, object_id='later', frame_index=20, mask=masks.copy())
    return Clip(96,frames,masks,(first,later))


def test_reference_cutout_preserves_exact_rgb_frame_metadata_and_hash():
    original = example()
    snapshots = [o.appearance.copy() for o in original.objects]
    got, records = apply_appearance_policy(original,'reference_cutout')
    assert got.frames is original.frames
    assert got.masks is original.masks
    for old,new,record,cached in zip(original.objects,got.objects,records,snapshots,strict=True):
        expected = original.frames[old.frame_index,1:3,1:4]
        np.testing.assert_array_equal(new.appearance,expected)
        assert not np.shares_memory(new.appearance,original.frames)
        assert (new.object_id,new.bbox,new.frame_index) == (old.object_id,old.bbox,old.frame_index)
        assert new.mask is old.mask
        assert record['source_frame_index'] == 38+old.frame_index
        assert record['appearance_rgb_sha256'] == hashlib.sha256(expected.tobytes()).hexdigest()
        assert record['appearance_input_policy'] == 'reference_cutout'
        np.testing.assert_array_equal(old.appearance,cached)


def test_prefix_excludes_later_objects_and_slices_every_guide_without_mutation():
    original = example()
    pixels, union = original.frames.copy(), original.masks.copy()
    object_masks = [o.mask.copy() for o in original.objects]
    prefix = slice_clip(original,12)
    assert prefix.n_frames == 12
    assert [o.object_id for o in prefix.objects] == ['first']
    np.testing.assert_array_equal(prefix.frames,original.frames[:12])
    np.testing.assert_array_equal(prefix.masks,original.masks[:12])
    np.testing.assert_array_equal(prefix.objects[0].mask,original.objects[0].mask[:12])
    assert prefix.objects[0].mask.shape[0] == 12
    np.testing.assert_array_equal(original.frames,pixels)
    np.testing.assert_array_equal(original.masks,union)
    for obj,snapshot in zip(original.objects,object_masks,strict=True):
        assert obj.mask.shape[0] == 96
        np.testing.assert_array_equal(obj.mask,snapshot)
    wider = slice_clip(original,48)
    assert [o.object_id for o in wider.objects] == ['first','later']
    assert all(o.mask.shape[0] == 48 for o in wider.objects)


@pytest.mark.parametrize('change',[
    {'bbox':(-1,1,2,3)}, {'bbox':(1,1,6,3)}, {'bbox':(1,3,4,3)},
    {'frame_index':-1}, {'frame_index':96}, {'frame_index':1.5},
    {'conditioning':object()}, {'supplied_crop':np.zeros((2,3,3),np.uint8)},
])
def test_invalid_raster_frame_or_alternate_input_fails_closed(change):
    clip = example()
    bad = replace(clip,objects=(replace(clip.objects[0],**change),))
    with pytest.raises(ValueError):
        apply_appearance_policy(bad,'reference_cutout')


def test_cached_policy_is_explicit_and_different_from_reference_pixels():
    clip = example()
    cached, cr = apply_appearance_policy(clip,'cached_associated')
    reference, rr = apply_appearance_policy(clip,'reference_cutout')
    np.testing.assert_array_equal(cached.objects[0].appearance,clip.objects[0].appearance)
    assert cr[0]['appearance_input_policy'] == 'cached_associated'
    assert rr[0]['appearance_input_policy'] == 'reference_cutout'
    assert cr[0]['appearance_rgb_sha256'] != rr[0]['appearance_rgb_sha256']
    assert not np.array_equal(cached.objects[0].appearance,reference.objects[0].appearance)
    with pytest.raises(ValueError):
        apply_appearance_policy(clip,'unregistered_policy')

import dataclasses
import numpy as np
import pytest
from experiments.gate_a_confirmation.fresh_window_sender import build_clip, slice_clip, rgb_digest
from src.pipeline.reconstruction.reconstruct import ObjectRequest

def object_for(source, *, frame=0, name='person_1'):
    mask = np.zeros(source.shape[:3], dtype=bool)
    mask[frame, 1:3, 1:3] = True
    return ObjectRequest(object_id=name, frame_index=frame, bbox=(1,1,3,3),
                         appearance=source[frame,1:3,1:3].copy(), mask=mask, object_class='person')

def test_single_source_boundary_union_class_and_rgb():
    source = np.zeros((96,4,4,3), dtype=np.uint8)
    source[...,0], source[...,2] = 10, 30
    objects = (object_for(source), object_for(source, frame=20, name='person_2'))
    clip, appearances = build_clip(source, objects, expected_shape=source.shape)
    assert clip.scene == 'scene_000' and clip.context_id == 'source000-rgb-' + rgb_digest(source)
    assert clip.masks.sum() == 8 and clip.masks[1].sum() == 0
    assert clip.objects[0].object_class == 'person'
    assert clip.objects[0].appearance[0,0].tolist() == [10,0,30]
    prefix = slice_clip(clip,12)
    assert prefix.frames.shape[0] == 12 and len(prefix.objects) == 1
    assert prefix.objects[0].mask.shape[0] == 12
    assert appearances[1]['source_frame_index'] == 58

def test_reject_wrong_pixels_class_and_mask():
    source=np.zeros((96,4,4,3),dtype=np.uint8)
    original=object_for(source)
    for wrong in (dataclasses.replace(original,appearance=np.ones((2,2,3),dtype=np.uint8)),
                  dataclasses.replace(original,object_class='player'),
                  dataclasses.replace(original,mask=original.mask[:12])):
        with pytest.raises(ValueError): build_clip(source,(wrong,),expected_shape=source.shape)
    with pytest.raises(ValueError): build_clip(source,(original,original),expected_shape=source.shape)

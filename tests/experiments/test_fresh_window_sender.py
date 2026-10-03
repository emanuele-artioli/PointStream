import dataclasses
import numpy as np
import pytest
from experiments.gate_a_confirmation.fresh_window_sender import build_clip, slice_clip, rgb_digest
from src.pipeline.reconstruction.reconstruct import ObjectRequest

def object_for(source, *, frame=0, name='player_near'):
    mask = np.zeros(source.shape[:3], dtype=bool)
    mask[frame, 1:3, 1:3] = True
    return ObjectRequest(object_id=name, frame_index=frame, bbox=(1,1,3,3),
                         appearance=source[frame,1:3,1:3].copy(), mask=mask)

def test_single_source_boundary_union_class_and_rgb():
    source = np.zeros((96,4,4,3), dtype=np.uint8)
    source[...,0], source[...,2] = 10, 30
    objects = (object_for(source), object_for(source, frame=20, name='player_far'))
    clip, appearances = build_clip(source, objects, expected_shape=source.shape)
    assert clip.scene == 'scene_000' and clip.context_id == 'source000-rgb-' + rgb_digest(source)
    assert clip.masks.sum() == 8 and clip.masks[1].sum() == 0
    assert clip.objects[0].object_id == 'player_near'
    assert clip.objects[0].appearance[0,0].tolist() == [10,0,30]
    prefix = slice_clip(clip,12)
    assert prefix.frames.shape[0] == 12 and len(prefix.objects) == 1
    assert prefix.objects[0].mask.shape[0] == 12
    assert appearances[1]['source_frame_index'] == 58

def test_reject_wrong_pixels_class_and_mask():
    source=np.zeros((96,4,4,3),dtype=np.uint8)
    original=object_for(source)
    for wrong in (dataclasses.replace(original,appearance=np.ones((2,2,3),dtype=np.uint8)),
                  dataclasses.replace(original,object_id='person'),
                  dataclasses.replace(original,mask=original.mask[:12])):
        with pytest.raises(ValueError): build_clip(source,(wrong,),expected_shape=source.shape)
    with pytest.raises(ValueError): build_clip(source,(original,original),expected_shape=source.shape)

# Minimal API extracted from clean274 reconstruct.py ObjectRequest. It has no
# object_class; tests against current production alone hid this incompatibility.
# Exact class-only AST extraction from clean274 src/pipeline/reconstruction/reconstruct.py.
_CLEAN274_CLASS_SOURCE = '@dataclass(frozen=True)\nclass ObjectRequest:\n    """One object the reconstruction may generate and place.\n\n    ``appearance`` is the crop the generator (or a supplied-pixel path) uses.\n    ``conditioning`` is what the generator declared it needs; when omitted,\n    a bundle is built from appearance, mask, bbox and identity.\n    """\n    object_id: str\n    appearance: np.ndarray\n    bbox: tuple[int, int, int, int]\n    mask: np.ndarray | None = None\n    frame_index: int = 0\n    conditioning: ConditioningBundle | None = None\n    supplied_crop: np.ndarray | None = None\n    \'When set, skip generation for this object and composite these pixels.\\n    Rigid shapes and already-decoded appearance crops use this.\''
_legacy_namespace={'dataclass':dataclasses.dataclass,'np':np,'ConditioningBundle':object}
exec(compile(_CLEAN274_CLASS_SOURCE,'clean274-ObjectRequest-fixture','exec'),_legacy_namespace)
Clean274ObjectRequest=_legacy_namespace['ObjectRequest']

def test_clean274_actual_object_api_build_and_subset():
    source=np.zeros((96,4,4,3),dtype=np.uint8)
    source[0,...,0]=17;source[20,...,2]=93
    objects=[]
    for frame,role in ((0,'player_near'),(20,'player_far')):
        mask=np.zeros(source.shape[:3],dtype=bool);mask[frame,1:3,1:3]=True
        objects.append(Clean274ObjectRequest(role,source[frame,1:3,1:3].copy(),(1,1,3,3),mask,frame))
    clip,crops=build_clip(source,tuple(objects),expected_shape=source.shape)
    short=slice_clip(clip,12)
    assert len(short.objects)==1 and isinstance(short.objects[0],Clean274ObjectRequest)
    assert short.objects[0].mask.shape==(12,4,4)
    assert np.array_equal(short.objects[0].appearance,source[0,1:3,1:3])
    assert objects[0].mask.shape==(96,4,4) and objects[1].frame_index==20
    assert crops[1]['source_frame_index']==58
    for field in ('conditioning','supplied_crop'):
        with pytest.raises(ValueError):
            build_clip(source,(dataclasses.replace(objects[0],**{field:object()}),),expected_shape=source.shape)

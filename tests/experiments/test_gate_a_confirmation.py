from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pytest
from experiments.gate_a_confirmation.legacy_recertify import slice_smoke, deployment_manifest
from experiments.gate_a_confirmation.native_anchor import raw_shape
@dataclass(frozen=True)
class Object:
    mask: np.ndarray
    frame_index: int
@dataclass(frozen=True)
class Clip:
    frames: np.ndarray
    masks: np.ndarray
    objects: tuple
    n_frames: int

def test_prefix_rejects_misaligned_object_masks():
    c=Clip(np.zeros((4,3,4,3),dtype=np.uint8),np.zeros((4,3,4),dtype=bool),(Object(np.zeros((3,3,4),dtype=bool),0),),4)
    with pytest.raises(ValueError,match='unaligned object'):slice_smoke(c,2)

def test_prefix_retains_only_registered_frames_and_objects():
    c=Clip(np.ones((4,3,4,3),dtype=np.uint8),np.ones((4,3,4),dtype=bool),(Object(np.ones((4,3,4),dtype=bool),0),Object(np.ones((4,3,4),dtype=bool),3)),4)
    sliced=slice_smoke(c,2)
    assert sliced.n_frames==2 and len(sliced.objects)==1 and sliced.objects[0].mask.shape[0]==2
    assert c.n_frames==4 and len(c.objects)==2

def test_deployment_manifest_excludes_audit_outputs_and_rejects_missing_scene():
    captured=[{'package':'scene-00.npz','output':'private-scorer.npy','output_sha256':'not-a-decoder-input','bytes':55}]
    result=deployment_manifest(captured,[[2,2160,3840,3]])
    assert 'private-scorer' not in str(result) and result['scenes'][0]['package']=='scene-00.npz'
    with pytest.raises(ValueError):deployment_manifest(captured,[])

def test_native_decode_rejects_missing_and_extra_frames(tmp_path):
    path=tmp_path/'decoded.rgb'
    path.write_bytes(bytes(2*3*4*3))
    assert raw_shape(path,2,3,4)==(2,3,4,3)
    for size in (3*4*3,3*3*4*3,2*3*4*3+1):
        path.write_bytes(bytes(size))
        with pytest.raises(ValueError,match='frame count'):raw_shape(path,2,3,4)

def test_unused_mask_pruning_preserves_other_arrays_and_placement_fields():
    import io, json
    from experiments.gate_a_confirmation.prune_unused_masks import prune
    metadata={'placements':[{'mask_key':'mask_0','frame_index':0,'bbox':[1,2,3,4],'object_id':'x'}],'frame_count':96}
    encoded=io.BytesIO()
    bg=np.array([1,7,9],dtype=np.uint8)
    np.savez_compressed(encoded,metadata=np.frombuffer(json.dumps(metadata).encode(),dtype=np.uint8),mask_0=np.ones((2,4,4),dtype=bool),background_payload=bg)
    raw,removed=prune(encoded.getvalue())
    with np.load(io.BytesIO(raw),allow_pickle=False) as packet:
        assert removed==['mask_0'] and 'mask_0' not in packet.files
        assert np.array_equal(packet['background_payload'],bg)
        got=json.loads(packet['metadata'].tobytes())
        metadata['placements'][0]['mask_key']=None
        assert got==metadata

def test_trajectory_box_preserves_seed_and_moves_scales_with_guide():
    from experiments.gate_a_confirmation.trajectory_packet import adapted_box, bounds
    base=[10,20,30,60];first=(12,24,28,56)
    assert adapted_box(base,first,first)==base
    assert adapted_box(base,first,(22,29,38,61))==[20,25,40,65]
    assert bounds(np.zeros((3,4),dtype=bool)) is None

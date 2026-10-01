"""Closed registered-background transport tests; native codecs injected locally."""
import io
import json
import zipfile
import cv2
import numpy as np
import pytest

from experiments.background import registered_plate_codec as r
from src.runner.client import ClientPlacement, serialize_client_request, reconstruct_serialized_client


def parent(count=3):
    mask=np.zeros((8,8),bool);mask[2:4,2:4]=True
    placements=tuple(ClientPlacement(encoded_crop=png_encoder(np.full((2,2,3),200,np.uint8)),bbox=(2,2,4,4),
        mask=mask,frame_index=i,object_id='actor') for i in range(count))
    return serialize_client_request(background=None,frame_count=count,height=8,width=8,placements=placements)


def png_encoder(bgr):
    ok, blob=cv2.imencode('.png',bgr);assert ok;return blob.tobytes()


class PngDecoder:
    def decode(self,blob):return cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR)


def test_median_excludes_parent_masks_and_records_fallback():
    frames=np.full((3,8,8,3),40,np.uint8);frames[:,2:4,2:4]=200
    masks=np.zeros((3,8,8),bool);masks[:,2:4,2:4]=True
    maps=np.repeat(np.eye(3,dtype=np.float32)[None],3,axis=0)
    plate,receipt=r.build_plate(frames,masks,maps,'median')
    assert receipt['uncovered_pixels']==4
    assert np.array_equal(plate,frames[0])
    masks[1,2:4,2:4]=False;frames[1,2:4,2:4]=50
    plate,receipt=r.build_plate(frames,masks,maps,'median')
    assert receipt['uncovered_pixels']==0 and np.all(plate[2:4,2:4]==50)


def test_registration_failure_is_explicit_identity():
    frames=np.full((3,8,8,3),40,np.uint8);masks=np.zeros((3,8,8),bool)
    maps,receipt=r.register(frames,masks)
    np.testing.assert_array_equal(maps,np.repeat(np.eye(3,dtype=np.float32)[None],3,axis=0))
    assert [x['status'] for x in receipt]==['reference','identity_fallback','identity_fallback']


def test_charged_package_no_raw_plate_and_fresh_client_parity(monkeypatch):
    import src.components.background.sidecar as sidecar
    monkeypatch.setattr(sidecar,'build_sidecar',lambda _:PngDecoder())
    original=parent();plate=np.broadcast_to(np.array([30,50,90],np.uint8),(8,8,3)).copy();maps=np.repeat(np.eye(3,dtype=np.float32)[None],3,axis=0)
    packet=r.encode_package(original,plate,maps,encode_plate=png_encoder)
    data=r.members(packet);meta=r.metadata(data)
    assert 'background_plate.npy' not in data and meta['background']['plate_key'] is None
    assert meta['background']['homographies']==[]
    assert np.load(io.BytesIO(data[r.MAPS]),allow_pickle=False).nbytes==3*9*4
    old=r.members(original)
    for name in old:
        if name.startswith(('crop_','encoded_crop_','ref_')):assert data[name]==old[name]
    decoded=r.reconstruct_package(packet)
    assert decoded.shape==(3,8,8,3)
    assert np.all(decoded[:,0,0]==[30,50,90]) and np.all(decoded[:,2:4,2:4]==200)
    assert packet==r.encode_package(original,plate,maps,encode_plate=png_encoder)
    with zipfile.ZipFile(io.BytesIO(packet)) as z:
        assert z.getinfo('background_payload_0.npy').compress_type==zipfile.ZIP_STORED
        assert sum(i.compress_size for i in z.infolist())<len(packet)


@pytest.mark.parametrize('interval',[12,24])
def test_real_reset_segments_complete_coverage_and_full_cost(monkeypatch,interval):
    import src.components.background.sidecar as sidecar
    monkeypatch.setattr(sidecar,'build_sidecar',lambda _:PngDecoder())
    original=parent(25);frames=np.full((25,8,8,3),50,np.uint8)
    frames[12:]=60;masks=r.parent_masks(original,25)
    bundle,receipts=r.encode_reset_bundle(original,frames,masks,interval=interval,encode_plate=png_encoder)
    data=r.members(bundle);spec=json.loads(data['reset_manifest.json'])
    assert len(spec['segments'])==(25+interval-1)//interval
    assert spec['segments'][-1]['end']==25 and receipts[-1]['end']==25
    assert len(bundle)>sum(x['bytes'] for x in spec['segments'])
    decoded=r.reconstruct_package(bundle);assert decoded.shape==(25,8,8,3)
    assert np.all(decoded[:,2:4,2:4]==200)
    assert np.all(decoded[-1,0,0]==60)
    # Segment identity corruption rejects before invoking its decoder.
    spec['segments'][0]['sha256']='0'*64;data['reset_manifest.json']=r._json(spec)
    with pytest.raises(ValueError,match='identity'):r.reconstruct_package(r._archive(data))


def test_bad_map_coverage_or_singular_map_rejected():
    plate=np.full((8,8,3),50,np.uint8)
    with pytest.raises(ValueError,match='homography'):r.encode_package(parent(),plate,np.eye(3)[None],encode_plate=png_encoder)
    maps=np.repeat(np.eye(3,dtype=np.float32)[None],3,axis=0);maps[2]=0
    with pytest.raises(ValueError,match='homography'):r.encode_package(parent(),plate,maps,encode_plate=png_encoder)

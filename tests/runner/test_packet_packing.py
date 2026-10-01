"""Opt-in transport packing: exact wire identity, lossy flags and client parity."""
import io
import json
import zipfile

import numpy as np
import pytest

from src.runner.client import ClientPlacement, reconstruct_serialized_client, serialize_client_request
from src.runner.mask_wire import decode_mask, encode_mask
from src.runner.packet_packing import MANIFEST, pack_client_envelope, packing_info, unpack_client_envelope


def envelope(mask=None, *, frames=3, placements=None):
    if mask is None:
        mask = np.zeros((5, 7), dtype=np.uint8)
        mask[1, 1] = 1
    if placements is None:
        placements = (ClientPlacement(crop=np.full((5, 7, 3), 180, dtype=np.uint8),
            bbox=(0, 0, 7, 5), mask=mask, frame_index=0),)
    return serialize_client_request(background=None, frame_count=frames, height=5, width=7,
                                    placements=placements)


def members(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def rewrite(payload, *, metadata=None, extra=None, spec=None):
    files = members(payload)
    if metadata is not None:
        out=io.BytesIO();np.save(out,np.frombuffer(json.dumps(metadata).encode(),dtype=np.uint8));files['metadata.npy']=out.getvalue()
    if spec is not None:
        files[MANIFEST]=json.dumps(spec).encode()
    for key,array in (extra or {}).items():
        out=io.BytesIO();np.save(out,array,allow_pickle=False);files[key+'.npy']=out.getvalue()
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as archive:
        for name,value in files.items():archive.writestr(name,value)
    return out.getvalue()


def meta(payload):
    with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
        return json.loads(arrays['metadata'].tobytes())


@pytest.mark.parametrize('codec',['psm1','rle'])
def test_lossless_client_parity_all_frames_and_deterministic_bytes(codec):
    original=envelope()
    packed=pack_client_envelope(original,mask_codec=codec)
    assert packed==pack_client_envelope(original,mask_codec=codec)
    restored=unpack_client_envelope(packed)
    before=reconstruct_serialized_client(original)
    after=reconstruct_serialized_client(restored)
    assert after.shape==(3,5,7,3)
    assert np.array_equal(after,before)
    assert np.all(after[1:]==0)  # Missing placements remain emitted black frames.
    info=packing_info(packed)
    assert info['complete_file_bytes']==len(packed)
    assert info['physical_member_bytes']+info['zip_framing_bytes']==len(packed)
    assert info['packing']['lossy_masks'] is False


@pytest.mark.parametrize('scale',[2,4,8])
@pytest.mark.parametrize('codec',['psm1','rle'])
def test_lossy_nearest_masks_preserve_odd_original_dimensions(scale,codec):
    mask=np.zeros((5,7),dtype=np.uint8);mask[0,0]=1;mask[1,1]=1;mask[4,6]=1
    packed=pack_client_envelope(envelope(mask),mask_scale=scale,mask_codec=codec)
    restored=unpack_client_envelope(packed)
    with np.load(io.BytesIO(restored),allow_pickle=False) as arrays:
        decoded=decode_mask(arrays['mask_0'].tobytes())
    expected=np.repeat(np.repeat(mask[::scale,::scale],scale,axis=0),scale,axis=1)[:5,:7]
    assert decoded.shape==mask.shape
    assert np.array_equal(decoded,expected)
    info=packing_info(packed)
    assert info['packing']['lossy_masks'] is True
    assert info['packing']['predictor_pixels_may_change'] is True
    assert info['packing']['changed_mask_pixels']==np.count_nonzero(mask!=expected)
    assert meta(restored)['placements'][0]['mask_wire']['shape']==[5,7]
    assert reconstruct_serialized_client(restored).shape==(3,5,7,3)


@pytest.mark.parametrize('codec',['psm1','rle'])
def test_empty_masks_and_multiple_objects_do_not_assume_one_mask_per_frame(codec):
    empty=np.zeros((5,7),dtype=np.uint8)
    placements=tuple(ClientPlacement(crop=np.ones((5,7,3),dtype=np.uint8),bbox=(0,0,7,5),
        mask=empty,frame_index=i%2,object_id=str(i)) for i in range(5))
    original=envelope(frames=2,placements=placements)
    packed=pack_client_envelope(original,mask_codec=codec,mask_scale=4)
    restored=unpack_client_envelope(packed)
    assert len(meta(restored)['placements'])==5
    assert np.array_equal(reconstruct_serialized_client(restored),np.zeros((2,5,7,3),dtype=np.uint8))
    assert packing_info(packed)['packing']['changed_mask_pixels']==0


def test_native_members_and_geometry_metadata_are_preserved():
    original=envelope();metadata=meta(original)
    metadata['background']={'geometry_header':'123456abcdef','wire_header_keys':['background_header_0']}
    extra={key:np.arange(17,dtype=np.uint8) for key in
           ['background_payload_0','encoded_crop_99','ref_player:0','residual_bitstream','background_header_0']}
    original=rewrite(original,metadata=metadata,extra=extra)
    packed=pack_client_envelope(original,mask_codec='rle')
    restored=unpack_client_envelope(packed)
    before,after=members(original),members(restored)
    for key in extra:
        assert before[key+'.npy']==after[key+'.npy']
    assert meta(restored)['background']==metadata['background']
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        assert archive.getinfo('residual_bitstream.npy').compress_type==zipfile.ZIP_STORED
        assert archive.getinfo('metadata.npy').compress_type==zipfile.ZIP_DEFLATED


def test_lossy_mask_really_changes_delivered_pixels():
    original=envelope()
    packed=pack_client_envelope(original,mask_scale=2)
    assert not np.array_equal(reconstruct_serialized_client(original),
                              reconstruct_serialized_client(unpack_client_envelope(packed)))


def test_residual_on_lossy_input_rejected_but_lossless_allowed():
    original=envelope();metadata=meta(original);metadata['residual']['present']=True
    original=rewrite(original,metadata=metadata,extra={'residual_bitstream':np.array([1,2],dtype=np.uint8)})
    pack_client_envelope(original,mask_scale=1)
    with pytest.raises(ValueError,match='recomputed correction'):
        pack_client_envelope(original,mask_scale=2)


@pytest.mark.parametrize('scale',[0,3,16,True])
def test_unsupported_mask_scales_rejected(scale):
    with pytest.raises(ValueError,match='unsupported'):
        pack_client_envelope(envelope(),mask_scale=scale)


def test_unsupported_formats_and_frame_count_rejected():
    with pytest.raises(ValueError):unpack_client_envelope(b'not an NPZ')
    original=envelope();metadata=meta(original);metadata['schema']=2
    with pytest.raises(ValueError,match='schema-1'):pack_client_envelope(rewrite(original,metadata=metadata))
    metadata['schema']=1;metadata['placements'][0]['frame_index']=3
    with pytest.raises(ValueError,match='frame count'):pack_client_envelope(rewrite(original,metadata=metadata))
    packed=pack_client_envelope(original);spec=packing_info(packed)['packing'];spec['version']=99
    with pytest.raises(ValueError,match='unsupported'):unpack_client_envelope(rewrite(packed,spec=spec))


def test_mask_stack_keeps_empty_frames_and_rejects_count_mismatch():
    volume=np.zeros((3,5,7),dtype=np.uint8);volume[0,0,0]=1;volume[2,4,6]=1
    original=envelope(volume)
    for codec in ['psm1','rle']:
        restored=unpack_client_envelope(pack_client_envelope(original,mask_codec=codec))
        with np.load(io.BytesIO(restored),allow_pickle=False) as arrays:
            assert np.array_equal(decode_mask(arrays['mask_0'].tobytes()),volume)
    with pytest.raises(ValueError,match='stack frame count'):
        pack_client_envelope(envelope(volume[:2]))


def test_unpacked_ordinary_envelope_is_validated_identity():
    original=envelope();assert unpack_client_envelope(original)==original
    metadata=meta(original);metadata['placements'][0]['mask_wire']['shape']=[6,7]
    with pytest.raises(ValueError,match='shape'):
        unpack_client_envelope(rewrite(original,metadata=metadata))

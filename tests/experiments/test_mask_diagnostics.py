"""Pooled parent-mask transport diagnostics; no independent segmentation labels."""
import numpy as np
import pytest
from experiments.packet_study import mask_diagnostics as m


def test_pooled_iou_and_fixed_positive_denominator_not_mean_frame_ratio():
    a=np.zeros((2,2,3),bool);b=a.copy()
    a[0,0,0]=True;a[1]=True;b[1]=True;b[0,0,1]=True
    d=m.compare_masks(a,b,shape=a.shape)
    assert d['fixed_original_positive_pixels']==7
    assert d['intersection_pixels']==6 and d['union_pixels']==8
    assert d['pooled_binary_iou']==.75  # Mean frame IoU would be .5.
    assert d['fn_pixels_relative_to_parent']==d['fp_pixels_relative_to_parent']==1
    assert d['fn_fraction_of_fixed_original_positive']==pytest.approx(1/7)
    assert d['fp_fraction_of_fixed_original_positive']==pytest.approx(1/7)
    assert d['changed_fraction_of_whole_raster']==pytest.approx(2/12)
    assert len(d['per_frame'])==2


def test_nonempty_to_empty_frames_and_empty_frames_all_preserved():
    a=np.zeros((3,2,2),bool);a[1,0,0]=True;b=np.zeros_like(a)
    d=m.compare_masks(a,b,shape=a.shape)
    assert d['frames']==3 and len(d['per_frame'])==3
    assert d['nonempty_parent_to_empty_delivered_frames']==[1]
    assert d['empty_union_frames']==[0,2]
    assert d['fn_fraction_of_fixed_original_positive']==1
    assert d['pooled_binary_iou']==0


def test_empty_parent_denominator_and_empty_union_are_explicit():
    a=np.zeros((2,2,2),bool);b=a.copy()
    d=m.compare_masks(a,b,shape=a.shape)
    assert d['pooled_binary_iou'] is None
    assert d['fn_fraction_of_fixed_original_positive'] is None
    assert d['fp_fraction_of_fixed_original_positive'] is None
    b[1,0,0]=True;d=m.compare_masks(a,b,shape=a.shape)
    assert d['pooled_binary_iou']==0 and d['fp_pixels_relative_to_parent']==1
    assert d['empty_parent_to_nonempty_delivered_frames']==[1]


def test_missing_frame_or_raster_and_nonbinary_values_rejected():
    with pytest.raises(ValueError,match='all registered'):
        m.compare_masks(np.zeros((47,360,640),bool),np.zeros((47,360,640),bool))
    with pytest.raises(ValueError,match='binary'):
        m.compare_masks(np.full((2,2,2),2,np.uint8),np.zeros((2,2,2),bool),shape=(2,2,2))


def report():
    rows=[]
    for setting,variant in sorted(m.EXPECTED_KEYS):
        scale=int(variant[-1]) if variant.startswith('rle_scale') else 1
        rows.append({'setting':setting,'variant':variant,'mask_scale':scale,'quality':{'frames':48}})
    return {'code_revision':m.REPORT_REVISION,'source_rgb_sha256':m.SOURCE_RGB_SHA,'source_shape':[48,360,640,3],'rows':rows}


def test_exact19_inventory_missing_duplicate_and_partial_frame_rejected():
    r=report();assert len(m.report_index(r))==19
    r['rows'].pop()
    with pytest.raises(ValueError,match='exact19'):m.report_index(r)
    r=report();r['rows'][-1]=dict(r['rows'][0])
    with pytest.raises(ValueError,match='duplicate'):m.report_index(r)
    r=report();r['rows'][0]['quality']['frames']=47
    with pytest.raises(ValueError,match='all48'):m.report_index(r)


def test_package_sha_and_byte_length_checked_before_decode(tmp_path):
    path=tmp_path/'packet';path.write_bytes(b'packet')
    identity={'path':str(path),'bytes':6,'sha256':m.sha(b'packet')}
    assert m.checked_payload(identity)==b'packet'
    identity['bytes']=5
    with pytest.raises(ValueError,match='bytes/SHA'):m.checked_payload(identity)
    identity['bytes']=6;identity['sha256']='0'*64
    with pytest.raises(ValueError,match='bytes/SHA'):m.checked_payload(identity)


def test_fail_all_inventory_before_any_mask_decode(monkeypatch):
    r=report();calls=[]
    for row in r['rows']:row['packet']={'path':'declared','bytes':1,'sha256':'a'}
    def checked(identity):
        calls.append('identity')
        if len(calls)==19:raise ValueError('bad last package')
        return b'packet'
    monkeypatch.setattr(m,'checked_payload',checked)
    monkeypatch.setattr(m,'decoded_union',lambda _:pytest.fail('decoded before complete inventory passed'))
    with pytest.raises(ValueError,match='last'):m.diagnose(r)
    assert len(calls)==19


def test_full48_wire_extraction_includes_empty_frames_and_lossy_disappearance():
    from src.runner.client import ClientPlacement,serialize_client_request
    from src.runner.packet_packing import pack_client_envelope
    mask=np.zeros((360,640),bool);mask[1,1]=True
    original=serialize_client_request(background=None,frame_count=48,height=360,width=640,
        placements=(ClientPlacement(encoded_crop=b'opaque appearance; no pixel decoding in this test',
            bbox=(0,0,2,2),mask=mask,frame_index=0),))
    lossy=pack_client_envelope(original,mask_codec='rle',mask_scale=4,batch_masks=True)
    before=m.decoded_union(original);after=m.decoded_union(lossy)
    assert before.shape==after.shape==(48,360,640)
    d=m.compare_masks(before,after)
    assert d['fixed_original_positive_pixels']==1 and d['fn_pixels_relative_to_parent']==1
    assert d['nonempty_parent_to_empty_delivered_frames']==[0]
    assert d['empty_union_frames']==list(range(1,48))

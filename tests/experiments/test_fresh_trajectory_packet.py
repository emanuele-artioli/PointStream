import copy
import io
import numpy as np
import pytest
from experiments.gate_a_confirmation.fresh_trajectory_packet import adapt_packet, encode_packet


def inputs():
    arrays={'encoded_crop_0':np.array([1,2,3],dtype=np.uint8),
            'background_payload':np.array([91,92],dtype=np.uint8),
            'mask_0':np.ones((4,6,8),dtype=np.uint8),
            'unused_preserved':np.array([77],dtype=np.uint8)}
    meta={'schema':1,'frame_count':4,'width':8,'height':6,
          'background':{'homographies':list(range(4))},
          'placements':[{'object_id':'player_near','frame_index':1,'bbox':[1,1,3,3],
                         'encoded_crop_key':'encoded_crop_0','crop_key':'crop_0','mask_key':'mask_0'}]}
    receipt={'status':'complete','frame_count':4,'source_frame_interval':[38,42],
             'objects':[{'object_id':'player_near','object_class':'player','frame_index':1,'bbox':[1,1,3,3]}]}
    records=[]
    for i in range(4):
        records.append({'frame_index':i,'source_frame_index':38+i,'status':'complete',
          'tracked':[{'track_id':'player_near','class_name':'player','bbox':[i,1,i+2,3],
                      'mask_missing':False,'history_held':False,'tracker_history_held':False,'stale':False}]})
    records[1]['tracked'][0]['bbox']=[1,1,3,3]
    return arrays,meta,records,receipt


def test_motion_shared_static_appearance_initialization_and_immutable_payload():
    arrays,meta,records,receipt=inputs();before=copy.deepcopy((arrays,meta,records,receipt))
    out,new,coverage=adapt_packet(arrays,meta,records,receipt,policy='opaque')
    assert coverage==[0,1,1,1]
    assert [r['bbox'] for r in new['placements']]==[[1,1,3,3],[2,1,4,3],[3,1,5,3]]
    assert all(r['encoded_crop_key']=='encoded_crop_0' for r in new['placements'])
    assert all(out[k] is arrays[k] for k in arrays)
    assert meta==before[1] and records==before[2] and receipt==before[3]
    assert all(np.array_equal(arrays[k],before[0][k]) for k in arrays)
    with np.load(io.BytesIO(encode_packet(out)),allow_pickle=False) as packet:
        assert np.array_equal(packet['encoded_crop_0'],arrays['encoded_crop_0'])
        assert np.array_equal(packet['background_payload'],arrays['background_payload'])
        assert 'unused_preserved' in packet


def test_missing_absent_held_null_and_alpha_are_explicit():
    arrays,meta,records,receipt=inputs()
    records[2]['tracked'][0]['mask_missing']=True
    records[3]['tracked'][0].update(history_held=True,stale=True)
    out,new,c=adapt_packet(arrays,meta,records,receipt,policy='alpha')
    assert c==[0,1,0,1] and out['fresh_static_alpha_player_near'].shape==(2,2)
    assert 'mask_0' in out and new['temporal_adapter']['appearance']=='shared_initial_reference_static'
    assert adapt_packet(arrays,meta,records,receipt,policy='opaque',held_policy='skip_recorded')[2]==[0,1,0,0]
    with pytest.raises(ValueError):adapt_packet(arrays,meta,records,receipt,policy='opaque',held_policy='reject')
    assert adapt_packet(arrays,meta,records,receipt,policy='null')[2]==[0]*4
    records[3]['tracked']=[]
    result=adapt_packet(arrays,meta,records,receipt,policy='opaque')[1]
    assert result['temporal_adapter']['role_decisions'][-1]['reason']=='role_absent'


@pytest.mark.parametrize('change', ['float_box','outside','role','duplicate','frame','init','stale','missing_flag'])
def test_invalid_geometry_roles_frames_and_flags_fail_closed(change):
    arrays,meta,records,receipt=inputs();row=records[2]['tracked'][0]
    if change=='float_box':row['bbox'][0]=2.25
    elif change=='outside':row['bbox'][2]=9
    elif change=='role':row['track_id']='spectator'
    elif change=='duplicate':records[2]['tracked'].append(copy.deepcopy(row))
    elif change=='frame':records[2]['source_frame_index']=100
    elif change=='init':receipt['objects'][0]['frame_index']=0
    elif change=='stale':row['history_held']=True
    elif change=='missing_flag':row.pop('mask_missing')
    with pytest.raises(ValueError):adapt_packet(arrays,meta,records,receipt,policy='opaque')


def test_two_roles_late_initialization_and_missing_sender_role():
    arrays,meta,records,receipt=inputs()
    arrays['encoded_crop_1']=np.array([4,5],dtype=np.uint8)
    meta['placements'].append({'object_id':'player_far','frame_index':2,'bbox':[4,3,6,5],
        'encoded_crop_key':'encoded_crop_1','crop_key':'crop_1','mask_key':None})
    receipt['objects'].append({'object_id':'player_far','object_class':'player','frame_index':2,'bbox':[4,3,6,5]})
    for record in records:
        record['tracked'].append({'track_id':'player_far','class_name':'player','bbox':[4,3,6,5],
            'mask_missing':False,'history_held':False,'tracker_history_held':False,'stale':False})
    out,new,c=adapt_packet(arrays,meta,records,receipt,policy='opaque')
    assert c==[0,1,2,2]
    assert [p['frame_index'] for p in new['placements'] if p['object_id']=='player_far']==[2,3]
    meta['placements'].pop()
    with pytest.raises(ValueError):adapt_packet(arrays,meta,records,receipt,policy='opaque')


def test_actual_guide_integral_float_box_representation():
    arrays,meta,records,receipt=inputs()
    for record in records:
        record['tracked'][0]['bbox']=[float(v) for v in record['tracked'][0]['bbox']]
    meta['placements'][0]['bbox']=[float(v) for v in meta['placements'][0]['bbox']]
    _,updated,coverage=adapt_packet(arrays,meta,records,receipt,policy='alpha')
    assert coverage==[0,1,1,1]
    assert all(type(v) is int for row in updated['placements'] for v in row['bbox'])
    assert updated['placements'][-1]['bbox']==[3,1,5,3]
    records[2]['tracked'][0]['bbox'][0]=float('nan')
    with pytest.raises(ValueError):adapt_packet(arrays,meta,records,receipt,policy='opaque')

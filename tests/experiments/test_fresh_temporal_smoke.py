import numpy as np
import pytest
from experiments.gate_a_confirmation.fresh_temporal_smoke import coverage_checks, visible_checks


def metadata():
    return {'frame_count':2,'placements':[{'object_id':'player_near','frame_index':0,'bbox':[0,0,2,2]}, {'object_id':'player_near','frame_index':1,'bbox':[1,0,3,2]}]}


def test_requires_motion_and_each_frame_role_coverage():
    meta=metadata();assert coverage_checks(meta,[{},{}])['moving_roles']==['player_near']
    meta['placements'][1]['bbox']=[0,0,2,2]
    with pytest.raises(ValueError):coverage_checks(meta,[{},{}])
    meta['placements'].pop()
    with pytest.raises(ValueError):coverage_checks(meta,[{},{}])


def test_visible_inside_bboxes_and_failures_not_finite_metric_proxy():
    null=np.zeros((2,3,4,3),dtype=np.uint8);fg=null.copy();fg[0,:2,:2]=23;fg[1,:2,1:3]=44
    assert [r['changed_pixels'] for r in visible_checks(fg,null,metadata())]==[4,4]
    fg[1,2,3]=1
    with pytest.raises(ValueError):visible_checks(fg,null,metadata())
    fg=null.copy()
    with pytest.raises(ValueError):visible_checks(fg,null,metadata())


def test_actual_receiver_policy_command_contract():
    import ast
    from pathlib import Path
    from experiments.gate_a_confirmation.fresh_temporal_smoke import receiver_command
    # Check the actual wrapper AST, rather than a fake consumer's API.
    wrapper=Path(__file__).parents[2]/'experiments/gate_a_confirmation/trajectory_receiver.py'
    tree=ast.parse(wrapper.read_text())
    allowed=next(node.comparators[0] for node in ast.walk(tree) if isinstance(node,ast.Compare)
        and isinstance(node.left,ast.Name) and node.left.id=='policy'
        and any(isinstance(op,ast.NotIn) for op in node.ops))
    assert ast.literal_eval(allowed)==['opaque','alpha']
    reg={'base_receiver':'base.py','mask_receiver':'trajectory_receiver.py'}
    for control in ('null','unadapted','opaque','alpha'):
        cmd=receiver_command(control,reg,package='p.npz',output='r.npy',root='legacy',data='protected')
        assert cmd[1]==('trajectory_receiver.py' if control in ast.literal_eval(allowed) else 'base.py')
        assert cmd[-2:]==['--deny-root','protected']
    with pytest.raises(ValueError):receiver_command('unknown',reg,package='p',output='r',root='l',data='d')


def test_unadapted_requires_same_rgb_file_and_receipt_identity():
    from experiments.gate_a_confirmation.fresh_temporal_smoke import check_unadapted_identity
    receipt={'frames_rgb_sha256':'rgb','frames_shape':[2,3,4,3],'package_sha256':'packet','package_bytes':17}
    report={'receiver':receipt.copy(),'receiver_file_sha256':'file'}
    check_unadapted_identity(receipt,'file',report)
    with pytest.raises(ValueError):check_unadapted_identity(receipt,'changed-file',report)
    for key,value in [('frames_rgb_sha256','changed-rgb'),('package_sha256','different-packet')]:
        modified=receipt.copy();modified[key]=value
        with pytest.raises(ValueError):check_unadapted_identity(modified,'file',report)

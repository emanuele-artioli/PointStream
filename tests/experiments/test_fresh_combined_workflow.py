import copy
from pathlib import Path

import pytest
from experiments.gate_a_confirmation.fresh_combined_workflow import derive_sender_registration, validate_temporal_gate


def evidence():
    joins={'rung':'C2','base_receiver_sha256':'base','mask_receiver_sha256':'mask','source96_rgb_sha256':'source','config_sha256':'config','sender_worker_sha256':'sender','adapter_sha256':'adapter','smoke_input_pins':{'/tmp/guide-reg':'guide'}}
    qualification={'schema':'pointstream.fresh_temporal_smoke.qualification.v1','passed':True,
        'visual_review':'passed','proof_sha256':'proof','registration_sha256':'reg','joins':joins.copy()}
    proof={'schema':'pointstream.fresh_temporal_smoke.proof.v1','mechanics_passed':True,
        'registration_sha256':'reg','controls':dict.fromkeys(('null','unadapted','opaque','alpha'))}
    smoke_reg={'frames':2,'rung':'C2','guide_qualification':{'passed':True,'frames':96},
        'base_receiver':'/tmp/base.py','mask_receiver':'/tmp/mask.py',
        'file_sha256':{str(Path('/tmp/base.py').resolve()):'base',str(Path('/tmp/mask.py').resolve()):'mask','/tmp/guide-reg':'guide',str(Path('/tmp/fresh_trajectory_packet.py').resolve()):'adapter',str(Path('/tmp/fresh_window_sender.py').resolve()):'sender'}}
    return qualification,proof,smoke_reg,joins


def test_longer_arms_require_exact_separately_qualified_proof():
    args=evidence()
    for frames in (12,48,96):validate_temporal_gate(frames,*args,'proof','reg')
    for key,value in (('passed',False),('visual_review','pending'),('proof_sha256','different'),('registration_sha256','different')):
        changed=list(copy.deepcopy(args));changed[0][key]=value
        with pytest.raises(ValueError):validate_temporal_gate(96,*changed,'proof','reg')
    for key in ('source96_rgb_sha256','config_sha256','base_receiver_sha256','mask_receiver_sha256'):
        changed=list(copy.deepcopy(args));changed[0]['joins'][key]='different'
        with pytest.raises(ValueError):validate_temporal_gate(96,*changed,'proof','reg')


def test_motion_mechanics_proof_cannot_self_promote_or_drop_controls():
    args=list(evidence()); args[0]=dict(args[0],visual_review='pending')
    with pytest.raises(ValueError):validate_temporal_gate(12,*args,'proof','reg')
    args=list(evidence());args[1]['controls'].pop('null')
    with pytest.raises(ValueError):validate_temporal_gate(48,*args,'proof','reg')
    args=list(evidence());args[2]['guide_qualification']['frames']=2
    with pytest.raises(ValueError):validate_temporal_gate(96,*args,'proof','reg')


def test_template_has_no_direct_longer_arm_and_derivation_preserves_input():
    template={'status':'frozen_before_execution','arms':[],'worker_sha256':'original'}
    arm={'frames':96,'source_window_index':0,'appearance_input_policy':'reference_cutout','rung':'C2'}
    derived=derive_sender_registration(template,arm)
    assert derived['arms']==[arm] and template['arms']==[]
    derived['arms'][0]['frames']=12
    assert arm['frames']==96
    with pytest.raises(ValueError):derive_sender_registration(dict(template,arms=[arm]),arm)


def test_two_frame_workflow_does_not_require_longer_qualification():
    validate_temporal_gate(2,None,None,None,None,None,None)


def test_wrong_smoke_guide_or_adapter_cannot_promote():
    for path in ('/tmp/guide-reg',str(Path('/tmp/fresh_trajectory_packet.py').resolve()),str(Path('/tmp/fresh_window_sender.py').resolve())):
        args=list(evidence());args[2]['file_sha256'][path]='changed'
        with pytest.raises(ValueError):
            validate_temporal_gate(96,*args,'proof','reg')


def test_each_role_required_every_frame_before_scoring():
    from experiments.gate_a_confirmation.fresh_temporal_smoke import coverage_checks
    rows=[{'object_id':role,'frame_index':i,'bbox':[i,0,i+2,2]} for i in range(12) for role in ('far','near')]
    assert coverage_checks({'frame_count':12,'placements':rows},[{}]*12)['roles']==['far','near']
    rows.pop()
    with pytest.raises(ValueError,match='all-role coverage'):
        coverage_checks({'frame_count':12,'placements':rows},[{}]*12)


def test_registered_four_control_wall_budget():
    from experiments.gate_a_confirmation.fresh_combined_workflow import validate_budget
    reg={'sender_timeout_seconds':600,'receiver_timeout_seconds':100,'score_timeout_seconds':200,'total_budget_seconds':3000}
    assert validate_budget(reg)==3000
    reg['total_budget_seconds']=2999
    with pytest.raises(ValueError,match='all four controls'):validate_budget(reg)
    reg['score_timeout_seconds']=3301
    with pytest.raises(ValueError,match='per-score'):validate_budget(reg)


def test_wrong_smoke_source_report_cannot_promote():
    from experiments.gate_a_confirmation.fresh_combined_workflow import validate_smoke_sender
    template={'guide_receipt_sha256':'guide','guide_registration_sha256':'guide-reg','legacy_files':{},
              'tools_and_libraries':{},'worker_sha256':'sender','receiver_sha256':'receiver','source_files':{'/source':'file'}}
    arm={'rung':'C2','config_sha256':'config'}
    prior={'complete':True,'arm':dict(arm,frames=2),'source96_rgb_sha256':'wrong'}
    with pytest.raises(ValueError,match='source/config/guide/code'):
        validate_smoke_sender(prior,arm,template,{})


def test_prior_stage_helper_roles_match_across_distinct_immutable_roots():
    for stage in ('/tmp/completed-temporal-stage','/tmp/other-completed-stage'):
        args=list(evidence());smoke_reg=args[2]
        smoke_reg['base_receiver']=stage+'/base.py';smoke_reg['mask_receiver']=stage+'/mask.py'
        smoke_reg['file_sha256']={str(Path(stage+'/base.py').resolve()):'base',
            str(Path(stage+'/mask.py').resolve()):'mask',
            str(Path(stage+'/fresh_window_sender.py').resolve()):'sender',
            str(Path(stage+'/fresh_trajectory_packet.py').resolve()):'adapter','/tmp/guide-reg':'guide'}
        validate_temporal_gate(96,*args,'proof','reg')
        smoke_reg['file_sha256'][str(Path(stage+'/fresh_trajectory_packet.py').resolve())]='wrong'
        with pytest.raises(ValueError,match='helper role hashes'):
            validate_temporal_gate(96,*args,'proof','reg')

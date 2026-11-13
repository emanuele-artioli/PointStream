"""Candidate construction tests; all source gates remain external approvals."""
import datetime
import hashlib
import json

import pytest

from experiments.tennis import k5d_svt_oracle as oracle, k5d_svt_profile as profile


def test_constructor_removes_neural_weights_and_preserves_original_svt_arguments(tmp_path,monkeypatch):
    base=json.loads(oracle.cf.BASE.read_text())
    monkeypatch.setattr(oracle,'source_admission',lambda *a:None)
    assets={}
    for name in sorted(oracle.ASSETS):
        path=tmp_path/(name+'.json')
        path.write_text('{}')
        assets[name]={'local':str(path),'remote':'/home/itec/emanuele/Datasets/pointstream-data/checks/'+path.name,
                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(oracle,'REFERENCE_SHA',assets['ada_reference']['sha256'])
    deadline=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=1)).isoformat()
    result=oracle.spec(base,assets=assets,identity={},deadline=deadline,unshare_sha256='a'*64,smoke_seconds=60,budget_seconds=120)
    names={x['name'] for x in result['staged_inputs']}
    assert not {'image_ckpt','video_ckpt'} & names
    assert names == ({x['name'] for x in base['staged_inputs']}-{'image_ckpt','video_ckpt'})|oracle.ASSETS
    assert '--image-ckpt' not in result['arguments'] and '--video-ckpt' not in result['arguments']
    assert result['gpu_memory_mib'] == 10240 and result['k5d_evaluation_profile'] == profile.PROFILE
    assert result['scale']['frames'] == {'smoke':16,'full':16}
    assert result['contention']['resume_attempts'] == 0
    assert '--lpips-backbone' in result['arguments'] and '--dists-backbone' in result['arguments']
    args=oracle.parser().parse_args(['run','--frames','16'])
    assert not hasattr(args,'image_ckpt') and not hasattr(args,'video_ckpt')
    assert '--image-ckpt' in base['arguments']  # Caller/default source remains intact.


def test_missing_source_or_denial_assets_never_render_candidate():
    with pytest.raises(ValueError,match='exact source'):
        oracle.spec({},assets={},identity={},deadline='later',unshare_sha256='a'*64,smoke_seconds=60,budget_seconds=120)


def test_non_smoke_validation_cannot_promote_an_oracle(tmp_path,monkeypatch):
    monkeypatch.setenv('PS_STAGE','full')
    monkeypatch.setenv('PS_STAGE_DIR',str(tmp_path/'full'))
    monkeypatch.setenv('PS_VALIDATION_PATH',str(tmp_path/'validation.json'))
    with pytest.raises(ValueError,match='smoke-only'):
        oracle.validate()
    assert json.loads((tmp_path/'validation.json').read_text())['passed'] is False


def test_validator_requires_actual_context_full_scores_and_every_native_cost(tmp_path,monkeypatch):
    identity={'source_sha256':'s'}
    context={'execution_identity':identity,'job_id':'actual','stage':'smoke','gpu':{'uuid':'GPU-real'}}
    points=[{'point':p,'receiver':{'decoded_yuv_sha256':hashlib.sha256(b'yuv').hexdigest(),
            'rendered_rgb_sha256':hashlib.sha256(b'rgb').hexdigest()},'streak':{'guard_off':True}} for p in (48,55,62)]
    for point in points:
        directory=tmp_path/f"svt-{point['point']}"
        evidence=directory/'cpu-receiver-evidence'
        evidence.mkdir(parents=True)
        for name,data in (('packet.psbl',b'wire'),('native.ivf',b'native'),('native.yuv',b'yuv'),('rendered.rgb',b'rgb')):
            (evidence/name).write_bytes(data)
        (directory/'stream.psbl').write_bytes(b'wire')
        (directory/'native.bin').write_bytes(b'native')
    report={'clip':oracle.cost.CLIP,'identities':{'lpips_backbone':{},'dists_backbone':{}},'points':points}
    record={'profile':profile.PROFILE,'execution_identity':identity,'producer_context':context,
            'points':[list(p) for p in profile.POINTS],'receiver_config_sha256':'cfg'}
    qualification={'identities':{key:{'sha256':sha} for key,sha in (
        ('exposure',oracle.fleet.EXPOSURE_SHA),('manifest',oracle.fleet.SOURCE_SHA),('archive',oracle.fleet.EXPECTED['archive']))}}
    names=['source.oracle_total','encoder.fit_streak','source.rgb_to_yuv','source.yuv_to_rgb_floor',
           'model.wasb_init','model.balltrack_init','model.lpips_init','model.dists_init']
    for p in (48,55,62):
        names += [f'native.svt.{p}.total',f'receiver.svt.{p}',f'encoder.svt.{p}.guard',
                  f'metrics.svt.{p}.baseline',f'judges.svt.{p}.baseline']
    timing={'events':[{'name':n,'completed':True,'wall_seconds':.1,'self_cpu_seconds':.1,'waited_children_cpu_seconds':0.} for n in names]}
    files={'pilot.json':report,'svt-oracle.json':record,'resources.json':{'resources':{'svt':{}}},
           'source-qualification.json':qualification,'svt-reference.json':{},'svt-resource.json':{'setup':{'gpu_uuid':'GPU-real'}},'timing.json':timing}
    for name,value in files.items():
        (tmp_path/name).write_text(json.dumps(value))
    result={'frames':16,'points':3,'full_metrics':True}
    monkeypatch.setattr(oracle.enc,'validate',lambda *a,**kw:result)
    monkeypatch.setattr(oracle.oracle,'validate_report',lambda *a:None)
    monkeypatch.setattr(profile,'validate_memory',lambda *a:None)
    monkeypatch.setattr(oracle,'compare_projection',lambda *a:None)
    checks=[]
    monkeypatch.setattr(oracle.isolated,'verify_receipt',lambda *a,**kw:checks.append(kw))
    assert oracle.validate_output(tmp_path,identity,producer_context=context,expected_reference={})['checks']['all_three_svt_points']
    assert len(checks)==3 and all(x['require_cpu_only'] is True for x in checks)
    with pytest.raises(ValueError,match='scope'):
        oracle.validate_output(tmp_path,identity,producer_context={**context,'job_id':'different'},expected_reference={})
    result['full_metrics']=False
    with pytest.raises(ValueError,match='scope'):
        oracle.validate_output(tmp_path,identity,producer_context=context,expected_reference={})
    result['full_metrics']=True
    timing['events']=[x for x in timing['events'] if x['name']!='receiver.svt.48']
    (tmp_path/'timing.json').write_text(json.dumps(timing))
    with pytest.raises(ValueError,match='cost coverage'):
        oracle.validate_output(tmp_path,identity,producer_context=context,expected_reference={})


@pytest.mark.parametrize('changed',('native','wire','render','zero','guard','full_metric','roi_metric','judge','source'))
def test_exclusive_reference_comparison_keeps_exact_semantic_rules(changed):
    import copy
    point={k:'same' for k in oracle.POINT_FIELDS}
    point.update(codec='svt',point=48,baseline={k:'same' for k in oracle.BASE_FIELDS},streak={'guard_off':False})
    for arm in ('baseline','streak'):
        point[arm].update(metrics=[{'full_lpips':.1,'crop_lpips':.2}],judges={'wasb':{'rows':[1]},'balltrack':{'rows':[2]}})
    points=[copy.deepcopy(point) for _ in range(3)]
    for p,num in zip(points,(48,55,62)):
        p['point']=num
    report={'frames':16,'clip':oracle.cost.CLIP,'size':[1280,720],'fps':'30','full_metrics':True,
            'conversion':{'yuv_sha256':'same','floor_psnr':30.},'points':points}
    source={'rows':[{'prediction':1}],'scores':{'wasb':1}}
    reference={'reference_archive_sha256':oracle.REFERENCE_SHA,'reference_execution_identity':oracle.ada.REFERENCE_IDENTITY,
               'reference_job':oracle.ada.REFERENCE_JOB,'contract':{k:report[k] for k in ('frames','clip','size','fps','full_metrics')},
               'conversion':copy.deepcopy(report['conversion']),'source_rows':copy.deepcopy(source['rows']),
               'source_scores':copy.deepcopy(source['scores']),'points':copy.deepcopy(points)}
    oracle.compare_projection(reference,report,source)
    p=points[0]
    if changed=='native':
        p['native_sha256']='changed'
    elif changed=='wire':
        p['wire_sha256']='changed'
    elif changed=='render':
        p['encoder_rendered_sha256']='changed'
    elif changed=='zero':
        p['baseline']['zero_ball_wire_sha256']='changed'
    elif changed=='guard':
        p['streak']['guard_off']=True
    elif changed=='full_metric':
        p['baseline']['metrics'][0]['full_lpips']+=1e-12
    elif changed=='roi_metric':
        p['streak']['metrics'][0]['crop_lpips']+=1e-12
    elif changed=='judge':
        p['baseline']['judges']['wasb']['rows']=[3]
    else:
        source['rows'][0]['prediction']=2
    with pytest.raises(ValueError,match='control'):
        oracle.compare_projection(reference,report,source)


def test_reference_reads_fixed_metadata_projection_without_neural_replay(tmp_path,monkeypatch):
    import io
    import tarfile
    archive=tmp_path/'reference.tar'
    records={
        'publish/fresh.json':{'execution_identity':oracle.ada.REFERENCE_IDENTITY,'gpu_profile':'rtx6000ada',
                              'device':{'name':oracle.ada.GPU_NAME,'capability':[8,9]}},
        'publish/prepared/same-job-receiver.json':{'producer_job':oracle.ada.REFERENCE_JOB,'producer_stage':'smoke'},
        'publish/scored/assembled/pilot.json':{'frames':16,'clip':oracle.cost.CLIP,'size':[64,64],'fps':'30','full_metrics':True,
            'conversion':{'yuv_sha256':'h','floor_psnr':30.},
            'points':[{'codec':kind,'point':p} for kind,p in (('dcvc',0),*profile.POINTS)]},
        'publish/scored/assembled/source-qualification.json':{'rows':[],'scores':{}},
    }
    with tarfile.open(archive,'w') as bundle:
        for name,record in records.items():
            data=json.dumps(record).encode()
            member=tarfile.TarInfo(name)
            member.size=len(data)
            bundle.addfile(member,io.BytesIO(data))
        data=b'not a stream'
        member=tarfile.TarInfo('publish/dcvc-0/stream.psdc')
        member.size=len(data)
        bundle.addfile(member,io.BytesIO(data))
    monkeypatch.setattr(oracle,'REFERENCE_SHA',hashlib.sha256(archive.read_bytes()).hexdigest())
    original=tarfile.TarFile.extractfile
    names=[]
    def extract(self,member):
        names.append(member.name)
        assert 'dcvc' not in member.name
        return original(self,member)
    monkeypatch.setattr(tarfile.TarFile,'extractfile',extract)
    result=oracle.reference_projection(archive)
    assert [(p['codec'],p['point']) for p in result['points']] == list(profile.POINTS)
    assert len(names)==4
    archive.write_bytes(archive.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='archive differs'):
        oracle.reference_projection(archive)

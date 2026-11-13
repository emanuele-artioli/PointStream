"""SVT-only16 oracle and opt-in registered study stages; no dispatch authority."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import resource
import tarfile
import time
from typing import Any

from experiments.tennis import k5d_encoder as enc, k5d_svt_profile as profile
from experiments.tennis import k5d_checkpoint as cp, k5d_checkpoint_fleet as cf
from experiments.tennis import k5d_ada_scale_worker as scale, k5d_ada_witness as witness
from experiments.tennis import k5d_isolated_receiver as isolated, k5d_cost_pilot as cost
from experiments.tennis import k5d_source_oracle as oracle, k5d_fleet as fleet
from experiments.tennis import k5d_ada_scale_adapter as ada
from experiments.tennis.k5d_timing import Timer

ASSETS = {'bridge','receiver_isolation_config','native_cpu_receipt','cpu_denial','fidelity_gate','ada_reference'}


REFERENCE_SHA = ada.REFERENCE_SHA256
POINT_FIELDS = ('codec','point','native_sha256','native_bytes','wire_sha256','wire_bytes',
                'encoder_decoded_sha256','encoder_rendered_sha256')
BASE_FIELDS = ('native_bytes','native_sha256','zero_ball_wire_bytes','zero_ball_wire_sha256')


def reference_projection(archive: Path) -> dict:
    """Read only pinned metadata; never decode any archived neural stream."""
    if oracle.sha(archive) != REFERENCE_SHA:
        raise ValueError('immutable exclusive Ada control archive differs')
    wanted = {'fresh':'publish/fresh.json', 'marker':'publish/prepared/same-job-receiver.json',
              'report':'publish/scored/assembled/pilot.json',
              'source':'publish/scored/assembled/source-qualification.json'}
    records = {}
    with tarfile.open(archive,'r:') as bundle:
        for key,name in wanted.items():
            matches = [m for m in bundle.getmembers() if m.name == name]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 32*2**20:
                raise ValueError('exclusive reference metadata missing or ambiguous')
            stream = bundle.extractfile(matches[0])
            if stream is None:
                raise ValueError('exclusive reference metadata unavailable')
            records[key] = json.loads(stream.read())
    fresh,marker = records['fresh'],records['marker']
    if (fresh['execution_identity'] != ada.REFERENCE_IDENTITY or fresh['gpu_profile'] != 'rtx6000ada'
            or fresh['device']['name'] != ada.GPU_NAME or fresh['device']['capability'] != [8,9]
            or marker['producer_job'] != ada.REFERENCE_JOB or marker['producer_stage'] != 'smoke'):
        raise ValueError('exclusive Ada control provenance differs')
    report = records['report']
    points = [p for p in report['points'] if p['codec']=='svt']
    if [(p['codec'],p['point']) for p in points] != list(profile.POINTS):
        raise ValueError('exclusive control three-point projection differs')
    return {'reference_archive_sha256':REFERENCE_SHA,'reference_job':ada.REFERENCE_JOB,
            'reference_execution_identity':ada.REFERENCE_IDENTITY,
            'contract':{k:report[k] for k in ('frames','clip','size','fps','full_metrics')},
            'conversion':{k:report['conversion'][k] for k in ('yuv_sha256','floor_psnr')},
            'source_rows':records['source']['rows'],'source_scores':records['source']['scores'],
            'points':points,'scope':'fixed SVT-only metadata projection; no archived DCVC decoding'}


def compare_projection(reference: dict,report: dict,source: dict) -> None:
    if (reference['reference_archive_sha256'] != REFERENCE_SHA
            or reference['reference_execution_identity'] != ada.REFERENCE_IDENTITY
            or reference['reference_job'] != ada.REFERENCE_JOB
            or reference['contract'] != {k:report[k] for k in ('frames','clip','size','fps','full_metrics')}
            or reference['conversion'] != {k:report['conversion'][k] for k in ('yuv_sha256','floor_psnr')}
            or reference['source_rows'] != source['rows'] or reference['source_scores'] != source['scores']
            or [(p['codec'],p['point']) for p in reference['points']] != list(profile.POINTS)
            or [(p['codec'],p['point']) for p in report['points']] != list(profile.POINTS)):
        raise ValueError('exclusive SVT control/source contract differs')
    for before,after in zip(reference['points'],report['points']):
        if any(before[k] != after[k] for k in POINT_FIELDS):
            raise ValueError('exclusive SVT native/wire/pixel control differs')
        if any(before['baseline'][k] != after['baseline'][k] for k in BASE_FIELDS):
            raise ValueError('exclusive SVT zero-Ball control differs')
        if before['streak']['guard_off'] != after['streak']['guard_off']:
            raise ValueError('exclusive SVT guard control differs')
        for arm in ('baseline','streak'):
            if before[arm]['metrics'] != after[arm]['metrics'] or before[arm]['judges'] != after[arm]['judges']:
                raise ValueError('exclusive SVT full/ROI metric or judge control differs')


def source_admission(args: argparse.Namespace, identity: dict[str,str]) -> None:
    scale.base_admission(args,identity)
    witness.cpu_denial_approval(args,identity)
    record = scale.checked(Path(args.cpu_denial),args.cpu_denial_sha256)
    isolated.verify_receipt(record['receipt'],expected_context=record['producer_context'],
                           expected_config_sha256=args.receiver_isolation_config_sha256,require_cpu_only=True)


def study_admission(args: argparse.Namespace, identity: dict[str, str], spec: dict) -> dict:
    """Authenticate a root's source-bound stage decision, never allocate or submit."""
    target = profile.scope(spec.get('k5d_study_stage'), spec.get('k5d_study_clip'), spec.get('k5d_study_frames'))
    gate = scale.checked(Path(args.study_gate), args.study_gate_sha256)
    needed = {
        'qualification16': set(),
        'cost48': {'qualification16'},
        'confirmation292': {'qualification16', 'cost48'},
        'confirmation586': {'qualification16', 'cost48', 'confirmation292'},
        'study': {'qualification16', 'cost48', 'confirmation292', 'confirmation586'},
    }[target['stage']]
    evidence = gate.get('evidence_sha256', {})
    if (gate.get('approved') is not True or gate.get('purpose') != 'SVT-only-study-stage'
            or gate.get('execution_identity') != identity or gate.get('selection_sha256') != fleet.SOURCE_SHA
            or gate.get('profile') != profile.STUDY_PROFILE
            or any(gate.get(key) != target[key] for key in ('stage', 'clip', 'frames'))
            or not isinstance(evidence, dict) or set(evidence) != needed
            or any(not scale.digest(value) for value in evidence.values())
            or gate.get('complete_forecast_fits') is not True):
        raise ValueError('exact source-bound SVT study stage approval required')
    if target['stage'] in ('confirmation292', 'confirmation586', 'study'):
        margin, reserve = gate.get('cost_margin'), gate.get('failure_reserve_seconds')
        if (isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 1.5
                or isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not math.isfinite(reserve) or reserve <= 0):
            raise ValueError('measured cost admission needs margin1.5 and failure reserve')
    return gate


def execution_scope(args: argparse.Namespace, spec: dict) -> dict | None:
    """Join the dispatched phase to its prospective target; only smoke uses gold16."""
    phase = os.environ['PS_STAGE']
    if args.study_stage is None:
        if phase != 'smoke' or args.frames != 16 or args.clip != cost.CLIP or spec.get('k5d_evaluation_profile') != profile.PROFILE:
            raise ValueError('SVT oracle admits only the original16-frame smoke')
        return None
    if spec.get('k5d_evaluation_profile') != profile.STUDY_PROFILE:
        raise ValueError('explicit opt-in SVT study profile required')
    target = profile.scope(spec.get('k5d_study_stage'), spec.get('k5d_study_clip'), spec.get('k5d_study_frames'))
    expected = ('qualification16', cost.CLIP, 16) if phase == 'smoke' else (target['stage'], target['clip'], target['frames'])
    if phase not in ('smoke', 'full') or (phase == 'full' and target['stage'] == 'qualification16'):
        raise ValueError('study execution phase differs from qualification/noop contract')
    if (args.study_stage, args.clip, args.frames) != expected:
        raise ValueError('dispatched study stage/clip/frames differ from saved target')
    return profile.scope(args.study_stage, args.clip, args.frames)


class MemoryTimer(Timer):
    def __init__(self,path: Path,torch: Any,setup: dict,report: dict,receipt_path: Path):
        super().__init__(path)
        self.torch,self.setup,self.report,self.receipt_path = torch,setup,report,receipt_path

    def observe(self,phase: str) -> None:
        self.report['snapshots'].append(profile.snapshot(self.torch,self.setup,phase))
        profile.save(self.receipt_path,self.report)

    def measure(self,name: str,function: Any) -> Any:
        value = super().measure(name,function)
        if name.startswith(('model.','native.','metrics.','judges.','receiver.')) or name in ('source.oracle_total','encoder.fit_streak'):
            self.observe(name)
        return value


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    own0,children0 = resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)
    stage,scratch = Path(os.environ['PS_STAGE_DIR']),Path(os.environ['PS_SCRATCH_DIR'])
    dispatch,specification = json.loads((stage/'dispatch.json').read_text()),json.loads((stage.parent/'spec.json').read_text())
    scope = execution_scope(args, specification)
    selected = profile.STUDY_PROFILE if scope else profile.PROFILE
    identity = cp.execution_identity()
    source_admission(args,identity)
    if args.ada_reference_sha256 != REFERENCE_SHA:
        raise ValueError('immutable exclusive Ada control asset required')
    if scope:
        study_admission(args, identity, specification)
    reference = reference_projection(Path(args.ada_reference)) if scope is None or scope['stage'] == 'qualification16' else None
    import torch
    setup = profile.admit(specification,dispatch,os.environ,torch)
    resource_dir = scratch/'svt-resource-evidence'
    resource_dir.mkdir(exist_ok=False)
    memory: dict[str,Any] = {'setup':setup,'snapshots':[],'scope':'observed snapshots and allocator cap; no absolute context/nonallocator peak guarantee'}
    timer = MemoryTimer(resource_dir/'timing.json',torch,setup,memory,resource_dir/'memory.json')
    timer.observe('before_models')
    data = scale.data_args(args,stage,scratch,dispatch)
    data.timer = timer
    if scope:
        data.clip, data.study_stage = scope['clip'], scope['stage']
        data.cost_pilot = scope['stage'] in ('qualification16', 'cost48')
        data.benchmark = not data.cost_pilot
        data.fidelity_purpose = 'first-292-pair-v1' if scope['stage'] == 'confirmation292' else None
        if data.benchmark:
            data.fidelity_gate, data.fidelity_gate_sha256 = args.study_fidelity_gate, args.study_fidelity_gate_sha256
    data.receiver_isolation_config = Path(args.receiver_isolation_config)
    data.receiver_producer_context = witness.context(stage,identity,{
        'name':torch.cuda.get_device_name(0),'uuid':os.environ['CUDA_VISIBLE_DEVICES'],'capability':[8,9]})
    data.receiver_protected_paths = {'source_archive':[Path(args.archive)],
        'ground_truth':[Path(data.manifest),Path(data.exposure)],
        'private_detector':[Path(args.balltrack_weight),Path(args.wasb_config),Path(args.wasb_weights)/'wasb_tennis_best.pth.tar']}
    # Fidelity rules and all metrics/judges remain the original ones.
    purpose = '48-frame-cost-pilot' if scope is None or data.cost_pilot else data.fidelity_purpose or 'full-retrospective-benchmark'
    enc.fidelity_gate(Path(data.fidelity_gate),data.fidelity_gate_sha256,data.archive_sha256,
                      purpose=purpose,clip=data.clip,frames=args.frames)
    enc.run(data,point_profile=selected)
    if reference is not None:
        compare_projection(reference,json.loads((data.output/'pilot.json').read_text()),
                           json.loads((data.output/'source-qualification.json').read_text()))
        profile.save(data.output/'svt-reference.json',reference)
    timer.observe('after_complete')
    own,children = resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)
    memory.update(self_cpu_seconds=own.ru_utime+own.ru_stime-own0.ru_utime-own0.ru_stime,
                  waited_children_cpu_seconds=children.ru_utime+children.ru_stime-children0.ru_utime-children0.ru_stime,
                  wall_seconds=time.monotonic()-started)
    if scope:
        profile.validate_memory(memory, expected_profile=selected)
    else:
        profile.validate_memory(memory)
    profile.save(data.output/'svt-resource.json',memory)
    profile.save(data.output/'svt-oracle.json',{'profile':selected,'execution_identity':identity,
        'cpu_denial_sha256':args.cpu_denial_sha256,'receiver_config_sha256':args.receiver_isolation_config_sha256,
        'producer_context':data.receiver_producer_context,'points':[list(p) for p in profile.POINTS],
        'scope':('SVT-only registered study stage; no aggregate/dualcodec or speed claim' if scope else '16-frame SVT-only feasibility/cost oracle;30clip/dualcodec goals not established; shared timings not speed'),
        **({'study_scope':scope} if scope else {})})
    validate_output(data.output,identity,producer_context=data.receiver_producer_context,expected_reference=reference,study_scope=scope)


def validate_output(output: Path,identity: dict[str,str],*,producer_context: dict,expected_reference: dict | None,study_scope: dict | None = None) -> dict:
    selected = profile.STUDY_PROFILE if study_scope else profile.PROFILE
    if study_scope and profile.scope(study_scope['stage'],study_scope['clip'],study_scope['frames']) != study_scope:
        raise ValueError('study scope differs from frozen selection')
    frames, clip = (study_scope['frames'],study_scope['clip']) if study_scope else (16,cost.CLIP)
    result = enc.validate(output,point_profile=selected)
    report = json.loads((output/'pilot.json').read_text())
    record = json.loads((output/'svt-oracle.json').read_text())
    if (result['frames'] != frames or result['points'] != 3 or result['full_metrics'] is not True
            or report['clip'] != clip or record['profile'] != selected
            or (study_scope is not None and record.get('study_scope') != study_scope)
            or record['execution_identity'] != identity or record['producer_context'] != producer_context
            or producer_context['execution_identity'] != identity or record['points'] != [list(p) for p in profile.POINTS]
            or set(json.loads((output/'resources.json').read_text())['resources']) != {'svt'}
            or set(report['identities']) != {'lpips_backbone','dists_backbone'}):
        raise ValueError('SVT-only exact16/fullmetrics/threepoint/resource scope differs')
    # Enc validates every registered metric and both judge counts against predictions.
    qualification = json.loads((output/'source-qualification.json').read_text())
    oracle.validate_report(qualification)
    if (qualification['identities']['exposure']['sha256'] != fleet.EXPOSURE_SHA
            or qualification['identities']['manifest']['sha256'] != fleet.SOURCE_SHA
            or qualification['identities']['archive']['sha256'] != fleet.EXPECTED['archive']):
        raise ValueError('registered judge exposure, population or source archive differs')
    coverage = None
    if study_scope:
        if qualification['frames'] != frames or len(qualification['rows']) != frames:
            raise ValueError('study source coverage lost frames')
        coverage = {name:sum(row['label_kind'] == kind for row in qualification['rows'])
                    for name,kind in (('absent',0),('visible',1),('excluded',-1))}
        if sum(coverage.values()) != frames:
            raise ValueError('study source label kind is invalid')
        if study_scope['full_context']:
            from experiments.tennis.k5d_full_study import selection
            counts = selection()[clip]['visibility_counts']
            if coverage != {'absent':counts['0'],'visible':counts['1']+counts['2'],'excluded':counts['3']}:
                raise ValueError('full-context absence/exclusion counts differ from frozen selection')
        elif study_scope['stage'] == 'cost48' and coverage != {'absent':0,'visible':48,'excluded':0}:
            raise ValueError('original48 calibration labels differ; no absence control substitution')
    memory = json.loads((output/'svt-resource.json').read_text())
    if study_scope:
        profile.validate_memory(memory, expected_profile=selected)
    else:
        profile.validate_memory(memory)
    gold = study_scope is None or study_scope['stage'] == 'qualification16'
    if gold:
        if expected_reference is None or json.loads((output/'svt-reference.json').read_text()) != expected_reference:
            raise ValueError('saved exclusive reference projection differs from immutable archive')
        compare_projection(expected_reference,report,qualification)
    elif expected_reference is not None or (output/'svt-reference.json').exists():
        raise ValueError('48/full study has no archived gold reference')
    if memory['setup']['gpu_uuid'] != producer_context['gpu']['uuid']:
        raise ValueError('model memory belongs to another admitted device')
    for point in report['points']:
        isolated.verify_receipt(point['receiver'],expected_context=record['producer_context'],
                               expected_config_sha256=record['receiver_config_sha256'],require_cpu_only=True)
        directory = output/f"svt-{point['point']}"
        evidence = directory/'cpu-receiver-evidence'
        if ((evidence/'packet.psbl').read_bytes() != (directory/'stream.psbl').read_bytes()
                or (evidence/'native.ivf').read_bytes() != (directory/'native.bin').read_bytes()
                or oracle.sha(evidence/'native.yuv') != point['receiver']['decoded_yuv_sha256']
                or oracle.sha(evidence/'rendered.rgb') != point['receiver']['rendered_rgb_sha256']):
            raise ValueError('retained actual CPU receiver bytes differ')
    timing = json.loads((output/'timing.json').read_text())
    needed = {'source.oracle_total','encoder.fit_streak','source.rgb_to_yuv','source.yuv_to_rgb_floor',
              'model.wasb_init','model.balltrack_init','model.lpips_init','model.dists_init'}
    for point in report['points']:
        tag = f"svt.{point['point']}"
        needed.update({f'native.{tag}.total',f'receiver.{tag}',f'encoder.{tag}.guard',
                       f'metrics.{tag}.baseline',f'judges.{tag}.baseline'})
        if not point['streak']['guard_off']:
            needed.update({f'metrics.{tag}.streak',f'judges.{tag}.streak'})
    events = timing['events']
    names = [x['name'] for x in events]
    if any(names.count(name) != 1 for name in needed) or any(not x['completed'] for x in events):
        raise ValueError('actual native/receiver/full metrics/judge cost coverage incomplete')
    for event in events:
        for key in ('wall_seconds','self_cpu_seconds','waited_children_cpu_seconds'):
            value = event[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid scoped cost interval')
    return {'passed':True,'checks':{'all_three_svt_points':True,'full_and_roi_both_arms':True,
            'registered_judges_and_exposure':True,'actual_cpu_only_isolation':True,
            'observed_memory_within_profile':True,'oracle_scoped_cost_receipt':True},
            'points':3,'frames':frames,'scope':selected,
            **({'study_scope':study_scope,'source_task_qualified':qualification['qualified'],
                'source_failures_retained':True,'label_counts':coverage} if study_scope else {}),'cost_scope':'self plus waited children; full lifecycle externally metered'}


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        if os.environ['PS_STAGE'] != 'smoke' and not (stage/'dispatch.json').is_file():
            raise ValueError('oracle validator is smoke-only')
        from experiments.tennis.k5d_cost_pilot import unpack
        staged = json.loads((stage/'staging.json').read_text())
        command = json.loads((stage/'dispatch.json').read_text())['command']
        if command[1:3] != ['-m','experiments.tennis.k5d_svt_oracle'] or command[3] != 'run':
            raise ValueError('actual dispatched oracle command differs')
        arguments = parser().parse_args(command[3:])
        if os.environ['PS_STAGE'] != 'smoke' and arguments.study_stage is None:
            raise ValueError('oracle validator is smoke-only')
        if arguments.ada_reference_sha256 != REFERENCE_SHA:
            raise ValueError('actual exclusive reference dispatch pin differs')
        specification = json.loads((stage.parent/'spec.json').read_text())
        scope = execution_scope(arguments,specification)
        reference = reference_projection(Path(arguments.ada_reference)) if scope is None or scope['stage'] == 'qualification16' else None
        with unpack(stage/'published.tar',staged['published']['sha256']) as output:
            identity = cp.execution_identity()
            if scope:
                study_admission(arguments,identity,specification)
            gpu = json.loads((stage/'dispatch.json').read_text())['environment']['gpu']
            actual_context = witness.context(stage,identity,{'uuid':gpu['uuid'],'name':gpu['name'],'capability':[8,9]})
            result = validate_output(output,identity,producer_context=actual_context,expected_reference=reference,study_scope=scope)
        target.write_text(json.dumps(result)+'\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed':False,'error':str(exc)})+'\n')
        raise


def spec(base: dict,*,assets: dict[str,dict[str,str]],identity: dict[str,str],deadline: str,
         unshare_sha256: str,smoke_seconds: int,budget_seconds: int,
         study_stage: str | None = None,clip: str = cost.CLIP,frames: int = 16,
         study_gate: dict[str,str] | None = None,study_fidelity_gate: dict[str,str] | None = None,
         full_seconds: int = 5) -> dict:
    """Candidate only; root must independently approve complete forecasts and release."""
    from experiments.jobs.inbox import validate_spec
    if (set(assets) != ASSETS or assets.get('ada_reference',{}).get('sha256') != REFERENCE_SHA
            or not 0 < smoke_seconds <= budget_seconds < 2700):
        raise ValueError('exact source/CPU/denial/fidelity assets and bounded stage required')
    args = argparse.Namespace(**{n:assets[n]['local'] for n in ASSETS},
        **{n+'_sha256':assets[n]['sha256'] for n in ASSETS},unshare_sha256=unshare_sha256)
    source_admission(args,identity)
    result = json.loads(json.dumps(base))
    removed = {'image_ckpt','video_ckpt'}
    result['staged_inputs'] = [x for x in result['staged_inputs'] if x['name'] not in removed]
    removed_paths = {x['path'] for x in base['staged_inputs'] if x['name'] in removed}
    result['inputs'] = [x for x in result['inputs'] if x['path'] not in removed_paths]
    arguments = []
    index = 0
    while index < len(base['arguments']):
        flag = base['arguments'][index]
        if flag in ('--image-ckpt','--video-ckpt'):
            index += 2
        else:
            arguments.append(flag)
            index += 1
    for name,item in assets.items():
        staged = cf.asset(name,item['remote'],item['sha256'],Path(item['local']),extract=False)
        result['staged_inputs'].append(staged)
        result['inputs'].append({k:staged[k] for k in ('path','sha256')})
        arguments += ['--'+name.replace('_','-'),'{staged:'+name+'}', '--'+name.replace('_','-')+'-sha256',item['sha256']]
    result.update(entrypoint=['-m','experiments.tennis.k5d_svt_oracle'],arguments=['run',*arguments,'--unshare-sha256',unshare_sha256],
        scale={'frames':{'smoke':16,'full':16}},hosts=['gpu6'],cpu_threads=1,single_cpu_campaign=True,
        cpu_affinity_index=0,defer_failed_publication=True,
        k5d_evaluation_profile=profile.PROFILE,gpu_models=['RTX 6000 Ada'],gpu_memory_mib=profile.CLAIM_MIB,
        gpu_sharing={'mode':'shared','basis':profile.BASIS,'margin_mib':8192},
        contention={'policy':'continue','basis':profile.BASIS,'resume_attempts':0},deadline=deadline,
        smoke={'seconds':smoke_seconds,'representative_basis':'All3 original SVT points/full metrics/both judges/CPU-isolated actual receiving.'},
        full={'seconds':5},budget_seconds=budget_seconds,validator_seconds=30,
        validator=['{python}','-m','experiments.tennis.k5d_svt_oracle','validate'],verify_staged_inputs=True)
    # The full stage is a no-op; scaling needs a separately released profile.
    result['scale']['oracle_mode']={'smoke':'run','full':'noop'}
    result['arguments'][0]='{oracle_mode}'
    if study_stage is not None:
        profile.scope(study_stage,clip,frames)
        if study_stage == 'qualification16' and full_seconds != 5:
            raise ValueError('qualification16 full stage is only the original5-second noop')
        if study_gate is None or not 0 < smoke_seconds <= 600 or not 0 < full_seconds or smoke_seconds+full_seconds+30 >= budget_seconds:
            raise ValueError('explicit study gate and complete bounded stage allowances required')
        result.update(k5d_evaluation_profile=profile.STUDY_PROFILE,k5d_study_stage=study_stage,k5d_study_clip=clip,k5d_study_frames=frames)
        for name,item in [('study_gate',study_gate), *([('study_fidelity_gate',study_fidelity_gate)] if study_fidelity_gate else [])]:
            staged = cf.asset(name,item['remote'],item['sha256'],Path(item['local']),extract=False)
            result['staged_inputs'].append(staged)
            result['inputs'].append({key:staged[key] for key in ('path','sha256')})
            result['arguments'] += ['--'+name.replace('_','-'),'{staged:'+name+'}','--'+name.replace('_','-')+'-sha256',item['sha256']]
        if study_stage in ('confirmation292','confirmation586','study') and study_fidelity_gate is None:
            raise ValueError('full registered clips require their independently scoped fidelity gate')
        if study_fidelity_gate is not None:
            purpose = 'first-292-pair-v1' if study_stage == 'confirmation292' else 'full-retrospective-benchmark'
            enc.fidelity_gate(Path(study_fidelity_gate['local']),study_fidelity_gate['sha256'],fleet.EXPECTED['archive'],purpose=purpose,clip=clip,frames=frames)
        result['arguments'] += ['--study-stage','{study_stage}','--clip','{study_clip}']
        result['scale'].update(frames={'smoke':16,'full':frames},study_stage={'smoke':'qualification16','full':study_stage},study_clip={'smoke':cost.CLIP,'full':clip})
        result['scale']['oracle_mode']['full'] = 'noop' if study_stage == 'qualification16' else 'run'
        result['full']['seconds'] = full_seconds
        gate_args = argparse.Namespace(study_gate=study_gate['local'],study_gate_sha256=study_gate['sha256'])
        study_admission(gate_args,identity,result)
    elif study_gate is not None or study_fidelity_gate is not None or clip != cost.CLIP or frames != 16 or full_seconds != 5:
        raise ValueError('legacy16 defaults require explicit study opt-in for changed scope')
    validate_spec(result)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('run','noop','validate'))
    for name in ('archive','balltrack-code','balltrack-weight','wasb-code','wasb-weights',
                 'wasb-config','lpips-backbone','dists-backbone'):
        p.add_argument('--'+name)
    for name in sorted(ASSETS):
        flag = name.replace('_','-')
        p.add_argument('--'+flag)
        p.add_argument('--'+flag+'-sha256')
    p.add_argument('--study-stage',choices=profile.STUDY_STAGES)
    p.add_argument('--clip',default=cost.CLIP)
    for name in ('study-gate','study-fidelity-gate'):
        p.add_argument('--'+name)
        p.add_argument('--'+name+'-sha256')
    p.add_argument('--unshare-sha256')
    p.add_argument('--frames',type=int)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.action == 'validate':
        validate()
    elif args.action == 'noop':
        if os.environ['PS_STAGE'] != 'full' or args.study_stage not in (None,'qualification16'):
            raise ValueError('full-only no-op')
        out = Path(os.environ['PS_SCRATCH_DIR'])/'publish'
        out.mkdir(exist_ok=False)
        (out/'noop.json').write_text('{"scope":"no second oracle"}\n')
    else:
        run(args)


if __name__=='__main__':
    main()

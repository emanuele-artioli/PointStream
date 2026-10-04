"""Registered CPU-only fresh sender -> trajectory controls -> original4K scoring.

New registration schema pointstream.fresh_combined_workflow.v1 has: status,
worker_sha256, file_sha256 (all code/receivers/native/guide/source/template/proof
inputs), legacy_root, data_root_canonical/logical, base_receiver, mask_receiver,
sender_template, sender_template_sha256, arms (exact original sender arms),
controls=['null','unadapted','opaque','alpha'], held_policy='preserve_recorded',
sender_timeout_seconds, receiver_timeout_seconds, score_timeout_seconds,
total_budget_seconds, and temporal_qualification. Total budget must cover sender
+ four receiver caps + four VMAF caps + 1200s conversion/metrics headroom.

Sender template is a NEW pinned sender registration with arms=[]: no longer
arm is runnable directly. The selected registration is derived only inside the
owned output after proof validation, prior to sender execution. Old registration
and artifacts are never modified. Qualified proof is required for frames>2.
Qualification schema pointstream.fresh_temporal_smoke.qualification.v1: passed,
visual_review='passed', proof_path/SHA, registration_path/SHA and joins equal
qualification_joins(template,reg). Root review must create and pin this file.
Static initial appearance + model guides do not establish pose/task fidelity.
"""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
from experiments.gate_a_confirmation import fresh_window_sender as sender_worker
from experiments.gate_a_confirmation import fresh_temporal_smoke as smoke
from experiments.gate_a_confirmation.fresh_trajectory_packet import adapt_packet, encode_packet
from experiments.gate_a_confirmation.score_receiver import psnr, REVISION, SOURCE_RGB
from experiments.gate_a_confirmation.fresh_window_sender import digest, rgb_digest, require


def qualification_joins(template, reg, arm):
    return {key: template[key] for key in ('source96_rgb_sha256', 'source_files',
            'guide_receipt_sha256', 'guide_registration_sha256', 'guide_artifact_sha256',
            'source000_guide_masks', 'source000_initial_appearance', 'legacy_files',
            'tools_and_libraries')} | {
        'config_sha256': arm['config_sha256'], 'rung': arm['rung'],
        'base_receiver_sha256': reg['file_sha256'][str(Path(reg['base_receiver']).resolve())],
        'mask_receiver_sha256': reg['file_sha256'][str(Path(reg['mask_receiver']).resolve())],
        'sender_worker_sha256': template['worker_sha256'],
        'smoke_input_pins': {
            str(Path(template['guide_root']).resolve()/'registration.json'):template['guide_registration_sha256'],
            str(Path(template['guide_root']).resolve()/'receipt.json'):template['guide_receipt_sha256'],
        },
        'adapter_sha256': reg['file_sha256'][str(Path(__file__).with_name('fresh_trajectory_packet.py').resolve())],
    }


def validate_temporal_gate(frames, qualification, proof, smoke_reg, expected_joins, proof_sha, registration_sha):
    if frames == 2:
        return
    require(frames in (12, 48, 96), 'registered frame count required')
    require(qualification.get('schema') == 'pointstream.fresh_temporal_smoke.qualification.v1'
            and qualification.get('passed') is True and qualification.get('visual_review') == 'passed',
            'separately root-qualified temporal smoke required before longer sender')
    require(qualification['proof_sha256'] == proof_sha and qualification['registration_sha256'] == registration_sha
            and qualification['joins'] == expected_joins, 'temporal qualification exact source/guide/config/code joins')
    require(proof.get('schema') == 'pointstream.fresh_temporal_smoke.proof.v1'
            and proof.get('mechanics_passed') is True and proof.get('registration_sha256') == registration_sha
            and set(proof['controls']) == set(smoke.CONTROLS), 'exact successful four-control temporal proof required')
    require(smoke_reg['frames'] == 2 and smoke_reg['rung'] == expected_joins['rung']
            and smoke_reg['guide_qualification']['passed'] is True
            and smoke_reg['guide_qualification']['frames'] == 96,
            'qualified smoke source000 guide/rung join')
    require(smoke_reg['file_sha256'][str(Path(smoke_reg['base_receiver']).resolve())] == expected_joins['base_receiver_sha256']
            and smoke_reg['file_sha256'][str(Path(smoke_reg['mask_receiver']).resolve())] == expected_joins['mask_receiver_sha256'],
            'qualified smoke receiver code join')
    pinmap=smoke_reg['file_sha256']
    prior_stage=Path(smoke_reg['base_receiver']).resolve().parent
    require(pinmap.get(str(prior_stage/'fresh_window_sender.py'))==expected_joins['sender_worker_sha256']
            and pinmap.get(str(prior_stage/'fresh_trajectory_packet.py'))==expected_joins['adapter_sha256'],
            'qualified smoke helper role hashes differ')
    for path,sha in expected_joins['smoke_input_pins'].items():
        require(pinmap.get(path)==sha, 'qualified smoke source/guide/adapter/sender pin join')


def validate_smoke_sender(prior,arm,template,proof):
    require(prior['complete'] is True and prior['arm']['frames']==2
            and prior['arm']['rung']==arm['rung'] and prior['arm']['config_sha256']==arm['config_sha256']
            and prior['source96_rgb_sha256']==SOURCE_RGB
            and prior['guide_receipt_sha256']==template['guide_receipt_sha256']
            and prior['guide_registration_sha256']==template['guide_registration_sha256']
            and prior['legacy_files']==template['legacy_files']
            and prior['tools_and_libraries']==template['tools_and_libraries']
            and prior['worker_sha256']==template['worker_sha256']
            and prior['receiver_script_sha256']==template['receiver_sha256'], 'qualified smoke sender source/config/guide/code joins')
    require(proof['controls']['unadapted']['receiver_receipt']==prior['receiver']
            and proof['controls']['unadapted']['receiver_file_sha256']==prior['receiver_file_sha256'],
            'qualified smoke unadapted receiver/report join')
    require({row['source_file']:row['source_file_sha256'] for row in prior['verified_original_sources']}==template['source_files'], 'qualified smoke original source file join')


def validate_budget(reg):
    caps = [reg[k] for k in ('sender_timeout_seconds','receiver_timeout_seconds','score_timeout_seconds','total_budget_seconds')]
    require(all(type(x) is int and x>0 for x in caps), 'positive registered walltime caps required')
    require(reg['score_timeout_seconds']<=3300, 'bounded registered per-score cap required')
    minimum=reg['sender_timeout_seconds'] + 4*reg['receiver_timeout_seconds'] + 4*reg['score_timeout_seconds'] + 1200
    require(reg['total_budget_seconds']>=minimum, 'prospective total budget must cover all four controls and conversion/metric headroom')
    return minimum


def derive_sender_registration(template, arm):
    require(template['status'] == 'frozen_before_execution' and template['arms'] == [],
            'new non-runnable sender template required')
    require(arm['frames'] in (2, 12, 48, 96) and arm['source_window_index'] == 0
            and arm['appearance_input_policy'] == 'reference_cutout', 'source000 static appearance arm required')
    return dict(template, arms=[dict(arm)])


def score_control(source, delivered, out, ff, timeout):
    """Same pinned RGB->YUV/VMAF/Y-MSE/SSIM definitions as score_receiver."""
    from src.components.codec.frames import rgb_to_luma
    from src.components.metrics.ssim import SsimMetric
    from src.components.metrics.vmaf import _write_y4m_clip
    out.mkdir()
    for name, clip in (('reference', source), ('delivered', delivered)):
        _write_y4m_clip(out / (name + '.y4m'), clip, ff.path)
    log = out / 'vmaf.json'
    filt = f'[1:v]format=yuv420p[dist];[0:v]format=yuv420p[ref];[dist][ref]libvmaf=model=version=vmaf_v0.6.1:log_path={log}:log_fmt=json:n_threads=8'
    cmd = [ff.path, '-hide_banner', '-loglevel', 'error', '-i', str(out/'reference.y4m'), '-i', str(out/'delivered.y4m'), '-filter_complex', filt, '-f', 'null', '-']
    with (out/'vmaf.log').open('wb') as stream:
        subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    frames = json.loads(log.read_text())['frames']
    require([f['frameNum'] for f in frames] == list(range(len(source))), 'all VMAF frames required')
    metrics = []; ssim = SsimMetric()
    for i, (ref, got) in enumerate(zip(source, delivered, strict=True)):
        v = float(frames[i]['metrics']['vmaf'])
        diff = rgb_to_luma(ref).astype(np.float64) - rgb_to_luma(got).astype(np.float64)
        mse = float(np.mean(diff**2)); s = float(ssim.score(ref[None], got[None]))
        require(all(math.isfinite(x) for x in (v, mse, s)), 'finite all-frame metrics required')
        metrics.append({'window_frame_index':i, 'source_frame_index':38+i, 'vmaf':v, 'mse_y':mse, 'psnr_y':psnr(mse), 'ssim':s})
    pooled = float(np.mean([x['mse_y'] for x in metrics]))
    result = {'per_frame':metrics, 'joined':{'vmaf':float(np.mean([x['vmaf'] for x in metrics])),
        'mse_y':pooled, 'psnr_y':psnr(pooled), 'ssim':float(np.mean([x['ssim'] for x in metrics]))},
        'vmaf_log_sha256':digest(log), 'vmaf_command':cmd}
    (out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', required=True); p.add_argument('--registration-sha256', required=True)
    p.add_argument('--out', required=True); p.add_argument('--frames', type=int, choices=(2,12,48,96), required=True)
    p.add_argument('--rung', choices=('C2','C3'), required=True); a = p.parse_args()
    began = time.monotonic(); regpath = Path(a.registration).resolve(); out = Path(a.out).resolve()
    require(sys.platform == 'linux' and 0 < len(os.sched_getaffinity(0)) <= 8, 'Linux CPU8 required')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PS_CODEC_THREADS','SSIM_THREADS'): os.environ[k]='8'
    require(digest(regpath) == a.registration_sha256, 'external frozen registration SHA required')
    reg = json.loads(regpath.read_text())
    require(reg['schema']=='pointstream.fresh_combined_workflow.v1' and reg['status']=='frozen_before_execution'
            and reg['worker_sha256']==digest(__file__), 'frozen combined worker required')
    validate_budget(reg)
    require(reg['controls']==list(smoke.CONTROLS) and reg['held_policy']=='preserve_recorded', 'exact controls/held policy required')
    root=Path(reg['legacy_root']).resolve(); data=Path(reg['data_root_canonical']).resolve()
    require(Path(reg['data_root_logical']).resolve()==data and not out.exists() and out.is_relative_to(data/'audits'), 'fresh canonical audit output required')
    require(subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==REVISION
            and not subprocess.check_output(['git','-C',str(root),'status','--porcelain']), 'clean274 required')
    templatepath=Path(reg['sender_template']).resolve()
    require(digest(templatepath)==reg['sender_template_sha256'], 'sender template pin')
    template=json.loads(templatepath.read_text())
    arms=[arm for arm in reg['arms'] if arm['frames']==a.frames and arm['rung']==a.rung]
    require(len(arms)==1, 'unique exact combined arm required'); arm=arms[0]
    derived=derive_sender_registration(template, arm)
    require(template['worker_sha256']==digest(sender_worker.__file__) and template['receiver_sha256']==digest(reg['base_receiver']), 'sender/receiver pins')
    require(template['legacy_commit']==REVISION and template['source96_rgb_sha256']==SOURCE_RGB
            and template['data_root_canonical']==str(data) and template['data_root_logical']==reg['data_root_logical'], 'source/root/revision joins')
    required={str(Path(__file__).resolve()),str(templatepath),str(Path(sys.executable).resolve())}
    required |= {str(Path(__file__).with_name(name).resolve()) for name in
                 ('fresh_window_sender.py','fresh_temporal_smoke.py','fresh_trajectory_packet.py','score_receiver.py')}
    required |= {str(Path(reg[k]).resolve()) for k in ('base_receiver','mask_receiver')}
    required |= set(template['source_files']) | set(template['tools_and_libraries'])
    required |= {str(root/rel) for rel in template['legacy_files']}
    guide=Path(template['guide_root']).resolve()
    required |= {str(guide/name) for name in ('registration.json','receipt.json','frames.json')}
    required |= {str(guide/name) for name in template['guide_artifact_sha256']}
    require(required <= set(reg['file_sha256']), 'all source/guide/code/native/template pins required')
    for path, sha in reg['file_sha256'].items(): require(digest(path)==sha, 'registered input changed '+path)
    joins=qualification_joins(template,reg,arm)
    if a.frames>2:
        qspec=reg['temporal_qualification']; qp=Path(qspec['path']).resolve()
        require(reg['file_sha256'].get(str(qp))==qspec['sha256'] and digest(qp)==qspec['sha256'], 'separate root qualification SHA pin')
        q=json.loads(qp.read_text()); pp=Path(q['proof_path']).resolve(); sp=Path(q['registration_path']).resolve()
        require(reg['file_sha256'].get(str(pp))==q['proof_sha256'] and reg['file_sha256'].get(str(sp))==q['registration_sha256'], 'qualified proof/registration pins')
        proof=json.loads(pp.read_text()); smoke_reg=json.loads(sp.read_text())
        validate_temporal_gate(a.frames,q,proof,smoke_reg,joins,digest(pp),digest(sp))
        prior_sender=Path(smoke_reg['sender_dir']).resolve(); prior_report_path=prior_sender/'report.json'
        require(reg['file_sha256'].get(str(prior_report_path))==smoke_reg['file_sha256'].get(str(prior_report_path))
                and digest(prior_report_path)==reg['file_sha256'][str(prior_report_path)], 'qualified smoke sender report pin')
        prior=json.loads(prior_report_path.read_text())
        validate_smoke_sender(prior,arm,template,proof)
    resource.setrlimit(resource.RLIMIT_AS,(arm['memory_gib']*1024**3,arm['memory_gib']*1024**3))
    out.mkdir(parents=True,exist_ok=False)
    sender_reg=out/'derived-sender-registration.json'; sender_reg.write_text(json.dumps(derived,indent=2)+'\n')
    sender=out/'sender-diagnostic'
    cmd=[sys.executable,str(Path(sender_worker.__file__).resolve()),'--registration',str(sender_reg),'--registration-sha256',digest(sender_reg),
         '--legacy-root',str(root),'--data-root',reg['data_root_logical'],'--receiver-script',reg['base_receiver'],
         '--out',str(sender),'--frames',str(a.frames),'--rung',a.rung,'--memory-gib',str(arm['memory_gib'])]
    subprocess.run(cmd,check=True,timeout=reg['sender_timeout_seconds'])
    report=json.loads((sender/'report.json').read_text()); state=json.loads((sender/'execution-status.json').read_text())
    require(state['status']=='complete' and state['report_sha256']==digest(sender/'report.json') and report['arm']==arm, 'fresh exact sender report join')
    require(report['guide_receipt_sha256']==template['guide_receipt_sha256'] and report['guide_registration_sha256']==template['guide_registration_sha256'], 'sender guide join')
    gr=json.loads((guide/'receipt.json').read_text()); records=json.loads((guide/'frames.json').read_text())[:a.frames]
    with np.load(sender/'scene.npz',allow_pickle=False) as packet: arrays={k:packet[k] for k in packet.files}
    meta=json.loads(arrays['metadata'].tobytes()); inputs={'unadapted':(arrays,meta)}; adaptation={}
    for control in ('null','opaque','alpha'):
        aa,mm,coverage=adapt_packet(arrays,meta,records,gr,policy=control,held_policy=reg['held_policy']); inputs[control]=(aa,mm); adaptation[control]=coverage
    smoke.coverage_checks(inputs['opaque'][1],records); smoke.coverage_checks(inputs['alpha'][1],records)
    sourcepath=Path(next(iter(template['source_files']))); source96=np.load(sourcepath,mmap_mode='r',allow_pickle=False)
    require(source96.shape==(96,2160,3840,3) and source96.dtype==np.uint8 and rgb_digest(source96)==SOURCE_RGB, 'original4K source000 identity')
    sys.path.insert(0,str(root)); from src.components.codec.tools import resolve_ffmpeg
    ff=resolve_ffmpeg(); require(template['tools_and_libraries'].get(ff.path)==digest(ff.path), 'resolved scorer FFmpeg pin')
    results={}
    for control in smoke.CONTROLS:
        require(time.monotonic()-began + reg['receiver_timeout_seconds'] + reg['score_timeout_seconds'] < reg['total_budget_seconds'], 'insufficient remaining registered budget; preserve outputs')
        dest=out/control; dest.mkdir(); aa,mm=inputs[control]
        payload=(sender/'scene.npz').read_bytes() if control=='unadapted' else encode_packet(aa)
        (dest/'scene.npz').write_bytes(payload); (dest/'manifest.json').write_text('{"schema":1,"fps":24,"package":"scene.npz"}\n')
        command=smoke.receiver_command(control,reg,package=dest/'scene.npz',output=dest/'receiver.npy',root=root,data=data)
        subprocess.run(command,check=True,timeout=reg['receiver_timeout_seconds'])
        rr=json.loads((dest/'receiver.receipt.json').read_text()); rgb=np.load(dest/'receiver.npy',mmap_mode='r',allow_pickle=False)
        require(rr['package_sha256']==digest(dest/'scene.npz') and rr['package_bytes']==len(payload)
                and rr['frames_shape']==[a.frames,2160,3840,3] and rgb.shape==(a.frames,2160,3840,3)
                and rgb.dtype==np.uint8 and rr['frames_rgb_sha256']==rgb_digest(rgb)
                and rr['violations']==[] and rr['denied_roots']==[str(data)], 'protected receiver payload/pixels/guard joins')
        if control=='unadapted': smoke.check_unadapted_identity(rr,digest(dest/'receiver.npy'),report)
        results[control]={'package_bytes':len(payload),'manifest_bytes':(dest/'manifest.json').stat().st_size,
            'physical_payload_bytes':len(payload)+(dest/'manifest.json').stat().st_size,'receiver':rr,
            'receiver_file_sha256':digest(dest/'receiver.npy'),'receiver_command':command,'adaptation':adaptation.get(control),
            'score':score_control(source96[:a.frames],rgb,out/(control+'-score'),ff,reg['score_timeout_seconds'])}
    null=np.load(out/'null/receiver.npy',mmap_mode='r',allow_pickle=False)
    for control in ('opaque','alpha'):
        results[control]['visible_pixels_by_frame']=smoke.visible_checks(np.load(out/control/'receiver.npy',mmap_mode='r',allow_pickle=False),null,inputs[control][1])
    for path,sha in reg['file_sha256'].items(): require(digest(path)==sha, 'upstream changed during workflow')
    require(digest(regpath)==a.registration_sha256, 'combined registration changed')
    result={'schema':'pointstream.fresh_combined_workflow.result.v1','complete':True,'paper_evidence':False,
        'arm':arm,'registration_sha256':a.registration_sha256,'worker_sha256':digest(__file__),
        'sender_command':cmd,'derived_sender_registration_sha256':digest(sender_reg),'sender_report_sha256':digest(sender/'report.json'),
        'qualification_joins':joins,'legacy_revision':REVISION,'source_file_sha256':digest(sourcepath),
        'source96_rgb_sha256':SOURCE_RGB,'source_selected_rgb_sha256':rgb_digest(source96[:a.frames]),
        'tools_and_libraries':template['tools_and_libraries'],'command':sys.argv,'controls':results,'resources':{'seconds':time.monotonic()-began,
        'affinity':sorted(os.sched_getaffinity(0)),'nice':os.getpriority(os.PRIO_PROCESS,0),
        'self_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'child_peak_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss},
        'scope':'Original sender receiver diagnostic; adapted protected receiver scored against original4K allframes. Initial static appearance and held boxes are not pose/task truth. Full96 guides condition shorter mechanics controls; no OS sandbox or winner claim.'}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()

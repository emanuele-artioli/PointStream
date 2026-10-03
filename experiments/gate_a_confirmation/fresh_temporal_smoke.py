"""Prospective protected receiver smoke; execution requires a frozen registration.

Consumes an already independently verified fresh_window_sender output; never
runs sender/models or reads original RGB/cache. All four controls retained.
Mechanics checks alone never promote longer sender arms: visual review required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from experiments.gate_a_confirmation.fresh_trajectory_packet import adapt_packet, encode_packet
from experiments.gate_a_confirmation.fresh_window_sender import digest, rgb_digest, require

CONTROLS = ('null','unadapted','opaque','alpha')


def receiver_command(control, reg, *, package, output, root, data):
    require(control in CONTROLS, 'registered receiver control required')
    # The mask wrapper supports only opaque/alpha. Null has no placements and
    # uses the exact base decoder, just as unadapted does.
    command = [sys.executable, reg['base_receiver']] if control in ('null','unadapted') else [sys.executable,reg['mask_receiver'],'--base-receiver',reg['base_receiver']]
    return command + ['--legacy-root',str(root),'--package',str(package),
                      '--output',str(output),'--deny-root',str(data)]


def check_unadapted_identity(receipt, file_sha256, sender_report):
    expected = sender_report['receiver']
    require(receipt['frames_rgb_sha256'] == expected['frames_rgb_sha256'] and
            receipt['frames_shape'] == expected['frames_shape'] and
            receipt['package_sha256'] == expected['package_sha256'] and
            receipt['package_bytes'] == expected['package_bytes'] and
            file_sha256 == sender_report['receiver_file_sha256'],
            'unadapted decode must equal independently qualified sender receiver')


def coverage_checks(metadata, records):
    """Strict temporal smoke: each initialized role present on every frame.

    Missing guide roles are valid recorded outcomes but hold this smoke rather
    than silently lowering its denominator. Held frames remain explicitly stale.
    """
    n = metadata['frame_count']; rows = metadata['placements']
    require(n >= 2, 'temporal smoke requires at least two frames')
    roles = {p['object_id'] for p in rows}
    require(roles and all({p['object_id'] for p in rows if p['frame_index']==f}==roles
                         for f in range(n)), 'all-role coverage required on every smoke frame')
    motion = {role: len({tuple(p['bbox']) for p in rows if p['object_id']==role}) > 1 for role in roles}
    require(any(motion.values()), 'demonstrable recorded bbox motion required')
    require(len(records)==n, 'all registered frame records required')
    return {'roles':sorted(roles),'moving_roles':sorted(r for r in roles if motion[r]),
            'placement_count_by_frame':[sum(p['frame_index']==f for p in rows) for f in range(n)]}


def visible_checks(foreground, null, metadata):
    require(foreground.shape==null.shape and foreground.dtype==null.dtype==np.uint8,
            'matching receiver uint8 shape required')
    measurements=[]
    for f in range(len(null)):
        changed=np.any(foreground[f]!=null[f],axis=2)
        union=np.zeros(changed.shape,dtype=bool);roles=[]
        for p in metadata['placements']:
            if p['frame_index']!=f:continue
            x1,y1,x2,y2=p['bbox'];union[y1:y2,x1:x2]=True
            count=int(changed[y1:y2,x1:x2].sum())
            require(count>0,'foreground must visibly alter pixels in each placed role bbox')
            roles.append({'object_id':p['object_id'],'bbox':p['bbox'],'changed_pixels_in_bbox':count})
        require(roles and not changed[~union].any(),'visible effect must stay inside declared placement union')
        measurements.append({'frame_index':f,'changed_pixels':int(changed.sum()),'roles':roles})
    return measurements


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',required=True);p.add_argument('--registration-sha256',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();regpath=Path(a.registration).resolve();out=Path(a.out).resolve()
    require(sys.platform=='linux' and 0<len(os.sched_getaffinity(0))<=8,'Linux CPU8 affinity required')
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)));resource.setrlimit(resource.RLIMIT_AS,(48*1024**3,48*1024**3))
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PS_CODEC_THREADS'):os.environ[k]='8'
    require(digest(regpath)==a.registration_sha256,'frozen external registration SHA required')
    reg=json.loads(regpath.read_text());require(reg['status']=='frozen_before_execution' and reg['schema']=='pointstream.fresh_temporal_smoke.v1','explicit smoke registration required')
    require(reg['controls']==list(CONTROLS) and reg['frames']==2 and reg['held_policy']=='preserve_recorded','exact four-control2frame instrument contract')
    require(reg['worker_sha256']==digest(__file__),'smoke worker pin')
    root=Path(reg['legacy_root']).resolve();data=Path(reg['data_root_canonical']).resolve();sender=Path(reg['sender_dir']).resolve();guide=Path(reg['guide_root']).resolve()
    require(Path(reg['data_root_logical']).resolve()==data and sender.is_relative_to(data/'audits') and guide.is_relative_to(data/'audits'),'canonical source000 audit roots required')
    require(not out.exists() and out.is_relative_to(data/'audits') and not out.is_relative_to(sender) and not out.is_relative_to(guide),'fresh separate smoke output required')
    require(subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()=='274638bdae7f5bd63c4f834a24804c0e14ed8d83' and not subprocess.check_output(['git','-C',str(root),'status','--porcelain']),'clean274 required')
    require(reg['file_sha256'] and any('libvmaf' in k for k in reg['file_sha256']),'code/native/input file pins required')
    required=[str(Path(__file__).resolve()),str(Path(__file__).with_name('fresh_trajectory_packet.py').resolve()),str(Path(__file__).with_name('fresh_window_sender.py').resolve()),str(Path(reg['base_receiver']).resolve()),str(Path(reg['mask_receiver']).resolve()),str(Path(sys.executable).resolve())]
    required += [str(sender/name) for name in ('scene.npz','manifest.json','report.json','receiver.receipt.json','receiver.npy','execution-status.json')]
    required += [str(guide/name) for name in ('registration.json','receipt.json','frames.json')]
    require(set(required)<=set(reg['file_sha256']),'all required code/tool/upstream pins required')
    for path,sha in reg['file_sha256'].items():require(digest(path)==sha,'registered input changed '+path)
    for name in ('sender_qualification','guide_qualification'):
        reviewed=reg[name];proof_path=Path(reviewed['path']).resolve()
        require(reg['file_sha256'].get(str(proof_path))==reviewed['sha256'] and digest(proof_path)==reviewed['sha256'],'reviewed independent qualification file pin')
    qualification=reg['sender_qualification'];require(qualification['passed'] is True and qualification['frames']==2 and qualification['rung']==reg['rung'],'independently qualified source/background/packing sender smoke required')
    gp=json.loads((guide/'registration.json').read_text());gr=json.loads((guide/'receipt.json').read_text())
    require(gp['frames']==gr['frame_count']==96 and gr['status']=='complete' and gp['source_rgb_sha256']=='09fbaee1f343b6046718853ef82b8b257c3bb8a9bbff8c52c23c368aeedce573','complete fresh source00096 guides required')
    require(reg['guide_qualification']['passed'] is True and reg['guide_qualification']['frames']==96,'reviewed full96 fresh guide qualification required')
    require(gr['registration_sha256']==digest(guide/'registration.json'),'guide registration receipt join')
    for name,sha in gr['artifact_sha256'].items():
        path=(guide/name).resolve();require(path.parent==guide and reg['file_sha256'].get(str(path))==sha and digest(path)==sha,'all fresh guide artifacts pinned')
    report=json.loads((sender/'report.json').read_text())
    require(report['arm']['frames']==2 and report['arm']['rung']==reg['rung'] and report['arm']['appearance_input_policy']=='reference_cutout','fresh sender arm/pixel policy required')
    state=json.loads((sender/'execution-status.json').read_text())
    require(state['status']=='complete' and state['report_sha256']==digest(sender/'report.json'),'sender completed report join')
    require(report['receiver']==json.loads((sender/'receiver.receipt.json').read_text()) and report['receiver_file_sha256']==digest(sender/'receiver.npy'),'sender report qualified receiver file/receipt join')
    require(report['data_root_canonical']==str(data) and Path(report['data_root_logical']).resolve()==data,'sender canonical/logical alias report join')
    require(report['input_context_contract']['source028_access'] is False and report['input_context_contract']['canonical_preparation_source_count']==1 and report['input_context_contract']['encoder_source_windows']==['scene_000'],'one source000 encoder context required')
    require(report['guide_receipt_sha256']==digest(guide/'receipt.json') and report['guide_registration_sha256']==digest(guide/'registration.json') and report['package_sha256']==digest(sender/'scene.npz'),'sender guide and packet joins')
    for path,sha in report['tools_and_libraries'].items():require(reg['file_sha256'].get(path)==sha and digest(path)==sha,'sender current native/tool identity')
    for rel,sha in report['legacy_files'].items():require(reg['file_sha256'].get(str(root/rel))==sha and digest(root/rel)==sha,'pinned legacy source identity')
    with np.load(sender/'scene.npz',allow_pickle=False) as packet:arrays={k:packet[k] for k in packet.files};meta=json.loads(arrays['metadata'].tobytes())
    require(meta['frame_count']==2 and (meta['height'],meta['width'])==(2160,3840),'registered original4K2frame packet required')
    records=json.loads((guide/'frames.json').read_text())[:2]
    control_inputs={'unadapted':(arrays,meta)}
    for policy in ('null','opaque','alpha'):
        aa,mm,_=adapt_packet(arrays,meta,records,gr,policy=policy,held_policy=reg['held_policy']);control_inputs[policy]=(aa,mm)
    scheduling=coverage_checks(control_inputs['opaque'][1],records)
    coverage_checks(control_inputs['alpha'][1],records)
    out.mkdir(parents=True,exist_ok=False);began=time.monotonic();results={}
    for control in CONTROLS:
        dest=out/control;dest.mkdir();aa,mm=control_inputs[control]
        payload=(sender/'scene.npz').read_bytes() if control=='unadapted' else encode_packet(aa)
        (dest/'scene.npz').write_bytes(payload)
        manifest={'schema':1,'fps':24,'package':'scene.npz'};(dest/'manifest.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')
        cmd=receiver_command(control,reg,package=dest/'scene.npz',output=dest/'receiver.npy',root=root,data=data)
        started=time.monotonic();subprocess.run(cmd,check=True,timeout=reg['receiver_timeout_seconds'])
        rr=json.loads((dest/'receiver.receipt.json').read_text());rgb=np.load(dest/'receiver.npy',mmap_mode='r',allow_pickle=False)
        require(rr['package_sha256']==digest(dest/'scene.npz') and rr['package_bytes']==len(payload) and rr['frames_shape']==[2,2160,3840,3] and rgb.shape==(2,2160,3840,3) and rgb_digest(rgb)==rr['frames_rgb_sha256'] and rr['violations']==[] and rr['denied_roots']==[str(data)],'fresh protected receiver receipt/shape/pixel joins')
        if control=='unadapted':check_unadapted_identity(rr,digest(dest/'receiver.npy'),report)
        results[control]={'physical_bytes':len(payload)+(dest/'manifest.json').stat().st_size,'package_bytes':len(payload),'manifest_bytes':(dest/'manifest.json').stat().st_size,'receiver_file_sha256':digest(dest/'receiver.npy'),'receiver_receipt':rr,'command':cmd,'seconds':time.monotonic()-started}
    null=np.load(out/'null/receiver.npy',mmap_mode='r',allow_pickle=False)
    for control in ('opaque','alpha'):
        results[control]['visible_pixels_by_frame']=visible_checks(np.load(out/control/'receiver.npy',mmap_mode='r',allow_pickle=False),null,control_inputs[control][1])
    for path,sha in reg['file_sha256'].items():require(digest(path)==sha,'upstream identity changed during smoke')
    evidence={'schema':'pointstream.fresh_temporal_smoke.proof.v1','mechanics_passed':True,'longer_sender_arms_held':True,'visual_review':'pending','registration_sha256':a.registration_sha256,'controls':results,'scheduling':scheduling,'resource_observation':{'seconds':time.monotonic()-began,'affinity':sorted(os.sched_getaffinity(0)),'nice':os.getpriority(os.PRIO_PROCESS,0),'address_space_gib':48,'self_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'child_peak_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss},'scope':'Static-reference placement only; foreground pixel changes and guide boxes are not true pose/task truth. Open audit guard is not OS sandbox. No automatic promotion, codec or quality claim.'}
    (out/'proof.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()

"""New physical-envelope legacy Gate A run; never historical-byte replay.

Invoke as a file, with a clean pinned legacy tree and the original data root.
Smoke and full share the exact loader, encoder, serialization and child receiver.
No baseline encoding is performed. Outputs must be a new directory.
"""
import argparse, dataclasses, hashlib, json, os, pathlib, resource, shutil, subprocess, sys

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def slice_smoke(clip,count=2):
    if not dataclasses.is_dataclass(clip):raise TypeError('legacy clip must be dataclass')
    if clip.frames.shape[0]<count or clip.masks.shape[0]!=clip.frames.shape[0]:raise ValueError('unaligned source/mask count')
    objects=[]
    for item in clip.objects:
        if item.mask is not None and item.mask.shape[0]!=clip.frames.shape[0]:raise ValueError('unaligned object mask count')
        if item.frame_index>=count:continue
        objects.append(dataclasses.replace(item,mask=item.mask[:count] if item.mask is not None else None))
    return dataclasses.replace(clip,n_frames=count,frames=clip.frames[:count],masks=clip.masks[:count],objects=tuple(objects))

def deployment_manifest(captured,shapes):
    return {'schema':1,'fps':24,'scenes':[{'package':x['package'],'shape':shape} for x,shape in zip(captured,shapes,strict=True)]}

def limit_cpu():
    os.nice(19)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
    resource.setrlimit(resource.RLIMIT_AS,(48*1024**3,48*1024**3))

def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);p.add_argument('--receiver-script',required=True);p.add_argument('--mode',choices=['smoke','pilot','full'],required=True);p.add_argument('--frames',type=int,choices=[2,12,48,96],required=True);p.add_argument('--rung',choices=['C2','C3'],required=True);p.add_argument('--expected-commit',default='274638bdae7f5bd63c4f834a24804c0e14ed8d83');a=p.parse_args()
    root=pathlib.Path(a.legacy_root).resolve();data=pathlib.Path(a.data_root).resolve();out=pathlib.Path(a.output).resolve();receiver=pathlib.Path(a.receiver_script).resolve()
    if out.exists():raise SystemExit('fresh output directory required')
    if (a.mode=='smoke' and a.frames!=2) or (a.mode=='full' and a.frames!=96) or (a.mode=='pilot' and a.frames not in (12,48)):
        raise SystemExit('mode/frame policy mismatch')
    os.environ['PS_CODEC_TIMEOUT_SECONDS']='600'
    os.environ['PS_CODEC_MAX_ATTEMPTS']='1'
    limit_cpu()
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if commit!=a.expected_commit:raise SystemExit('legacy revision mismatch '+commit)
    dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True)
    if dirty.strip():raise SystemExit('legacy checkout must be clean')
    # Contracts paths must resolve only the explicitly assigned original root.
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PS_DATA_ROOT']=str(data)
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VMAF_THREADS','PS_CODEC_THREADS']:os.environ[key]='8'
    sys.path.insert(0,str(root));os.chdir(root)
    import numpy as np
    from experiments.tier.gate_a_identity import verify_source_clips, verify_manifest, MEASUREMENT_FILES
    from experiments.tier.gate_a_long_context import RUNGS, configure_rung
    from experiments.tier.low_rate_sweep import pointstream_e1
    from src.runner.config_io import load_tier
    from src.contracts import paths
    import src.runner.client as client
    # Refuse silently using the legacy repository's old assets/outputs aliases.
    resolved_outputs=paths.outputs().resolve()
    if not(resolved_outputs==data or data in resolved_outputs.parents):raise SystemExit('outputs not under selected data root: '+str(resolved_outputs))
    clips=verify_source_clips(n_frames=96);manifest=verify_manifest()
    full_hashes=[hashlib.sha256(np.ascontiguousarray(c.frames).data).hexdigest() for c in clips]
    if a.frames!=96:
        clips=[slice_smoke(c,a.frames) for c in clips]
    for c in clips:
        if not c.objects:raise RuntimeError('prefix has no foreground objects; cannot qualify foreground path')
    out.mkdir(parents=True);captured=[];original=client.serialize_client_request
    def serialize(**kwargs):
        payload=original(**kwargs);index=len(captured);package=out/f'scene-{index:02d}.npz';package.write_bytes(payload)
        decoded=out/f'scene-{index:02d}.receiver.npy'
        command=[sys.executable,str(receiver),'--legacy-root',str(root),'--package',str(package),'--output',str(decoded),'--deny-root',str(data),'--deny-root',str(out/'encoder-checkpoints')]
        subprocess.run(command,check=True,env=dict(os.environ),timeout=600)
        frames=np.load(decoded,allow_pickle=False,mmap_mode='r')
        captured.append({'package':package.name,'bytes':len(payload),'sha256':sha(package),'output':decoded.name,'output_sha256':sha(decoded),'frames':frames})
        return payload
    client.serialize_client_request=serialize
    rung=next(x for x in RUNGS if x.name==a.rung);config=configure_rung(load_tier('balanced'),rung)
    # Original timing receiver still executes; scientific DAG remains original.
    result=pointstream_e1(clips,config,checkpoint_dir=None)
    if len(captured)!=len(clips):raise RuntimeError('not every scene emitted a serialized request')
    from experiments.tier.low_rate_measure import score_headlines
    source=np.concatenate([c.frames for c in clips]); delivered=np.concatenate([x['frames'] for x in captured])
    if source.shape!=delivered.shape:raise RuntimeError('receiver shape mismatch')
    from src.components.metrics.vmaf import _write_y4m_clip, _read_vmaf_mean
    from experiments.tier.low_rate_measure import y_psnr
    from src.components.metrics.ssim import SsimMetric
    ffmpeg=os.environ.get('FFMPEG_BIN') or shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('native ffmpeg required')
    ref_y4m=out/'reference.y4m';pred_y4m=out/'receiver.y4m';log=out/'receiver-vmaf.json'
    _write_y4m_clip(ref_y4m,source,ffmpeg);_write_y4m_clip(pred_y4m,delivered,ffmpeg)
    metric_filter=f'[1:v]format=yuv420p[dist];[0:v]format=yuv420p[ref];[dist][ref]libvmaf=model=version=vmaf_v0.6.1:log_path={log}:log_fmt=json:n_threads=8'
    metric_command=[ffmpeg,'-hide_banner','-loglevel','error','-i',str(ref_y4m),'-i',str(pred_y4m),'-filter_complex',metric_filter,'-f','null','-']
    subprocess.run(metric_command,check=True,timeout=600,capture_output=True)
    metric_log=json.loads(log.read_text())
    metric_frames=metric_log.get('frames',[])
    if len(metric_frames)!=len(source) or any(not np.isfinite(x['metrics']['vmaf']) for x in metric_frames):
        raise RuntimeError('VMAF must score every registered frame finitely')
    scores={'psnr_y':y_psnr(source,delivered),'ssim':float(SsimMetric().score(source,delivered)),'vmaf':_read_vmaf_mean(log)}
    metric_receipt={'command':metric_command,'filter':metric_filter,'model':'vmaf_v0.6.1','pool':'mean','log_sha256':sha(log),'ffmpeg_path':ffmpeg,'ffmpeg_sha256':sha(pathlib.Path(ffmpeg).resolve())}
    (out/'metric-identity.json').write_text(json.dumps(metric_receipt,indent=2)+'\n')
    envelope_manifest={'schema':1,'label':'new legacy-derived physical-envelope run; not exact historical byte replay','mode':a.mode,'rung':a.rung,'legacy_commit':commit,'legacy_measurement_sha256':{f:sha(root/f) for f in MEASUREMENT_FILES},'manifest_identity':manifest,'source96_rgb_sha256':full_hashes,'selected_shapes':[list(c.frames.shape) for c in clips],'config':dataclasses.asdict(config),'receiver_script_sha256':sha(receiver),'packages':[{k:v for k,v in x.items() if k!='frames'} for x in captured],'model_policy':'explicit built-in vmaf_v0.6.1, binary receipt; linked library identity in startup preflight','isolation':'attended only; Python open-deny hook does not certify native OS isolation'}
    (out/'provenance.json').write_text(json.dumps(envelope_manifest,sort_keys=True,indent=2,default=str)+'\n')
    wire_manifest=deployment_manifest(captured,[list(c.frames.shape) for c in clips])
    encoded=(json.dumps(wire_manifest,sort_keys=True,separators=(',',':'))+'\n').encode();(out/'manifest.json').write_bytes(encoded)
    report={'physical_payload_bytes':sum(x['bytes'] for x in captured)+len(encoded),'physical_package_bytes':sum(x['bytes'] for x in captured),'physical_manifest_bytes':len(encoded),'receiver_scores':scores,'legacy_encoder_diagnostic':result,'scored_boundary':'actual saved fresh-child receiver output','complete':True,'paper_evidence':False,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'metric_frames':len(metric_frames),'frames_per_scene':a.frames,'memory_limit_gib':48,'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None}
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
if __name__=='__main__':main()

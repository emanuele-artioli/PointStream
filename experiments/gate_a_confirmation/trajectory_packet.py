"""Bounded foreground-conditioning repair probe; no codec-win claim.

Reuse a preserved full96-scene background/crop packet, encode model-guide-derived
per-frame boxes and one alpha template per actor. Static reference appearance is
not dynamic pose reconstruction. Prefixes are mechanics controls, not RD points.
"""
import argparse,hashlib,io,json,os,resource,subprocess,sys
from pathlib import Path
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def bounds(mask):
    ys=np.flatnonzero(mask.any(axis=1));xs=np.flatnonzero(mask.any(axis=0))
    return None if not len(xs) or not len(ys) else (int(xs[0]),int(ys[0]),int(xs[-1])+1,int(ys[-1])+1)

def adapted_box(base,first,current):
    sx=(current[2]-current[0])/(first[2]-first[0]);sy=(current[3]-current[1])/(first[3]-first[1])
    x=round(current[0]+(base[0]-first[0])*sx);y=round(current[1]+(base[1]-first[1])*sy)
    return [x,y,x+max(1,round((base[2]-base[0])*sx)),y+max(1,round((base[3]-base[1])*sy))]

def make_packet(raw,objects,count,policy,foreground=True):
    with np.load(io.BytesIO(raw),allow_pickle=False) as packet:
        meta=json.loads(packet['metadata'].tobytes());arrays={k:packet[k] for k in packet.files if k!='metadata' and not k.startswith('mask_') and (foreground or not k.startswith(('crop_','encoded_crop_')))}
    assert meta['frame_count']==96 and len(meta['background']['homographies'])==96
    meta['frame_count']=count;meta['background']['homographies']=meta['background']['homographies'][:count];meta['mask_policy']=policy
    by_id={o.object_id:o for o in objects};placements=[];coverage=[0]*count
    for i,old in enumerate(meta['placements'] if foreground else []):
        obj=by_id[old['object_id']];mask=np.asarray(obj.mask,dtype=bool)
        if mask.shape!=(96,meta['height'],meta['width']):raise ValueError('registered all-frame source guide required')
        first=bounds(mask[old['frame_index']])
        if first is None:raise ValueError('missing initial guide')
        x1,y1,x2,y2=old['bbox'];key=f'actor_alpha_{i}'
        if policy=='alpha':arrays[key]=mask[old['frame_index'],y1:y2,x1:x2].astype(np.uint8)
        for f in range(count):
            current=bounds(mask[f])
            if current is None:continue
            item=dict(old);item.update(frame_index=f,bbox=adapted_box(old['bbox'],first,current),mask_key=key if policy=='alpha' else None)
            placements.append(item);coverage[f]+=1
    meta['placements']=placements;arrays['metadata']=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8)
    out=io.BytesIO();np.savez_compressed(out,**arrays)
    return out.getvalue(),coverage,meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--original-packet',required=True);p.add_argument('--receiver-script',required=True);p.add_argument('--base-receiver',required=True);p.add_argument('--registration',required=True);p.add_argument('--out',required=True);p.add_argument('--frames',type=int,choices=[2,12,48,96],required=True);p.add_argument('--mask-policy',choices=['opaque','alpha'],required=True);p.add_argument('--null-foreground',action='store_true');a=p.parse_args()
    reg=json.loads(Path(a.registration).read_text());arm={'frames':a.frames,'mask_policy':a.mask_policy,'foreground':not a.null_foreground}
    if reg.get('status')!='frozen_before_execution' or arm not in reg['arms'] or digest(__file__)!=reg['worker_sha256'] or digest(a.receiver_script)!=reg['receiver_sha256'] or digest(a.base_receiver)!=reg['base_receiver_sha256'] or digest(a.original_packet)!=reg['original_packet_sha256']:raise ValueError('frozen identities/arm required')
    for path,sha in reg['tools_and_libraries'].items():
        if digest(path)!=sha:raise ValueError('native dependency changed')
    if len(os.sched_getaffinity(0))>8:raise ValueError('isolated CPU allowance required')
    os.nice(19);resource.setrlimit(resource.RLIMIT_AS,(48*1024**3,48*1024**3));os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PS_DATA_ROOT']=a.data_root
    for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PS_CODEC_THREADS']:os.environ[k]='8'
    root=Path(a.legacy_root);revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if revision!='274638bdae7f5bd63c4f834a24804c0e14ed8d83' or subprocess.check_output(['git','-C',str(root),'status','--porcelain']):raise ValueError('clean pinned legacy required')
    sys.path.insert(0,str(root));from experiments.tier.gate_a_identity import verify_source_clips
    clips=verify_source_clips(n_frames=96);clip=clips[0]
    encoded,coverage,meta=make_packet(Path(a.original_packet).read_bytes(),clip.objects,a.frames,a.mask_policy,not a.null_foreground)
    if not a.null_foreground and any(c==0 for c in coverage):raise ValueError('guide-positive coverage missing registered frame')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);packet=out/'scene.npz';packet.write_bytes(encoded);decoded=out/'receiver.npy'
    subprocess.run([sys.executable,a.receiver_script,'--base-receiver',a.base_receiver,'--legacy-root',str(root),'--package',str(packet),'--output',str(decoded),'--deny-root',a.data_root],check=True,timeout=600)
    rr=json.loads(decoded.with_suffix('.receipt.json').read_text());assert rr['frames_shape']==[a.frames,2160,3840,3] and not rr['violations']
    manifest={'schema':1,'fps':24,'package':'scene.npz'};(out/'manifest.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')
    (out/'report.json').write_text(json.dumps({'arm':arm,'complete':True,'paper_evidence':False,'physical_bytes':len(encoded)+(out/'manifest.json').stat().st_size,'guide_placement_count_by_frame':coverage,'source96_rgb_sha256':hashlib.sha256(clip.frames.data).hexdigest(),'source_guide_masks':[{"object_id":o.object_id,"shape":list(o.mask.shape),"sha256":hashlib.sha256(np.ascontiguousarray(o.mask).data).hexdigest(),"first_bbox":list(o.bbox),"first_frame":o.frame_index} for o in clip.objects],'registration_sha256':digest(a.registration),'original_packet_sha256':digest(a.original_packet),'receiver':rr,'scope':'Posthoc conditioning adaptation; first source/window only, full96 offline background reused even in prefix controls; model guides are not independent task truth; static appearance/silhouette is not true pose reconstruction. No baseline/quality/generalization win.'},indent=2)+'\n')
if __name__=='__main__':main()

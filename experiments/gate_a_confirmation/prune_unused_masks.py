"""Prospective pixel-parity control for the pinned legacy rectangular receiver.

Drops only mask arrays referenced by placements and clears those references.
This is unused-field pruning, NOT a bitwise or semantic lossless packet format.
Original packets remain unchanged. No quality/compression win is inferred here.
"""
import argparse, hashlib, io, json, os, subprocess
from pathlib import Path
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()

def prune(payload):
    with np.load(io.BytesIO(payload),allow_pickle=False) as data:
        metadata=json.loads(data['metadata'].tobytes())
        mask_keys={p['mask_key'] for p in metadata['placements'] if p['mask_key'] is not None}
        for key in mask_keys:
            if not key.startswith('mask_') or key not in data.files: raise ValueError('invalid placement mask reference')
        arrays={key:data[key] for key in data.files if key not in mask_keys and key!='metadata'}
        for p in metadata['placements']: p['mask_key']=None
        arrays['metadata']=np.frombuffer(json.dumps(metadata).encode('utf-8'),dtype=np.uint8)
    out=io.BytesIO();np.savez_compressed(out,**arrays)
    return out.getvalue(), sorted(mask_keys)

def main():
    p=argparse.ArgumentParser();p.add_argument('--frames',type=int,choices=[2,12,48,96],required=True);p.add_argument('--original',required=True);p.add_argument('--out',required=True);p.add_argument('--legacy-root',required=True);p.add_argument('--receiver-script',required=True);p.add_argument('--python',required=True);p.add_argument('--registration',required=True);p.add_argument('--deny-root',required=True);a=p.parse_args()
    original=Path(a.original).resolve();out=Path(a.out).resolve();reg=json.loads(Path(a.registration).read_text())
    if reg.get('status')!='frozen_before_execution' or reg.get('worker_sha256')!=digest(__file__) or reg.get('receiver_sha256')!=digest(a.receiver_script): raise ValueError('frozen worker/receiver registration required')
    if reg.get('frames_per_scene')!=a.frames or reg.get('original')!=str(original):raise ValueError('unregistered original')
    if os.environ.get('CUDA_VISIBLE_DEVICES','')!='' or len(os.sched_getaffinity(0))>8:raise ValueError('CPU-only isolated cores required')
    if reg.get('original_report_sha256')!=digest(original/'report.json'):raise ValueError('original score report changed')
    revision=subprocess.check_output(['git','-C',a.legacy_root,'rev-parse','HEAD'],text=True).strip()
    if revision!='274638bdae7f5bd63c4f834a24804c0e14ed8d83' or subprocess.check_output(['git','-C',a.legacy_root,'status','--porcelain']):raise ValueError('clean pinned legacy source required')
    for path,expected in reg.get('tools_and_libraries',{}).items():
        if digest(path)!=expected:raise ValueError('native dependency changed')
    if not reg.get('tools_and_libraries'):raise ValueError('native identities required')
    report=json.loads((original/'report.json').read_text())
    if not report['complete'] or report['metric_frames']!=2*a.frames:raise ValueError('complete registered original receiver gate required')
    os.nice(19)
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PS_CODEC_THREADS']:os.environ[key]='8'
    out.mkdir(parents=True,exist_ok=False);rows=[]
    for source in sorted(original.glob('scene-*.npz')):
        expected=reg['packages'].get(source.name)
        if expected!=digest(source):raise ValueError('original packet identity differs')
        encoded,keys=prune(source.read_bytes());packet=out/source.name;packet.write_bytes(encoded)
        decoded=out/(source.stem+'.receiver.npy')
        subprocess.run([a.python,a.receiver_script,'--legacy-root',a.legacy_root,'--package',str(packet),'--output',str(decoded),'--deny-root',a.deny_root],check=True,timeout=600)
        old=json.loads((original/(source.stem+'.receiver.receipt.json')).read_text());new=json.loads((out/(source.stem+'.receiver.receipt.json')).read_text())
        if old['frames_rgb_sha256']!=new['frames_rgb_sha256'] or old['frames_shape']!=new['frames_shape'] or new['frames_shape']!=[a.frames,2160,3840,3] or new['violations']:raise ValueError('fresh receiver pixel-parity failure')
        rows.append({'package':packet.name,'original_sha256':expected,'new_sha256':digest(packet),'original_bytes':source.stat().st_size,'new_bytes':len(encoded),'removed_keys':keys,'decoded_rgb_sha256':new['frames_rgb_sha256'],'frames_shape':new['frames_shape']})
    if len(rows)!=2:raise ValueError('two original scene packets required')
    (out/'manifest.json').write_bytes((original/'manifest.json').read_bytes())
    result={'complete':True,'pixel_parity':True,'semantic_packet_lossless':False,'physical_payload_bytes':sum(r['new_bytes'] for r in rows)+(out/'manifest.json').stat().st_size,'original_report_sha256':digest(original/'report.json'),'registration_sha256':digest(a.registration),'packages':rows,'quality_policy':'Exact RGB parity permits reuse of pinned original fresh-receiver scores; no new native scores claimed','original_receiver_scores':report['receiver_scores'],'scope':'Fixed historical rectangular receiver only; no motion/task/generalization or matched-anchor win inferred','paper_evidence':False}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()

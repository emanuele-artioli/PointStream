"""Size-only search for strong 640p ball controls; all results require fleet validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import zlib
from fractions import Fraction

import cv2
import numpy as np

from experiments.tennis import ball_confirmation as prior, ball_confirmation_score as scorer, w1_roi
from experiments.tennis.ball_starved_target269 import copy_member, score_frame
from src.codecs import quality, svtav1
from src.pointstream import tracked_patch

CONFIG = {
    'match209_000': (64, 25, [54.61875, 60.6]),
    'match269_000': (64, 25, [53.10625, 53.906, 60.6]),
    'match235_000': (154, 60, [38.52155844, 44.428, 60.6]),
}
MANIFEST = 'df0dabf48b61ed3a096a9a6b4e54d823ca3b279c86d6a09e69ef0cd9bf2ceeb1'
BINARIES = {'ffmpeg': '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2', 'SvtAv1EncApp': '4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4', 'dav1d': 'e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6'}


def run(args):
    import torch
    started = time.monotonic()
    full, fps, targets = CONFIG[args.source]
    n = args.frames
    if n != (16 if os.environ['PS_STAGE'] == 'smoke' else full) or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('Wrong stage or CPU allocation')
    work = Path(os.environ['PS_SCRATCH_DIR']) / 'controls640'
    work.mkdir()
    for name, target in [('publish/source.yuv','source.yuv'), ('publish/source.rgb','source.rgb'), ('publish/bases/sender.json','sender.json'), ('publish/receipt.json','receipt.json')]:
        copy_member(args.bases, name, work / target)
    receipt = json.loads((work / 'receipt.json').read_text())
    if receipt['source_id'] != args.source or receipt['frames'] != full or prior.sha(work/'source.yuv') != receipt['result']['source_yuv_sha256'] or prior.sha(work/'source.rgb') != receipt['source']['rgb_sha256']:
        raise ValueError('source identity')
    sender = json.loads((work/'sender.json').read_text())
    prior.validate_sender(sender, Fraction(fps), full)
    track = prior.checked_track(next(x for x in sender['points'] if x['codec']=='svt' and x['point']==62)['centres'][:n], n)
    manifest, raw = scorer.authenticated_manifest(args.sources)
    if hashlib.sha256(raw).hexdigest() != MANIFEST:
        raise ValueError('manifest')
    labels = scorer.labels_for_source(manifest,args.source)[:n]
    source = np.memmap(work/'source.rgb',np.uint8,'r',shape=(full,720,1280,3))[:n]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    net = quality.load_dists(args.weights,'cpu')
    tools = {k:svtav1.tool(k) for k in BINARIES}
    if any(tools[k]['sha256'] != v for k,v in BINARIES.items()):
        raise ValueError('native binary identity')
    masks = np.zeros((n,360,640),bool)
    for i,p in enumerate(track):
        if p is not None:
            x,y=p[0]//2,p[1]//2
            masks[i,max(0,y-16):min(360,y+16),max(0,x-16):min(640,x+16)] = True
    rows=w1_roi.roi_text(masks,640,360,-40,outside=True).splitlines()
    for i,p in enumerate(track):
        if p is None:
            rows[i]=str(i)+' '+' '.join(['0']*60)
    roi=work/'roi.txt'
    roi.write_text('\n'.join(rows)+'\n')
    small=work/'small.yuv'
    subprocess.run([tools['ffmpeg']['path'],'-v','error','-f','rawvideo','-pixel_format','yuv420p','-video_size','1280x720','-framerate',str(fps),'-i',str(work/'source.yuv'),'-vf','scale=640:360:flags=area,format=yuv420p','-frames:v',str(n),'-f','rawvideo',str(small)],check=True,timeout=300)
    if small.stat().st_size != n*640*360*3//2:
        raise ValueError('scaled frame count')
    publish=Path(os.environ['PS_SCRATCH_DIR'])/'publish'
    publish.mkdir()
    trials=[]
    cached={}
    def encode(mode,q):
        key=(mode,round(q*4)/4)
        if key in cached:
            return cached[key]
        folder=work/f'{mode}-{key[1]}'
        folder.mkdir()
        r=svtav1.code(small,folder,width=640,height=360,fps=Fraction(fps),frames=n,crf=key[1],preset=4,threads=1,full_range=False,encoder=tools['SvtAv1EncApp']['path'],decoder=tools['dav1d']['path'],timeout=600,extra=['--roi-map-file',str(roi)] if mode=='roi' else None)
        data=Path(r['stream']).read_bytes()
        packed=zlib.compress(data,6)
        wire=(b'\x01'+packed) if len(packed)<len(data) else (b'\x00'+data)
        recovered=zlib.decompress(wire[1:]) if wire[0] else wire[1:]
        if recovered != data or len(svtav1.ivf_frames(data)) != n or r['decoded_frames'] != n:
            raise ValueError('transport or native count')
        row={'mode':mode,'crf':key[1],'kbps':len(wire)*8*fps/n/1000,'bytes':len(wire),'raw_sha256':hashlib.sha256(data).hexdigest(),'wire_sha256':hashlib.sha256(wire).hexdigest(),'decoded_sha256':prior.sha(Path(r['decoded'])),'decoded':r['decoded'],'stream':r['stream']}
        cached[key]=row
        trials.append(row)
        return row
    for mode in ['plain','roi']:
        if n==16:
            encode(mode,54)
            continue
        for target in targets:
            lo,hi=30.0,76.0
            for q in [lo,hi]:
                encode(mode,q)
            for _ in range(8):
                q=round((lo+hi)*2)/4
                if q in (lo,hi):
                    break
                r=encode(mode,q)
                if r['kbps'] > target:
                    lo=q
                else:
                    hi=q
    selected=[]
    for mode in ['plain','roi']:
        points=[r for r in trials if r['mode']==mode]
        for target in (targets if n!=16 else [points[0]['kbps']]):
            below=[r for r in points if r['kbps']<=target]
            above=[r for r in points if r['kbps']>=target]
            for row in ([max(below,key=lambda x:x['kbps'])] if below else [])+([min(above,key=lambda x:x['kbps'])] if above else []):
                if row not in selected:
                    selected.append(row)
    for i,row in enumerate(selected):
        native=np.memmap(row['decoded'],np.uint8,'r',shape=(n,540,640))
        row['native_frame_sha256']=[hashlib.sha256(f.tobytes()).hexdigest() for f in native]
        rgb=np.stack([cv2.resize(tracked_patch.yuv_to_rgb(f,640,360,False),(1280,720),interpolation=cv2.INTER_CUBIC) for f in native])
        row['crop']=score_frame(net,source,rgb,labels)
        row['mean_dists']=float(np.mean([x['dists'] for x in row['crop']])) if row['crop'] else None
        dest=publish/f'candidate-{i}.ivf'
        shutil.copyfile(row['stream'],dest)
        row['file']=dest.name
    for row in trials:
        del row['decoded'],row['stream']
    report={'source':args.source,'frames':n,'fps':fps,'targets':targets,'source_sha256':receipt['source']['rgb_sha256'],'manifest_sha256':MANIFEST,'visible_labels':sum(x[0]==1 for x in labels),'tools':tools,'trials':trials,'selected':selected,'wall_seconds':time.monotonic()-started}
    data=json.dumps(report,indent=2)+'\n'
    (publish/'result.json').write_text(data)
    (Path(os.environ['PS_STAGE_DIR'])/'result.json').write_text(data)


def validate():
    stage=Path(os.environ['PS_STAGE_DIR'])
    report=json.loads((stage/'result.json').read_text())
    n=report['frames']
    checks={'labels':report['visible_labels']>0,'both_modes':{x['mode'] for x in report['selected']}=={'plain','roi'},'forecast':True}
    with tarfile.open(stage/'published.tar') as tar:
        for i,r in enumerate(report['selected']):
            data=tar.extractfile('publish/'+r['file']).read()
            checks[str(i)]=(hashlib.sha256(data).hexdigest()==r['raw_sha256'] and len(svtav1.ivf_frames(data))==n and len(r['native_frame_sha256'])==n and len(r['crop'])==report['visible_labels'])
    # Smoke extrapolation includes a conservative allowance for 16 encodes per mode.
    projection=report['wall_seconds']*(CONFIG[report['source']][0]/16)*16*1.5 if n==16 else None
    if n==16:
        checks['forecast']=projection<1400
    result={'passed':all(checks.values()),'checks':checks,'projected_full_seconds':projection}
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:
        raise SystemExit(1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bases',type=Path)
    p.add_argument('--sources',type=Path)
    p.add_argument('--weights',type=Path)
    p.add_argument('--source',choices=CONFIG)
    p.add_argument('--frames',type=int)
    p.add_argument('--validate',action='store_true')
    a=p.parse_args()
    if a.validate:
        validate()
    else:
        run(a)


if __name__=='__main__':
    main()

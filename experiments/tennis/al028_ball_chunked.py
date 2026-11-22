"""Track a full-context Al028 source in independently checkpointed even chunks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

import cv2
import numpy as np
import torch

from experiments.tennis import al028_ball_sender as sender
from experiments.tennis import fresh_ball_sender, k5b_trackers


def prepare_context(frames):
    n=len(frames)
    pick=np.linspace(0,n-1,min(n,k5b_trackers.MEDIAN_FRAMES)).astype(int)
    median=np.median(np.asarray(frames)[pick],axis=0).astype(np.uint8)
    small=np.stack([cv2.resize(np.ascontiguousarray(f[:,:,::-1]),(512,288)) for f in frames]).astype(np.float32)/255.0
    med=np.moveaxis(cv2.resize(np.ascontiguousarray(median[:,:,::-1]),(512,288)).astype(np.float32)/255.0,-1,0)
    return med,small,pick


def predict(tracker,context,start,end):
    med,small,_=context
    if start<0 or end>len(small) or start>=end or start%2:
        raise ValueError('invalid global chunk bounds')
    out=[]
    with torch.no_grad():
        for begin in range(start,end,2):
            batch=[]
            for tick in range(begin,min(end,begin+2)):
                seq=[np.moveaxis(small[max(0,j)],-1,0) for j in range(tick-3,tick+1)]
                batch.append(np.concatenate([med,*seq],0))
            hm,_,_=tracker.net(frames=torch.from_numpy(np.stack(batch)).to(tracker.device))
            out.extend(tracker.blobs(hm.float().cpu().numpy()[:,0],1280/512,720/288))
    return out


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run(a):
    start=time.monotonic()
    if len(os.sched_getaffinity(0))!=1 or a.frames != (24 if os.environ['PS_STAGE']=='smoke' else 120):
        raise ValueError('wrong stage/allocation')
    if sender.prior.sha(a.archive)!=fresh_ball_sender.MODEL_CODE or sender.prior.sha(a.weights)!=fresh_ball_sender.MODEL_WEIGHT:
        raise ValueError('model identity')
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    root=Path(os.environ['PS_SCRATCH_DIR'])/'chunked-ball'
    root.mkdir()
    with tarfile.open(a.archive) as tar:
        tar.extractall(root/'model',filter='data')
    before=time.monotonic()
    tracker=k5b_trackers.BallTrack(root/'model',a.weights,device='cpu')
    if tracker.net.training or next(tracker.net.parameters()).device.type!='cpu':
        raise ValueError('wrong model device')
    def finite(_module, _inputs, output):
        if not torch.isfinite(output[0]).all():
            raise ValueError('non-finite model output')
    tracker.net.register_forward_hook(finite)
    frames=sender.source_frames(a.source,240,a.scene)
    source_hash=sha(frames)
    full=prepare_context(frames)
    setup=time.monotonic()-before
    if os.environ['PS_STAGE']=='smoke':
        short=frames[:24]
        original=tracker.run(short)
        local=prepare_context(short)
        combined=predict(tracker,local,0,12)+predict(tracker,local,12,24)
        if original!=combined:
            raise ValueError('24-frame prediction or split differs from original')
        sample_start=time.monotonic()
        continuous=predict(tracker,full,116,124)
        sample=time.monotonic()-sample_start
        boundary=predict(tracker,full,116,120)+predict(tracker,full,120,124)
        if continuous!=boundary:
            raise ValueError('240-frame global-context boundary differs')
        predictions=combined
        forecast=1.5*(setup+sample*15)+60
    else:
        predictions=predict(tracker,full,a.start,a.start+120)
        forecast=None
    result={'source_sha256':sender.SOURCE_PROFILES[a.scene][0],'source_rgb_sha256':source_hash,'source_frames':240,'fps':25,
            'mode':os.environ['PS_STAGE'],'start':a.start,'frames':a.frames,'predictions':predictions,
            'context_median_sha256':sha(full[0]),'context_small_sha256':sha(full[1]),
            'context_picks':full[2].tolist(),'forecast_seconds':forecast,'seconds':time.monotonic()-start,
            'raw_prediction_coordinates':'1280x720 historical convention; convert x*960/1280, y*544/720',
            'model_code_sha256':fresh_ball_sender.MODEL_CODE,'model_weight_sha256':fresh_ball_sender.MODEL_WEIGHT}
    (Path(os.environ['PS_STAGE_DIR'])/'chunk.json').write_text(json.dumps(result,indent=2)+'\n')


def validate(a):
    stage=Path(os.environ['PS_STAGE_DIR'])
    r=json.loads((stage/'chunk.json').read_text())
    n=24 if os.environ['PS_STAGE']=='smoke' else 120
    checks={'source':r['source_sha256']==sender.SOURCE_PROFILES[a.scene][0],'count':len(r['predictions'])==n==r['frames'],
            'median':len(r['context_picks'])==100,'model':r['model_weight_sha256']==fresh_ball_sender.MODEL_WEIGHT,
            'forecast':n!=24 or r['forecast_seconds']<1300}
    value={'passed':all(checks.values()),'checks':checks,'forecast_seconds':r['forecast_seconds']}
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(value,indent=2)+'\n')
    if not value['passed']:
        raise SystemExit(1)


def main():
    p=argparse.ArgumentParser()
    for k in ('source','archive','weights'):
        p.add_argument('--'+k,type=Path)
    p.add_argument('--start',type=int,choices=(0,120),default=0)
    p.add_argument('--frames',type=int,choices=(24,120))
    p.add_argument('--scene',choices=sender.SOURCE_PROFILES,default='al028')
    p.add_argument('--validate',action='store_true')
    a=p.parse_args()
    validate(a) if a.validate else run(a)


if __name__=='__main__':
    main()

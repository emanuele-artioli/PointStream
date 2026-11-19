"""Bounded CPU screen of native-size reviewed subject regions using the project DISTS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import cv2
import numpy as np
import torch
from src.codecs import quality

EXPECTED = '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'
DEFAULT_WEIGHTS = Path('/local/users/emanuele/pointstream/cache')/EXPECTED/'vgg16-397923af.pth'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def image(z, name, grey=False):
    buf = np.frombuffer(z.read(name+'.png'),np.uint8)
    pix = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE if grey else cv2.IMREAD_COLOR)
    return pix if grey else pix[:,:,::-1].copy()


def score_tick(net, z, t, arms):
    rows=[]
    for r in ('p1','p2','ball'):
        prefix=f'{t:03d}/{r}/'
        ref=image(z,prefix+'source')
        y=torch.from_numpy(ref).permute(2,0,1).unsqueeze(0).float()/255.
        mask=torch.from_numpy(image(z,prefix+'mask',True)>0)[None,None]
        assert mask.any() and (r!='ball' or mask.all())
        for arm in arms:
            x=torch.from_numpy(image(z,prefix+arm)).permute(2,0,1).unsqueeze(0).float()/255.
            v=float(quality.masked_dists(net,x,y,mask).item())
            if not np.isfinite(v) or (arm=='source' and abs(v)>1e-5):
                raise ValueError('metric failed')
            rows.append(dict(t=t,region=r,arm=arm,dists=v))
    return rows


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    assert sha(a.weights)==EXPECTED and sha(a.zip)==a.sha256
    net=quality.load_dists(a.weights,'cpu')
    with zipfile.ZipFile(a.zip) as z:
        arms=sorted({Path(x).stem for x in z.namelist()}-{'source','mask'})
        ticks=list(range(10,240,20))
        assert arms
        start=time.process_time()
        smoke=score_tick(net,z,10,['source',arms[0]])
        smoke_seconds=time.process_time()-start
        forecast=smoke_seconds*len(ticks)*len(arms)*0.5*1.8
        if forecast > a.cap_seconds:
            raise RuntimeError(f'smoke forecasts {forecast:.1f} CPU s, cap {a.cap_seconds}')
        rows=[]
        for tick in ticks:
            rows+=score_tick(net,z,tick,arms)
            if time.process_time()-start > a.cap_seconds:
                raise RuntimeError('CPU budget consumed')
    means={arm:{region:float(np.mean([v['dists'] for v in rows if v['arm']==arm and v['region']==region])) for region in ('p1','p2','ball')} for arm in arms}
    receipt=dict(host=os.uname().nodename,device='cpu',python=sys.executable,torch=torch.__version__,
                 opencv=cv2.__version__,scope='matched human body masks and centered 64px ball crop; 12 frames, one rally',
                 metric=quality.dists_record(a.weights),quality_file_sha256=sha(Path(quality.__file__)),
                 scorer_sha256=sha(Path(__file__)),crops_sha256=a.sha256,smoke=smoke,
                 forecast_cpu_seconds=forecast,smoke_cpu_seconds=smoke_seconds,inference_cpu_seconds=time.process_time()-start,means=means,rows=rows)
    with a.output.open('x') as f:
        json.dump(receipt,f,indent=2)
        f.write('\n')
    print(json.dumps(dict(arms=arms,rows=len(rows),means=means,forecast=forecast,seconds=receipt['inference_cpu_seconds'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('zip',type=Path)
    p.add_argument('sha256')
    p.add_argument('output',type=Path)
    p.add_argument('--cap-seconds',type=int,required=True)
    p.add_argument('--weights',type=Path,default=DEFAULT_WEIGHTS)
    run(p.parse_args())

"""Frozen WASB and transmitted coordinates on preselected Adelaide packets."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
import time
import zlib

import cv2
import numpy as np
import torch

from experiments.tennis import adelaide_ball_sender as sender, k5b_trackers as trackers
from experiments.tennis import k5d_source_oracle as oracle
from src.codecs import svtav1
from src.pointstream import ball, ball_stream, tracked_patch

WEIGHT='9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb'
CONFIG='6dbcad60e207bf684d0094f59aa922b107be1c1ce83998dc5b167ed59b0d1e0d'


def read_tar(path,name):
    with tarfile.open(path) as tar:
        file=tar.extractfile(name)
        if file is None:
            raise ValueError('missing archive member')
        return file.read()


def run(a):
    start=time.monotonic()
    if torch.cuda.device_count()!=1 or len(os.sched_getaffinity(0))!=1:
        raise ValueError('GPU/CPU isolation')
    if os.environ.get('PS_GPU_SHARING')=='shared':
        torch.cuda.set_per_process_memory_fraction(min(1.0,int(os.environ['PS_GPU_MEMORY_LIMIT_MIB'])*1024**2/torch.cuda.get_device_properties(0).total_memory))
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    oracle.bind_config(a.wasb_code,CONFIG)
    if oracle.sha(a.wasb_weights/'wasb_tennis_best.pth.tar')!=WEIGHT or oracle.sha(a.labels)!=sender.LABELS:
        raise ValueError('judge or labels identity')
    report=json.loads(read_tar(a.packets,'publish/result.json'))
    decision=json.loads(read_tar(a.packets,'publish/rate-decision.json'))
    if report['source_sha256']!=sender.SOURCE or report['frames']!=64:
        raise ValueError('source identity')
    if set(report['chosen'])!={'ordinary','metadata'}:
        raise ValueError('no feasible metadata and ordinary comparison')
    for mode,row in decision['chosen'].items():
        if any(row[k]!=report['chosen'][mode][k] for k in ('width','crf','bytes','packet_sha256')):
            raise ValueError('rate decisions changed after scoring')
    labels=[]
    with a.labels.open() as f:
        for row in csv.DictReader(f):
            tick,visibility,x,y=(int(row[k]) for k in ('Frame','Visibility','X','Y'))
            if tick<a.frames:
                labels.append({'tick':tick,'visibility':visibility,'x':x,'y':y})
    work=Path(os.environ['PS_SCRATCH_DIR'])/'adelaide-machine'
    work.mkdir()
    decoder=svtav1.tool('dav1d')
    if decoder['sha256']!='e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6':
        raise ValueError('decoder mismatch')
    model=trackers.Zoo(a.wasb_code,a.wasb_weights,names=('wasb-step1',))
    settings=model.settings['wasb-step1']
    if next(settings['net'].parameters()).device.type!='cuda':
        raise ValueError('wrong model device')
    source=sender.source_frames(a.source,64)
    sent=json.loads(a.sender.read_text())
    if hashlib.sha256(source).hexdigest()!=sent['rgb_sha256']:
        raise ValueError('source reconstruction mismatch')
    images={'source':source}
    positions=None
    pos_width=None
    for mode,row in report['chosen'].items():
        packet=read_tar(a.packets,'publish/'+row['file'])
        if hashlib.sha256(packet).hexdigest()!=row['packet_sha256']:
            raise ValueError('packet mismatch')
        payload=zlib.decompress(packet[1:]) if packet[0] else packet[1:]
        if mode=='metadata':
            stream=ball_stream.unpack(payload)
            native=stream.native
            positions,_=ball.decode_frames(stream.ball_payload,64,row['width'],row['height'])
            pos_width=row['width']
        else:
            native=payload
        if hashlib.sha256(native).hexdigest()!=row['native_sha256']:
            raise ValueError('native identity')
        out=work/(mode+'.yuv')
        decoded=tracked_patch.decode_native(native,out,{'width':row['width'],'height':row['height'],'fps':[25,1],'frames':64},decoder['path'])
        if decoded['decoded_yuv_sha256']!=row['decoded_sha256']:
            raise ValueError('decoded native identity')
        frames=np.memmap(out,np.uint8,'r',shape=(64,row['height']*3//2,row['width']))
        rgb=np.stack([cv2.resize(tracked_patch.yuv_to_rgb(f,row['width'],row['height'],False),(1280,720),interpolation=cv2.INTER_CUBIC) for f in frames])
        if hashlib.sha256(rgb).hexdigest()!=row['rgb_sha256']:
            raise ValueError('decoded RGB differs from rate study')
        images[mode]=rgb
    cache={}
    results=[]
    for mode,rgb in images.items():
        key=hashlib.sha256(rgb).hexdigest()
        if key not in cache:
            cache[key]=model.run('wasb-step1',rgb[:a.frames])
        pred=cache[key]
        records=[]
        for label in labels:
            present,x,y=pred[label['tick']]
            error=math.hypot(1.5*x-label['x'],1.5*y-label['y']) if present and label['visibility']==1 else None
            point=positions[label['tick']] if mode=='metadata' else None
            scale=1920/pos_width if mode=='metadata' else None
            direct=math.hypot(scale*point[0]-label['x'],scale*point[1]-label['y']) if point is not None and label['visibility']==1 else None
            records.append({'tick':label['tick'],'visibility':label['visibility'],'detected':bool(present),'native_error':error,'direct_present':point is not None,'direct_native_error':direct})
        results.append({'mode':mode,'rows':records,'rgb_sha256':key,'wire_kbps':report['chosen'][mode]['kbps'] if mode!='source' else None})
    result={'frames':a.frames,'labels':labels,'results':results,'gpu':torch.cuda.get_device_name(0),
            'checkpoint_sha256':WEIGHT,'settings':{'frames_in':settings['cfg']['model']['frames_in'],'frames_out':settings['cfg']['model']['frames_out'],'step':settings['cfg']['detector']['step'],'threshold':settings['cfg']['detector']['postprocessor']['score_threshold']},'seconds':time.monotonic()-start}
    publish=Path(os.environ['PS_SCRATCH_DIR'])/'publish'
    publish.mkdir()
    data=json.dumps(result,indent=2,allow_nan=False)+'\n'
    (publish/'result.json').write_text(data)
    (Path(os.environ['PS_STAGE_DIR'])/'result.json').write_text(data)


def validate():
    r=json.loads((Path(os.environ['PS_STAGE_DIR'])/'result.json').read_text())
    n=16 if os.environ['PS_STAGE']=='smoke' else 64
    checks={'frames':r['frames']==n,'arms':{x['mode'] for x in r['results']}=={'source','ordinary','metadata'},'settings':r['settings']=={'frames_in':3,'frames_out':3,'step':1,'threshold':0.5},'labels':len(r['labels'])>0 and all(len(x['rows'])==len(r['labels']) for x in r['results'])}
    forecast=1.5*r['seconds']*4+45 if n==16 else None
    checks['forecast']=forecast is None or forecast<650
    value={'passed':all(checks.values()),'checks':checks,'forecast_seconds':forecast}
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(value,indent=2)+'\n')
    if not value['passed']:
        raise SystemExit(1)


def main():
    p=argparse.ArgumentParser()
    for name in ('packets','source','sender','labels','wasb-code','wasb-weights'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--frames',type=int,choices=(16,64))
    p.add_argument('--validate',action='store_true')
    a=p.parse_args()
    validate() if a.validate else run(a)


if __name__=='__main__':
    main()

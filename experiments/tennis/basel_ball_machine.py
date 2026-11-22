"""Frozen WASB and paid tile centers on the outcome-unopened Basel clip."""
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

from experiments.tennis import basel_ball_sender as sender, k5b_trackers as trackers
from experiments.tennis import k5d_source_oracle as oracle
from experiments.tennis.ball_starved_scaled import render_scaled, unpack_scaled
from src.codecs import svtav1
from src.pointstream import tracked_patch

WEIGHT = '9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb'
CONFIG = '6dbcad60e207bf684d0094f59aa922b107be1c1ce83998dc5b167ed59b0d1e0d'
ARCHIVE = '5f41e38effb3506da7668049eaf84ed0db401baafe7607dfd7edeba50b023ba9'


def read_tar(path, name):
    with tarfile.open(path) as archive:
        member = archive.extractfile(name)
        if member is None:
            raise ValueError(f'missing {name}')
        return member.read()


def run(a):
    start = time.monotonic()
    if torch.cuda.device_count() != 1 or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('GPU/CPU isolation')
    if oracle.sha(a.packets) != ARCHIVE or oracle.sha(a.labels) != sender.LABELS:
        raise ValueError('input identity')
    torch.set_num_threads(1)
    if os.environ.get('PS_GPU_SHARING') == 'shared':
        limit = int(os.environ['PS_GPU_MEMORY_LIMIT_MIB']) * 1024 * 1024
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, limit / total))
    cv2.setNumThreads(1)
    oracle.bind_config(a.wasb_code, CONFIG)
    if oracle.sha(a.wasb_weights / 'wasb_tennis_best.pth.tar') != WEIGHT:
        raise ValueError('WASB checkpoint identity')
    report = json.loads(read_tar(a.packets, 'publish/result.json'))
    if report['source_sha256'] != sender.SOURCE or report['frames'] != 64 or report['cap'] != 60.6:
        raise ValueError('source or fixed budget mismatch')
    rows=[]
    for width, mode in [(320,'tile'), (1280,'plain'), (1280,'roi')]:
        choices = [r for r in report['rows'] if r['width']==width and r['mode']==mode and r['kbps'] <= 60.6]
        if len(choices) != 1:
            raise ValueError('missing or ambiguous preselected packet')
        row=choices[0]
        packet=read_tar(a.packets, 'publish/'+row['file'])
        if len(packet)!=row['bytes'] or hashlib.sha256(packet).hexdigest()!=row['sha256']:
            raise ValueError('packet identity')
        data=zlib.decompress(packet[1:]) if packet[0] == 1 else packet[1:]
        rows.append((row,data))
    rawlabels=[]
    with a.labels.open() as stream:
        for r in csv.DictReader(stream):
            tick, visibility, x, y = (int(r[k]) for k in ('Frame','Visibility','X','Y'))
            if tick < a.frames:
                rawlabels.append({'tick':tick,'visibility':visibility,'x':x,'y':y})
    if not rawlabels or any(x['visibility'] not in (0,1) for x in rawlabels):
        raise ValueError('unexpected labels')
    work=Path(os.environ['PS_SCRATCH_DIR'])/'basel-machine'
    work.mkdir()
    decoder=svtav1.tool('dav1d')
    if decoder['sha256'] != 'e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6':
        raise ValueError('decoder identity')
    zoo=trackers.Zoo(a.wasb_code,a.wasb_weights,names=('wasb-step1',))
    settings=zoo.settings['wasb-step1']
    if next(settings['net'].parameters()).device.type != 'cuda':
        raise ValueError('WASB device')
    source=sender.source_frames(a.source,64)
    sender_report=json.loads(a.sender.read_text())
    if sender_report['rgb_sha256'] != hashlib.sha256(source).hexdigest():
        raise ValueError('source decoded RGB identity')
    inputs=[('source',source,None,None)]
    tile_positions=None
    for row, data in rows:
        folder=work/(row['mode']+str(row['width']))
        folder.mkdir()
        if row['mode']=='tile':
            rgb=render_scaled(data,folder,decoder['path'])
            meta,_,_=unpack_scaled(data)
            tile_positions=meta['positions']
        else:
            output=folder/'decoded.yuv'
            tracked_patch.decode_native(data,output,{'width':1280,'height':720,'fps':[25,1],'frames':64},decoder['path'])
            raw=np.memmap(output,np.uint8,'r',shape=(64,1080,1280))
            rgb=np.stack([tracked_patch.yuv_to_rgb(frame,1280,720,False) for frame in raw])
        if hashlib.sha256(rgb).hexdigest()!=row['rgb_sha256']:
            raise ValueError('receiver RGB differs from the frozen quality result')
        inputs.append((row['mode'],rgb,row['kbps'],row['sha256']))
    outputs=[]
    for mode, images, rate, sha in inputs:
        began=time.monotonic()
        predictions=zoo.run('wasb-step1', images[:a.frames])
        measured=[]
        for label in rawlabels:
            tick=label['tick']
            present,x,y=predictions[tick]
            error=math.hypot(1.5*x-label['x'],1.5*y-label['y']) if present and label['visibility']==1 else None
            position=tile_positions[tick] if mode=='tile' else None
            direct=math.hypot(1.5*(position[0]+16)-label['x'],1.5*(position[1]+16)-label['y']) if position is not None and label['visibility']==1 else None
            measured.append({'tick':tick,'visibility':label['visibility'],'detected':bool(present),
                             'native_error':error,'direct_present':position is not None,'direct_native_error':direct})
        outputs.append({'mode':mode,'wire_kbps':rate,'packet_sha256':sha,'rows':measured,'seconds':time.monotonic()-began})
    result={'frames':a.frames,'labels':rawlabels,'results':outputs,'checkpoint_sha256':WEIGHT,
            'gpu':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
            'settings':{'frames_in':settings['cfg']['model']['frames_in'],'frames_out':settings['cfg']['model']['frames_out'],
                        'step':settings['cfg']['detector']['step'],'threshold':settings['cfg']['detector']['postprocessor']['score_threshold']},
            'seconds':time.monotonic()-start}
    publish=Path(os.environ['PS_SCRATCH_DIR'])/'publish'
    publish.mkdir()
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n'
    (publish/'result.json').write_text(encoded)
    (Path(os.environ['PS_STAGE_DIR'])/'result.json').write_text(encoded)


def validate():
    r=json.loads((Path(os.environ['PS_STAGE_DIR'])/'result.json').read_text())
    n=16 if os.environ['PS_STAGE']=='smoke' else 64
    expected={'frames_in':3,'frames_out':3,'step':1,'threshold':0.5}
    checks={'frames':r['frames']==n,'settings':r['settings']==expected,
            'arms':{x['mode'] for x in r['results']}=={'source','tile','plain','roi'},
            'labels':len(r['labels'])>0 and all(len(x['rows'])==len(r['labels']) for x in r['results'])}
    forecast=1.5*r['seconds']*4+45 if n==16 else None
    checks['forecast']=forecast is None or forecast<650
    payload={'passed':all(checks.values()),'checks':checks,'forecast_seconds':forecast}
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(payload,indent=2)+'\n')
    if not payload['passed']:
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

"""Prospective size-only native AV1 and point-metadata ladder for Adelaide."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zlib
from fractions import Fraction

import cv2
import numpy as np
import torch

from experiments.tennis import adelaide_ball_sender as sender
from experiments.tennis import ball_confirmation as prior, k5d_encoder
from experiments.tennis.ball_starved_scaled import score_frame
from src.codecs import quality, svtav1
from src.pointstream import ball, ball_stream, tracked_patch

CAP=60.6
WIDTHS=(1280,640,320)
QUALITIES=(40,50,60,65,70)


def wire(raw):
    compressed=zlib.compress(raw,6)
    return b'\x01'+compressed if len(compressed)<len(raw) else b'\x00'+raw


def unwrap(packet):
    if packet[0] not in (0,1):
        raise ValueError('invalid wire selector')
    return zlib.decompress(packet[1:]) if packet[0] else packet[1:]


def run(a):
    started=time.monotonic()
    n=a.frames
    if n != (16 if os.environ['PS_STAGE']=='smoke' else 64) or len(os.sched_getaffinity(0))!=1:
        raise ValueError('wrong stage or allocation')
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    source=sender.source_frames(a.source,n)
    receipt=json.loads(a.sender.read_text())
    if receipt['source_sha256']!=sender.SOURCE or receipt['frames']!=64 or (n==64 and hashlib.sha256(source).hexdigest()!=receipt['rgb_sha256']):
        raise ValueError('sender/source mismatch')
    track=prior.checked_track(receipt['positions'][:n],n)
    tools={k:svtav1.tool(k) for k in ('ffmpeg','SvtAv1EncApp','dav1d')}
    if any(tools[k]['sha256'] != receipt['tools'][k]['sha256'] for k in tools):
        raise ValueError('codec identity')
    root=Path(os.environ['PS_SCRATCH_DIR'])
    work=root/'adelaide-ladder'
    work.mkdir()
    publish=root/'publish'
    publish.mkdir()
    yuv=work/'source.yuv'
    k5d_encoder.convert_rgb(tools['ffmpeg']['path'],source,yuv,Fraction(25))
    trials=[]
    chosen={}
    expected_points={}
    for width in WIDTHS:
        height=width*9//16
        small=yuv
        if width!=1280:
            small=work/f'source-{width}.yuv'
            subprocess.run([tools['ffmpeg']['path'],'-v','error','-f','rawvideo','-pixel_format','yuv420p','-video_size','1280x720','-framerate','25','-i',str(yuv),'-vf',f'scale={width}:{height}:flags=area,format=yuv420p','-frames:v',str(n),'-f','rawvideo',str(small)],check=True,timeout=180)
        scale=1280//width
        scaled=[]
        for point in track:
            if point is None:
                scaled.append(None)
            else:
                hint=[min(width-1,max(0,round(point[0]/scale))),min(height-1,max(0,round(point[1]/scale)))]
                origin=tracked_patch.origin(hint,width,height)
                scaled.append((origin[0]+16,origin[1]+16))
        expected_points[width]=scaled
        placeholder=ball.Appearance((0,0,0),0,0,0)
        appearances=[placeholder if x is not None else None for x in scaled]
        payload=ball.encode_frames(scaled,appearances,width,height)
        for q in QUALITIES:
            folder=work/f'{width}-{q}'
            folder.mkdir()
            encoded=svtav1.code(small,folder,width=width,height=height,fps=Fraction(25),frames=n,crf=q,preset=4,threads=1,full_range=False,encoder=tools['SvtAv1EncApp']['path'],decoder=tools['dav1d']['path'],timeout=300)
            native=Path(encoded['stream']).read_bytes()
            stream=ball_stream.Stream(width,height,Fraction(25),n,'svt-av1-ivf','dav1d-1.5.4',native,payload)
            hybrid=wire(ball_stream.pack(stream))
            ordinary=wire(native)
            record={'width':width,'height':height,'crf':q,'ordinary_bytes':len(ordinary),'hybrid_bytes':len(hybrid),'native_sha256':hashlib.sha256(native).hexdigest(),'decoded_sha256':prior.sha(Path(encoded['decoded']))}
            trials.append(record)
            for mode,packet in [('ordinary',ordinary),('metadata',hybrid)]:
                if mode not in chosen and len(packet)*8*25/n/1000<=CAP:
                    chosen[mode]={'width':width,'height':height,'crf':q,'bytes':len(packet),'kbps':len(packet)*8*25/n/1000,'packet_sha256':hashlib.sha256(packet).hexdigest(),'native_sha256':record['native_sha256'],'decoded_sha256':record['decoded_sha256'],'file':mode+'.packet'}
                    (publish/(mode+'.packet')).write_bytes(packet)
        if len(chosen)==2:
            break
    decision={'source_sha256':sender.SOURCE,'frames':n,'fps':25,'cap':CAP,'ladder':{'widths':WIDTHS,'crfs':QUALITIES},'trials':trials,'chosen':chosen,'tracked_positions':sum(x is not None for x in track)}
    (publish/'rate-decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    # Only now are official labels and scores opened.
    if prior.sha(a.labels)!=sender.LABELS:
        raise ValueError('label identity')
    labels=[(-1,0,0)]*n
    with a.labels.open() as f:
        for row in csv.DictReader(f):
            tick,kind,x,y=(int(row[k]) for k in ('Frame','Visibility','X','Y'))
            if tick<n:
                labels[tick]=(kind,round(2*x/3),round(2*y/3))
    net=quality.load_dists(a.weights,'cpu')
    for mode,row in chosen.items():
        raw=unwrap((publish/row['file']).read_bytes())
        if mode=='metadata':
            stream=ball_stream.unpack(raw)
            native=stream.native
            points,appearances=ball.decode_frames(stream.ball_payload,n,row['width'],row['height'])
            if points!=expected_points[row['width']] or appearances!=[ball.Appearance((0,0,0),0,0,0) if x is not None else None for x in points]:
                raise ValueError('decoded metadata count')
            row['positions']=points
        else:
            native=raw
        if hashlib.sha256(native).hexdigest()!=row['native_sha256']:
            raise ValueError('native payload changed')
        output=work/(mode+'.yuv')
        dec=tracked_patch.decode_native(native,output,{'width':row['width'],'height':row['height'],'fps':[25,1],'frames':n},tools['dav1d']['path'])
        if dec['decoded_yuv_sha256']!=row['decoded_sha256']:
            raise ValueError('receiver differs')
        native_frames=np.memmap(output,np.uint8,'r',shape=(n,row['height']*3//2,row['width']))
        rgb=np.stack([cv2.resize(tracked_patch.yuv_to_rgb(frame,row['width'],row['height'],False),(1280,720),interpolation=cv2.INTER_CUBIC) for frame in native_frames])
        row['rgb_sha256']=hashlib.sha256(rgb).hexdigest()
        row['cropped']=score_frame(net,source,rgb,labels)
        row['dists']=float(np.mean([x['dists'] for x in row['cropped']])) if row['cropped'] else None
        row['whole_rgb_psnr']=k5d_encoder.pooled_psnr(source,rgb)
    result={**decision,'labels':labels,'seconds':time.monotonic()-started,'tools':tools,'metric':quality.dists_record(a.weights)}
    data=json.dumps(result,indent=2,allow_nan=False)+'\n'
    (publish/'result.json').write_text(data)
    (Path(os.environ['PS_STAGE_DIR'])/'result.json').write_text(data)


def validate():
    root=Path(os.environ['PS_STAGE_DIR'])
    r=json.loads((root/'result.json').read_text())
    n=16 if os.environ['PS_STAGE']=='smoke' else 64
    checks={'source':r['source_sha256']==sender.SOURCE,'frames':r['frames']==n,'trials':len(r['trials'])>=5,
            'chosen':all(x['kbps']<=CAP and x['bytes']>0 and (n==16 or x['cropped']) for x in r['chosen'].values()),
            'labels':n==16 or any(x[0]==1 for x in r['labels'])}
    forecast=1.5*r['seconds']*4+85 if n==16 else None
    checks['forecast']=forecast is None or forecast<1350
    result={'passed':all(checks.values()),'checks':checks,'forecast_seconds':forecast}
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:
        raise SystemExit(1)


def main():
    p=argparse.ArgumentParser()
    for k in ('source','sender','labels','weights'):
        p.add_argument('--'+k,type=Path)
    p.add_argument('--frames',type=int,choices=(16,64))
    p.add_argument('--validate',action='store_true')
    a=p.parse_args()
    validate() if a.validate else run(a)


if __name__=='__main__':
    main()

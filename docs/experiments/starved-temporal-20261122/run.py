"""Judge adjacent source and decoded frames with the project's backward DIS flow."""
import argparse
import base64
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np
from experiments.visor.h3 import dis_flow, warp
from experiments.visor.h3b import luma
from experiments.tennis import common_player_ball as ball
from src.pointstream.tracked_patch import yuv_to_rgb

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'starved-base-local-20261122'))
import crops as replay  # noqa: E402


def load(p):
    return json.loads(p.read_text())


def main(out):
    cv2.setNumThreads(1)
    start=time.process_time()
    receipt=load(HERE.parent/'starved-base-local-20261122/base-receipt.json')
    delivered=next(r for r in receipt['outputs'] if r['arm']=='clean' and r['crf']==60)
    alts=load(HERE.parent/'starved-stroke-20261122/result.json')['selected']
    decoded={}
    for mode,r in dict(**alts,pointstream=delivered).items():
        w,h=(240,136) if mode=='pointstream' else (480,272)
        path=Path(r['decoded'])
        assert replay.file_sha(path)==r['decoded_sha256']
        decoded[mode]=(np.memmap(path,np.uint8,'r',shape=(240,h*3//2,w)),w,h)
    source=replay.SOURCE/'ps-native-ctl-20261118-full/source.yuv'
    assert replay.file_sha(source)==receipt['inputs']['original']
    native=np.memmap(source,np.uint8,'r',shape=(240,816,960))
    d=Path(delivered['stream']).parent
    positions,data,_=ball._open((d/'players.pscp').read_bytes(),(d/'ball.sidecar').read_bytes(),receipt['tools'][2]['sha256'])
    tile=np.frombuffer(subprocess.check_output(['/opt/homebrew/bin/ffmpeg','-v','error','-c:v','libdav1d','-f','ivf','-i','pipe:0','-pix_fmt','yuv420p','-f','rawvideo','pipe:1'],input=data),np.uint8).reshape(240,96,64)
    finish=load(replay.STAGE/'finish/finish.json')
    assert load(replay.STAGE/'finish-full-audit.json')['all_frames_exact']
    seg=Path('/private/tmp/ps-starved-player-seg-20261122')
    assert replay.file_sha(seg/'frames.zip')=='4c891777b33b10ba92fd94fa1a59ba360aacf175e7bfd67c382206058d59550a'
    keys={(arm,t):k for k,(arm,t) in load(seg/'key.json').items()}
    assert replay.file_sha(replay.LABELS)=='9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a'
    labels={r['tick']:r for r in load(replay.LABELS)['frames']}
    rows=[]
    yy,xx=np.mgrid[:544,:960]
    with ExitStack() as stack:
        archived=stack.enter_context(zipfile.ZipFile(seg/'frames.zip'))
        gens=dict(replay.bind_generation(stack,replay.AUDIT/f'player{i}-new-full.tar',replay.STAGE/f'player{i}-full.json',finish) for i in (1,2))
        masks={}
        for u in replay.UNITS:
            p=replay.STAGE/'finish'/(u+'-masks.npz')
            assert replay.file_sha(p)==finish['mask_files'][u]
            z=stack.enter_context(zipfile.ZipFile(p))
            assert np.array_equal(replay.read_small(z,'t'),np.arange(240))
            masks[u]=stack.enter_context(replay.array_frames(z,'masks',(240,784,512),np.bool_))
        previous={}
        for t in range(240):
            tick_started=time.process_time()
            current={'source':yuv_to_rgb(native[t],960,544,False)}
            for arm,(video,w,h) in decoded.items():
                pixels=cv2.resize(yuv_to_rgb(video[t],w,h,False),(1920,1080),interpolation=cv2.INTER_CUBIC)
                if arm=='pointstream':
                    support=np.zeros((1080,1920),bool)
                    for u in replay.UNITS:
                        n,status=replay.overlay(pixels,support,next(gens[u][1]),gens[u][0][t],next(masks[u]))
                        assert n==finish['rows'][t]['projected_pixels'][u] and status==finish['rows'][t]['status'][u]
                    if positions[t] is not None:
                        x,y=positions[t];pixels[y:y+32,x:x+32]=yuv_to_rgb(tile[t],64,64,False)[16:48,16:48]
                if t in labels:
                    data=archived.read(keys[arm,t]+'.png')
                    assert hashlib.sha256(data).hexdigest()==keys[arm,t]
                    assert np.array_equal(pixels,cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)[:,:,::-1])
                current[arm]=cv2.resize(pixels,(960,544),interpolation=cv2.INTER_AREA)
            if t:
                flow=dis_flow(cv2.cvtColor(current['source'],cv2.COLOR_RGB2GRAY),cv2.cvtColor(previous['source'],cv2.COLOR_RGB2GRAY))
                valid=(xx+flow[:,:,0]>=0)&(yy+flow[:,:,1]>=0)&(xx+flow[:,:,0]<959)&(yy+flow[:,:,1]<543)
                supports={'whole':valid}
                if t in labels:
                    f=labels[t]
                    for i,p in enumerate(f['players']):
                        b=np.frombuffer(base64.b64decode(p['mask']['png_base64']),np.uint8)
                        m=cv2.resize(cv2.imdecode(b,0),(960,544),interpolation=cv2.INTER_NEAREST)>0
                        supports[f'p{i+1}']=valid&m
                    x,y=f['ball']['center_xy']
                    supports['ball']=valid&(abs(xx-x*.5)<16)&(abs(yy-y*544/1080)<32*544/1080)
                residual={k:luma(v)-luma(warp(previous[k],flow)) for k,v in current.items()}
                for arm,delta in residual.items():
                    for region,mask in supports.items():
                        error=float(np.mean(np.abs(delta[mask]-residual['source'][mask])))
                        energy=float(np.mean(np.abs(delta[mask])))
                        if arm=='source':assert error==0
                        rows.append(dict(tick=t,region=region,arm=arm,error=error,energy=energy))
                if t==1:
                    forecast=(time.process_time()-tick_started)*239*1.5+45
                    assert forecast<500,forecast
            previous=current
            assert time.process_time()-start<500
    means={}
    for arm in previous:
        means[arm]={}
        for region in ('whole','p1','p2','ball'):
            vals=[r for r in rows if r['arm']==arm and r['region']==region]
            assert len(vals)==(239 if region=='whole' else 12)
            means[arm][region]={k:float(np.mean([r[k] for r in vals])) for k in ('error','energy')}
    result=dict(means=means,rows=rows,smoke_projected_cpu_s=forecast,cpu_s=time.process_time()-start,
                source_sha256=replay.file_sha(source),opencv=cv2.__version__,matched_decoded_frames=36,script_sha256=replay.file_sha(Path(__file__)))
    with out.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(means)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('out',type=Path)
    main(cli.parse_args().out)

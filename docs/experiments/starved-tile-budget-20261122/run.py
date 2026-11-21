"""Produce cheaper paid sidecars, bracket their running rate, and crop decoded frames."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import zipfile

import cv2
import numpy as np
from experiments.tennis import common_player_ball as ball
from src.codecs import svtav1
from src.pointstream.tracked_patch import yuv_to_rgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'starved-base-local-20261122'))
from crops import LABELS, supports  # noqa: E402

SEG = Path('/private/tmp/ps-starved-player-seg-20261122')
STRONG = Path('/private/tmp/ps-starved-rally-tiles-strong-control-b-20261122')


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def cpu():
    return sum(resource.getrusage(k).ru_utime+resource.getrusage(k).ru_stime
               for k in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN))


def load(p):
    return json.loads(p.read_text())


def main(out):
    out.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    t0=cpu()
    previous=load(HERE.parent/'starved-base-local-20261122/base-receipt.json')
    row=next(r for r in previous['outputs'] if r['arm']=='clean' and r['crf']==60)
    path=Path(row['stream']).parent
    carrier=(path/'players.pscp').read_bytes()
    decoder=svtav1.tool('dav1d'); encoder=svtav1.tool('SvtAv1EncApp')
    assert decoder['sha256']==previous['tools'][2]['sha256'] and encoder['sha256']==previous['tools'][1]['sha256']
    positions,original,size=ball._open(carrier,(path/'ball.sidecar').read_bytes(),decoder['sha256'])
    assert size==(1920,1080) and sum(p is not None for p in positions)==195
    assert hashlib.sha256(original).hexdigest()=='cfa40919ce0bdb64b8b98ae5ed3b0425b19cd0c020ce9c0e7ff08dec3f8e95a8'
    (out/'original.ivf').write_bytes(original)
    raw=out/'tile.yuv'
    subprocess.run(svtav1.decode_command(decoder['path'],out/'original.ivf',raw,threads=1),check=True,capture_output=True,timeout=40)
    assert raw.stat().st_size==240*96*64
    original_raw_digest=sha(raw)
    assert original_raw_digest==load(Path('/private/tmp/ps-low-ball-stage-20261118/input.json'))['tile_yuv_sha256']
    all_new=[]
    def encode(src,mode,q,w,h,n,extra=None):
        dest=out/f'{mode}-{q:g}-{n}'
        dest.mkdir()
        r=svtav1.code(src,dest,width=w,height=h,fps=Fraction(50),frames=n,crf=q,preset=4,threads=2,
                      full_range=False,encoder=encoder['path'],decoder=decoder['path'],timeout=65,extra=extra)
        assert r['decoded_frames']==n and r['temporal_units']==n
        r.update(mode=mode,q=q,n=n,decoded_sha256=sha(Path(r['decoded'])))
        all_new.append(r)
        return r
    smoke=encode(raw,'tile',48,64,64,24)
    assert (smoke['encode_cpu_seconds']+smoke['decode_cpu_seconds'])*15+30 < 100
    rungs={}
    for q in [36,48,60]:
        r=encode(raw,'tile',q,64,64,240)
        if q==36:
            assert (r['encode_cpu_seconds']+r['decode_cpu_seconds'])*4.5+30 < 100
        dest=Path(r['stream']).parent
        bitstream=Path(r['stream']).read_bytes()
        side=ball.pack(carrier,positions,bitstream,decoder_sha256=decoder['sha256'])
        (dest/'ball.sidecar').write_bytes(side)
        assert ball._open(carrier,side,decoder['sha256'])==(positions,bitstream,size)
        total=len(side)+len(carrier)+(path/'receiver.manifest').stat().st_size
        r.update(cold=total/600,running=(total-row['reference_bytes'])/600,sidecar_bytes=len(side),
                 sidecar_sha256=sha(dest/'ball.sidecar'))
        rungs[f'jointQ{q}']=r
    chosen=min(rungs,key=lambda k:abs(rungs[k]['running']-50))
    target=round(rungs[chosen]['running']*600)
    controls=load(HERE.parent/'starved-rally-tiles-20261122/strong-result.json')
    assert sha(STRONG/'480.yuv')==controls['scaled_source_sha256'] and sha(STRONG/'roi.txt')==controls['roi_map_sha256']
    sizes=load(Path('/private/tmp/ps-starved-rally-tiles-20261122/sizes.json'))
    source240=path.parent/'original.yuv'
    assert sha(source240)==sizes['source_yuv_sha256']
    anchors={}
    for mode,width,height,src in [('plain',480,272,STRONG/'480.yuv'),('roi',480,272,STRONG/'480.yuv'),('plain240',240,136,source240)]:
        if mode=='plain240':
            known={r['crf']:dict(stream=r['stream'],decoded=str(Path(r['stream']).parent/'decoded.yuv'),
                            stream_sha256=r['base_sha256'],file_bytes=r['plain_bytes']) for r in sizes['rows']}
        else:
            known={r['q']:r for r in controls['all_encodes'] if r['mode']==mode and r['n']==240}
        def get(q):
            if q not in known:
                known[q]=encode(src,mode,q,width,height,240,['--roi-map-file',str(STRONG/'roi.txt')] if mode=='roi' else None)
            return known[q]
        for _ in range(8):
            ordered=sorted(known)
            best=min(known.values(),key=lambda r:abs(r['file_bytes']-target))
            if abs(best['file_bytes']/target-1)<=.005:
                break
            candidates=[(a,b) for a,b in zip(ordered,ordered[1:]) if (known[a]['file_bytes']-target)*(known[b]['file_bytes']-target)<0 and b-a>=.5]
            if not candidates:
                break
            a,b=min(candidates,key=lambda v:v[1]-v[0])
            if cpu()-t0>400:
                raise RuntimeError('CPU budget')
            get(round((a+b)*2)/4)
        best=min(known.values(),key=lambda r:abs(r['file_bytes']-target))
        selected=[('match',best)] if abs(best['file_bytes']/target-1)<=.005 else [
            (tag,min(items,key=lambda z:abs(z['file_bytes']-target)))
            for tag,items in [('under',[r for r in known.values() if r['file_bytes']<=target]),('over',[r for r in known.values() if r['file_bytes']>=target])] if items]
        for tag,r in selected:
            assert sha(Path(r['stream']))==r['stream_sha256']
            assert Path(r['decoded']).is_file()
            anchors[f'{mode}_{tag}']=dict(**{k:r[k] for k in ('decoded','stream','stream_sha256','file_bytes')},width=width,height=height,
                      decoded_sha256=sha(Path(r['decoded'])),kbps=r['file_bytes']/600)
    assert sha(SEG/'frames.zip')=='4c891777b33b10ba92fd94fa1a59ba360aacf175e7bfd67c382206058d59550a'
    assert sha(LABELS)=='9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a'
    key={(arm,t): k for k,(arm,t) in load(SEG/'key.json').items()}
    labels=load(LABELS)['frames']
    tiles={k:np.memmap(r['decoded'],'uint8','r',shape=(240,96,64)) for k,r in rungs.items()}
    bases={k:np.memmap(r['decoded'],'uint8','r',shape=(240,r['height']*3//2,r['width'])) for k,r in anchors.items()}
    squared={}
    with zipfile.ZipFile(SEG/'frames.zip') as prev,zipfile.ZipFile(out/'crops.zip','x') as cropzip:
        for f in labels:
            t=f['tick']
            def rgb(arm):
                blob=prev.read(key[arm,t]+'.png');assert hashlib.sha256(blob).hexdigest()==key[arm,t]
                return cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR)[:,:,::-1].copy()
            src=rgb('source')
            old=rgb('pointstream')
            imgs=dict(source=src)
            for name,stack in tiles.items():
                img=old.copy()
                if positions[t] is not None:
                    x,y=positions[t]
                    img[y:y+32,x:x+32]=yuv_to_rgb(stack[t],64,64,False)[16:48,16:48]
                imgs[name]=img
            for name,stack in bases.items():
                r=anchors[name]
                imgs[name]=cv2.resize(yuv_to_rgb(stack[t],r['width'],r['height'],False),size,interpolation=cv2.INTER_CUBIC)
            for name,img in imgs.items():
                if name!='source':
                    squared.setdefault(name,[]).append(float(np.mean((src.astype(np.float32)-img)**2,dtype=np.float64)))
            for region,y,x,mask in supports(f):
                for name,array in {'mask':mask,**{k:v[y,x,::-1] for k,v in imgs.items()}}.items():
                    ok,png=cv2.imencode('.png',array);assert ok
                    cropzip.writestr(f'{t:03d}/{region}/{name}.png',png.tobytes())
    assert cpu()-t0<500
    receipt=dict(chosen=chosen,target_bytes=target,rungs=rungs,anchors=anchors,original_tile_sha256=hashlib.sha256(original).hexdigest(),
                 decoded_input_sha256=original_raw_digest,new_encodes=all_new,crops_sha256=sha(out/'crops.zip'),
                 whole_rgb_psnr={a:float(10*np.log10(255**2/np.mean(e))) for a,e in squared.items()},cpu_seconds=cpu()-t0,
                 encoder=encoder,decoder=decoder,driver_sha256=sha(Path(__file__)))
    (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(chosen,{k:round(r['running'],3) for k,r in rungs.items()},{k:round(r['kbps'],3) for k,r in anchors.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    main(p.parse_args().out)

"""CPU-only source/blurred-mask inpaint byte screen; no review labels or quality."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import resource
import subprocess
import tarfile
import time

import cv2
import numpy as np

from experiments.tennis.k6b_rate_streams import decode_masks
from src.codecs import svtav1

SOURCE=Path('/private/tmp/ps-al028-review-source-20261122.mp4')
TRANSPORT=Path('/private/tmp/ps-al028-review-mattes-20261122.tar')
OUT=Path('/private/tmp/ps-al028-source-rate-oracle-20261122')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu():
    return sum(resource.getrusage(r).ru_utime+resource.getrusage(r).ru_stime for r in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN))


def main():
    start=cpu()
    cv2.setNumThreads(1)
    if sha(SOURCE)!='c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934' or sha(TRANSPORT)!='8b0871ab633edcf4c77cb810cf2d3617a2a9593f0aa51af451c2a2ec21e30607':
        raise ValueError('source identity')
    OUT.mkdir(exist_ok=False)
    with tarfile.open(TRANSPORT) as tar:
        prefix='publish/packets/alcaraz_hurkacz__028@120/'
        names=[x.name for x in tar if x.name.startswith(prefix)]
        refs=[n for n in names if n.endswith('.png.ref')]
        if len(refs)!=2:
            raise ValueError('reference count')
        refsizes={n:len(tar.extractfile(n).read()) for n in refs}
        keypoints=tar.extractfile(prefix+'keypoints.kps').read()
        matte=tar.extractfile(prefix+'exact-rle.matte').read()
    _,masks=decode_masks(matte)
    if masks.shape!=(240,2,540,960):
        raise ValueError('mask source shape')
    tools={x:svtav1.tool(x) for x in ('ffmpeg','SvtAv1EncApp','dav1d')}
    ordinary=OUT/'ordinary.yuv'
    subprocess.run([tools['ffmpeg']['path'],'-v','error','-threads','1','-i',str(SOURCE),'-vf','select=between(n\\,0\\,239),scale=160:96:flags=area:out_range=tv,format=yuv420p','-fps_mode','passthrough','-frames:v','240','-f','rawvideo',str(ordinary)],check=True,timeout=90)
    if ordinary.stat().st_size!=240*160*96*3//2:
        raise ValueError('source frame count')
    source=np.memmap(ordinary,np.uint8,'r',shape=(240,96*3//2,160))
    clean=OUT/'spatial-inpaint.yuv'
    with clean.open('xb') as out:
        for i in range(240):
            mask=cv2.resize(np.any(masks[i],axis=0).astype(np.uint8)*255,(160,96),interpolation=cv2.INTER_NEAREST)
            mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
            chroma=cv2.resize(mask,(80,48),interpolation=cv2.INTER_AREA)
            frame=source[i].reshape(-1)
            y=frame[:160*96].reshape(96,160)
            u=frame[160*96:160*96+80*48].reshape(48,80)
            v=frame[160*96+80*48:].reshape(48,80)
            for image,region in ((y,mask),(u,chroma),(v,chroma)):
                out.write(cv2.inpaint(image,region,2,cv2.INPAINT_TELEA).tobytes())
    def encode(path,kind,q,n):
        dest=OUT/f'{kind}-{q}-{n}'
        dest.mkdir()
        row=svtav1.code(path,dest,width=160,height=96,fps=Fraction(25),frames=n,crf=q,preset=4,threads=1,full_range=False,encoder=tools['SvtAv1EncApp']['path'],decoder=tools['dav1d']['path'],timeout=120)
        if row['decoded_frames']!=n or row['temporal_units']!=n:
            raise ValueError('decode frame count')
        return row
    before=cpu()
    encode(clean,'smoke',70,24)
    cost=cpu()-before
    projection=(cpu()-start)+cost*10*4*1.5+60
    if projection>900:
        raise ValueError(f'full forecast exceeds budget: {projection}')
    output=[]
    for name,path in [('ordinary',ordinary),('spatial_inpaint',clean)]:
        for q in (70,50):
            row=encode(path,name,q,240)
            total=row['file_bytes']+len(keypoints)
            output.append({'mode':name,'crf':q,'ivf_bytes':row['file_bytes'],'ivf_kbps':row['file_bytes']/1200,'warm_lower_bound_bytes':total,'warm_lower_bound_kbps':total/1200,'cold_lower_bound_bytes':total+sum(refsizes.values()),'cold_lower_bound_kbps':(total+sum(refsizes.values()))/1200,'decoded_sha256':sha(Path(row['decoded'])),'stream_sha256':sha(Path(row['stream']))})
    r={'source_sha256':sha(SOURCE),'transport_sha256':sha(TRANSPORT),'frames':240,'fps':25,'width':160,'height':96,'reference_bytes':refsizes,'reference_total_bytes':sum(refsizes.values()),'keypoint_bytes':len(keypoints),'keypoints_sha256':hashlib.sha256(keypoints).hexdigest(),'source_yuv_sha256':sha(ordinary),'inpaint_yuv_sha256':sha(clean),'mask_matte_sha256':hashlib.sha256(matte).hexdigest(),'smoke_seconds_cpu':cost,'full_forecast_seconds_cpu':projection,'cpu_seconds':cpu()-start,'wall_seconds':time.time(),'tools':tools,'results':output,'scope':'Source-only byte screen. Spatial inpainting is unvalidated and not the donor-cleaned receiver. Warm lower bound omits ball, operational framing and other costs; cold adds only two reference blobs.'}
    (OUT/'result.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({'output':str(OUT/'result.json'),'reference_bytes':r['reference_total_bytes'],'keypoint_bytes':r['keypoint_bytes'],'forecast':projection,'results':output},indent=2))


if __name__=='__main__':
    main()

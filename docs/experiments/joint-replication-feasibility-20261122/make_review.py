"""Build the private source-only Al028 review packet from frozen source and masks."""
import hashlib
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np
from PIL import Image

from experiments.tennis.k6b_rate_streams import decode_masks

HERE=Path(__file__).resolve().parent
SOURCE=Path('/private/tmp/ps-al028-review-source-20261122.mp4')
MASKS=Path('/private/tmp/ps-al028-review-mattes-20261122.tar')
OUTPUT=Path('/private/tmp/ps-al028-source-annotations-20261122')
TICKS=list(range(10,240,20))
SOURCE_SHA='c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934'
MASKS_SHA='8b0871ab633edcf4c77cb810cf2d3617a2a9593f0aa51af451c2a2ec21e30607'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sha(SOURCE)!=SOURCE_SHA or sha(MASKS)!=MASKS_SHA:
        raise ValueError('source or archived proposals changed')
    with tarfile.open(MASKS) as tar:
        member=tar.extractfile('publish/packets/alcaraz_hurkacz__028@120/exact-rle.matte')
        if member is None:
            raise ValueError('missing source-derived proposal')
        mask_blob=member.read()
    meta,masks=decode_masks(mask_blob)
    if masks.shape!=(240,2,540,960):
        raise ValueError('unexpected proposal grid')
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT/'assets').mkdir()
    capture=cv2.VideoCapture(str(SOURCE),cv2.CAP_FFMPEG)
    if [capture.get(p) for p in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS,cv2.CAP_PROP_FRAME_COUNT)] != [1920,1080,25,492]:
        raise ValueError('source geometry/cadence mismatch')
    frames=[]
    try:
        for tick in range(TICKS[-1]+1):
            ok,bgr=capture.read()
            if not ok:
                raise ValueError('short source decode')
            if tick not in TICKS:
                continue
            frame=OUTPUT/'assets'/f'source-{tick:03d}.png'
            if not cv2.imwrite(str(frame),bgr,[cv2.IMWRITE_PNG_COMPRESSION,3]):
                raise ValueError('source frame write failed')
            players=[]
            for person,color in enumerate(((0,132,199),(192,38,211))):
                proposal=cv2.resize(masks[tick,person].astype(np.uint8),(1920,1080),interpolation=cv2.INTER_NEAREST)*255
                path=OUTPUT/'assets'/f'proposal-{tick:03d}-{person}.png'
                Image.fromarray(proposal).save(path)
                rgba=np.zeros((1080,1920,4),np.uint8)
                rgba[:,:,:3]=color
                rgba[:,:,3]=(proposal>0).astype(np.uint8)*85
                overlay=OUTPUT/'assets'/f'overlay-{tick:03d}-{person}.png'
                Image.fromarray(rgba).save(overlay)
                players.append({'id':f'archived-track-{person}','proposal':'assets/'+path.name,'sha256':sha(path),'overlay':'assets/'+overlay.name,'overlay_sha256':sha(overlay),'pixels':int(np.count_nonzero(proposal))})
            frames.append({'id':f'al028-tick-{tick:03d}','frame_index':tick,'source':'assets/'+frame.name,'source_sha256':sha(frame),'players':players})
    finally:
        capture.release()
    if len(frames)!=12:
        raise ValueError('missing selected frame')
    review={'version':1,'review_id':'al028-source-only-20261122','source_group':'alcaraz_hurkacz/028@120','source_video_sha256':SOURCE_SHA,'source_video_fps':[25,1],'window_start_in_clip':0,'matte_transport_sha256':MASKS_SHA,'matte_member_sha256':hashlib.sha256(mask_blob).hexdigest(),'proposal_provenance':'historical source-derived model masks, not human reviewed','matte_meta':meta,'frames':frames}
    canonical=json.dumps(review,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    review['review_sha256']=hashlib.sha256(canonical).hexdigest()
    (OUTPUT/'review.json').write_text(json.dumps(review,indent=2)+'\n')
    template=(HERE/'review_template.html').read_text()
    (OUTPUT/'index.html').write_text(template.replace('__REVIEW_JSON__',json.dumps(review,separators=(',',':'))))
    print(json.dumps({'path':str(OUTPUT),'frames':len(frames),'review_sha256':review['review_sha256'],'matte_sha256':review['matte_member_sha256'],'ticks':TICKS,'opencv':cv2.__version__},indent=2))


if __name__=='__main__':
    main()

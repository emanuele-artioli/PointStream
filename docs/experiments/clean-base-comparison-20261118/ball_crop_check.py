"""Measure the original base ball and paid tile at twelve reviewed centers."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image
from src.pointstream.tracked_patch import yuv_to_rgb

DATA = Path('/Users/manu/Datasets/pointstream-submission-20261118')
LABELS = DATA / 'foreground-human-reviewed-20261118/9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a.json'
BUNDLE = DATA / 'ps-development-source-refs-20261118-native/source-only-review.zip'
BASE = DATA / 'ps-native-ctl-20261118-full/plain-48/decoded.yuv'
BALL = Path('/private/tmp/ps-low-ball-stage-20261118/ball-input.npz')
OUT = Path('docs/experiments/clean-base-comparison-20261118/ball-crop-results.json')


def main():
    cv2.setNumThreads(1)
    def digest(path):
        with path.open('rb') as handle:
            return hashlib.file_digest(handle,'sha256').hexdigest()
    expected={LABELS:'9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a',
              BUNDLE:'5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb',
              BASE:'4f549befc1445ae4baca3a8da55c822261a78b08f98a8b212fc702f81998499b',
              BALL:'3a010a990d7cb434e7027e35216b41ba21a64829c0ad0281998117c4147786fc'}
    if any(digest(p)!=h for p,h in expected.items()):
        raise ValueError('input hash')
    frames=np.memmap(BASE,np.uint8,'r',shape=(240,816,960))
    blobs=np.load(BALL,allow_pickle=False)
    labels=json.loads(LABELS.read_text())
    rows=[]
    with zipfile.ZipFile(BUNDLE) as source:
        for label in labels['frames']:
            tick=label['tick']
            raw=source.read(f'frame-{tick:03d}/source.png')
            if hashlib.sha256(raw).hexdigest()!=label['source_png_sha256']:
                raise ValueError('source identity')
            orig=np.asarray(Image.open(BytesIO(raw)).convert('RGB'))
            native=cv2.resize(yuv_to_rgb(frames[tick],960,544,False),(1920,1080),interpolation=cv2.INTER_CUBIC)
            x,y=label['ball']['center_xy']
            x0,y0=min(max(x-16,0),1888),min(max(y-16,0),1048)
            target=orig[y0:y0+32,x0:x0+32].astype(np.float32)
            before=native[y0:y0+32,x0:x0+32].astype(np.float32)
            px,py=blobs['coords'][tick]
            after=native.copy()
            if px>=0:
                after[py:py+32,px:px+32]=blobs['patches'][tick]
            after=after[y0:y0+32,x0:x0+32].astype(np.float32)
            rows.append(dict(tick=tick,center=[x,y],tile=[int(px),int(py)] if px>=0 else None,
                             center_covered=bool(px>=0 and px<=x<px+32 and py<=y<py+32),
                             base_mse=float(np.square(before-target).mean()),
                             with_tile_mse=float(np.square(after-target).mean())))
    result=dict(scope='12 reviewed visible centers on one development rally, RGB MSE over fixed 32x32 source-centered crop',
                rows=rows,base_mse=float(np.mean([r['base_mse'] for r in rows])),
                tile_mse=float(np.mean([r['with_tile_mse'] for r in rows])),
                covered=sum(r['center_covered'] for r in rows),placements_240=int((blobs['coords'][:,0]>=0).sum()))
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__ == '__main__':
    main()

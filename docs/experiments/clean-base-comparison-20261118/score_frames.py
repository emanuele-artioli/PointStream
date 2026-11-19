"""Score the predeclared body, boundary, ball and complement on reviewed frames."""
import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DATA = Path('/Users/manu/Datasets/pointstream-submission-20261118')
LABELS = DATA / 'foreground-human-reviewed-20261118/9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a.json'
BALL = Path('/private/tmp/ps-low-ball-stage-20261118/ball-input.npz')
BASE = DATA / 'ps-native-ctl-20261118-full/plain-48/decoded.yuv'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mse(a,b,mask):
    return float(np.square(a[mask].astype(np.float32)-b[mask].astype(np.float32)).mean())


def run(root):
    if sha(LABELS) != '9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a':
        raise ValueError('label identity')
    info = json.loads((root/'composition.json').read_text())
    labels = json.loads(LABELS.read_text())
    balls = np.load(BALL,allow_pickle=False)
    rows = []
    for frame, record in zip(labels['frames'],info['rows'],strict=True):
        tick=frame['tick']
        if tick != record['tick']:
            raise ValueError('tick mismatch')
        source = np.asarray(Image.open(root/'source'/f'{tick:03d}.png').convert('RGB'))
        masks = [np.asarray(Image.open(BytesIO(base64.b64decode(p['mask']['png_base64']))).convert('L')) > 0
                 for p in frame['players']]
        foreground = masks[0] | masks[1]
        dilated = cv2.dilate(foreground.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))>0
        ring = dilated & ~foreground
        x,y = frame['ball']['center_xy']
        x0 = min(max(x-16,0),1920-32)
        y0 = min(max(y-16,0),1080-32)
        ball = np.zeros((1080,1920),bool)
        ball[y0:y0+32,x0:x0+32]=True
        regions={'player':foreground,'boundary':ring,'ball':ball,'complement':~(dilated | ball)}
        row={'tick':tick,'ball_xy':[x,y],'placement':balls['coords'][tick].tolist()}
        px,py=balls['coords'][tick]
        row['ball_center_covered']=bool(px >= 0 and px <= x < px+32 and py <= y < py+32)
        for arm in ('original','clean'):
            path=root/arm/f'{tick:03d}.png'
            image=np.asarray(Image.open(path).convert('RGB'))
            if hashlib.sha256(image.tobytes()).hexdigest()!=record[arm+'_rgb_sha256']:
                raise ValueError('output pixels')
            row[arm]={k:mse(source,image,mask) for k,mask in regions.items()}
        rows.append(row)
    result={'scope':'one development rally, 12 source-reviewed instants, 9 assisted and 3 unprompted; RGB MSE',
            'rows':rows,'averages':{arm:{region:float(np.mean([r[arm][region] for r in rows]))
                                        for region in ('player','boundary','ball','complement')}
                                   for arm in ('original','clean')},
            'coverage':sum(r['ball_center_covered'] for r in rows)}
    (root/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('scope','averages','coverage')}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    run(p.parse_args().root)

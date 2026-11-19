"""Extract identical native-resolution RGB supports for each source/decoded arm."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from src.pointstream.tracked_patch import yuv_to_rgb

ROOT = Path('/Users/manu/Datasets/pointstream-submission-20261118')
LABEL = '9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a'
BUNDLE = '5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb'
SOURCE = 'b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def supports(frame):
    for index, player in enumerate(frame['players'], start=1):
        assert player['status'] == 'visible' and player['id'] == f'person:{index-1:012d}'
        data = np.frombuffer(base64.b64decode(player['mask']['png_base64']), np.uint8)
        mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        ys, xs = np.nonzero(mask)
        y = slice(max(0, ys.min()-32), min(1080, ys.max()+33))
        x = slice(max(0, xs.min()-32), min(1920, xs.max()+33))
        assert len(xs) == player['mask']['foreground_pixels']
        yield 'p'+str(index), y, x, mask[y,x]
    assert frame['ball']['status'] == 'visible'
    x, y = frame['ball']['center_xy']
    assert 32 <= x < 1888 and 32 <= y < 1048
    yield 'ball', slice(y-32,y+32), slice(x-32,x+32), np.full((64,64),255,np.uint8)


def build(a):
    cv2.setNumThreads(1)
    labels = ROOT/'foreground-human-reviewed-20261118'/f'{LABEL}.json'
    source = ROOT/'ps-development-source-refs-20261118-native'/'source-only-review.zip'
    assert sha(labels) == LABEL and sha(source) == BUNDLE
    frames = json.loads(labels.read_text())['frames']
    assert [f['tick'] for f in frames] == list(range(10,240,20))
    anchors, inventory = {}, {}
    if a.controls:
        for folder, mode, crfs in [('ps-native-ctl-20261118-full','plain',[36,48,60]),('ps-joint-roi-20261118-full','roi',[36,48,60]),
                                   ('ps-upper-roi-20261118-full','plain',[28]),('ps-upper-roi-20261118-full','roi',[28])]:
            report=json.loads((ROOT/folder/'report.json').read_text())
            assert report['source_sha256'] == SOURCE and report['frames'] == 240
            for crf in crfs:
                name = f'{mode}-{crf}'
                row=next(r for r in report['rows'] if r['mode']==mode and r['crf']==crf)
                loc=ROOT/folder/name
                assert sha(loc/'stream.ivf') == row['stream_sha256'] and sha(loc/'decoded.yuv') == row['decoded_sha256']
                anchors[name] = np.memmap(loc/'decoded.yuv',np.uint8,'r',shape=(240,816,960))
                inventory[name] = {k:row[k] for k in ('file_bytes','complete_kbps','stream_sha256','decoded_sha256')}
    if a.upper:
        folder = ROOT/'selected-rate-support-linux-png'/'full'
        report = json.loads((folder/'result.json').read_text())
        assert report['passed'] and report['same_source'] == SOURCE and report['frames'] == 240
        assert {row['mode'] for row in report['rows']} == {'plain', 'roi'}
        for row in report['rows']:
            mode = row['mode']
            assert row['crf'] == 18 and row['decoded_frames'] == 240
            stream, decoded = folder/mode/'stream.ivf', folder/mode/'decoded.yuv'
            assert sha(stream) == row['stream_sha256'] and sha(decoded) == row['decoded_sha256']
            assert stream.stat().st_size == row['file_bytes'] and decoded.stat().st_size == 240*816*960
            name = f'{mode}-18'
            anchors[name] = np.memmap(decoded, np.uint8, 'r', shape=(240,816,960))
            inventory[name] = {k:row[k] for k in ('file_bytes','complete_kbps','stream_sha256','decoded_sha256')}
    a.output.mkdir(exist_ok=False,parents=True)
    rows=[]
    with zipfile.ZipFile(source) as z, zipfile.ZipFile(a.output/'crops.zip','x') as out:
        for f in frames:
            t=f['tick']
            png=z.read(f'frame-{t:03d}/source.png')
            assert hashlib.sha256(png).hexdigest()==f['source_png_sha256']
            images={'source':cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_COLOR)[:,:,::-1]}
            for name, stack in anchors.items():
                images[name]=cv2.resize(yuv_to_rgb(stack[t],960,544,False),(1920,1080),interpolation=cv2.INTER_CUBIC)
            for name, folder in a.arm:
                path=Path(folder)/f'{t:03d}.png'
                images[name]=cv2.imread(str(path),cv2.IMREAD_COLOR)[:,:,::-1]
                assert images[name].shape==(1080,1920,3)
                inventory.setdefault(name,{})[str(t)]=hashlib.sha256(images[name].tobytes()).hexdigest()
            for who, y, x, mask in supports(f):
                group=f'{t:03d}/{who}/'
                rows.append(dict(tick=t,region=who,bounds=[int(v) for v in (x.start,y.start,x.stop,y.stop)],mask_area=int(np.count_nonzero(mask))))
                for name, img in {'mask':mask, **{k:v[y,x,::-1] for k,v in images.items()}}.items():
                    ok, encoded=cv2.imencode('.png',img,[cv2.IMWRITE_PNG_COMPRESSION,5])
                    assert ok
                    out.writestr(group+name+'.png',encoded.tobytes())
    meta=dict(source=SOURCE,labels=LABEL,opencv=cv2.__version__,arms=inventory,regions=rows,zip_sha256=sha(a.output/'crops.zip'))
    (a.output/'inputs.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',nargs=2,action='append',default=[])
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--controls',action='store_true')
    p.add_argument('--upper',action='store_true')
    build(p.parse_args())

"""Verify all 240 paid tile placements and score 12 reviewed RGB frames."""
import hashlib
import json
import math
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np

from experiments.tennis import common_player_ball as ball
from src.codecs import svtav1
from src.pointstream.tracked_patch import yuv_to_rgb

ROOT = Path('/Users/manu/Datasets/pointstream-submission-20261118')
DATA = Path('/private/tmp/ps-matched-ball-20261122-crf50')
OUT = Path('/private/tmp/ps-matched-ball-20261122-frames')
TICKS = list(range(10, 231, 20))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    cv2.setNumThreads(1)
    rec = json.loads((DATA/'receipt.json').read_text())['row']
    carrier, sidecar = (DATA/'native.pscp').read_bytes(), (DATA/'ball.sidecar').read_bytes()
    assert hashlib.sha256(carrier).hexdigest() == rec['carrier_sha256']
    assert hashlib.sha256(sidecar).hexdigest() == rec['sidecar_sha256']
    assert sha(DATA/'decoded.yuv') == rec['decoded_sha256']
    original = ROOT/'ps-native-ctl-20261118-full/plain-48/decoded.yuv'
    assert sha(original) == '4f549befc1445ae4baca3a8da55c822261a78b08f98a8b212fc702f81998499b'
    source = ROOT/'ps-development-source-refs-20261118-native/source-only-review.zip'
    assert sha(source) == '5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb'
    dec, ff = svtav1.tool('dav1d'), svtav1.tool('ffmpeg')
    origins, tile, size = ball._open(carrier, sidecar, dec['sha256'])
    assert size == (1920, 1080)
    raw = subprocess.check_output([ff['path'], '-v','error','-threads','1','-c:v','libdav1d','-f','ivf',
                                   '-i','pipe:0','-pix_fmt','yuv420p','-f','rawvideo','pipe:1'], input=tile, timeout=30)
    tiles = np.frombuffer(raw,np.uint8).reshape((240,96,64))
    frames = np.memmap(DATA/'decoded.yuv',np.uint8,'r',shape=(240,816,960))
    base = np.memmap(original,np.uint8,'r',shape=(240,816,960))
    source_y = np.memmap(ROOT/'ps-native-ctl-20261118-full/source.yuv',np.uint8,'r',shape=(240,816,960))
    assert sha(ROOT/'ps-native-ctl-20261118-full/source.yuv') == '6cae4bd7a46bfa4bd2a7a7ec5c926647ea43d9e7a5366eb96e464eaffe1e603c'
    OUT.mkdir(exist_ok=False)
    err = [0,0]
    frame_error = [0,0]
    whole = hashlib.sha256()
    got = []
    with zipfile.ZipFile(source) as archive:
        def canvases():
            for t, yuv in enumerate(frames):
                yield t, cv2.resize(yuv_to_rgb(yuv,960,544,False),size,interpolation=cv2.INTER_CUBIC)
        for t, image in ball.render(carrier,sidecar,canvases(),scratch=OUT,decoder=dec['path']):
            plain = cv2.resize(yuv_to_rgb(base[t],960,544,False),size,interpolation=cv2.INTER_CUBIC)
            reference = cv2.resize(yuv_to_rgb(frames[t],960,544,False),size,interpolation=cv2.INTER_CUBIC)
            if origins[t] is not None:
                x,y = origins[t]
                reference[y:y+32,x:x+32] = yuv_to_rgb(tiles[t],64,64,False)[16:48,16:48]
            assert np.array_equal(image,reference)
            whole.update(image)
            for i, array in enumerate((base[t,:544],frames[t,:544])):
                diff = array.astype(np.int32) - source_y[t,:544].astype(np.int32)
                err[i] += int(np.sum(diff.astype(np.int64)**2))
            if t in TICKS:
                png=archive.read(f'frame-{t:03d}/source.png')
                truth=cv2.cvtColor(cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
                for i, array in enumerate((plain,image)):
                    diff=array.astype(np.int32)-truth.astype(np.int32)
                    frame_error[i] += int(np.sum(diff.astype(np.int64)**2))
                path=OUT/f'{t:03d}.png'
                assert cv2.imwrite(str(path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                got.append({'tick':t,'sha256':sha(path),'ball_origin':origins[t]})
    assert len(got)==12
    all_pixels=240*544*960
    rgb_pixels=12*1080*1920*3
    receipt={'candidate_total_bytes':rec['total_bytes'],'baseline_total_bytes':182055,
             'candidate_kbps':rec['total_kbps'],'baseline_kbps':182055/600,
             'rgb_psnr_12_reviewed':[10*math.log10(255**2*rgb_pixels/e) for e in frame_error],
             'native_luma_psnr_240':[10*math.log10(255**2*all_pixels/e) for e in err],
             'order':['plain48','matched50_plus_tiles'], 'all_frames_rgb_sha256':whole.hexdigest(),
             'placements':sum(x is not None for x in origins),'saved':got,
             'scope':'one opened 240-frame development rally', 'source_bundle_sha256':sha(source)}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='saved'}))


if __name__ == '__main__':
    main()

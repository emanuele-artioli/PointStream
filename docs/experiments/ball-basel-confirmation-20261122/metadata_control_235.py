"""Post-outcome paid centers alongside the fixed 640p plain AV1 control."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import zlib
from fractions import Fraction

from experiments.tennis.ball_starved_scaled import unpack_scaled
from src.pointstream import ball, ball_stream

ROOT=Path(__file__).resolve().parents[3]
BASE=Path('/private/tmp/ps-235-640-published.tar')
TILE=Path('/private/tmp/ps-235-tile-published.tar')
OUTPUT=Path('/private/tmp/ps-235-center-control')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    if sha(BASE.read_bytes())!='0f4978a87db1f451e61456bddd791b6c12c6836e54a3fdaeec79fa0b22f95f4f':
        raise ValueError('base archive identity')
    with tarfile.open(BASE) as tar:
        report=json.load(tar.extractfile('publish/result.json'))
        row=next(x for x in report['selected'] if x['mode']=='plain' and x['crf']==67.5)
        native=tar.extractfile('publish/'+row['file']).read()
        if sha(native)!=row['raw_sha256']:
            raise ValueError('base packet identity')
    with tarfile.open(TILE) as tar:
        report=json.load(tar.extractfile('publish/result.json'))
        rowtile=next(x for x in report['results'] if x['sha256']=='338c8d17c3bbaaedb70a9c82c75bf2567ac0acfb9e0b0289c73ad9151bedd4e5')
        tile=tar.extractfile('publish/'+rowtile['file']).read()
        if sha(tile)!=rowtile['sha256']:
            raise ValueError('tile packet identity')
    meta,_,_=unpack_scaled(tile)
    centers=[(p[0]+16,p[1]+16) if p is not None else None for p in meta['positions']]
    if any(p is not None and (p[0]%2 or p[1]%2) for p in centers):
        raise ValueError('cannot represent paid centers exactly on 640 canvas')
    scaled=[(p[0]//2,p[1]//2) if p is not None else None for p in centers]
    placeholder=ball.Appearance((0,0,0),0,0,0)
    appearances=[placeholder if p else None for p in scaled]
    payload=ball.encode_frames(scaled,appearances,640,360)
    stream=ball_stream.Stream(640,360,Fraction(60),154,'svt-av1-ivf','dav1d-1.5.4',native,payload)
    body=ball_stream.pack(stream)
    compressed=zlib.compress(body,6)
    wire=b'\x01'+compressed if len(compressed)<len(body) else b'\x00'+body
    recovered=ball_stream.unpack(zlib.decompress(wire[1:]) if wire[0] else wire[1:])
    decoded,_=ball.decode_frames(recovered.ball_payload,154,640,360)
    restored=[(p[0]*2,p[1]*2) if p is not None else None for p in decoded]
    if recovered.native!=native or restored!=centers:
        raise ValueError('native or paid position mismatch')
    OUTPUT.mkdir(exist_ok=True)
    for name,data in [('original',native),('recovered',recovered.native)]:
        (OUTPUT/(name+'.ivf')).write_bytes(data)
        subprocess.run(['dav1d','-i',str(OUTPUT/(name+'.ivf')),'-o',str(OUTPUT/(name+'.yuv')),'--threads','1','--quiet'],check=True)
    before=(OUTPUT/'original.yuv').read_bytes()
    after=(OUTPUT/'recovered.yuv').read_bytes()
    if before!=after or len(before)!=154*640*360*3//2:
        raise ValueError('decoded visual frames differ')
    m=json.load(open(ROOT/'docs/experiments/ball-machine-anchors-20261122/235-result.json'))
    tile_result=next(x for x in m['results'] if x['mode']=='tracked')
    labels=tile_result['rows']
    result={'status':'post-outcome control','frames':154,'fps':60,'base':'640x360 plain AV1',
            'base_wire_kbps':row['kbps'],'base_dists':row['mean_dists'],'base_wire_bytes':row['bytes'],
            'complete_wire_bytes':len(wire),'complete_wire_kbps':len(wire)*8*60/154/1000,
            'complete_sha256':sha(wire),'ball_payload_bytes':len(payload),'base_ivf_unchanged':True,
            'decoded_bytes_identical':True,'decoded_yuv_sha256':sha(before),'paid_centers_exact':True,
            'paid_missing':sum(x is None for x in centers),'labels':len(labels),
            'hits_10':sum(x['visibility']==1 and x['direct_hit_10px'] for x in labels),
            'tile_packet_sha256':rowtile['sha256'],'base_packet_sha256':row['raw_sha256'],
            'fixed_placeholder':'black zero indices, syntax only; not rendered'}
    (ROOT/'docs/experiments/ball-basel-confirmation-20261122/metadata-control-235.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

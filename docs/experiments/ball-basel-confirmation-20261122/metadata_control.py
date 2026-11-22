"""Post-outcome byte-exact native AV1 plus pre-existing paid ball centers."""
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
ARCHIVE=Path('/private/tmp/ps-basel-score-published.tar')
EXPECTED='5f41e38effb3506da7668049eaf84ed0db401baafe7607dfd7edeba50b023ba9'
OUT=Path('/private/tmp/ps-basel-center-control')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unwrap(w):
    if not w or w[0] not in (0,1):
        raise ValueError('invalid wire selector')
    return zlib.decompress(w[1:]) if w[0] == 1 else w[1:]


def wire(raw):
    compressed=zlib.compress(raw,6)
    return b'\x01'+compressed if len(compressed)<len(raw) else b'\x00'+raw


def main():
    raw=ARCHIVE.read_bytes()
    if sha(raw) != EXPECTED:
        raise ValueError('input archive mismatch')
    with tarfile.open(ARCHIVE) as tar:
        r=json.load(tar.extractfile('publish/result.json'))
        def packet(row):
            p=tar.extractfile('publish/'+row['file']).read()
            if sha(p)!=row['sha256'] or len(p)!=row['bytes']:
                raise ValueError('selected packet mismatch')
            return p
        base=next(x for x in r['rows'] if x['mode']=='plain' and x['width']==1280 and x['kbps']<=60.6)
        tile=next(x for x in r['rows'] if x['mode']=='tile' and x['kbps']<=60.6)
        native=unwrap(packet(base))
        meta,_,_=unpack_scaled(unwrap(packet(tile)))
    centers=[(p[0]+16,p[1]+16) if p is not None else None for p in meta['positions']]
    placeholder=ball.Appearance((0,0,0),0,0,0)
    appearances=[placeholder if p is not None else None for p in centers]
    payload=ball.encode_frames(centers,appearances,1280,720)
    stream=ball_stream.Stream(1280,720,Fraction(25),64,'svt-av1-ivf','dav1d-1.5.4',native,payload)
    complete=wire(ball_stream.pack(stream))
    recovered=ball_stream.unpack(unwrap(complete))
    decoded,decoded_appearances=ball.decode_frames(recovered.ball_payload,64,1280,720)
    if recovered.native != native or decoded != centers or decoded_appearances != appearances:
        raise ValueError('receiver did not preserve bytes and coordinates')
    OUT.mkdir(exist_ok=True)
    (OUT/'original.ivf').write_bytes(native)
    (OUT/'recovered.ivf').write_bytes(recovered.native)
    (OUT/'metadata.psblwire').write_bytes(complete)
    for name in ('original','recovered'):
        subprocess.run(['dav1d','-i',str(OUT/(name+'.ivf')),'-o',str(OUT/(name+'.yuv')),'--threads','1','--quiet'],check=True)
    original=(OUT/'original.yuv').read_bytes()
    new=(OUT/'recovered.yuv').read_bytes()
    if original != new or len(original)!=64*1280*720*3//2:
        raise ValueError('decoded video differs')
    machine=json.loads((ROOT/'docs/experiments/ball-basel-confirmation-20261122/machine-result.json').read_text())
    tile_machine=next(x for x in machine['results'] if x['mode']=='tile')
    result={'status':'post-outcome control','source':'Basel Rune–Etcheverry','frames':64,'fps':25,
            'base_wire_bytes':base['bytes'],'base_wire_kbps':base['kbps'],'base_dists':base['dists'],
            'complete_wire_bytes':len(complete),'complete_wire_kbps':len(complete)*8*25/64/1000,
            'complete_sha256':sha(complete),'container_bytes':len(ball_stream.pack(stream)),
            'ball_payload_bytes':len(payload),'base_packet_unchanged':recovered.native==native,
            'video_decoded_bytes_identical':True,'decoded_yuv_sha256':sha(original),
            'native_frame_sha256':[sha(original[i*1280*720*3//2:(i+1)*1280*720*3//2]) for i in range(64)],
            'paid_centers_exact':decoded==centers,'paid_missing':sum(p is None for p in decoded),
            'visible_labeled':sum(x['visibility']==1 for x in tile_machine['rows']),
            'direct_hit_10px':sum(x['visibility']==1 and x['direct_native_error'] is not None and x['direct_native_error']<=10 for x in tile_machine['rows']),
            'direct_hit_5px':sum(x['visibility']==1 and x['direct_native_error'] is not None and x['direct_native_error']<=5 for x in tile_machine['rows']),
            'base_packet_sha256':base['sha256'],'source_tile_packet_sha256':tile['sha256'],
            'dav1d_version':subprocess.run(['dav1d','--version'],capture_output=True,text=True,check=True).stderr.strip(),
            'placeholder_appearance':'fixed black zero indices, syntax only; not rendered'}
    (ROOT/'docs/experiments/ball-basel-confirmation-20261122/metadata-control.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('native_frame_sha256',)},indent=2))


if __name__=='__main__':
    main()

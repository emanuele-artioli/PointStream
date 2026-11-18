"""Serialize all pilot ablations and record their exact sizes for anchor planning."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.tennis import common_player_ball as ball, common_player_packet as player


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('references','native','ball-only','track-witness','sidecar','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    witness=json.loads(a.track_witness.read_text())
    dec=witness['decoder']['sha256']
    seed=(a.references/'q70'/'packet.pscp').read_bytes()
    if hashlib.sha256(seed).hexdigest()!=witness['carrier_sha256'] or digest(a.sidecar)!=witness['sidecar_sha256']:
        raise ValueError('Input identity')
    positions,tile,_=ball._open(seed,a.sidecar.read_bytes(),dec)
    rows=[]
    for obj in json.loads(a.ball_only.read_text())['rows']:
        crf=obj['crf']
        raw=a.native/str(crf)/'native.pscp'
        if digest(raw)!=obj['native_sha256']:
            raise ValueError('Native carrier identity changed')
        base=player.unpack(raw.read_bytes())[0]['base']
        for q in (70,40):
            root=a.references/f'q{q}'
            orig=(root/'packet.pscp').read_bytes()
            if hashlib.sha256(orig).hexdigest()!={70:witness['carrier_sha256'],40:'be27d15fbda67845727b02562a780f379573e67df4dc6047a0aca2e5081ad371'}[q]:
                raise ValueError('Reference variant identity')
            parts,_,_=player.unpack(orig)
            profile=player.inspect(orig)['profile']
            profile['coded_size']=[960,544]
            wire=player.build(profile,base,parts['keypoints'],{k:v for k,v in parts.items() if k.startswith('ref')})
            rebuilt,_,_=player.unpack(wire)
            if rebuilt!={**parts,'base':base}:
                raise ValueError('Contents changed')
            sc=ball.pack(wire,positions,tile,decoder_sha256=dec)
            stats=ball.inspect(wire,sc,decoder_sha256=dec)
            out=a.output/f'base{crf}-q{q}'
            out.mkdir()
            (out/'players.pscp').write_bytes(wire)
            (out/'ball.sidecar').write_bytes(sc)
            rows.append(dict(base_crf=crf,jpeg=q,native_only_bytes=obj['native_bytes'],ball_only_bytes=obj['total_bytes'],
                             player_only_bytes=len(wire),joint_bytes=stats['total_bytes'],joint_kbps=stats['total_bytes']/600,
                             player_sha256=digest(out/'players.pscp'),sidecar_sha256=digest(out/'ball.sidecar')))
    (a.output/'rates.json').write_text(json.dumps(dict(rows=rows,seconds=4.8,profile='Mac cold-start',gpu_seconds=0,
                                                      runner_sha256=digest(Path(__file__))),indent=2)+'\n')
    print([(r['base_crf'],r['jpeg'],round(r['joint_kbps'],2)) for r in rows])


if __name__=='__main__':
    main()

"""Replace only the native base in the verified Linux WebP delivery and rebind the ball."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
from experiments.tennis import common_player_packet as player
from experiments.tennis import common_receiver_manifest as manifest
from experiments.tennis import common_player_ball as ball

DAV1D='e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6'
MIDDLE=Path('/local/users/emanuele/pointstream/appearance-webp-packet-6905036f/middle')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(a):
    assert not os.environ['CUDA_VISIBLE_DEVICES'] and np.__version__=='2.5.4'
    orig=(MIDDLE/'players.pscp').read_bytes()
    oldside=(MIDDLE/'ball.sidecar').read_bytes()
    assert sha(orig)=='7dcb2dbb45b3595c66c2d7e018fe7c8cf49ca2b7e1bae972ae59b450ec4afe84' and sha(oldside)=='81bf192ad5c668d7094eabb91d5cfa1ec86ed25fbc9cbdb3f810f14cd339cca6'
    manifest.verify(MIDDLE/'receiver.manifest',sha(orig))
    original,_,_=player.unpack(orig)
    profile=player._embedded_profile(orig)
    assert profile['coded_size']==[960,544] and profile['recipe']==player.LOSSLESS_RECIPE
    positions,tile,size=ball._open(orig,oldside,DAV1D)
    ivf=a.ivf.read_bytes()
    assert sha(ivf)==a.sha256
    ref={k:v for k,v in original.items() if k not in ('base','keypoints')}
    carrier=player.build(profile,ivf,original['keypoints'],ref)
    side=ball.pack(carrier,positions,tile,decoder_sha256=DAV1D)
    a.output.mkdir()
    data={'players.pscp':carrier,'receiver.manifest':manifest.encode(sha(carrier)),'ball.sidecar':side}
    for k,v in data.items():
        with (a.output/k).open('xb') as f:
            f.write(v)
    reopened=(a.output/'players.pscp').read_bytes()
    sections,_,_=player.unpack(reopened)
    assert sections=={**original,'base':ivf}
    manifest.verify(a.output/'receiver.manifest',sha(reopened))
    assert ball._open(reopened,(a.output/'ball.sidecar').read_bytes(),DAV1D)==(positions,tile,size)
    total=sum(len(v) for v in data.values())
    print(json.dumps(dict(source_carrier_sha256=sha(orig),source_sidecar_sha256=sha(oldside),coded_base_sha256=sha(ivf),
                          files={k:{'sha256':sha(v),'bytes':len(v)} for k,v in data.items()},
                          total_bytes=total,kbps=total/600,placements=sum(p is not None for p in positions),
                          coded_base_bytes=len(ivf),reference_bytes=sum(map(len,ref.values())))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('ivf',type=Path)
    p.add_argument('sha256')
    p.add_argument('output',type=Path)
    run(p.parse_args())

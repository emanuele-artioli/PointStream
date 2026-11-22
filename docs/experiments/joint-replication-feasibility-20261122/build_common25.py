"""Package real Al028 conditioning into the separately versioned 25 fps profile."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from experiments.tennis import common_player_packet as packet
from experiments.tennis import k6b_rate_streams as rates

SOURCE=Path('/private/tmp/ps-al028-review-mattes-20261122.tar')
PREP=Path('/private/tmp/ps-al028-prepared-20261122.tar')
BASE=Path('/private/tmp/ps-al028-source-rate-oracle-20261122/spatial_inpaint-70-240/stream.ivf')
NADAL=Path('/private/tmp/agentctl-artifacts/Emanueles-MacBook-Pro.local/9897d39530594990a4b5cc4ab12ceaf9/output/full-quarter/clean-60/players.pscp')
OUT=Path('/private/tmp/ps-al028-common25-packet-20261122')
NATIVE_DECODER=None
CODED_SIZE=[160,96]


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    if sha(SOURCE.read_bytes())!='8b0871ab633edcf4c77cb810cf2d3617a2a9593f0aa51af451c2a2ec21e30607' or sha(PREP.read_bytes())!='e30c6d97a6a8406f85bcc6c6928d9b7fb21ddf8280ad25a629f78f178478cfdb':
        raise ValueError('archive identity')
    if sha(NADAL.read_bytes())!='c5d62404913130a759cbe0ced630dc8fa9743140e3dacbc74078d7afd0d5e14f':
        raise ValueError('common profile identity')
    prefix='publish/packets/alcaraz_hurkacz__028@120/'
    with tarfile.open(SOURCE) as tar:
        kps=tar.extractfile(prefix+'keypoints.kps').read()
        oldmeta,units=rates.decode_keypoints(kps)
        refs={}
        for i,suffix in enumerate(('000000000004','000000000006')):
            name='alcaraz_hurkacz__028_at120__person_'+suffix
            refs[name]=tar.extractfile(prefix+name+'.png.ref').read()
    units.sort(key=lambda u:u['name'])
    names=[u['name'] for u in units]
    if oldmeta['fps']!=[25,1] or oldmeta['native_frames']!=240:
        raise ValueError('old native cadence mismatch')
    coded=rates.encode_keypoints_kps2({'native_frames':240},units)
    _,again=rates.decode_keypoints(coded)
    if not all(a['name']==b['name'] and np.array_equal(a['t'],b['t']) and np.array_equal(a['kp'],b['kp']) for a,b in zip(units,again,strict=True)):
        raise ValueError('KPS2 transcoding changed coordinates')
    profile=packet._embedded_profile(NADAL.read_bytes())
    profile['recipe']=packet.RECIPE_25.copy()
    profile.pop('reference_rgb_sha256')
    profile['resources'].pop('reference_decoder')
    if NATIVE_DECODER is not None:
        profile['resources']['native_decoder']=NATIVE_DECODER
    profile['coded_size']=CODED_SIZE.copy()
    profile['order']=names
    profile['cuts']=[]
    profile['runs']=[{'unit':name,'start':0,'end':240,'reference':f'ref{i}'} for i,name in enumerate(names)]
    references={f'ref{i}':refs[name] for i,name in enumerate(names)}
    base=BASE.read_bytes()
    encoded=packet.build(profile,base,coded,references)
    decoded,accounting,players=packet.unpack(encoded)
    if decoded['base']!=base or decoded['keypoints']!=coded:
        raise ValueError('packet section changed')
    with tarfile.open(PREP) as tar:
        checks={}
        for name in names:
            prepared=np.load(io.BytesIO(tar.extractfile('publish/units/'+name+'.npz').read()),allow_pickle=False)
            run=players[name][0]
            if not np.array_equal(run['t'],prepared['t']) or not np.array_equal(run['reference'],prepared['reference']) or not np.allclose(run['boxes'],prepared['box'],rtol=0,atol=1e-10):
                raise ValueError('conditions differ from prepared source')
            pose=prepared['pose']
            mismatches=sum(not np.array_equal(packet.pose_for_tick(run,i),pose[i]) for i in range(240))
            if mismatches:
                raise ValueError('pose raster mismatch')
            checks[name]={'ticks':240,'pose_mismatches':mismatches,'max_abs_box_error':float(np.max(np.abs(run['boxes']-prepared['box']))),'seed':run['seed'],'reference_exact':True}
    OUT.mkdir(exist_ok=False)
    (OUT/'players.pscp').write_bytes(encoded)
    result={'packet_sha256':sha(encoded),'packet_bytes':len(encoded),'fps':[25,1],'frames':240,'coded_size':CODED_SIZE.copy(),'source_sha256':'c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934','base_sha256':sha(base),'base_bytes':len(base),'keypoints_bytes':len(coded),'original_keypoints_bytes':len(kps),'accounting':accounting,'checks':checks,'scope':'CPU packet and source-conditioning parity only; no new common model rendering, source denial, ball or donor-clean quality validation.'}
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

"""On the campus CPU, launch the source-denied worker then verify its hashes outside it."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    src = root / 'source'
    sys.path.insert(0, str(src))
    from experiments.tennis import common_receiver_isolation as iso
    from experiments.tennis import player_requalification as replay
    import numpy as np

    registry = json.loads((root/'linux-carriers.json').read_text())
    frozen = json.loads((root/'prepared-manifest.json').read_text())
    protected = json.loads((root/'denied.json').read_text())
    report = {}
    for q in ('70','40'):
        packet = root/f'q{q}.pscp'
        manifest = root/f'q{q}.manifest'
        units = [r['unit'] for r in frozen[q]['units']]
        config = {'carrier':str(packet),'manifest':str(manifest),'expected_sha256':registry['carriers'][q]['sha256'],
                  'frames':240,'units':units}
        archive = Path('/home/itec/emanuele/Datasets/pointstream-data/tennis/player-reference-pilot-20261117')/f'q{q}/conditioning.tar'
        denied = [Path(s) for s in protected] + [archive]
        receipt = iso.launch(source_root=src,output=root/f'out-q{q}',worker_config=config,
                             inputs=[packet,manifest],denied=denied,seconds=90,kind='conditioning')
        data = archive.read_bytes()
        if hashlib.sha256(data).hexdigest() != frozen[q]['archive_sha256']:
            raise ValueError('external validator archive differs')
        checks = []
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            for row in receipt['outputs']:
                member = tar.extractfile('publish/'+row['unit']+'.npz').read()
                with np.load(io.BytesIO(member),allow_pickle=False) as npz:
                    sums = {k:replay.array_sha(npz[k]) for k in ('pose','reference','box','t')}
                checks.append({'unit':row['unit'],'array_parity':{k:row['array_sha256'][k] == v for k,v in sums.items()},
                               'actual':row['array_sha256'],'expected':sums})
        report[q] = {'worker':receipt,'external_validator':checks}
    with (root/'all.json').open('x') as out:
        json.dump(report,out,indent=2)
        out.write('\n')
    print({q:[x['array_parity'] for x in v['external_validator']] for q,v in report.items()})


if __name__ == '__main__':
    main()

"""Build reviewed-support crops and one numeric reproduction control; no labels at inference."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'starved-base-local-20261122'))
from crops import LABELS, supports  # noqa: E402


def main():
    root=Path('/private/tmp/ps-starved-dcvc-20261122')
    cheap=HERE.parent/'starved-tile-budget-20261122'
    scored=json.loads((cheap/'scores.json').read_text())
    src=Path('/private/tmp/ps-cheaper-tiles-20261122/crops.zip')
    assert hashlib.sha256(src.read_bytes()).hexdigest()==scored['crops_sha256']
    assert hashlib.sha256(LABELS.read_bytes()).hexdigest()=='9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a'
    records=[]
    with zipfile.ZipFile(src) as inputs,zipfile.ZipFile(root/'supports.zip','x') as out:
        for frame in json.loads(LABELS.read_text())['frames']:
            t=frame['tick']
            for region,y,x,_ in supports(frame):
                base=f'{t:03d}/{region}/'
                control=next(r['dists'] for r in scored['rows'] if (r['t'],r['region'],r['arm'])==(t,region,'plain_match'))
                rec=dict(t=t,region=region,rect=[x.start,y.start,x.stop,y.stop],av1_dists=control)
                for name in ('source','mask','plain_match'):
                    out.writestr(base+name+'.png',inputs.read(base+name+'.png'))
                records.append(rec)
        out.writestr('regions.json',json.dumps(records,default=int))
    print(hashlib.sha256((root/'supports.zip').read_bytes()).hexdigest())


if __name__=='__main__':
    main()

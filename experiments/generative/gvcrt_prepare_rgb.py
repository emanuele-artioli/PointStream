"""Export the registered 48-frame uint8 RGB target, with exact identity gates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

FILE_SHA='9234147b68ace7217043c2761701f3bc915a0371db9fc83be6fea3973ae20191'
RGB_SHA='1f02475a5bbc3d94e4bae2e904dc29c3af3082be0c0c160e027b706a6950f6f8'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def prepare(source,output):
    import numpy as np
    from PIL import Image
    source=Path(source).resolve(strict=True)
    if digest(source)!=FILE_SHA:raise ValueError('Registered NPY file identity mismatch')
    rgb=np.load(source,allow_pickle=False)
    if rgb.dtype!=np.uint8 or rgb.shape!=(48,360,640,3):raise ValueError('Registered RGB geometry/type mismatch')
    if hashlib.sha256(rgb.tobytes()).hexdigest()!=RGB_SHA:raise ValueError('Registered RGB pixel identity mismatch')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    frames=[]
    for n,frame in enumerate(rgb):
        path=output/f'frame_{n:06d}.png';Image.fromarray(frame).save(path)
        with Image.open(path) as image:
            if image.mode!='RGB' or image.size!=(640,360) or image.tobytes()!=frame.tobytes():raise ValueError('PNG pixel parity failed')
        frames.append({'path':str(path),'sha256':digest(path),'rgb_pixel_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),'source_frame_index':n})
    manifest=output/'frames.json';manifest.write_text(json.dumps(frames,indent=2,sort_keys=True)+'\n')
    receipt={'source':str(source),'source_file_sha256':FILE_SHA,'source_rgb_sha256':RGB_SHA,'shape':[48,360,640,3],'dtype':'uint8','fps':12,'cadence_status':'registered prepared window declaration','frame_count':48,'png_pixel_parity':True,'frames_json_sha256':digest(manifest),'helper_sha256':digest(__file__),'argv':sys.argv,'numpy':np.__version__}
    (output/'preparation_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();print(json.dumps(prepare(a.source,a.output_dir)))

if __name__=='__main__':main()

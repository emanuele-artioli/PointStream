"""Export the twelve mechanically selected source-only review frames."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import zipfile

import cv2
import numpy as np

from experiments.tennis.k4 import upsample
from experiments.tennis.k4b_pose import player_masks
from src.segmentation.masks import ClipMasks


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def png(frame):
    if frame.shape[:2] != (1080,1920):
        raise ValueError("Source and every proposal must share the native pixel grid")
    ok, data = cv2.imencode('.png', frame)
    if not ok or not np.array_equal(cv2.imdecode(data,cv2.IMREAD_UNCHANGED),frame):
        raise ValueError('PNG round trip failed')
    return data.tobytes()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('source', 'archive', 'track', 'output'):
        p.add_argument('--'+arg, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    source_sha = digest(a.source)
    archive_sha = digest(a.archive)
    track_sha = digest(a.track)
    if (source_sha != 'b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d'
            or archive_sha != '965b8879a821b90dd1e6afba415b8eed269a35a768e1f9891e0f4df96a37a51a'
            or track_sha != '8abbacb923215d2f93b54275eef43cc16ba2c63ec6f46e9c35898a094add7ca6'):
        raise ValueError('Input identity mismatch')
    track = json.loads(a.track.read_text())
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(a.archive) as src:
        prefix = 'publish/nadal_kyrgios__000@120/'
        window = json.load(src.extractfile(prefix+'window.json'))
        for name in ['features.json', 'sam-masks.rle']:
            path = Path(tmp)/name
            path.write_bytes(src.extractfile(prefix+name).read())
            if digest(path) != window['files'][name]:
                raise ValueError(name)
        masks = ClipMasks.load(Path(tmp)/'sam-masks.rle')
        aliases = json.loads((Path(tmp)/'features.json').read_text())['aliases']
    if window['id'] != track['window'] or len(masks.frames) != 240 or len(track['centers']) != 240:
        raise ValueError('Source window mismatch')
    cap = cv2.VideoCapture(str(a.source),cv2.CAP_FFMPEG)
    rows = []
    overview = {'source_sha256':source_sha,'archive_sha256':archive_sha,'old_job_id':'20261031T155454Z-e86a61b6',
                'track_sha256':track_sha,'ticks':list(range(10,240,20)),'unprompted':[10,90,170],
                'canvas':[1920,1080],'fps':[50,1],'geometry':'source-only model proposals on a 960x540 grid, nearest-neighbor expanded to source 1920x1080; all unreviewed'}
    with zipfile.ZipFile(a.output/'source-only-review.zip','x') as z:
        for t in range(240):
            ok,img=cap.read()
            if not ok or img.shape != (1080,1920,3):
                raise ValueError('Native source frame')
            if t not in overview['ticks']:
                continue
            root = f'frame-{t:03d}/'
            z.writestr(root+'source.png',png(img))
            row = dict(tick=t,blind=t in overview['unprompted'],human_status='unreviewed')
            if t not in overview['unprompted']:
                bodies = player_masks(masks.frames[t], aliases, window['players'])
                row['player_proposals'] = []
                for unit in window['players']:
                    if unit in bodies:
                        data = upsample(bodies[unit],(1920,1080)).astype(np.uint8)*255
                        z.writestr(root+unit.replace(':','-')+'.png',png(data))
                        row['player_proposals'].append(unit)
                rackets = [item.mask() for item in masks.frames[t] if item.class_name == 'racket']
                for i,mask in enumerate(rackets):
                    z.writestr(root+f'racket-{i}.png',png(upsample(mask,(1920,1080)).astype(np.uint8)*255))
                row.update(racket_proposals=len(rackets), ball_center_proposal=track['centers'][t])
            rows.append(row)
        cap.release()
        overview['frames']=rows
        overview['instruction']='Inspect native source. Mark visible body/clothing and ball, keep racket separate. Explicitly record uncertainty, confirmed absence and missed proposals. An omitted proposal never implies confirmed absence.'
        z.writestr('review.json',json.dumps(overview,indent=2))
    with zipfile.ZipFile(a.output/'source-only-review.zip') as z:
        if z.testzip() or sum(n.endswith('source.png') for n in z.namelist())!=12:
            raise ValueError('Review archive validation')
    overview['zip_sha256']=digest(a.output/'source-only-review.zip')
    (a.output/'receipt.json').write_text(json.dumps(overview,indent=2)+'\n')
    print(overview['zip_sha256'],len(rows),'source frames')


if __name__ == '__main__':
    main()

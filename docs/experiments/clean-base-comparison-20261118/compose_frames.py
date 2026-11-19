"""Compose labeled frames from authenticated paid foreground and a supplied AV1 base."""
import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'docs/experiments/receiver-replay-20261118'))
from independent_crop_audit import array_frames, file_sha, read_small, require  # noqa: E402
from independent_finish_audit import UNITS, bind_generation, overlay  # noqa: E402
from src.pointstream.tracked_patch import yuv_to_rgb  # noqa: E402

STAGE = Path('/private/tmp/ps-joint-campus-stage-20261118')
AUDIT = Path('/private/tmp/pointstream-independent-idle-nov18')
DATA = Path('/Users/manu/Datasets/pointstream-submission-20261118')
LABELS = DATA / 'foreground-human-reviewed-20261118/9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a.json'
BUNDLE = DATA / 'ps-development-source-refs-20261118-native/source-only-review.zip'
BASE = DATA / 'ps-native-ctl-20261118-full/plain-48/decoded.yuv'
BALL = Path('/private/tmp/ps-low-ball-stage-20261118/ball-input.npz')
OLD_RECEIPT = Path('/Users/manu/.codex/visualizations/2026/11/18/01a1edc1-dd4e-7ab3-b819-fbd88da8db05/matched-player-ball-20261118/receipt.json')
TICKS = [10,30,50,70,90,110,130,150,170,190,210,230]
BASE_SHA = '4f549befc1445ae4baca3a8da55c822261a78b08f98a8b212fc702f81998499b'
BALL_SHA = '3a010a990d7cb434e7027e35216b41ba21a64829c0ad0281998117c4147786fc'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def run(output, clean, original):
    cv2.setNumThreads(1)
    require(file_sha(LABELS) == '9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a', 'labels')
    require(file_sha(BUNDLE) == '5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb', 'source bundle')
    require(file_sha(BASE) == BASE_SHA and file_sha(BALL) == BALL_SHA, 'base or paid ball inputs')
    if clean is not None:
        require(clean.stat().st_size == BASE.stat().st_size, 'clean base geometry')
    original_path = original or BASE
    require(original_path.stat().st_size == BASE.stat().st_size, 'original base geometry')
    identical_original = file_sha(original_path) == BASE_SHA
    inputs = {'original': np.memmap(original_path, np.uint8, 'r', shape=(240,816,960))}
    if clean is not None:
        inputs['clean'] = np.memmap(clean, np.uint8, 'r', shape=(240,816,960))
    labels = json.loads(LABELS.read_text())
    old = json.loads(OLD_RECEIPT.read_text())
    require([f['tick'] for f in labels['frames']] == TICKS, 'review ticks')
    metadata = json.loads((STAGE / 'finish/finish.json').read_text())
    audit = json.loads((STAGE / 'finish-full-audit.json').read_text())
    require(audit['all_frames_exact'] and audit['carrier_sha256'] == metadata['carrier_sha256'], 'finish audit')
    ball = np.load(BALL, allow_pickle=False)
    patches, coords = ball['patches'], ball['coords']
    require(patches.shape == (240,32,32,3) and coords.shape == (240,2), 'ball coverage')
    output.mkdir(parents=True, exist_ok=False)
    for kind in [*inputs, 'source']:
        (output / kind).mkdir()
    rows = []
    with ExitStack() as stack:
        source = stack.enter_context(zipfile.ZipFile(BUNDLE))
        generations = dict(bind_generation(stack, AUDIT / f'player{i}-new-full.tar',
                                             STAGE / f'player{i}-full.json', metadata) for i in (1,2))
        masks = {}
        for unit in UNITS:
            path = STAGE / 'finish' / (unit + '-masks.npz')
            require(file_sha(path) == metadata['mask_files'][unit], 'paid mask identity')
            zipped = stack.enter_context(zipfile.ZipFile(path))
            require(np.array_equal(read_small(zipped, 't'), np.arange(240)), 'mask timeline')
            masks[unit] = stack.enter_context(array_frames(zipped, 'masks', (240,784,512), np.bool_))
        wanted = set(TICKS)
        for tick in range(max(TICKS)+1):
            crops = {u: next(generations[u][1]) for u in UNITS}
            ms = {u: next(masks[u]) for u in UNITS}
            if tick not in wanted:
                continue
            source_bytes = source.read(f'frame-{tick:03d}/source.png')
            label = next(f for f in labels['frames'] if f['tick']==tick)
            require(sha(source_bytes) == label['source_png_sha256'], 'source frame identity')
            (output/'source'/f'{tick:03d}.png').write_bytes(source_bytes)
            row = {'tick': tick, 'source_png_sha256': sha(source_bytes)}
            for kind, video in inputs.items():
                canvas = cv2.resize(yuv_to_rgb(video[tick],960,544,False),(1920,1080),interpolation=cv2.INTER_CUBIC)
                support = np.zeros((1080,1920),bool)
                for unit in UNITS:
                    area, status = overlay(canvas,support,crops[unit],generations[unit][0][tick],ms[unit])
                    require(status == metadata['rows'][tick]['status'][unit] and
                            area == metadata['rows'][tick]['projected_pixels'][unit], 'mask projection')
                if kind == 'original' and identical_original:
                    require(sha(canvas.tobytes()) == old['rows'][tick]['player_rgb_sha256'], 'original player pixels')
                x,y = coords[tick]
                if x >= 0:
                    canvas[y:y+32,x:x+32] = patches[tick]
                if kind == 'original' and identical_original:
                    require(sha(canvas.tobytes()) == old['rows'][tick]['joint_rgb_sha256'], 'original joint pixels')
                target = output/kind/f'{tick:03d}.png'
                Image.fromarray(canvas).save(target)
                row[kind+'_rgb_sha256'] = sha(canvas.tobytes())
                row[kind+'_png_sha256'] = file_sha(target)
            rows.append(row)
    result = {'rows':rows, 'labels_sha256':file_sha(LABELS), 'bundle_sha256':file_sha(BUNDLE),
              'original_base_sha256':file_sha(original_path), 'clean_base_sha256':file_sha(clean) if clean else None,
              'ball_input_sha256':BALL_SHA,'opencv':cv2.__version__,'numpy':np.__version__}
    (output/'composition.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(output),'ticks':len(rows),'clean':bool(clean)}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--clean',type=Path)
    p.add_argument('--original',type=Path)
    a=p.parse_args()
    run(a.output,a.clean,a.original)

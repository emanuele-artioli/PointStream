"""Measure the best body MSE achievable by changing only omitted body pixels."""
import argparse
import base64
from contextlib import ExitStack
from io import BytesIO
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

import cv2
import numpy as np
from PIL import Image

ITEM = Path(__file__).resolve().parent
BASE = ITEM.parent
sys.path.insert(0, str(BASE / 'receiver-replay-20261118'))
from independent_crop_audit import array_frames, file_sha, read_small, require  # noqa: E402
from independent_finish_audit import bind_generation, overlay  # noqa: E402

LABEL_HASH = '9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a'


def read(path):
    return json.loads(path.read_text())


def checked_pixels(path, expected):
    require(file_sha(path) == expected, f'PNG digest: {path.name}')
    with Image.open(path) as image:
        rgb = np.array(image.convert('RGB'))
    require(rgb.shape == (1080, 1920, 3), 'native RGB shape')
    return rgb


def run(args):
    started, cpu = time.monotonic(), time.process_time()
    cv2.setNumThreads(1)
    label_path = args.data / 'foreground-human-reviewed-20261118' / (LABEL_HASH + '.json')
    require(file_sha(label_path) == LABEL_HASH, 'source-only mask digest')
    labels = {r['tick']: r for r in read(label_path)['frames']}
    joint_file = BASE / 'clean-court-regional-screen-20261118/refined-composition.json'
    joint = {r['tick']: r for r in read(joint_file)['rows']}
    native_file = BASE / 'ball-only-screen-20261119/receipt.json'
    native = next(r for r in read(native_file)['rows'] if r['crf'] == 48)
    coverage_file = BASE / 'clean-base-comparison-20261118/mask-coverage.json'
    coverage = {r['tick']: r['players'] for r in read(coverage_file)['rows']}
    meta = read(args.stage / 'finish/finish.json')
    require(sorted(labels) == sorted(joint) == list(range(10, 240, 20)), 'all review ticks')
    rows = []
    zero_crop = np.zeros((784, 512, 3), np.uint8)
    with ExitStack() as stack:
        projections = {}
        for index in (1, 2):
            key, (boxes, _) = bind_generation(stack, args.archives / f'player{index}-new-full.tar',
                    args.stage / f'player{index}-full.json', meta)
            path = args.stage / 'finish' / (key + '-masks.npz')
            require(file_sha(path) == meta['mask_files'][key], 'receiver mask identity')
            archive = stack.enter_context(zipfile.ZipFile(path))
            require(np.array_equal(read_small(archive, 't'), np.arange(240)), 'mask frame order')
            projections[key] = boxes, stack.enter_context(array_frames(archive, 'masks', (240,784,512), np.bool_))
        for tick in range(max(labels) + 1):
            masks = {key: next(series) for key, (_, series) in projections.items()}
            if tick not in labels:
                continue
            filename = f'{tick:03d}.png'
            source = checked_pixels(args.frames/'source'/filename, labels[tick]['source_png_sha256'])
            clean = checked_pixels(args.frames/'clean'/filename, joint[tick]['clean_png_sha256'])
            anchor = checked_pixels(args.native/filename, native['saved'][str(tick)]['sha256'])
            for player in labels[tick]['players']:
                identity = player['id']
                unit = 'nadal_kyrgios__000_at120__person_' + identity.split(':')[1]
                with Image.open(BytesIO(base64.b64decode(player['mask']['png_base64']))) as label:
                    body = np.asarray(label.convert('L')) > 0
                projected = np.zeros(body.shape, bool)
                pixels, status = overlay(np.zeros_like(source), projected, zero_crop, projections[unit][0][tick], masks[unit])
                require(pixels == meta['rows'][tick]['projected_pixels'][unit] and status == meta['rows'][tick]['status'][unit], 'projected mask')
                missing = body & ~projected
                require(int(missing.sum()) == coverage[tick][identity]['omitted_pixels'], 'independent coverage')
                error = np.square(clean[body].astype(np.int32) - source[body])
                native_error = np.square(anchor[body].astype(np.int32) - source[body])
                sse = dict(clean=int(error.sum()), perfect_edge=int(error[~missing[body]].sum()), native=int(native_error.sum()))
                count = int(body.sum())
                require(count == player['mask']['foreground_pixels'] and count > 0, 'body support')
                rows.append(dict(tick=tick, player=identity, body_pixels=count, omitted_pixels=int(missing.sum()), sse=sse))
            require(time.process_time()-cpu < 90 and time.monotonic()-started < 120, 'local budget')
    means = {}
    for identity in sorted({r['player'] for r in rows}):
        data = [r for r in rows if r['player'] == identity]
        require(len(data) == 12, 'paired ticks')
        means[identity] = {arm: dict(mean_frame_mse=float(np.mean([r['sse'][arm]/(3*r['body_pixels']) for r in data])),
                            pooled_mse=sum(r['sse'][arm] for r in data)/(3*sum(r['body_pixels'] for r in data)))
                           for arm in ('native','clean','perfect_edge')}
    receipt = dict(estimand='native RGB body MSE on 12 reviewed ticks of one opened rally', device='CPU',
                   platform=platform.platform(), numpy=np.__version__, opencv=cv2.__version__, script_sha256=file_sha(Path(__file__)),
                   input_receipts={p.name:file_sha(p) for p in (coverage_file,native_file,joint_file,label_path)},
                   clean_base_stream_sha256=next(r['stream_sha256'] for r in read(BASE/'clean-court-grabcut-20261119/receipt.json')['rows'] if r['mode']=='clean'),
                   receiver_mask_sha256=meta['mask_files'], anchor_kbps=native['total_kbps'], rows=rows, means=means,
                   cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-started)
    with args.output.open('x') as out:
        json.dump(receipt, out, indent=2)
        out.write('\n')
    print(json.dumps(means))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/Users/manu/Datasets/pointstream-submission-20261118'))
    parser.add_argument('--frames', type=Path, default=Path('/private/tmp/pointstream-20261119-grabcut-score'))
    parser.add_argument('--native', type=Path, default=Path('/private/tmp/ps-regional-ball-only-frames-20261119/48'))
    parser.add_argument('--stage', type=Path, default=Path('/private/tmp/ps-joint-campus-stage-20261118'))
    parser.add_argument('--archives', type=Path, default=Path('/private/tmp/pointstream-independent-idle-nov18'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())

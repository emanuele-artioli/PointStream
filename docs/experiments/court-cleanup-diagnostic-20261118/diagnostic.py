"""Probe court cleanup with reviewed player masks and neighboring decoded frames."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


BASE_HASH = '4f549befc1445ae4baca3a8da55c822261a78b08f98a8b212fc702f81998499b'
LABEL_HASH = '9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='240 decoded 960x544 YUV420 frames')
    parser.add_argument('--labels', type=Path, required=True, help='reviewed source-frame labels')
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    args = parser.parse_args()
    if sha256(args.base) != BASE_HASH or sha256(args.labels) != LABEL_HASH:
        raise ValueError('input hash mismatch')
    if args.base.stat().st_size != 240 * 816 * 960:
        raise ValueError('unexpected decoded-base size')
    args.output.mkdir(parents=True, exist_ok=False)

    labels = {row['tick']: row for row in json.loads(args.labels.read_text())['frames']}
    yuv = np.memmap(args.base, np.uint8, 'r', shape=(240, 816, 960))
    ticks = (10, 90, 170)
    frames = {tick: cv2.cvtColor(yuv[tick], cv2.COLOR_YUV2BGR_I420) for tick in ticks}
    masks = {}
    for tick in ticks:
        mask = np.zeros((1080, 1920), np.uint8)
        for player in labels[tick]['players']:
            data = base64.b64decode(player['mask']['png_base64'], validate=True)
            decoded = cv2.imdecode(np.frombuffer(data, np.uint8), 0)
            if decoded is None or decoded.shape != mask.shape:
                raise ValueError(f'invalid mask at tick {tick}')
            mask = cv2.bitwise_or(mask, decoded)
        mask = cv2.resize(mask, (960, 544), interpolation=cv2.INTER_NEAREST)
        masks[tick] = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))

    results = []
    for target_tick, donors in ((90, (10, 170)), (10, (90, 170))):
        target = frames[target_tick]
        remaining = masks[target_tick] > 0
        target_pixels = int(remaining.sum())
        clean = target.copy()
        orb = cv2.ORB_create(nfeatures=12000)
        target_keys, target_descriptors = orb.detectAndCompute(target, cv2.bitwise_not(masks[target_tick]))
        matches_by_donor = []
        for tick in donors:
            keys, descriptors = orb.detectAndCompute(frames[tick], cv2.bitwise_not(masks[tick]))
            pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors, target_descriptors, k=2)
            matches = [first for first, second in pairs if first.distance < 0.7 * second.distance]
            source = np.float32([keys[match.queryIdx].pt for match in matches])
            destination = np.float32([target_keys[match.trainIdx].pt for match in matches])
            transform, inliers = cv2.findHomography(source, destination, cv2.RANSAC, 2.5)
            if transform is None or inliers is None:
                raise ValueError(f'could not align tick {tick} to {target_tick}')
            warped = cv2.warpPerspective(frames[tick], transform, (960, 544))
            valid = cv2.warpPerspective(
                cv2.bitwise_not(masks[tick]), transform, (960, 544),
                flags=cv2.INTER_NEAREST, borderValue=0,
            ) > 0
            filled = remaining & valid
            clean[filled] = warped[filled]
            remaining &= ~filled
            matches_by_donor.append({
                'tick': tick, 'matches': len(matches), 'inliers': int(inliers.sum()),
                'filled_pixels': int(filled.sum()), 'homography': transform.tolist(),
            })
        cv2.imwrite(str(args.output / f'tick-{target_tick}-before.png'), target)
        cv2.imwrite(str(args.output / f'tick-{target_tick}-after.png'), clean)
        results.append({
            'target_tick': target_tick, 'target_pixels': target_pixels,
            'unfilled_pixels': int(remaining.sum()), 'donors': matches_by_donor,
        })
    result = {'base_sha256': BASE_HASH, 'labels_sha256': LABEL_HASH,
              'opencv': cv2.__version__, 'numpy': np.__version__, 'results': results}
    (args.output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

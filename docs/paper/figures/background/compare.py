"""Compare court-filling policies on fixed, artificially hidden visible court."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import warnings

import cv2
import numpy as np

DATA = Path('/Users/manu/Datasets/pointstream-submission-20261118/ps-native-ctl-20261118-full/source.yuv')
MASKS = Path('/private/tmp/ps-clean-court-grabcut-20261119/removal-masks.npy')
EXPECTED = '6cae4bd7a46bfa4bd2a7a7ec5c926647ea43d9e7a5366eb96e464eaffe1e603c'
MASK_SHA = 'd7746266853240916e476e57b2d9f233b4c6068d965d0c42d473a99979f9cb28'
RECEIPT = Path('docs/experiments/clean-court-grabcut-20261119/receipt.json')
TICKS = (30, 90, 150, 210)
DONORS = tuple(np.linspace(0, 239, 17).round().astype(int).tolist())
RECTANGLES = ((260, 300, 324, 404), (635, 300, 699, 404))
H, W = 544, 960


def sha(path):
    with path.open('rb') as inp:
        return hashlib.file_digest(inp, 'sha256').hexdigest()


def elapsed_cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def reconstruct(target, target_mask, frames, masks, ticks):
    orb = cv2.ORB_create(nfeatures=8000)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    key_t, des_t = orb.detectAndCompute(target_gray, (~target_mask).astype(np.uint8)*255)
    warped = []
    alignments = []
    for tick in ticks:
        donor = cv2.cvtColor(frames[tick], cv2.COLOR_YUV2BGR_I420)
        gray = cv2.cvtColor(donor, cv2.COLOR_BGR2GRAY)
        key_d, des_d = orb.detectAndCompute(gray, (~masks[tick]).astype(np.uint8)*255)
        if des_d is None or des_t is None:
            raise ValueError('missing ORB descriptors')
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_d, des_t, k=2)
        good = [a for a, b in pairs if a.distance < 0.7*b.distance]
        a = np.float32([key_d[m.queryIdx].pt for m in good])
        b = np.float32([key_t[m.trainIdx].pt for m in good])
        hom, inlier = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
        if hom is None or int(inlier.sum()) < 50:
            raise ValueError('insufficient target-specific alignment')
        image = cv2.warpPerspective(donor, hom, (W, H), flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(masks[tick].astype(np.uint8)*255, hom, (W, H), flags=cv2.INTER_NEAREST,
                                    borderValue=255) == 0
        warped.append((image, valid))
        alignments.append({'tick': tick, 'matches': len(good), 'inliers': int(inlier.sum()),
                           'homography': (hom/hom[2, 2]).tolist()})
    return warped, alignments


def predict(target, region, donors, withhold):
    # Use a separate target image for spatial inpainting; masked values are
    # ignored by OpenCV. Donors exclude the target frame entirely.
    spatial = cv2.inpaint(target, region.astype(np.uint8)*255, 5, cv2.INPAINT_TELEA)
    stack = []
    for donor, valid in donors:
        values = donor.astype(np.float32)
        valid = valid.copy()
        if withhold:
            valid[:, 492:] &= ~region[:, 492:]
        values[~valid] = np.nan
        stack.append(values)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        temporal = np.nanmedian(np.stack(stack), axis=0)
    seen = np.isfinite(temporal).all(axis=2)
    # Apply the production missing-plate policy, while retaining actual
    # unmasked target pixels as fixed boundary conditions.
    combined = target.copy()
    observed = region & seen
    combined[observed] = np.clip(np.rint(temporal[observed]), 0, 255).astype(np.uint8)
    missing = region & ~seen
    combined[missing] = 0
    hybrid = cv2.inpaint(combined, missing.astype(np.uint8)*255, 5, cv2.INPAINT_TELEA)
    return spatial, temporal, hybrid, seen


def error(image, truth, mask):
    d = image[mask].astype(np.float64)-truth[mask].astype(np.float64)
    return {'sse': float(np.sum(d*d)), 'components': int(d.size),
            'mse': float(np.mean(d*d))}


def run(args):
    start, cpu = time.monotonic(), elapsed_cpu()
    cv2.setNumThreads(1)
    assert sha(DATA) == EXPECTED
    info = json.loads(RECEIPT.read_text())
    expected_mask_sha = info['preparation'].get('mask_sha256')
    mask_sha = sha(MASKS)
    assert mask_sha == MASK_SHA
    if expected_mask_sha:
        assert expected_mask_sha == mask_sha
    masks = np.load(MASKS, mmap_mode='r', allow_pickle=False)
    frames = np.memmap(DATA, np.uint8, 'r', shape=(240, 816, 960))
    assert masks.shape == (240,H,W) and masks.dtype == np.bool_
    assert int(masks.sum()) == info['preparation']['masks_pixels']
    rows = []
    for tick in TICKS:
        target = cv2.cvtColor(frames[tick], cv2.COLOR_YUV2BGR_I420)
        region = np.zeros((H,W), np.uint8)
        for x0, y0, x1, y1 in RECTANGLES:
            region[y0:y1,x0:x1] = 255
        region = (region > 0) & (masks[tick] == 0)
        assert region.sum() > 3000
        donors = [d for d in DONORS if d != tick]
        assert tick not in donors
        aligned, alignments = reconstruct(target, masks[tick] | region, frames, masks, donors)
        for condition in ('available', 'half-withheld'):
            spatial, temporal, hybrid, seen = predict(target, region, aligned, condition == 'half-withheld')
            visible, missing = region & seen, region & ~seen
            assert visible.any()
            if condition == 'half-withheld':
                assert missing.any()
            scores = {'spatial_all': error(spatial, target, region),
                      'hybrid_all': error(hybrid, target, region),
                      'spatial_seen': error(spatial, target, visible),
                      'hybrid_seen': error(hybrid, target, visible),
                      'temporal_seen': error(temporal, target, visible)}
            if missing.any():
                scores['spatial_missing'] = error(spatial, target, missing)
                scores['hybrid_missing'] = error(hybrid, target, missing)
            rows.append({'tick':tick, 'condition':condition, 'mask_pixels':int(region.sum()),
                         'covered_pixels':int(visible.sum()), 'scores':scores,
                         'alignment':alignments})
            if args.visual and tick == 90:
                shown = target.copy()
                shown[region] = (20, 20, 20)
                # Private check panel; preserves the source in scratch only.
                tiles = [target, shown, spatial, hybrid]
                cv2.imwrite(str(args.visual / f'background-{condition}.png'),
                            np.concatenate([v[280:420,240:720] for v in tiles], axis=1))
        if tick == TICKS[0]:
            forecast = (elapsed_cpu()-cpu)*len(TICKS)*1.5
            if forecast > 300 or time.monotonic()-start > 90:
                raise RuntimeError(f'smoke forecast exceeds local budget: {forecast:.1f}')
        if elapsed_cpu()-cpu > 300 or time.monotonic()-start > 360:
            raise RuntimeError('local CPU/wall budget exceeded')
    result = {'source_sha256':sha(DATA), 'mask_sha256':mask_sha, 'prior_receipt_sha256':sha(RECEIPT),
              'script_sha256':sha(Path(__file__)), 'cpu_seconds':elapsed_cpu()-cpu,
              'wall_seconds':time.monotonic()-start, 'host':os.uname().nodename,
              'opencv':cv2.__version__, 'numpy':np.__version__, 'rows':rows}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps({'cpu_seconds':result['cpu_seconds'], 'wall_seconds':result['wall_seconds'],
                      'coverage': [round(r['covered_pixels']/r['mask_pixels'],3) for r in rows]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--visual', type=Path)
    run(parser.parse_args())

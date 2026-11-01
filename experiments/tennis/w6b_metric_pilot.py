"""Time the real W1 player judges on retained W6 source/base data, CPU only."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ('anchor', 'transport', 'source', 'base', 'lpips', 'dists', 'scratch', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    started, cpu_started = time.monotonic(), time.process_time()
    import cv2
    import numpy as np
    import torch

    from experiments.tennis import w1_masks
    from experiments.tennis.w6 import plane_rgb
    from experiments.visor.h3b import dists_crops, load_dists, lpips_reduce
    from src.codecs.quality import load_lpips

    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    args.scratch.mkdir(parents=True, exist_ok=False)
    ticks = [0, 120, 237]
    masks = w1_masks.build(args.anchor, args.transport, args.scratch, ticks)
    source = np.memmap(args.source, np.uint8, 'r', shape=(240, 1620, 1920))
    base = np.memmap(args.base, np.uint8, 'r', shape=(240, 408, 480))
    nets = {'lpips': load_lpips(args.lpips, 'cpu'), 'dists': load_dists(args.dists, 'cpu')}
    results = []
    for i, t in enumerate(ticks):
        truth = plane_rgb(source, t, 1920, 1080, torch, 'cpu')
        image = cv2.resize(plane_rgb(base, t, 480, 272, torch, 'cpu'), (1920, 1080), interpolation=cv2.INTER_CUBIC)
        mask = w1_masks.unpack(masks['s'][i])
        ys, xs = np.nonzero(mask)
        crop = (slice(max(0, ys.min() - 64), min(1080, ys.max() + 65)),
                slice(max(0, xs.min() - 64), min(1920, xs.max() + 65)))
        wall0, cpu0 = time.monotonic(), time.process_time()
        lp = lpips_reduce(nets['lpips'], [image], truth, [[mask]], 'cpu')[0][0]
        ds = dists_crops(nets['dists'], [image[crop]], truth[crop], mask[crop], 'cpu')[0]
        results.append({'tick': t, 'lpips': lp, 'dists': ds, 'wall_s': time.monotonic() - wall0,
                        'cpu_s': time.process_time() - cpu0})
        args.out.write_text(json.dumps({'results': results, 'cpu_s': time.process_time() - cpu_started,
                                        'wall_s': time.monotonic() - started}, indent=2) + '\n')


if __name__ == '__main__':
    main()

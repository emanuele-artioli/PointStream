"""W1 CPU job: the fixed foreground masks R and S of the Al window, rebuilt from saved inputs.

The real-base judge scores "fixed-foreground" quality inside S = R plus the pixels the
warp-drawn players cover (k6b_real_base_judge). S depends only on the frozen source mattes
and the coded keypoints, never on a method's pixels, so it is rebuilt once here, checked
against the hashes the original K6 window recorded, and handed to the GPU scorer as packed
bits. No pixel of the source video is read: the drawing is run on a blank canvas, which
changes no mask.
"""
from __future__ import annotations

import argparse
import sys
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time
from fractions import Fraction
from typing import Any

import numpy as np

from experiments.tennis.w1_roi import AL, ANCHOR_SHA, TICKS, TRANSPORT_SHA, WINDOWS, Window, digest


def pack_bits(mask: np.ndarray) -> bytes:
    return np.packbits(mask).tobytes()


def build(anchor: Path, transport: Path, scratch: Path, ticks: list[int], window: Window = AL) -> dict[str, Any]:
    import cv2

    from experiments.tennis import k6, k6b
    from experiments.tennis.k4 import safe
    from experiments.tennis.k6b_rate_streams import decode_keypoints, decode_masks
    from src.segmentation.masks import decompress

    stages = k6b.extract_stages([str(anchor)], scratch / 'anchor')
    native = k6b.k6_window(stages[window.id])
    row = native['row']
    if row['id'] != window.id or row['judged'] != TICKS:
        raise ValueError('K6 window or judged ticks changed')
    with tarfile.open(transport) as tar:
        member = tar.extractfile(window.packet_dir + 'exact-rle.matte')
        keypoints = tar.extractfile(window.packet_dir + 'keypoints.kps')
        if member is None or keypoints is None:
            raise ValueError('transport members missing')
        _, source_masks = decode_masks(member.read())
        kp_meta, _ = decode_keypoints(keypoints.read())
    players = row['players']
    region_all = [{p: cv2.resize(source_masks[t, i].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
                   for i, p in enumerate(players) if source_masks[t, i].any()} for t in range(240)]
    heights = {(p, t): float(np.ptp(np.nonzero(r[p].any(1))[0]) + 1) if p in r and r[p].any() else 1.0
               for t, r in enumerate(region_all) for p in players}
    coded = json.loads(decompress((stages[window.id] / 'keypoints.json.zst').read_bytes()))['decoded']
    decoded = {key: {(name.rsplit('@', 1)[0], int(name.rsplit('@', 1)[1])): np.asarray(v, np.float64)
                     for name, v in values.items()} for key, values in coded.items()}
    blank = np.zeros((1080, 1920, 3), np.uint8)
    gap = max(1, round(k6.REF_GAP_S * float(Fraction(row['fps']))))
    r_bits, s_bits, counts = [], [], []
    for t in ticks:
        r = np.logical_or.reduce([*region_all[t].values(), np.zeros((1080, 1920), bool)])
        _, drawn = k6b.k6_oracle(t, native, k6.variant_names()[-9:], _Blank(blank), region_all, blank,
                                 kp_meta['compositing_order'], row['scored'], heights, decoded, gap)
        s = r.copy()
        for mask in drawn.values():
            s |= mask
        expected = row['scoring_masks'][str(t)]
        if (hashlib.sha256(pack_bits(r)).hexdigest() != expected['r']
                or hashlib.sha256(pack_bits(s)).hexdigest() != expected['s']):
            raise ValueError(f'rebuilt scoring masks differ from the recorded ones at tick {t}')
        r_bits.append(np.frombuffer(pack_bits(r), np.uint8))
        s_bits.append(np.frombuffer(pack_bits(s), np.uint8))
        counts.append((int(r.sum()), int(s.sum())))
    return {'ticks': ticks, 'r': np.stack(r_bits), 's': np.stack(s_bits), 'counts': counts, 'safe': safe(window.id),
            'hashes': {str(t): row['scoring_masks'][str(t)] for t in ticks}}


class _Blank:
    """A source stand-in: every frame is the same blank image (colours never reach a mask)."""

    def __init__(self, image: np.ndarray) -> None:
        self.image = image

    def __getitem__(self, index: int) -> np.ndarray:
        return self.image


def unpack(bits: np.ndarray) -> np.ndarray:
    return np.unpackbits(bits)[:1080 * 1920].reshape(1080, 1920).astype(bool)


def run(a: argparse.Namespace) -> None:
    began = time.monotonic()
    frozen = a.window in ('al', 'med')  # manifest windows (W7) have their own anchor and transport archives, identified by the fleet's staged hashes
    for path, expected in ((a.anchor, ANCHOR_SHA), (a.transport, TRANSPORT_SHA)):
        if frozen and digest(path) != expected:
            raise ValueError('pinned input changed: ' + path.name)
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch / 'publish'
    publish.mkdir()
    ticks = TICKS[:a.judged or None]
    result = build(a.anchor, a.transport, scratch, ticks, WINDOWS[a.window])
    np.savez_compressed(publish / 'masks.npz', ticks=np.asarray(ticks), r=result['r'], s=result['s'])
    summary = {'ticks': ticks, 'counts': result['counts'], 'hashes': result['hashes'], 'matches_k6_record': True,
               'seconds': time.monotonic() - began, 'cpus': len(os.sched_getaffinity(0)),
               'inputs': {'anchor': digest(a.anchor), 'transport': digest(a.transport)},
               'masks_sha256': digest(publish / 'masks.npz')}
    (publish / 'masks.json').write_text(json.dumps(summary, indent=1))
    (Path(os.environ['PS_STAGE_DIR']) / 'masks-summary.json').write_text(json.dumps(summary, indent=1))


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    summary = json.loads((stage / 'masks-summary.json').read_text())
    spec = json.loads((stage.parent / 'spec.json').read_text())
    checks: dict[str, bool] = {'ticks': summary['ticks'] in (TICKS[:3], TICKS),
                               'k6_hashes_matched': summary['matches_k6_record'] and len(summary['hashes']) == len(summary['ticks']),
                               'inputs': all(summary['inputs'][x['name']] == x['sha256'] for x in spec['staged_inputs']
                                             if x['name'] in summary['inputs']),
                               'foreground_nonempty': all(0 < r <= s for r, s in summary['counts'])}
    with tarfile.open(stage / 'published.tar') as tar:
        member = tar.extractfile('publish/masks.npz')
        assert member is not None
        blob = member.read()
        checks['archive_identity'] = hashlib.sha256(blob).hexdigest() == summary['masks_sha256']
        with np.load(io.BytesIO(blob)) as z:
            checks['shape'] = z['r'].shape == (len(summary['ticks']), 1080 * 1920 // 8) and z['s'].shape == z['r'].shape
            checks['s_contains_r'] = bool(((z['r'] & ~z['s']) == 0).all())
            checks['pixel_counts'] = [int(unpack(z['s'][i]).sum()) for i in range(len(z['s']))] == [c[1] for c in summary['counts']]
    record = {'passed': all(checks.values()), 'checks': checks}
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(record))
    if not record['passed']:
        raise SystemExit(1)


def main() -> None:
    from experiments.tennis.w1_roi import preload_windows
    preload_windows(sys.argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--windows-file', type=Path, help='window manifest (window_manifest.py); registers its windows next to al/med')
    p.add_argument('--window-state', type=Path, help='what earlier stages measured per manifest window (w1_roi.window_from_spec)')
    p.add_argument('--validate', action='store_true')
    p.add_argument('--anchor', type=Path)
    p.add_argument('--transport', type=Path)
    p.add_argument('--window', choices=sorted(WINDOWS), default='al')
    p.add_argument('--judged', type=int, default=3, help='first N ticks; 0 means all 80')
    a = p.parse_args()
    if a.validate:
        validate()
    elif a.anchor is None or a.transport is None or a.judged not in (0, 3):
        p.error('anchor, transport and 3 or 0 judged ticks are required')
    else:
        run(a)


if __name__ == '__main__':
    main()

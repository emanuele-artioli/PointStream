"""Rebill fixed H5p 4a composites only for bit-exact replacement masks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from experiments.visor import h5
from src.pointstream.bitstream import leb128


def main() -> int:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--shape', type=Path, action='append', default=[])
    p.add_argument('--decoder', type=Path, action='append', default=[])
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.scores.read_text())
    shapes = {r['item']: r for path in args.shape for r in [json.loads(path.read_text())]}
    decoders = {r['item']: r for path in args.decoder for r in [json.loads(path.read_text())]}
    result: dict[str, Any] = {'inputs': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [args.scores, *args.shape, *args.decoder]}, 'items': {}, 'summary': {}}
    metrics = ('wdists_fill', 'wlpips_fill', 'dists_frame', 'wpsnr_fill')
    for item, selection in sorted(data['selection'].items()):
        chosen = {(pair, coder) for pair, coder, ms in selection if 'wdists_fill' in ms}
        rows = [r for stage in data['scores'] if stage['item'] == item for r in stage['rows']
                if (r['pair'], r['tag'].removeprefix('h5p4-')) in chosen]
        anchors = [r for stage in data['anchors'] if stage['item'] == item for r in stage['rows']
                   if r['codec'] == 'dcvc-venue']
        if len(rows) != len(chosen) or not anchors:
            raise ValueError('incomplete fixed pair or anchor set')
        variants: dict[str, list[dict[str, Any]]] = {'lzma': rows}
        if item in shapes:
            source = shapes[item]
            assert all(r['roundtrip'] and r['accuracy_identical'] for r in source['rows'])
            best = min(source['rows'], key=lambda r: r['bytes'])
            variants['shape'] = []
            for row in rows:
                payload_size = best['bytes']
                size = row['bytes'] - row['breakdown']['mask'] + payload_size + 1 + len(leb128(payload_size))
                variants['shape'].append({**row, 'kbps': size*8/(source['frames']/source['fps'])/1000})
        if item in decoders:
            source = decoders[item]
            lookup = {(r['arm'], r['point']): r for r in source['rows']}
            variants['decoder'] = []
            for row in rows:
                arm = int(row['tag'] == 'h5p4-venue')
                correction = lookup[(arm, row['foreground'])]
                if not correction['exact'] or correction['frames'] != 240:
                    raise ValueError('inexact or partial correction cannot reuse quality')
                payload_size = correction['bytes']
                size = row['bytes'] - row['breakdown']['mask'] + payload_size + 1 + len(leb128(payload_size))
                variants['decoder'].append({**row, 'kbps': size*8/(source['frames']/source['fps'])/1000})
        result['items'][item] = {v: {m: h5.compare(h5.curve(points, m), h5.curve(anchors, m), h5.METRICS[m][3])
                                   for m in metrics} for v, points in variants.items()}
    for variant in ('lzma', 'shape', 'decoder'):
        for metric in metrics:
            per = [r[variant][metric] for r in result['items'].values() if variant in r]
            bds = [r['bd'] for r in per if r['bd'] is not None]
            result['summary'][variant+'|'+metric] = {'items': len(per), 'items_below_zero': sum(h5.below_zero(r) for r in per),
                                                    'mean_bd': float(np.mean(bds)) if bds else None}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['summary'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

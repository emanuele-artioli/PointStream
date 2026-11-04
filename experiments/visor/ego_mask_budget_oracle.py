"""Fixed-composite H5p mask-rate ceiling; see ego-peer-feasibility-20261104.md.

The zero and 2 kbps scenarios are idealized total carrier costs. The shape
scenario uses actual exact-roundtrip payload sizes and container framing. All
three retain the same 4a composites and therefore do not bound changed masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from experiments.background import g5
from experiments.visor import h5
from src.pointstream import bitstream
from src.pointstream.bitstream import leb128

BUNDLE_SHA = '383b34b10c838b883e21308e22344cdd28dba4c0eb404d3e12734a52719a0565'
ROOT = 'h5p4d-report-2026-10-24/'
ITEMS = ('P01_107_0000003049', 'P02_02_0000006080',
         'P03_10_0000000616', 'P09_106_0000006303')
METRICS = ('wdists_fill', 'wlpips_fill')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_member(bundle: tarfile.TarFile, name: str) -> Any:
    member = bundle.getmember(ROOT + name)
    if not member.isfile() or member.size > 20_000_000:
        raise ValueError('invalid report member')
    stream = bundle.extractfile(member)
    if stream is None:
        raise ValueError('missing report member')
    return json.load(stream)


def points(data: dict[str, Any], item: str) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    chosen = {(pair, coder) for pair, coder, metrics in data['selection'][item] if 'wdists_fill' in metrics}
    rows = [r for stage in data['scores'] if stage['item'] == item for r in stage['rows']
            if (r['pair'], r['tag'].removeprefix('h5p4-')) in chosen]
    anchors = [r for stage in data['anchors'] if stage['item'] == item for r in stage['rows']]
    venue = [row for row in anchors if row['codec'] == 'dcvc-venue']
    if not chosen or len(rows) != len(chosen) or len(venue) < 2 or len(anchors) < 2:
        raise ValueError(f'incomplete points or anchors for {item}')
    return rows, {'venue': venue, 'pooled': anchors}


def rebill(rows: list[dict[str, Any]], shape: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    if shape.get('frames') != 240 or shape.get('fps', 0) <= 0 or shape.get('scale') != 8:
        raise ValueError('incorrect shape configuration')
    methods = shape.get('rows', [])
    if len(methods) != 3 or any(not r.get('roundtrip') or not r.get('accuracy_identical') for r in methods):
        raise ValueError('shape payloads do not have exact-roundtrip receipts')
    size = min(r['bytes'] for r in methods)
    if type(size) is not int or size <= 0:
        raise ValueError('invalid shape payload size')
    duration = shape['frames'] / shape['fps']
    result = []
    for row in rows:
        total = row['bytes']
        mask = row['breakdown']['mask']
        if type(total) is not int or type(mask) is not int or not 0 < mask < total:
            raise ValueError('invalid physical byte accounting')
        if not math.isclose(row['kbps'], total * 8 / duration / 1000, rel_tol=1e-7):
            raise ValueError('row rate disagrees with byte count and cadence')
        base = (total - mask) * 8 / duration / 1000
        costs = {'free': 0.0, 'ideal_2kbps': 2.0,
                 'shape': (size + 1 + len(leb128(size))) * 8 / duration / 1000}
        result.append({**row, 'kbps': base + costs[variant]})
    return result


def run(bundle_path: Path) -> dict[str, Any]:
    if digest(bundle_path) != BUNDLE_SHA:
        raise ValueError('report archive SHA-256 mismatch')
    with tarfile.open(bundle_path, 'r:gz') as bundle:
        data = read_member(bundle, 'h5p-scoring-inputs.json')
        shapes = {item: read_member(bundle, f'h5p-cpu-{item}.json') for item in ITEMS}
    if set(data['selection']) != set(ITEMS) or any(shapes[x].get('item') != x for x in ITEMS):
        raise ValueError('report item identities mismatch')
    items: dict[str, Any] = {}
    for item in ITEMS:
        rows, anchors = points(data, item)
        comparisons: dict[str, Any] = {}
        for variant in ('free', 'ideal_2kbps', 'shape'):
            candidate = rebill(rows, shapes[item], variant)
            comparisons[variant] = {anchor: {metric: h5.compare(
                h5.curve(candidate, metric), h5.curve(anchor_rows, metric), h5.METRICS[metric][3])
                for metric in METRICS} for anchor, anchor_rows in anchors.items()}
        items[item] = {'selected_pairs': len(rows), 'shape_bytes': min(r['bytes'] for r in shapes[item]['rows']),
                       'shape_source_sha256': shapes[item]['source_sha256'], 'comparisons': comparisons}
    summary: dict[str, Any] = {}
    for variant in ('free', 'ideal_2kbps', 'shape'):
        for anchor in ('venue', 'pooled'):
            for metric in METRICS:
                cells = [items[item]['comparisons'][variant][anchor][metric] for item in ITEMS]
                values = [cell['bd'] for cell in cells if cell['bd'] is not None]
                mean = float(np.mean(values)) if len(values) == len(ITEMS) else None
                wins = sum(h5.below_zero(cell) for cell in cells)
                summary[f'{variant}|{anchor}|{metric}'] = {
                    'mean_bd': mean, 'items_with_bd': len(values), 'items_below_zero': wins,
                    'passes_dists_rule': metric == 'wdists_fill' and mean is not None and mean <= -0.10 and wins >= 3}
    overall = {variant: all(summary[f'{variant}|{anchor}|wdists_fill']['passes_dists_rule']
                            for anchor in ('venue', 'pooled'))
               for variant in ('free', 'ideal_2kbps', 'shape')}
    revision = subprocess.run(['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'],
                              check=True, capture_output=True, text=True).stdout.strip()
    return {'schema': 1, 'input_archive_sha256': BUNDLE_SHA, 'items': items, 'summary': summary,
            'overall_dists_rule': overall, 'git_revision': revision,
            'source_sha256': {'oracle': digest(Path(__file__)), 'h5': digest(Path(h5.__file__)),
                              'g5': digest(Path(g5.__file__)), 'bitstream': digest(Path(bitstream.__file__))},
            'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'scope': 'Fixed reduced masks, fixed H5p 4a composites and selected pairs; ideal costs are not actual streams.'}


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform == 'linux' and len(os.sched_getaffinity(0)) != 1:
        raise ValueError('one-CPU affinity required')
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    result = run(args.bundle)
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'summary': result['summary']}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

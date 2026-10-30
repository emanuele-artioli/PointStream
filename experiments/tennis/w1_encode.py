"""W1 CPU job: SVT-AV1 encodes of the Al window for one resolution, ROI offsets 0/-20/-40.

For each offset a rate-only search (`w1_roi.next_crf`) finds the adjacent CRFs whose rates
bracket the hybrid's 22.737962 kbps and one neighbour; every encode is a full 240-frame
encode, so the search is the preflight and the selected three are final. No quality is
read here. Streams, their rates and the decoded-file hashes are published for the scorer.
With `--controls` the plain CRF 63 bases of scale 8/4/2 are re-encoded and must match the
bytes of the existing anchors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis.w1_roi import (AL, CODED, CONTROL_CRF, FRAMES, OFFSETS, TARGET_KBPS, TOOLS, TRANSPORT_SHA,
                                       WINDOWS, Window, digest, next_crf, policy_name, rate_kbps, roi_text,
                                       select_crfs, start_crf)


def encode_point(raw: Path, work: Path, *, scale: int, frames: int, crf: int, regions: np.ndarray | None,
                 offset: int, cpus: list[int], outside: bool = False, window: Window = AL) -> dict[str, Any]:
    """One encode and an independent dav1d decode; the decoded file is hashed and removed."""
    from src.codecs import svtav1

    width, height = CODED[scale]
    work.mkdir(parents=True)
    extra: list[str] = []
    roi_sha = None
    if offset:
        assert regions is not None
        roi = work / 'roi.txt'
        roi.write_text(roi_text(regions[:frames], width, height, offset, outside))
        roi_sha = digest(roi)
        extra = ['--roi-map-file', str(roi)]
    started = time.monotonic()
    rec = svtav1.code(raw, work, width=width, height=height, fps=Fraction(window.fps), frames=frames, crf=crf, preset=4,
                      threads=min(4, len(cpus)), full_range=False, encoder='SvtAv1EncApp', decoder='dav1d',
                      timeout=2400, cpus=cpus, extra=extra)
    stream, decoded = Path(rec['stream']), Path(rec['decoded'])
    again = work / 'fresh.yuv'
    subprocess.run(svtav1.decode_command('dav1d', stream, again, threads=1), check=True, capture_output=True, timeout=600)
    decoded_sha = digest(decoded)
    if digest(again) != decoded_sha:
        raise ValueError('independent dav1d decode disagrees')
    sizes = svtav1.ivf_frames(stream.read_bytes())
    if len(sizes) != frames or decoded.stat().st_size != frames * width * height * 3 // 2:
        raise ValueError('frame count or decoded size wrong')
    decoded.unlink()
    again.unlink()
    nbytes = stream.stat().st_size
    return {'scale': scale, 'offset': offset, 'crf': crf, 'size': [width, height], 'frames': frames,
            'outside': outside, 'ivf_bytes': nbytes, 'payload_bytes': sum(sizes), 'rate_kbps': rate_kbps(nbytes, frames, window.fps),
            'stream_sha256': digest(stream), 'decoded_sha256': decoded_sha, 'roi_sha256': roi_sha,
            'encode_seconds': rec['encode_seconds'], 'encode_cpu_seconds': rec['encode_cpu_seconds'],
            'decode_seconds': rec['decode_seconds'], 'cpus': cpus, 'encode_command': rec['encode_command'],
            'decode_command': rec['decode_command'], 'wall_seconds': time.monotonic() - started,
            '_stream': str(stream)}


class Ledger:
    """Finished encodes, published as they complete so a stop loses only the item in progress."""

    def __init__(self, publish: Path) -> None:
        self.publish = publish
        (publish / 'streams').mkdir(parents=True)
        self.rows: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def add(self, row: dict[str, Any], name: str) -> dict[str, Any]:
        from experiments.jobs.monitor import publish_progress

        row = {**row}
        source = Path(row.pop('_stream'))
        target = self.publish / 'streams' / f'{name}.ivf'
        target.write_bytes(source.read_bytes())
        source.unlink()
        row['stream'] = f'streams/{name}.ivf'
        with self.lock:
            self.rows.append(row)
            (self.publish / 'encodes.json').write_text(json.dumps(self.rows, indent=1))
            publish_progress(os.environ['PS_STAGE'], len(self.rows))
        return row


def search_offset(raw: Path, work: Path, ledger: Ledger, *, scale: int, offset: int, frames: int, regions: np.ndarray,
                  cpus: list[int], max_encodes: int, target: float, outside: bool = False, window: Window = AL) -> dict[str, Any]:
    known: dict[int, float] = {}
    order: list[int] = []
    status = 'budget'
    while True:
        status, crf = next_crf(known, target, start_crf(scale, offset, outside, window))
        if crf is None:
            break
        if len(known) >= max_encodes:
            status = 'budget'
            break
        row = encode_point(raw, work / f'crf{crf}', scale=scale, frames=frames, crf=crf, regions=regions,
                           offset=offset, cpus=cpus, outside=outside, window=window)
        known[crf] = row['rate_kbps']
        order.append(crf)
        ledger.add(row, f'{policy_name(scale, offset)}_crf{crf}')
    return {'scale': scale, 'offset': offset, 'outside': outside, 'status': status, 'order': order,
            'rates': {str(c): known[c] for c in sorted(known)}, 'selected': select_crfs(known, target)}


def controls(raw_full: Path, scratch: Path, ledger: Ledger, frames: int, cpus: list[int],
             window: Window = AL) -> list[dict[str, Any]]:
    from experiments.tennis.k6 import downscaled_source

    out = []
    for scale in (8, 4, 2):
        raw = downscaled_source(raw_full, frames, scale, scratch / f'control-s{scale}.yuv')
        row = encode_point(raw, scratch / f'control-s{scale}', scale=scale, frames=frames, crf=CONTROL_CRF,
                           regions=None, offset=0, cpus=cpus, window=window)
        raw.unlink()
        row = ledger.add(row, f'control_s{scale}_crf{CONTROL_CRF}')
        row['control_expected_bytes'] = window.control_ivf_bytes[scale]
        row['control_bytes_match'] = row['ivf_bytes'] == window.control_ivf_bytes[scale] if frames == FRAMES else None
        out.append(row)
    return out


def run(a: argparse.Namespace) -> None:
    from experiments.tennis.k4b_features import convert_command
    from experiments.tennis.k6 import downscaled_source
    from experiments.tennis.k6b_rate_streams import decode_masks
    from src.codecs import svtav1

    began = time.monotonic()
    window = WINDOWS[a.window]
    for path, expected in ((a.al, window.source_sha), (a.transport, TRANSPORT_SHA)):
        if digest(path) != expected:
            raise ValueError('pinned input changed: ' + path.name)
    tools = {name: svtav1.tool(name) for name in TOOLS}
    if any(tools[n]['sha256'] != d for n, d in TOOLS.items()):
        raise ValueError('pinned native tool changed')
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch / 'publish'
    publish.mkdir()
    cpus = sorted(os.sched_getaffinity(0))
    frames = a.frames
    full = scratch / 'source-full.yuv'
    command = convert_command('ffmpeg', a.al, full, window.source_start, frames)
    subprocess.run(command, check=True, capture_output=True, timeout=900)
    if frames == FRAMES and digest(full) != window.source_yuv_sha:
        raise ValueError('source conversion changed')
    with tarfile.open(a.transport) as tar:
        member = tar.extractfile(window.packet_dir + 'exact-rle.matte')
        assert member is not None
        _, masks = decode_masks(member.read())
    if masks.shape != (240, 2, 540, 960) or masks.dtype != bool:
        raise ValueError('transport matte changed')
    regions = masks.any(axis=1)[:frames]
    del masks
    raw = full if a.scale == 1 else downscaled_source(full, frames, a.scale, scratch / f'source-s{a.scale}.yuv')
    ledger = Ledger(publish)
    offsets = tuple(a.offsets)
    workers = len(offsets)
    lanes = workers + (1 if a.controls else 0)
    share = max(1, len(cpus) // lanes)
    cpu_sets = [cpus[i * share:(i + 1) * share] or cpus for i in range(lanes)]
    searches: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers + (1 if a.controls else 0)) as pool:
        futures = [pool.submit(search_offset, raw, scratch / policy_name(a.scale, o), ledger, scale=a.scale, offset=o,
                               frames=frames, regions=regions, cpus=cpu_sets[i], max_encodes=a.max_encodes,
                               target=a.target_kbps, outside=a.convention == 'outside', window=window) for i, o in enumerate(offsets)]
        if a.controls:
            control_future = pool.submit(controls, full, scratch, ledger, frames, cpu_sets[-1], window)
        searches = [f.result() for f in futures]
        if a.controls:
            control_rows = control_future.result()
    seconds = time.monotonic() - began
    summary = {'window': window.name, 'scale': a.scale, 'convention': a.convention, 'offsets': list(offsets), 'frames': frames, 'max_encodes': a.max_encodes, 'target_kbps': a.target_kbps,
               'searches': searches, 'controls': control_rows, 'tools': tools, 'cpus': cpus, 'cpu_sets': cpu_sets,
               'source_command': command, 'host': os.uname().nodename, 'seconds': seconds,
               'encodes': len(ledger.rows), 'inputs': {window.name: digest(a.al), 'transport': digest(a.transport)},
               'scope': f'rate-only search; no quality read; {window.name} window of 240 frames, preset 4, keyint -1'}
    (publish / 'encode.json').write_text(json.dumps(summary, indent=1))
    (Path(os.environ['PS_STAGE_DIR']) / 'encode-summary.json').write_text(json.dumps(summary, indent=1))


def projection(summary: dict[str, Any], full_encodes: int, full_seconds: float) -> dict[str, Any]:
    """Cost of the full stage from the smoke: per-frame encode seconds at the smoke's operating
    points, scaled to 240 frames, for the longest search chain, inflated 35%."""
    per_frame = max(r['wall_seconds'] / r['frames'] for r in summary['rows'])
    chain = per_frame * FRAMES * full_encodes
    setup = summary['setup_seconds']
    projected = 1.35 * (setup + chain + (summary['control_seconds'] * 10 if summary['has_controls'] else 0))
    return {'projected_full_seconds': projected, 'per_frame_seconds': per_frame, 'under_allowance': projected < full_seconds}


def validate() -> None:
    from src.codecs import svtav1

    stage = Path(os.environ['PS_STAGE_DIR'])
    summary = json.loads((stage / 'encode-summary.json').read_text())
    spec = json.loads((stage.parent / 'spec.json').read_text())
    frames = summary['frames']
    smoke = frames != FRAMES
    fps = WINDOWS[summary.get('window', 'al')].fps
    target = summary['target_kbps']
    checks: dict[str, bool] = {
        'frames': frames in (24, FRAMES) and spec['scale']['frames'] == {'smoke': 24, 'full': FRAMES},
        'multi_cpu': len(summary['cpus']) >= 3 * 2 and all(len(c) >= 2 for c in summary['cpu_sets']),
        'tools_pinned': all(summary['tools'][n]['sha256'] == d for n, d in TOOLS.items()),
        'inputs': all(summary['inputs'][x['name']] == x['sha256'] for x in spec['staged_inputs'] if x['name'] in summary['inputs']),
        'offsets': [s['offset'] for s in summary['searches']] == summary['offsets'] and set(summary['offsets']) <= set(OFFSETS)}
    rows: list[dict[str, Any]] = []
    with tarfile.open(stage / 'published.tar') as tar:
        member = tar.extractfile('publish/encodes.json')
        assert member is not None
        rows = json.load(member)
        ok = len(rows) == summary['encodes'] and bool(rows)
        for row in rows:
            blob = tar.extractfile('publish/' + row['stream'])
            assert blob is not None
            data = blob.read()
            ok &= hashlib.sha256(data).hexdigest() == row['stream_sha256'] and len(data) == row['ivf_bytes']
            ok &= len(svtav1.ivf_frames(data)) == frames and row['frames'] == frames
            ok &= abs(row['rate_kbps'] - rate_kbps(len(data), frames, fps)) < 1e-9 and row['rate_kbps'] > 0
            ok &= len(row['decoded_sha256']) == 64
        checks['streams_match_rows'] = ok
    by = {(r['offset'], r['crf']): r['rate_kbps'] for r in rows if r['scale'] == summary['scale']}
    first = {s['offset']: s['order'][0] for s in summary['searches'] if s['order']}
    same = len(set(first.values())) == 1
    inside_full = summary['convention'] == 'inside' and set(first) == set(OFFSETS)
    checks['roi_raises_rate'] = (not inside_full) or (not same) or (by[(-40, first[-40])] > by[(0, first[0])] and by[(-20, first[-20])] >= by[(0, first[0])])
    for s in summary['searches']:
        known = {int(c): r for c, r in s['rates'].items()}
        checks[f"offset{s['offset']}:search_consistent"] = (
            s['selected'] == select_crfs(known, target)
            and all(known[c] == by[(s['offset'], c)] for c in known))
        if not smoke:
            checks[f"offset{s['offset']}:status"] = s['status'] in ('complete', 'unbracketed_high', 'unbracketed_low', 'budget')
            checks[f"offset{s['offset']}:selected_bracket"] = (s['status'] != 'complete') or (
                s['selected'] is not None and len(s['selected']) == 3
                and any(known[c] >= target > known[c + 1] for c in s['selected'] if c + 1 in s['selected']))
    if summary['controls']:
        checks['controls_scales'] = [c['scale'] for c in summary['controls']] == [8, 4, 2]
        if not smoke:
            checks['controls_bytes_match_existing_anchors'] = all(c['control_bytes_match'] for c in summary['controls'])
    if smoke:
        encode_rows = [r for r in rows if r['crf'] != CONTROL_CRF or r['offset'] or r['scale'] == summary['scale']]
        full_encodes = spec['scale'].get('max_encodes', {}).get('full', 8)
        timing = {'rows': encode_rows, 'setup_seconds': summary['seconds'] - max(r['wall_seconds'] for r in rows),
                  'control_seconds': sum(c['wall_seconds'] for c in summary['controls']),
                  'has_controls': bool(summary['controls'])}
        proj = projection(timing, full_encodes, spec['full']['seconds'])
        checks['projection'] = proj['under_allowance']
    else:
        proj = None
    record = {'passed': all(checks.values()), 'checks': checks, 'projection': proj}
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(record, indent=1))
    if not record['passed']:
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--al', '--source', dest='al', type=Path, help='the window\'s source video')
    p.add_argument('--window', choices=sorted(WINDOWS), default='al')
    p.add_argument('--target-kbps', type=float, default=TARGET_KBPS, help='the rate the CRF search brackets (the hybrid\'s total)')
    p.add_argument('--transport', type=Path)
    p.add_argument('--scale', type=int, choices=(1, 2, 4), default=2)
    p.add_argument('--frames', type=int, choices=(24, FRAMES), default=24)
    p.add_argument('--max-encodes', type=int, default=2)
    p.add_argument('--convention', choices=('inside', 'outside'), default='inside',
                   help='outside: the relative offset as 0 in the blocks R touches and +|offset| elsewhere')
    p.add_argument('--offsets', type=int, nargs='+', default=list(OFFSETS))
    p.add_argument('--controls', action='store_true', help='also re-encode the plain CRF 63 bases of scale 8/4/2')
    a = p.parse_args()
    if a.validate:
        validate()
    elif a.al is None or a.transport is None:
        p.error('the Al source and the transport archive are required')
    else:
        run(a)


if __name__ == '__main__':
    main()

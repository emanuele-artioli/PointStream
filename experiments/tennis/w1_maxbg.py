"""W1 CPU job: the strongest SVT allocation, background at the encoder's maximum quantizer.

Step 1 (mode probe, half resolution): find an encoder mode in which a negative QP offset inside
R raises the quality of R's blocks while the background stays put (the default CRF mode with
aq-mode 2 ignores it). Pre-registered criterion: offset -40 raises the luma PSNR inside the
touched 64x64 blocks by at least 1.0 dB over offset 0 and moves the PSNR outside them by at
most 0.3 dB. Modes are tried in the order of `MODES`; the first that passes is used. If none
passes the job records that and stops.
Step 2 (rate-only search): background at quantizer 70, offset -(70 - k) inside R, k found by the
same bracketing search as the CRF (`next_crf` over k in 7..70); half and quarter resolution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis.w1_encode import Ledger, controls
from experiments.tennis.w1_roi import (CODED, FPS, FRAMES, MAXBG, MAXBG_BASE_Q, PACKET_DIR, SOURCE_SHA, SOURCE_START,
                                       SOURCE_YUV_SHA, TARGET_KBPS, TOOLS, TRANSPORT_SHA, digest, maxbg_roi_value,
                                       next_crf, policy_name, rate_kbps, roi_text, select_crfs)
from experiments.tennis.w1_roicheck import region_psnr

MODES = ('aq0', 'cqp')
PROBE = 98  # offset label of the probe rows (never selected for scoring)
K_LOW = MAXBG_BASE_Q - 63
MIN_GAIN_DB = 1.0
MAX_BACKGROUND_DB = 0.3
START_K = {2: 44, 4: 44}


def command(mode: str, raw: Path, stream: Path, width: int, height: int, frames: int, roi: Path | None) -> list[str]:
    from src.codecs import svtav1

    extra = ['--aq-mode', '0'] + (['--roi-map-file', str(roi)] if roi else [])
    cmd = svtav1.encode_command('SvtAv1EncApp', raw, stream, width=width, height=height, fps=Fraction(FPS), frames=frames,
                                crf=MAXBG_BASE_Q, preset=4, full_range=False, extra=extra)
    if mode == 'cqp':
        i = cmd.index('--crf')
        cmd[i:i + 2] = ['--cqp', str(MAXBG_BASE_Q)]
    return cmd


def encode_variant(raw: Path, work: Path, *, scale: int, frames: int, mode: str, k: int, regions: np.ndarray,
                   cpus: list[int], label: int = MAXBG) -> dict[str, Any]:
    from src.codecs import svtav1

    width, height = CODED[scale]
    work.mkdir(parents=True)
    value = maxbg_roi_value(k)
    roi = None
    if value:
        roi = work / 'roi.txt'
        roi.write_text(roi_text(regions[:frames], width, height, value))
    stream, decoded = work / 'stream.ivf', work / 'decoded.yuv'
    cmd = command(mode, raw, stream, width, height, frames, roi)
    started = time.monotonic()
    timing = svtav1.run(cmd, 2400, cpus)
    dec = svtav1.decode_command('dav1d', stream, decoded, threads=2)
    svtav1.run(dec, 600, cpus)
    again = work / 'fresh.yuv'
    subprocess.run(svtav1.decode_command('dav1d', stream, again, threads=1), check=True, capture_output=True, timeout=600)
    data = stream.read_bytes()
    sizes = svtav1.ivf_frames(data)
    decoded_sha = digest(decoded)
    if digest(again) != decoded_sha or len(sizes) != frames or decoded.stat().st_size != frames * width * height * 3 // 2:
        raise ValueError('decode check failed')
    psnr = region_psnr(raw, decoded, regions, width, height, frames)
    decoded.unlink()
    again.unlink()
    return {'scale': scale, 'offset': label, 'crf': k, 'mode': mode, 'base_q': MAXBG_BASE_Q, 'roi_value': value,
            'size': [width, height], 'frames': frames, 'ivf_bytes': len(data), 'payload_bytes': sum(sizes),
            'rate_kbps': rate_kbps(len(data), frames), 'stream_sha256': hashlib.sha256(data).hexdigest(),
            'decoded_sha256': decoded_sha, 'roi_sha256': digest(roi) if roi else None, 'psnr': psnr,
            'encode_seconds': timing['wall'], 'encode_command': cmd, 'wall_seconds': time.monotonic() - started,
            '_stream': str(stream)}


def probe(raw: Path, scratch: Path, ledger: Ledger, regions: np.ndarray, frames: int, cpu_sets: list[list[int]]) -> dict[str, Any]:
    """Offsets 0, -20, -40 at quantizer 70 in each mode; the first mode meeting the criterion wins."""
    ks = (70, 50, 30)
    with ThreadPoolExecutor(max_workers=len(cpu_sets)) as pool:
        jobs = {(mode, k): pool.submit(encode_variant, raw, scratch / f'probe-{mode}-{k}', scale=2, frames=frames, mode=mode, k=k,
                                       regions=regions, cpus=cpu_sets[i % len(cpu_sets)], label=PROBE)
                for i, (mode, k) in enumerate((m, k) for m in MODES for k in ks)}
        rows = {key: job.result() for key, job in jobs.items()}
    table = []
    chosen = None
    for mode in MODES:
        plain, strong = rows[(mode, 70)]['psnr'], rows[(mode, 30)]['psnr']
        gain = strong['in_blocks'] - plain['in_blocks']
        background = strong['out_blocks'] - plain['out_blocks']
        passes = bool(gain >= MIN_GAIN_DB and abs(background) <= MAX_BACKGROUND_DB)
        table.append({'mode': mode, 'gain_in_blocks_db': gain, 'background_change_db': background, 'passes': passes,
                      'rows': [{'k': k, 'roi_value': rows[(mode, k)]['roi_value'], 'ivf_bytes': rows[(mode, k)]['ivf_bytes'],
                                'rate_kbps': rows[(mode, k)]['rate_kbps'], 'psnr': rows[(mode, k)]['psnr']} for k in ks]})
        if passes and chosen is None:
            chosen = mode
    for key, row in rows.items():
        ledger.add(row, f'probe_{key[0]}_k{key[1]}')
    return {'criterion': {'min_gain_db': MIN_GAIN_DB, 'max_background_change_db': MAX_BACKGROUND_DB}, 'modes': table, 'chosen': chosen}


def search(raw: Path, work: Path, ledger: Ledger, regions: np.ndarray, *, scale: int, frames: int, mode: str,
           cpus: list[int], max_encodes: int) -> dict[str, Any]:
    known: dict[int, float] = {}
    order: list[int] = []
    status = 'budget'
    while True:
        status, k = next_crf(known, TARGET_KBPS, START_K[scale], lo=K_LOW, hi=MAXBG_BASE_Q)
        if k is None:
            break
        if len(known) >= max_encodes:
            status = 'budget'
            break
        row = encode_variant(raw, work / f'k{k}', scale=scale, frames=frames, mode=mode, k=k, regions=regions, cpus=cpus)
        known[k] = row['rate_kbps']
        order.append(k)
        ledger.add(row, f'{policy_name(scale, MAXBG)}_k{k}')
    return {'scale': scale, 'offset': MAXBG, 'mode': mode, 'status': status, 'order': order,
            'rates': {str(c): known[c] for c in sorted(known)}, 'selected': select_crfs(known, TARGET_KBPS)}


def run(a: argparse.Namespace) -> None:
    from experiments.tennis.k4b_features import convert_command
    from experiments.tennis.k6 import downscaled_source
    from experiments.tennis.k6b_rate_streams import decode_masks
    from src.codecs import svtav1

    began = time.monotonic()
    for path, expected in ((a.al, SOURCE_SHA), (a.transport, TRANSPORT_SHA)):
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
    cmd = convert_command('ffmpeg', a.al, full, SOURCE_START, frames)
    subprocess.run(cmd, check=True, capture_output=True, timeout=900)
    if frames == FRAMES and digest(full) != SOURCE_YUV_SHA:
        raise ValueError('source conversion changed')
    with tarfile.open(a.transport) as tar:
        member = tar.extractfile(PACKET_DIR + 'exact-rle.matte')
        assert member is not None
        _, masks = decode_masks(member.read())
    regions = masks.any(axis=1)[:frames]
    raws = {2: downscaled_source(full, frames, 2, scratch / 'source-s2.yuv'), 4: downscaled_source(full, frames, 4, scratch / 'source-s4.yuv')}
    ledger = Ledger(publish)
    lanes = 6
    share = max(1, len(cpus) // lanes)
    cpu_sets = [cpus[i * share:(i + 1) * share] or cpus for i in range(lanes)]
    verdict = probe(raws[2], scratch, ledger, regions, frames, cpu_sets)
    searches: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    if verdict['chosen']:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(search, raws[s], scratch / f'search-s{s}', ledger, regions, scale=s, frames=frames,
                                   mode=verdict['chosen'], cpus=cpu_sets[i], max_encodes=a.max_encodes) for i, s in enumerate((2, 4))]
            control_future = pool.submit(controls, full, scratch, ledger, frames, cpu_sets[2])
            searches = [f.result() for f in futures]
            control_rows = control_future.result()
    summary = {'scale': 2, 'frames': frames, 'max_encodes': a.max_encodes, 'target_kbps': TARGET_KBPS, 'probe': verdict,
               'searches': searches, 'controls': control_rows, 'tools': tools, 'cpus': cpus, 'host': os.uname().nodename,
               'source_command': cmd, 'seconds': time.monotonic() - began, 'encodes': len(ledger.rows),
               'inputs': {'al': digest(a.al), 'transport': digest(a.transport)},
               'scope': 'background at quantizer 70, offset inside R by rate-only search; no quality read'}
    (publish / 'encode.json').write_text(json.dumps(summary, indent=1))
    (Path(os.environ['PS_STAGE_DIR']) / 'maxbg-summary.json').write_text(json.dumps(summary, indent=1))


def validate() -> None:
    from src.codecs import svtav1

    stage = Path(os.environ['PS_STAGE_DIR'])
    summary = json.loads((stage / 'maxbg-summary.json').read_text())
    spec = json.loads((stage.parent / 'spec.json').read_text())
    frames = summary['frames']
    smoke = frames != FRAMES
    probe_rows = summary['probe']['modes']
    checks: dict[str, bool] = {
        'frames': frames in (24, FRAMES) and spec['scale']['frames'] == {'smoke': 24, 'full': FRAMES},
        'tools_pinned': all(summary['tools'][n]['sha256'] == d for n, d in TOOLS.items()),
        'inputs': all(summary['inputs'][x['name']] == x['sha256'] for x in spec['staged_inputs'] if x['name'] in summary['inputs']),
        'probe_complete': [m['mode'] for m in probe_rows] == list(MODES) and all(len(m['rows']) == 3 for m in probe_rows),
        'probe_finite': all(math.isfinite(r['psnr'][k]) for m in probe_rows for r in m['rows'] for k in ('in_r', 'out_r', 'in_blocks', 'out_blocks')),
        'probe_rule_applied': all(m['passes'] == (m['gain_in_blocks_db'] >= MIN_GAIN_DB and abs(m['background_change_db']) <= MAX_BACKGROUND_DB)
                                  for m in probe_rows)}
    chosen = summary['probe']['chosen']
    checks['mode_is_first_passing'] = chosen == next((m['mode'] for m in probe_rows if m['passes']), None)
    with tarfile.open(stage / 'published.tar') as tar:
        member = tar.extractfile('publish/encodes.json')
        assert member is not None
        rows = json.load(member)
        ok = len(rows) == summary['encodes']
        for row in rows:
            blob = tar.extractfile('publish/' + row['stream'])
            assert blob is not None
            data = blob.read()
            ok &= hashlib.sha256(data).hexdigest() == row['stream_sha256'] and len(svtav1.ivf_frames(data)) == frames
        checks['streams_match_rows'] = ok
    if chosen:
        checks['searches'] = [s['scale'] for s in summary['searches']] == [2, 4] and all(s['mode'] == chosen for s in summary['searches'])
        checks['controls'] = [c['scale'] for c in summary['controls']] == [8, 4, 2]
        if not smoke:
            checks['controls_bytes_match_existing_anchors'] = all(c['control_bytes_match'] for c in summary['controls'])
            checks['search_status'] = all(s['status'] in ('complete', 'unbracketed_high', 'unbracketed_low', 'budget') for s in summary['searches'])
    else:
        checks['no_search_without_mode'] = not summary['searches']
    record = {'passed': all(checks.values()), 'checks': checks}
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(record, indent=1))
    if not record['passed']:
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--al', type=Path)
    p.add_argument('--transport', type=Path)
    p.add_argument('--frames', type=int, choices=(24, FRAMES), default=24)
    p.add_argument('--max-encodes', type=int, default=2)
    a = p.parse_args()
    if a.validate:
        validate()
    elif a.al is None or a.transport is None:
        p.error('the Al source and the transport archive are required')
    else:
        run(a)


if __name__ == '__main__':
    main()

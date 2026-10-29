"""W1 CPU validity check of the SVT-AV1 ROI map (no quality metric networks, no GPU).

Question: do the ROI offsets written by `w1_roi.roi_text` put their bits where the players are?
(1) overlays of the map on the coded frame beside R, (2) luma PSNR inside and outside R and the
bytes for one CRF per resolution under several offsets and under the opposite convention
(a positive offset outside R), (3) the encoder's own help for the ROI and AQ options.
"""
from __future__ import annotations

import argparse
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

from experiments.tennis.w1_roi import (CODED, FPS, FRAMES, PACKET_DIR, SOURCE_SHA, SOURCE_START, SOURCE_YUV_SHA,
                                       TRANSPORT_SHA, digest, rate_kbps, roi_text, touched_blocks)

#: (name, where, value, extra encoder arguments)
CASES = [('plain', 'none', 0, []), ('in10', 'inside', -10, []), ('in20', 'inside', -20, []), ('in40', 'inside', -40, []),
         ('in63', 'inside', -63, []), ('out20', 'outside', 20, []), ('out40', 'outside', 40, []),
         ('in40_aq0', 'inside', -40, ['--aq-mode', '0'])]
CRF = {2: 67, 4: 62}


def outside_text(regions: np.ndarray, width: int, height: int, value: int) -> str:
    """The opposite convention: 0 in every block touched by R and `value` elsewhere."""
    lines = []
    for t, region in enumerate(regions):
        touched = touched_blocks(region, width, height)
        lines.append(str(t) + ' ' + ' '.join('0' if x else str(value) for x in touched.ravel()))
    return '\n'.join(lines) + '\n'


def parse_text(text: str, rows: int, cols: int) -> list[np.ndarray]:
    grids = []
    for t, line in enumerate(text.strip().split('\n')):
        parts = line.split()
        if int(parts[0]) != t or len(parts) != 1 + rows * cols:
            raise ValueError('ROI map line index or length wrong')
        grids.append(np.array([int(x) for x in parts[1:]]).reshape(rows, cols))
    return grids


def psnr(sse: float, count: int) -> float | None:
    if count == 0:
        return None
    return 99.0 if sse == 0 else 10 * math.log10(255.0 ** 2 * count / sse)


def region_psnr(source: Path, decoded: Path, regions: np.ndarray, width: int, height: int, frames: int) -> dict[str, Any]:
    """Luma PSNR over all frames inside / outside R and inside / outside the touched 64x64 blocks."""
    import cv2

    ys = np.memmap(source, np.uint8, 'r', shape=(frames, height * 3 // 2, width))
    yd = np.memmap(decoded, np.uint8, 'r', shape=(frames, height * 3 // 2, width))
    acc = {k: [0.0, 0] for k in ('in_r', 'out_r', 'in_blocks', 'out_blocks')}
    for t in range(frames):
        r = cv2.resize(regions[t].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        blocks = np.kron(touched_blocks(regions[t], width, height), np.ones((64, 64), bool))[:height, :width]
        err = (ys[t, :height].astype(np.float64) - yd[t, :height].astype(np.float64)) ** 2
        for key, mask in (('in_r', r), ('out_r', ~r), ('in_blocks', blocks), ('out_blocks', ~blocks)):
            acc[key][0] += float(err[mask].sum())
            acc[key][1] += int(mask.sum())
    return {k: psnr(*v) for k, v in acc.items()} | {'pixels': {k: v[1] for k, v in acc.items()}}


def overlay(source: Path, regions: np.ndarray, grids: list[np.ndarray], width: int, height: int, tick: int, out: Path) -> None:
    """Left: coded luma with R in red. Right: coded luma with the blocks carrying a nonzero offset shaded
    green; the column and row of the first and last such block are written in the image title."""
    import cv2

    luma = np.memmap(source, np.uint8, 'r', shape=(len(regions), height * 3 // 2, width))[tick, :height]
    grey = cv2.cvtColor(np.ascontiguousarray(luma), cv2.COLOR_GRAY2BGR)
    r = cv2.resize(regions[tick].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    left = grey.copy()
    left[r] = (0.4 * left[r] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    right = grey.copy()
    grid = grids[tick] != 0
    cells = np.kron(grid, np.ones((64, 64), bool))[:height, :width]
    right[cells] = (0.6 * right[cells] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    for i in range(0, height, 64):
        cv2.line(right, (0, i), (width, i), (255, 255, 255), 1)
    for j in range(0, width, 64):
        cv2.line(right, (j, 0), (j, height), (255, 255, 255), 1)
    both = np.concatenate([left, right], axis=1)
    cv2.putText(both, f'tick {tick}: R (left), nonzero ROI blocks (right)', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(str(out), both)


def run_case(case: tuple[str, str, int, list[str]], raw: Path, work: Path, regions: np.ndarray, scale: int, frames: int,
             cpus: list[int]) -> dict[str, Any]:
    from src.codecs import svtav1

    name, where, value, extra = case
    width, height = CODED[scale]
    work.mkdir(parents=True)
    args = list(extra)
    if where != 'none':
        text = roi_text(regions[:frames], width, height, value) if where == 'inside' else outside_text(regions[:frames], width, height, value)
        (work / 'roi.txt').write_text(text)
        args += ['--roi-map-file', str(work / 'roi.txt')]
    rec = svtav1.code(raw, work, width=width, height=height, fps=Fraction(FPS), frames=frames, crf=CRF[scale], preset=4,
                      threads=2, full_range=False, encoder='SvtAv1EncApp', decoder='dav1d', timeout=1800, cpus=cpus, extra=args)
    stats = region_psnr(raw, Path(rec['decoded']), regions, width, height, frames)
    Path(rec['decoded']).unlink()
    nbytes = Path(rec['stream']).stat().st_size
    return {'case': name, 'where': where, 'value': value, 'extra': extra, 'scale': scale, 'crf': CRF[scale],
            'ivf_bytes': nbytes, 'rate_kbps': rate_kbps(nbytes, frames), 'psnr': stats, 'encode_seconds': rec['encode_seconds'],
            'encode_command': rec['encode_command']}


def run(a: argparse.Namespace) -> None:
    from experiments.tennis.k4b_features import convert_command
    from experiments.tennis.k6 import downscaled_source
    from experiments.tennis.k6b_rate_streams import decode_masks
    from src.codecs import svtav1

    began = time.monotonic()
    for path, expected in ((a.al, SOURCE_SHA), (a.transport, TRANSPORT_SHA)):
        if digest(path) != expected:
            raise ValueError('pinned input changed: ' + path.name)
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch / 'publish'
    publish.mkdir()
    cpus = sorted(os.sched_getaffinity(0))
    frames = a.frames
    full = scratch / 'source-full.yuv'
    subprocess.run(convert_command('ffmpeg', a.al, full, SOURCE_START, frames), check=True, capture_output=True, timeout=900)
    if frames == FRAMES and digest(full) != SOURCE_YUV_SHA:
        raise ValueError('source conversion changed')
    with tarfile.open(a.transport) as tar:
        member = tar.extractfile(PACKET_DIR + 'exact-rle.matte')
        assert member is not None
        _, masks = decode_masks(member.read())
    regions = masks.any(axis=1)[:frames]
    ticks = [0, 120, 237] if frames == FRAMES else [0, 12, 23]
    help_text = subprocess.run(['SvtAv1EncApp', '--help'], capture_output=True, text=True, timeout=60).stdout
    help_lines = [ln.strip() for ln in help_text.splitlines() if any(k in ln.lower() for k in ('roi', 'aq-mode', '--crf', 'delta-q', 'deltaq', 'qp-offset'))]
    # (1) overlays and the map read back from the text exactly as the encoder will read it
    checks: dict[str, Any] = {}
    full_luma = np.memmap(full, np.uint8, 'r', shape=(frames, 1620, 1920))
    import cv2

    for scale in (2, 4):
        width, height = CODED[scale]
        raw = downscaled_source(full, frames, scale, scratch / f'source-s{scale}.yuv')
        text = roi_text(regions, width, height, -40)
        rows, cols = (height + 63) // 64, (width + 63) // 64
        grids = parse_text(text, rows, cols)
        checks[f's{scale}'] = {'rows': rows, 'cols': cols,
                               'map_equals_touched_blocks': all(((g != 0) == touched_blocks(regions[t], width, height)).all() for t, g in enumerate(grids)),
                               'nonzero_values': sorted({int(v) for g in grids for v in np.unique(g)}),
                               'first_lines': text.splitlines()[0][:80]}
        for tick in ticks:
            overlay(raw, regions, grids, width, height, tick, publish / f'overlay-s{scale}-t{tick:03d}.png')
    for tick in ticks:
        image = cv2.cvtColor(np.ascontiguousarray(full_luma[tick, :1080]), cv2.COLOR_GRAY2BGR)
        mask = cv2.resize(regions[tick].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
        image[mask] = (0.4 * image[mask] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(str(publish / f'overlay-full-t{tick:03d}.png'), image)
    # (2) the encodes
    rows_out: list[dict[str, Any]] = []
    lanes = 4
    share = max(1, len(cpus) // lanes)
    cpu_sets = [cpus[i * share:(i + 1) * share] or cpus for i in range(lanes)]
    jobs = []
    with ThreadPoolExecutor(max_workers=lanes) as pool:
        for scale in (2, 4):
            raw = scratch / f'source-s{scale}.yuv'
            for i, case in enumerate(CASES):
                jobs.append(pool.submit(run_case, case, raw, scratch / f's{scale}-{case[0]}', regions, scale, frames, cpu_sets[i % lanes]))
        rows_out = [j.result() for j in jobs]
    summary = {'frames': frames, 'ticks': ticks, 'checks': checks, 'cases': rows_out, 'help_lines': help_lines,
               'version': svtav1.tool('SvtAv1EncApp')['version'], 'cpus': len(cpus), 'host': os.uname().nodename,
               'seconds': time.monotonic() - began, 'source_start': SOURCE_START,
               'inputs': {'al': digest(a.al), 'transport': digest(a.transport)}}
    (publish / 'roicheck.json').write_text(json.dumps(summary, indent=1))
    (Path(os.environ['PS_STAGE_DIR']) / 'roicheck-summary.json').write_text(json.dumps(summary, indent=1))


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    summary = json.loads((stage / 'roicheck-summary.json').read_text())
    with tarfile.open(stage / 'published.tar') as tar:
        names = tar.getnames()
    cases = summary['cases']
    checks = {'cases': len(cases) == 2 * len(CASES), 'overlays': sum(n.endswith('.png') for n in names) == 3 * 3,
              'maps_match': all(c['map_equals_touched_blocks'] for k, c in summary['checks'].items() if k.startswith('s')),
              'psnr_finite': all(v is not None and math.isfinite(v) for c in cases for k, v in c['psnr'].items() if k != 'pixels'),
              'help_captured': any('roi' in ln.lower() for ln in summary['help_lines'])}
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
    a = p.parse_args()
    if a.validate:
        validate()
    elif a.al is None or a.transport is None:
        p.error('the Al source and the transport archive are required')
    else:
        run(a)


if __name__ == '__main__':
    main()

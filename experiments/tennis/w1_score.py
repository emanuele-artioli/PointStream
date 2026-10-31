"""W1 GPU job: score the selected SVT ROI encodes of the Al window and apply the frozen rule.

Whole-frame and fixed-foreground (S, from `w1_masks`) LPIPS and DISTS on the 80 saved ticks,
with the real-base judge's functions and weights. The plain CRF 63 bases of scale 8/4/2 are
scored first and must reproduce the hybrid job's recorded per-tick scores within 1e-4.
"""
from __future__ import annotations

import argparse
import sys
import io
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis.w1_roi import (CONTROL_CRF, CONTROL_ROW, FRAMES, control_tolerance, HYBRID, OFFSETS,
                                       TARGET_KBPS, TICKS, TOOLS, WEIGHTS, WINDOWS, Window, analyse,
                                       best_foreground, decide, digest, policy_at_target, policy_name, rebilled_kbps)

#: the hybrid job's names for the four per-tick series, and ours in the same order.
PRIOR_SUMMARY_SHA = '44d4bbc32aa74ac742091b55ba816f7025005138f7969ee161ca88f28b783a30'  # score job 20261029T095649Z-43181442
SERIES = ('full_lpips', 'foreground_lpips', 'full_dists', 'foreground_dists')
OURS = ('whole_lpips', 'foreground_lpips', 'whole_dists', 'foreground_dists')


def extract(archive: Path, root: Path) -> Path:
    root.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter='data')
    return root / 'publish'


def selected_points(published: Path) -> list[dict[str, Any]]:
    """Rows of the encodes a search selected, plus (scale job 2) the plain controls."""
    summary = json.loads((published / 'encode.json').read_text())
    rows = json.loads((published / 'encodes.json').read_text())
    points = []
    for search in summary['searches']:
        for crf in search['selected'] or []:
            row = next(r for r in rows if r['scale'] == search['scale'] and r['offset'] == search['offset']
                       and r['crf'] == crf and 'control' not in r['stream'])
            points.append({**row, 'kind': 'roi', 'policy': policy_name(row['scale'], row['offset']),
                           'name': f"{policy_name(row['scale'], row['offset'])}_crf{crf}", 'path': published / row['stream']})
    for row in summary['controls']:
        points.append({**row, 'kind': 'control', 'policy': f"control_s{row['scale']}",
                       'name': f"control_s{row['scale']}_crf{CONTROL_CRF}", 'path': published / row['stream']})
    return points


def control_errors(scores: np.ndarray, hybrid: Any, scale: int, n: int) -> dict[str, float]:
    return {name: float(np.max(np.abs(scores[i, :n] - hybrid[name][CONTROL_ROW[scale], :n]))) for i, name in enumerate(SERIES)}


def score_point(point: dict[str, Any], scratch: Path, ticks: list[int], truth: list[np.ndarray], masks: Any,
                nets: dict[str, Any], torch: Any) -> dict[str, Any]:
    import cv2

    from experiments.tennis.k6 import DISTS_PAD
    from experiments.visor.h3b import dists_crops, lpips_reduce
    from src.codecs import svtav1
    from src.codecs.quality import yuv420_to_rgb

    width, height = point['size']
    decoded = scratch / (point['name'] + '.yuv')
    subprocess.run(svtav1.decode_command('dav1d', point['path'], decoded, threads=2), check=True, capture_output=True, timeout=900)
    if digest(decoded) != point['decoded_sha256']:
        raise ValueError('decoded pixels differ from the encoder job: ' + point['name'])
    pixels = np.memmap(decoded, np.uint8, 'r', shape=(FRAMES, height * 3 // 2, width))
    all_mask = np.ones((1080, 1920), bool)
    values = np.full((len(SERIES), len(ticks)), np.nan)
    started = time.monotonic()
    for j, tick in enumerate(ticks):
        plane = torch.from_numpy(np.array(pixels[tick], copy=True)[None]).cuda()
        image = yuv420_to_rgb(plane, width, height, full_range=False)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        if (width, height) != (1920, 1080):
            image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        s = np.unpackbits(masks['s'][j])[:1080 * 1920].reshape(1080, 1920).astype(bool)
        ys, xs = np.nonzero(s)
        crop = (slice(max(0, ys.min() - DISTS_PAD), min(1080, ys.max() + 1 + DISTS_PAD)),
                slice(max(0, xs.min() - DISTS_PAD), min(1920, xs.max() + 1 + DISTS_PAD)))
        true = truth[j]
        lp = lpips_reduce(nets['lpips'], [image], true, [[all_mask, s]], 'cuda')[0]
        values[0, j], values[1, j] = lp
        values[2, j] = dists_crops(nets['dists'], [image], true, all_mask, 'cuda')[0]
        values[3, j] = dists_crops(nets['dists'], [image[crop]], true[crop], s[crop], 'cuda')[0]
    decoded.unlink()
    if not np.isfinite(values).all():
        raise ValueError('nonfinite score: ' + point['name'])
    return {'values': values, 'seconds': time.monotonic() - started}


def hybrid_point(quality: dict[str, Any], window: Window) -> dict[str, Any]:
    """The hybrid's scale 4 means and rates (KPS1 as recorded, and re-billed with KPS2) from the real-base judge's
    quality.json; the scale 4 method is series row 3 of the judge's arrays."""
    row = next(r for r in quality['rows'] if r['id'] == window.id)
    rate = next(r for r in row['rates'] if r['scale'] == 4)
    if rate['kps_bytes'] not in (window.kps1_bytes, window.kps2_bytes):
        raise ValueError('the hybrid job billed a different keypoint packet than the pinned KPS1 or KPS2')
    billed_kps2 = rate['kps_bytes'] == window.kps2_bytes != window.kps1_bytes  # W7 windows: the judge pays the KPS2 packet directly
    means = {'lpips': {'whole': row['means']['full_lpips'][3], 'foreground': row['means']['foreground_lpips'][3]},
             'dists': {'whole': row['means']['full_dists'][3], 'foreground': row['means']['foreground_dists'][3]}}
    total = rate['total_kbps']
    if billed_kps2:
        return {'hybrid': means, 'kps1_kbps': rebilled_kbps(total, window.kps2_bytes, window.kps1_bytes, window), 'kps2_kbps': total, 'rate': rate}
    return {'hybrid': means, 'kps1_kbps': total, 'kps2_kbps': rebilled_kbps(total, window.kps1_bytes, window.kps2_bytes, window), 'rate': rate}


def assemble(points: list[dict[str, Any]], scores: dict[str, np.ndarray], scales: tuple[int, ...] = (),
             carried: dict[str, list[dict[str, Any]]] | None = None, target: float = TARGET_KBPS,
             hybrid: dict[str, dict[str, float]] = HYBRID) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        if point['kind'] != 'roi':
            continue
        mean = scores[point['name']].mean(axis=1)
        by_policy.setdefault(point['policy'], []).append(
            {'crf': point['crf'], 'rate_kbps': point['rate_kbps'], **{name: float(mean[i]) for i, name in enumerate(OURS)}})
    by_policy = {**(carried or {}), **by_policy}
    policies = {name: policy_at_target(rows, target) for name, rows in by_policy.items()}
    for scale in sorted({int(p['policy'].split('_')[0][1:]) for p in points if p['kind'] == 'roi'} | set(scales)):
        for offset in OFFSETS:
            policies.setdefault(policy_name(scale, offset), None)
    return {'by_policy': by_policy, 'target_kbps': target, 'at_target': policies, 'decision': decide(policies, hybrid),
            'best_foreground': best_foreground(policies, hybrid)}


def scales_of(published: Path) -> tuple[int, ...]:
    return tuple(sorted({int(x['scale']) for x in json.loads((published / 'encode.json').read_text())['searches']}))


def qualified_gpu(name: str, allow_ada: bool) -> bool:
    return 'A6000' in name or (allow_ada and 'RTX 6000 Ada' in name)


def run(a: argparse.Namespace) -> None:
    import torch

    from experiments.jobs.monitor import publish_progress
    from experiments.tennis.k4b_features import convert_command
    from src.codecs import svtav1
    from src.codecs.quality import load_lpips, yuv420_to_rgb
    from experiments.visor.h3b import load_dists

    began = time.monotonic()
    window = WINDOWS[a.window]
    if a.allow_ada and (a.windows_file is None or control_tolerance(window.name) != 1e-3):
        raise ValueError('Ada requires a manifest window with the 1e-3 control gate')
    if not torch.cuda.is_available() or not qualified_gpu(torch.cuda.get_device_name(0), a.allow_ada):
        raise ValueError('a qualified RTX A6000 or explicitly enabled RTX 6000 Ada is required')
    torch.set_num_threads(4)
    limit = int(os.environ.get('PS_GPU_MEMORY_LIMIT_MIB', '8000'))
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    for path, expected in ((a.al, window.source_sha), (a.hybrid, window.hybrid_archive_sha), (a.lpips, WEIGHTS['lpips']),
                           (a.dists, WEIGHTS['dists'])):
        if expected is not None and digest(path) != expected:
            raise ValueError('pinned input changed: ' + path.name)
    tools = {name: svtav1.tool(name) for name in TOOLS}
    if any(tools[n]['sha256'] != d for n, d in TOOLS.items()):
        raise ValueError('pinned native tool changed')
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch / 'publish'
    publish.mkdir()
    half = extract(a.encode_half, scratch / 'half')
    full = extract(a.encode_full, scratch / 'full') if a.encode_full else None
    more = [extract(path, scratch / f'more{i}') for i, path in enumerate(a.encode_extra)]
    masks_root = extract(a.masks, scratch / 'masks')
    hybrid_root = extract(a.hybrid, scratch / 'hybrid')
    with np.load(masks_root / 'masks.npz') as z:
        masks = {k: z[k] for k in ('ticks', 'r', 's')}
    quality = json.loads((hybrid_root / 'quality.json').read_text())
    anchor_rates = hybrid_point(quality, window)
    primary = anchor_rates['kps2_kbps'] if a.primary == 'kps2' else anchor_rates['kps1_kbps']
    with np.load(hybrid_root / window.hybrid_dir / 'scores.npz') as z:
        hybrid = {k: z[k] for k in SERIES}
        hybrid_ticks = z['ticks'].tolist()
    ticks = TICKS[:a.judged or None]
    if masks['ticks'].tolist()[:len(ticks)] != ticks or hybrid_ticks[:len(ticks)] != ticks:
        raise ValueError('tick cohorts disagree')
    points = selected_points(half) + (selected_points(full) if full else []) + [p for root in more for p in selected_points(root)]
    controls = [p for p in points if p['kind'] == 'control']
    if [c['scale'] for c in controls] != [8, 4, 2]:
        raise ValueError('control encodes missing')
    ordered = controls + [p for p in points if p['kind'] == 'roi']
    yuv = scratch / 'source.yuv'
    subprocess.run(convert_command('ffmpeg', a.al, yuv, window.source_start, FRAMES), check=True, capture_output=True, timeout=900)
    if digest(yuv) != window.source_yuv_sha:
        raise ValueError('source conversion changed')
    raw = np.memmap(yuv, np.uint8, 'r', shape=(FRAMES, 1620, 1920))
    truth = []
    for tick in ticks:
        plane = torch.from_numpy(np.array(raw[tick], copy=True)[None]).cuda()
        truth.append(yuv420_to_rgb(plane, 1920, 1080, full_range=False)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy())
    nets = {'lpips': load_lpips(a.lpips, 'cuda'), 'dists': load_dists(a.dists, 'cuda')}
    if not all(next(net.parameters()).is_cuda for net in nets.values()):
        raise ValueError('metric networks must be on CUDA')
    scores: dict[str, np.ndarray] = {}
    errors: dict[str, dict[str, float]] = {}
    seconds: dict[str, float] = {}
    for i, point in enumerate(ordered):
        result = score_point(point, scratch, ticks, truth, masks, nets, torch)
        scores[point['name']] = result['values']
        seconds[point['name']] = result['seconds']
        if point['kind'] == 'control':
            errors[point['name']] = control_errors(result['values'], hybrid, point['scale'], len(ticks))
            if max(errors[point['name']].values()) > control_tolerance(window.name):
                raise ValueError(f"control {point['name']} does not reproduce the recorded scores: {errors[point['name']]}")
        np.savez_compressed(publish / 'scores.npz', ticks=np.asarray(ticks), **scores)
        publish_progress(os.environ['PS_STAGE'], i + 1)
    torch.cuda.synchronize()
    carried: dict[str, list[dict[str, Any]]] = {}
    if a.prior is not None:
        if digest(a.prior) != PRIOR_SUMMARY_SHA:
            raise ValueError('prior summary changed')
        old = json.loads(a.prior.read_text())['analysis']['by_policy']
        carried = {n: rows for n, rows in old.items() if n not in {p['policy'] for p in ordered if p['kind'] == 'roi'}}
    scales = scales_of(half) + (scales_of(full) if full else ()) + tuple(x for root in more for x in scales_of(root))
    analysis = assemble(ordered, scores, scales, carried, primary, anchor_rates['hybrid'])
    other = anchor_rates['kps1_kbps'] if a.primary == 'kps2' else anchor_rates['kps2_kbps']
    analysis['other_rate'] = analyse(analysis['by_policy'], tuple(sorted(set(scales))), other, anchor_rates['hybrid'])
    analysis['rates'] = {'primary': a.primary, 'kps1_kbps': anchor_rates['kps1_kbps'], 'kps2_kbps': anchor_rates['kps2_kbps']}
    rows = [{k: v for k, v in p.items() if k not in ('path',)} for p in ordered]
    summary = {'window': window.name, 'hybrid': anchor_rates, 'ticks': ticks, 'points': rows, 'carried_policies': sorted(carried), 'control_errors': errors, 'score_seconds': seconds, 'analysis': analysis,
               'series': SERIES, 'tools': tools, 'target_kbps': primary, 'window_id': window.id,
               'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
               'cudnn': torch.backends.cudnn.version(), 'cuda': True,
               'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
               'seconds': time.monotonic() - began, 'cpus': len(os.sched_getaffinity(0)),
               'inputs': {**{(window.name if k == 'al' else k): digest(getattr(a, k)) for k in ('al', 'masks', 'hybrid', 'lpips', 'dists', 'encode_half', 'encode_full')
                            if getattr(a, k) is not None}, **{f'encode_x{i}': digest(path) for i, path in enumerate(a.encode_extra)}},
               'encode_summaries': {'half': json.loads((half / 'encode.json').read_text()),
                                    'full': json.loads((full / 'encode.json').read_text()) if full else None,
                                    'extra': [json.loads((root / 'encode.json').read_text()) for root in more]}}
    (publish / 'score.json').write_text(json.dumps(summary, indent=1))
    (Path(os.environ['PS_STAGE_DIR']) / 'score-summary.json').write_text(json.dumps(summary, indent=1))


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    summary = json.loads((stage / 'score-summary.json').read_text())
    spec = json.loads((stage.parent / 'spec.json').read_text())
    execution = json.loads((stage / 'execution.json').read_text()) if (stage / 'execution.json').exists() else {}
    smoke = len(summary['ticks']) != len(TICKS)
    checks: dict[str, bool] = {
        'ticks': summary['ticks'] in (TICKS[:3], TICKS),
        'cuda_qualified': bool(summary['cuda']) and qualified_gpu(summary['gpu'], '--allow-ada' in spec['arguments'])
        and ('A6000' in summary['gpu'] or control_tolerance(summary['window']) == 1e-3),
        'tools_pinned': all(summary['tools'][n]['sha256'] == d for n, d in TOOLS.items()),
        'inputs': all(summary['inputs'][x['name']] == x['sha256'] for x in spec['staged_inputs'] if x['name'] in summary['inputs']),
        'controls_reproduced': sorted(summary['control_errors']) == ['control_s2_crf63', 'control_s4_crf63', 'control_s8_crf63']
        and all(max(e.values()) <= control_tolerance(summary['window']) for e in summary['control_errors'].values()),
        'memory': 0 < summary['peak_allocated_mib'] < 8000,
        'rates_positive': all(p['rate_kbps'] > 0 for p in summary['points'])}
    with tarfile.open(stage / 'published.tar') as tar:
        member = tar.extractfile('publish/scores.npz')
        assert member is not None
        with np.load(io.BytesIO(member.read())) as z:
            checks['scores_complete'] = (sorted(k for k in z.files if k != 'ticks') == sorted(p['name'] for p in summary['points'])
                                         and all(z[p['name']].shape == (4, len(summary['ticks'])) and np.isfinite(z[p['name']]).all()
                                                 for p in summary['points']))
    roi = [p for p in summary['points'] if p['kind'] == 'roi']
    checks['roi_points_present'] = len(roi) >= 3 and all(p['policy'].startswith('s') for p in roi)
    if smoke:
        per_point_tick = sum(summary['score_seconds'].values()) / (len(summary['points']) * len(summary['ticks']))
        projected = 1.35 * (summary['seconds'] - sum(summary['score_seconds'].values())
                            + per_point_tick * len(TICKS) * len(summary['points']))
        checks['projection'] = projected < spec['full']['seconds']
        projection: float | None = projected
    else:
        projection = None
        checks['decision_present'] = summary['analysis']['decision']['outcome'] in ('A', 'B')
    record = {'passed': all(checks.values()), 'checks': checks, 'projected_full_seconds': projection,
              'execution_seconds': execution.get('seconds')}
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(record, indent=1))
    if not record['passed']:
        raise SystemExit(1)


def main() -> None:
    from experiments.tennis.w1_roi import preload_windows
    preload_windows(sys.argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--windows-file', type=Path, help='window manifest (window_manifest.py); registers its windows next to al/med')
    p.add_argument('--window-state', type=Path, help='what earlier stages measured per manifest window (w1_roi.window_from_spec)')
    p.add_argument('--validate', action='store_true')
    p.add_argument('--allow-ada', action='store_true', help='allow Ada for manifest windows with the 1e-3 control gate')
    for name in ('masks', 'hybrid', 'lpips', 'dists', 'encode-half', 'encode-full'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--al', '--source', dest='al', type=Path, help='the window\'s source video')
    p.add_argument('--window', choices=sorted(WINDOWS), default='al')
    p.add_argument('--encode-extra', type=Path, nargs='*', default=[], help='further encode archives')
    p.add_argument('--primary', choices=('kps1', 'kps2'), default='kps1', help='the rate the embedded decision uses')
    p.add_argument('--prior', type=Path, help='earlier score summary whose offset-0 policies are carried over')
    p.add_argument('--judged', type=int, choices=(0, 3), default=3, help='first N ticks; 0 means all 80')
    a = p.parse_args()
    if a.validate:
        validate()
    elif any(getattr(a, n.replace('-', '_')) is None for n in ('al', 'masks', 'hybrid', 'lpips', 'dists', 'encode-half')):
        p.error('every frozen input is required')
    else:
        run(a)


if __name__ == '__main__':
    main()

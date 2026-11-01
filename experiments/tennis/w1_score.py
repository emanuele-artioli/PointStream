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

from experiments.tennis.w1_roi import (CODED, CONTROL_CRF, CONTROL_ROW, FRAMES, control_tolerance, HYBRID, OFFSETS,
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



def repair_points(manifest: dict[str, Any], archives: list[tuple[Path, Path]], window: Window,
                  transport_sha256: str) -> list[dict[str, Any]]:
    """Resolve exactly the frozen stream identities, including unselected encoder trials."""
    if set(manifest) != {"window", "points", "controls"} or manifest["window"] != window.name:
        raise ValueError("invalid repair manifest window or fields")
    entries, controls = manifest["points"], manifest["controls"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 3 or not isinstance(controls, list) or len(controls) != 3:
        raise ValueError("repair requires one to three points and exactly three controls")
    archive_map = {digest(path): root for path, root in archives}
    if len(archive_map) != len(archives):
        raise ValueError("duplicate repair archive")
    from experiments.tennis.k4b_features import convert_command

    expected_command = convert_command("ffmpeg", Path("source"), Path("output"), window.source_start, FRAMES)
    expected_filter = expected_command[expected_command.index("-vf") + 1]
    for root in archive_map.values():
        summary = json.loads((root / 'encode.json').read_text())
        if (summary['window'] != window.name or summary['frames'] != FRAMES
                or summary['inputs'].get(window.name) != window.source_sha
                or summary['inputs'].get('transport') != transport_sha256
                or summary['source_command'][summary['source_command'].index('-vf') + 1] != expected_filter
                or any(summary['tools'][name]['sha256'] != sha for name, sha in TOOLS.items())):
            raise ValueError('repair archive window, source, transport or native tool mismatch')
    expected_keys = {"policy", "scale", "offset", "crf", "rate_kbps", "bytes", "stream_sha256", "decoded_sha256", "archive_sha256", "member"}
    result = []
    seen: set[tuple[str, str]] = set()
    for kind, rows in (("control", controls), ("roi", entries)):
        for entry in rows:
            if not isinstance(entry, dict) or set(entry) != expected_keys:
                raise ValueError("invalid repair point fields")
            identity = (entry["archive_sha256"], entry["member"])
            if identity in seen or identity[0] not in archive_map:
                raise ValueError("duplicate point or missing repair archive")
            seen.add(identity)
            root = archive_map[identity[0]]
            member = entry["member"]
            if not isinstance(member, str) or not member.startswith("publish/") or ".." in Path(member).parts:
                raise ValueError("invalid stream member")
            relative = member[len("publish/"):]
            metadata = json.loads((root / ("encode.json" if kind == "control" else "encodes.json")).read_text())
            candidates = metadata["controls"] if kind == "control" else metadata
            matching = [row for row in candidates if row["stream"] == relative]
            if len(matching) != 1:
                raise ValueError("repair stream missing or ambiguous in archive metadata")
            row = matching[0]
            policy = f"control_s{row['scale']}" if kind == "control" else policy_name(row["scale"], row["offset"])
            if any(row[k] != entry[k] for k in ("scale", "offset", "crf", "rate_kbps", "stream_sha256", "decoded_sha256")) or row["ivf_bytes"] != entry["bytes"] or entry["policy"] != policy:
                raise ValueError("repair stream metadata mismatch")
            if kind == "roi" and "control" in row["stream"]:
                raise ValueError("control cannot be a repair point")
            if row["frames"] != FRAMES or tuple(row["size"]) != CODED.get(row["scale"]):
                raise ValueError("repair stream frame count or geometry mismatch")
            if kind == "roi" and (row["scale"], row["offset"], row["outside"]) not in ((2, 0, False), (4, 0, False), (2, -20, True)):
                raise ValueError("repair policy differs from frozen W7 policies")
            if kind == "control" and (row["crf"] != CONTROL_CRF or row["offset"] != 0 or row["ivf_bytes"] != window.control_ivf_bytes[row["scale"]]):
                raise ValueError("invalid control")
            path = root / relative
            if path.stat().st_size != entry["bytes"] or digest(path) != entry["stream_sha256"]:
                raise ValueError("repair stream hash or bytes mismatch")
            expected_rate = entry["bytes"] * 8 * window.fps / FRAMES / 1000
            if abs(entry["rate_kbps"] - expected_rate) > 1e-9:
                raise ValueError("repair stream rate mismatch")
            name = f"control_s{row['scale']}_crf{CONTROL_CRF}" if kind == "control" else f"repair_{entry['stream_sha256']}"
            result.append({**row, "kind": kind, "policy": policy, "name": name, "path": path,
                           "archive_sha256": identity[0], "member": member})
    if [r["scale"] for r in result[:3]] != [8, 4, 2]:
        raise ValueError("repair controls must be ordered scales 8, 4, 2")
    if set(archive_map) != {entry["archive_sha256"] for entry in controls + entries}:
        raise ValueError("unreferenced repair archive")
    return result

def repair_summary_matches(manifest: dict[str, Any], points: list[dict[str, Any]]) -> bool:
    controls, roi = manifest['controls'], manifest['points']
    if len(controls) != 3 or not 1 <= len(roi) <= 3 or [x['scale'] for x in controls] != [8, 4, 2]:
        return False
    if any(x['crf'] != CONTROL_CRF or x['offset'] != 0 or x['policy'] != f"control_s{x['scale']}" for x in controls):
        return False
    expected = [('control', x) for x in controls] + [('roi', x) for x in roi]
    if len(points) != len(expected) or len({(x['archive_sha256'], x['member']) for _, x in expected}) != len(expected):
        return False
    for point, (kind, entry) in zip(points, expected):
        if point['kind'] != kind or any(point[k] != entry[k] for k in ('policy', 'scale', 'offset', 'crf', 'rate_kbps', 'stream_sha256', 'decoded_sha256', 'archive_sha256', 'member')) or point['ivf_bytes'] != entry['bytes']:
            return False
    return True


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




def audit_repair_controls(repair_archive: Path, repair_sha256: str, hybrid_archive: Path,
                          hybrid_sha256: str, hybrid_dir: str) -> dict[str, dict[str, float]]:
    """Recompute every control error from verified raw outputs and original hybrid arrays."""
    if digest(repair_archive) != repair_sha256 or digest(hybrid_archive) != hybrid_sha256:
        raise ValueError('control audit archive hash mismatch')
    if Path(hybrid_dir).name != hybrid_dir:
        raise ValueError('invalid hybrid directory')
    with tarfile.open(repair_archive) as tar:
        summary_file = tar.extractfile('publish/score.json')
        scores_file = tar.extractfile('publish/scores.npz')
        if summary_file is None or scores_file is None:
            raise ValueError('repair score outputs missing')
        summary = json.load(summary_file)
        with np.load(io.BytesIO(scores_file.read())) as z:
            scores = {k: z[k] for k in z.files}
    with tarfile.open(hybrid_archive) as tar:
        hybrid_file = tar.extractfile(f'publish/{hybrid_dir}/scores.npz')
        if hybrid_file is None:
            raise ValueError('original hybrid scores missing')
        with np.load(io.BytesIO(hybrid_file.read())) as z:
            hybrid = {k: z[k] for k in z.files}
    if summary.get('repair_manifest') is None or not repair_summary_matches(summary['repair_manifest'], summary['points']):
        raise ValueError('repair control identity mismatch')
    if summary['ticks'] != TICKS or scores['ticks'].tolist() != TICKS or hybrid['ticks'].tolist() != TICKS:
        raise ValueError('control audit requires all original ticks')
    results = {}
    for point in summary['points']:
        if point['kind'] != 'control':
            continue
        value = scores[point['name']]
        if value.shape != (4, len(TICKS)) or not np.isfinite(value).all():
            raise ValueError('invalid raw control scores')
        error = control_errors(value, hybrid, point['scale'], len(TICKS))
        reported = summary['control_errors'].get(point['name'], {})
        if (set(reported) != set(SERIES) or any(not np.isfinite(v) or v > 1e-3
                or not np.isclose(v, reported[k], rtol=0, atol=1e-12) for k, v in error.items())):
            raise ValueError('raw control errors fail or disagree with summary')
        results[point['name']] = error
    if sorted(results) != ['control_s2_crf63', 'control_s4_crf63', 'control_s8_crf63']:
        raise ValueError('three original controls required')
    return results

def merge_repair_scores(old_archive: Path, old_sha256: str, repair_archive: Path, repair_sha256: str,
                        target: float) -> dict[str, Any]:
    """Merge verified per-stream points, then interpolate only within measured brackets."""
    if not np.isfinite(target) or target <= 0:
        raise ValueError("invalid corrected target")
    if digest(old_archive) != old_sha256 or digest(repair_archive) != repair_sha256:
        raise ValueError("score archive hash mismatch")
    with tarfile.open(old_archive) as tar:
        source = tar.extractfile('publish/score.json')
        if source is None:
            raise ValueError("old score summary missing")
        old = json.load(source)
    with tarfile.open(repair_archive) as tar:
        source = tar.extractfile('publish/score.json')
        raw = tar.extractfile('publish/scores.npz')
        if source is None or raw is None:
            raise ValueError("repair score outputs missing")
        repair = json.load(source)
        with np.load(io.BytesIO(raw.read())) as z:
            scores = {k: z[k] for k in z.files}
    if old['window'] != repair['window'] or repair.get('repair_manifest') is None or repair['analysis'] is not None:
        raise ValueError("repair and original window/profile mismatch")
    if not repair_summary_matches(repair['repair_manifest'], repair['points']):
        raise ValueError('repair summary differs from manifest')
    errors = repair['control_errors']
    if sorted(errors) != ['control_s2_crf63', 'control_s4_crf63', 'control_s8_crf63'] or any(set(v) != set(SERIES) or not all(np.isfinite(x) and 0 <= x <= 1e-3 for x in v.values()) for v in errors.values()):
        raise ValueError('repair control gate failed')
    if repair['ticks'] != TICKS or scores['ticks'].tolist() != TICKS:
        raise ValueError("repair must include all original ticks")
    merged: dict[str, list[dict[str, Any]]] = {}
    for policy, points in old['analysis']['by_policy'].items():
        for point in points:
            matches = [r for r in old['points'] if r['kind'] == 'roi' and r['policy'] == policy
                       and r['crf'] == point['crf'] and r['rate_kbps'] == point['rate_kbps']]
            if len(matches) != 1 or 'stream_sha256' not in matches[0]:
                raise ValueError("old point lacks unambiguous stream identity")
            existing = merged.setdefault(policy, [])
            if any(p['rate_kbps'] == point['rate_kbps'] or p['stream_sha256'] == matches[0]['stream_sha256'] for p in existing):
                raise ValueError('duplicate stream or rate in old policy')
            existing.append({**point, 'stream_sha256': matches[0]['stream_sha256']})
    roi = [p for p in repair['points'] if p['kind'] == 'roi']
    if len(roi) != len(repair['repair_manifest']['points']) or set(scores) != {'ticks', *(p['name'] for p in repair['points'])}:
        raise ValueError("repair score point set mismatch")
    for point in repair['points']:
        values = scores[point['name']]
        if values.shape != (4, len(TICKS)) or not np.isfinite(values).all():
            raise ValueError('invalid repair scores')
    for point in roi:
        values = scores[point['name']]
        if values.shape != (4, len(TICKS)) or not np.isfinite(values).all():
            raise ValueError("invalid repair scores")
        mean = values.mean(axis=1)
        row = {'crf': point['crf'], 'rate_kbps': point['rate_kbps'], 'stream_sha256': point['stream_sha256'],
               'archive_sha256': point['archive_sha256'], 'member': point['member'],
               **{name: float(mean[i]) for i, name in enumerate(OURS)}}
        existing = merged.setdefault(point['policy'], [])
        if any(p['stream_sha256'] == row['stream_sha256'] or p['rate_kbps'] == row['rate_kbps'] for p in existing):
            raise ValueError("duplicate stream or rate in merged policy")
        existing.append(row)
    return {'by_policy': merged, 'target_kbps': target,
            'at_target': {policy: policy_at_target(points, target) for policy, points in merged.items()}}

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
    repair = json.loads(a.repair_manifest.read_text()) if a.repair_manifest else None
    archives = [(a.encode_half, half)] + ([(a.encode_full, full)] if full else []) + list(zip(a.encode_extra, more))
    if repair is not None:
        mask_summary = json.loads((masks_root / 'masks.json').read_text())
        if digest(masks_root / 'masks.npz') != mask_summary['masks_sha256']:
            raise ValueError('repair source mask hash mismatch')
        points = repair_points(repair, archives, window, mask_summary['inputs']['transport'])
    else:
        points = (
        selected_points(half) + (selected_points(full) if full else []) + [p for root in more for p in selected_points(root)])
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
    analysis = None
    if repair is None:
        analysis = assemble(ordered, scores, scales, carried, primary, anchor_rates['hybrid'])
        other = anchor_rates['kps1_kbps'] if a.primary == 'kps2' else anchor_rates['kps2_kbps']
        analysis['other_rate'] = analyse(analysis['by_policy'], tuple(sorted(set(scales))), other, anchor_rates['hybrid'])
        analysis['rates'] = {'primary': a.primary, 'kps1_kbps': anchor_rates['kps1_kbps'], 'kps2_kbps': anchor_rates['kps2_kbps']}
    rows = [{k: v for k, v in p.items() if k not in ('path',)} for p in ordered]
    summary = {'window': window.name, 'hybrid': anchor_rates, 'ticks': ticks, 'points': rows, 'carried_policies': sorted(carried), 'control_errors': errors, 'score_seconds': seconds, 'analysis': analysis,
               'series': SERIES, 'tools': tools, 'window_id': window.id,
               'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
               'cudnn': torch.backends.cudnn.version(), 'cuda': True,
               'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
               'seconds': time.monotonic() - began, 'cpus': len(os.sched_getaffinity(0)),
               'inputs': {**{(window.name if k == 'al' else k): digest(getattr(a, k)) for k in ('al', 'masks', 'hybrid', 'lpips', 'dists', 'encode_half', 'encode_full')
                            if getattr(a, k) is not None}, **({'repair_manifest': digest(a.repair_manifest)} if a.repair_manifest else {}), **{f'encode_x{i}': digest(path) for i, path in enumerate(a.encode_extra)}},
               'encode_summaries': {'half': json.loads((half / 'encode.json').read_text()),
                                    'full': json.loads((full / 'encode.json').read_text()) if full else None,
                                    'extra': [json.loads((root / 'encode.json').read_text()) for root in more]}}
    if repair is None:
        summary['target_kbps'] = primary
    else:
        summary['repair_manifest'] = repair
        summary['historical_hybrid_target_kbps'] = primary
    (publish / 'score.json').write_text(json.dumps(summary, indent=1))
    (Path(os.environ['PS_STAGE_DIR']) / 'score-summary.json').write_text(json.dumps(summary, indent=1))


def repair_manifest_matches_stage(summary: dict[str, Any], spec: dict[str, Any], staging: dict[str, Any]) -> bool:
    declared = [x for x in spec['staged_inputs'] if x['name'] == 'repair_manifest']
    actual = [x for x in staging['inputs'] if x['name'] == 'repair_manifest']
    if len(declared) != 1 or len(actual) != 1 or actual[0].get('extract', False):
        return False
    path = Path(actual[0]['path'])
    return (digest(path) == declared[0]['sha256'] == actual[0]['sha256'] == summary['inputs'].get('repair_manifest')
            and json.loads(path.read_text()) == summary['repair_manifest'])


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
    repair = summary.get('repair_manifest')
    checks['roi_points_present'] = (len(roi) == len(repair['points']) if repair is not None else len(roi) >= 3) and all(p['policy'].startswith('s') for p in roi)
    if repair is not None:
        checks['repair_manifest_pinned'] = repair_manifest_matches_stage(summary, spec, json.loads((stage / 'staging.json').read_text()))
        checks['repair_no_decision'] = summary['analysis'] is None
        checks['repair_exact_selection'] = repair_summary_matches(repair, summary['points'])
    if smoke:
        per_point_tick = sum(summary['score_seconds'].values()) / (len(summary['points']) * len(summary['ticks']))
        projected = 1.35 * (summary['seconds'] - sum(summary['score_seconds'].values())
                            + per_point_tick * len(TICKS) * len(summary['points']))
        checks['projection'] = projected < spec['full']['seconds']
        projection: float | None = projected
    else:
        projection = None
        if repair is None:
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
    p.add_argument('--repair-manifest', type=Path, help='frozen exact point and control identities for repair scoring')
    p.add_argument('--judged', type=int, choices=(0, 3), default=3, help='first N ticks; 0 means all 80')
    a = p.parse_args()
    if a.validate:
        validate()
    elif any(getattr(a, n.replace('-', '_')) is None for n in ('al', 'masks', 'hybrid', 'lpips', 'dists', 'encode-half')):
        p.error('every frozen input is required')
    elif a.repair_manifest and (a.prior or a.windows_file is None or control_tolerance(a.window) != 1e-3):
        p.error('repair requires a manifest window with 1e-3 controls and does not accept --prior')
    else:
        run(a)


if __name__ == '__main__':
    main()

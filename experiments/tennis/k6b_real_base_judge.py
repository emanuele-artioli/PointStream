"""Actual-base saved-player quality pilot, gated on prior-control reproduction."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import tarfile
import tempfile
import math
import time
from typing import Any

import numpy as np

from experiments.tennis.k6b_rate_transport import array_sha, sha, write
from experiments.tennis.k6b_real_base_compose import compose


def extract(archive: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter='data')


def plan(a: argparse.Namespace) -> dict[str, Any]:
    """Windows, sources and generated archives: the frozen Al/Med pair (`--al --med --gen-al --gen-med`) or a window manifest
    (`--windows-file`, `--window`, `--source`, `--gen`, one each per selected window, in order)."""
    if a.windows_file is None:
        from experiments.tennis.k6b_foreground_video import PILOTS, SOURCE_SHA
        return {'manifest': False, 'ids': list(PILOTS), 'shas': list(SOURCE_SHA), 'prefixes': ['al', 'med'],
                'sources': [a.al, a.med], 'gens': [a.gen_al, a.gen_med]}
    from experiments.tennis import window_manifest as wm

    windows = wm.select(wm.load(a.windows_file), a.window or None)
    if len(a.source or []) != len(windows) or len(a.gen or []) != len(windows):
        raise ValueError('one --source and one --gen per selected window, in order')
    return {'manifest': True, 'ids': [w.id for w in windows], 'shas': [w.source_sha256 for w in windows],
            'prefixes': [w.name for w in windows], 'sources': list(a.source), 'gens': list(a.gen)}


def rate_components(a: argparse.Namespace, previous: dict[str, Any], wid: str) -> dict[str, float]:
    """Bank and adapter kbps of the 7200 s amortisation and the paid keypoint rate for window `wid`: from the prior run's
    assessment (frozen pair) or from `--rates` (JSON {window id: {...}} written by the generalised rate stage)."""
    if a.windows_file is None:
        return next(r for r in previous['assessment']['rows'] if r['window'] == wid)['rate_components']
    return json.loads(a.rates.read_text())[wid]


def identity_controls(images: list[np.ndarray], base: np.ndarray, composed: np.ndarray, again: np.ndarray,
                      scores: tuple[Any, Any]) -> dict[str, bool]:
    """In-job controls for windows with no prior run: the hybrid with every generated crop removed is the paid base,
    and scoring the same plain base twice gives identical numbers."""
    return {'hybrid_without_players_is_base': bool(np.array_equal(composed, base)),
            'plain_base_scored_twice_identical': bool(np.array_equal(scores[0], scores[1])) and bool(np.array_equal(images[0], again))}


def run(a: argparse.Namespace) -> None:
    P = plan(a)
    import cv2
    import torch
    from experiments.audit.env_smoke import device_record
    from experiments.tennis import k6, k6b, k6b_matte_judge as judge
    from experiments.tennis.k4 import safe
    from experiments.tennis.k4b_features import convert
    from experiments.tennis.k6b_matte_cpu import box_maps
    from experiments.tennis.k6b_rate_streams import decode_keypoints, decode_masks
    from experiments.tennis.k6b_receiver_sam import project, transport_root
    from experiments.visor.h3 import rgb_memmap
    from experiments.visor.h3b import dists_crops, load_dists, lpips_reduce
    from src.codecs.quality import load_lpips
    from src.segmentation.masks import decompress

    began = time.monotonic()
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    if not torch.cuda.is_available():
        raise ValueError('real CUDA metrics required')
    torch.cuda.set_per_process_memory_fraction(min(1.0, 20000 * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    device = device_record('RTX A6000' if 'A6000' in torch.cuda.get_device_name() else 'RTX 6000 Ada')
    if not device['class_matches']:
        raise ValueError('unqualified CUDA class')
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch / 'publish'
    publish.mkdir()
    anchors = judge.extract([a.anchor], scratch / 'anchors')
    controls = judge.extract([a.control], scratch / 'old-controls') if a.control is not None else {}
    extract(a.transport, scratch / 'wire')
    wire = transport_root(scratch / 'wire')
    extract(a.masks, scratch / 'masks')
    extract(a.gpu_base, scratch / 'gpu-base')
    with tarfile.open(a.base) as outer:
        member = outer.extractfile('publish/bundle.tar')
        assert member is not None
        with tarfile.open(fileobj=member, mode='r:') as inner:
            base_manifest_file = inner.extractfile('base/manifest.json')
            assert base_manifest_file is not None
            source_base_manifest = json.load(base_manifest_file)
    gpu_base = scratch / 'gpu-base' / 'publish'
    diag = json.loads((gpu_base / 'diagnostic.json').read_text())
    if len(diag['base']) != 6 or any(not r['source_matches'] or len(r['ticks']) != 80 for r in diag['source']):
        raise ValueError('GPU source control or base coverage missing')
    for row in diag['base']:
        path = gpu_base / f"{row['window']}-s{row['scale']}-gpu.npy"
        if sha(path.read_bytes()) != row['sha256']:
            raise ValueError('GPU base input hash changed')
    generated = k6b.load_generated([str(g) for g in P['gens']], scratch / 'generated')
    previous = json.loads(a.prior.read_text())
    prior_rows = {r['id']: r for r in previous['rows']}
    source_map = {sha(path.read_bytes()): path for path in P['sources']}
    nets = {'lpips': load_lpips(a.lpips, 'cuda'), 'dists': load_dists(a.dists, 'cuda')}
    if not all(next(net.parameters()).is_cuda for net in nets.values()):
        raise ValueError('metric network not on CUDA')
    controls_root = publish / 'calibration'
    controls_root.mkdir()
    masks_by_window: dict[str, dict[int, dict[int, np.ndarray]]] = {}
    units_by_window: dict[str, Any] = {}
    calibration: list[dict[str, Any]] = []
    # Both windows' original granted-matte and saved-prediction controls must
    # pass before any actual-base hybrid quality is opened.
    for wid, digest in zip(P['ids'], P['shas']):
        kp_meta, units = decode_keypoints((wire / 'packets' / safe(wid) / 'keypoints.kps').read_bytes())
        boxes = box_maps(units)
        masks: dict[int, dict[int, np.ndarray]] = {}
        for i, (unit, box) in enumerate(zip(units, boxes)):
            file = scratch / 'masks' / 'publish' / (unit['name'] + '-masks.npz')
            with np.load(file) as z:
                original_prediction = next(q for q in previous['prediction']['runs'][0]['rows'] if q['name'] == unit['name'])
                if z['t'].tolist() != list(range(0, 240, 3)) or array_sha(z['masks']) != original_prediction['output_sha256']:
                    raise ValueError('saved SAM ticks or receiver prediction hash changed')
                masks[i] = {int(t): project(m, box[int(t)]) for t, m in zip(z['t'], z['masks'])}
        if P['manifest']:  # no prior run to reproduce: the in-job controls below replace the gate
            calibration.append({'id': wid, 'gate': 'in-job controls (identity_controls), no prior run'})
            masks_by_window[wid] = masks
            units_by_window[wid] = (kp_meta, units, boxes)
            continue
        rec = judge.judge_window(wid, anchors[wid], controls[wid], wire / 'packets', scratch, {}, generated,
                                 source_map[digest], digest, nets, scratch, controls_root, a.judged,
                                 control_variant='full', predicted_masks=masks)
        old = prior_rows[wid]
        ticks = rec['judged']
        if any(rec['composite_hashes'][str(t)] != old['composite_hashes'][str(t)] for t in ticks):
            raise ValueError('old granted/predicted composite RGB hash mismatch')
        old_scores = scratch / 'masks' / 'publish' / safe(wid) / 'scores.npz'
        new_scores = controls_root / safe(wid) / 'scores.npz'
        with np.load(old_scores) as prior_score, np.load(new_scores) as current_score:
            for metric in ('lpips', 'dists'):
                if np.max(np.abs(prior_score[metric][:, ticks] - current_score[metric][:, ticks])) > 1e-4:
                    raise ValueError('original granted/predicted score control mismatch')
        calibration.append({'id': wid, 'ticks': ticks, 'exact_control': True, 'prediction_control': True,
                            'seconds': rec['seconds'], 'scored_frame_seconds': rec['scored_frame_seconds']})
        masks_by_window[wid] = masks
        units_by_window[wid] = (kp_meta, units, boxes)
    control_elapsed = time.monotonic() - began
    rows: list[dict[str, Any]] = []
    identity: dict[str, Any] = {}
    for wid, digest, prefix in zip(P['ids'], P['shas'], P['prefixes']):
        native = k6b.k6_window(anchors[wid])
        row = native['row']
        kp_meta, units, boxes = units_by_window[wid]
        source_masks = decode_masks((wire / 'packets' / safe(wid) / 'exact-rle.matte').read_bytes())[1]
        truth_dir = scratch / ('truth-' + prefix)
        truth_dir.mkdir()
        converted = convert('ffmpeg', {'id': wid, 'start': row['start'], 'frames': 240}, source_map[digest], truth_dir / 'source.yuv')
        truth = rgb_memmap(truth_dir / 'source.yuv', truth_dir / 'source.npy', 240, 1920, 1080, False, 'cuda')
        region_all = [{p: cv2.resize(source_masks[t, i].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
                       for i, p in enumerate(row['players']) if source_masks[t, i].any()} for t in range(240)]
        heights = {(p, t): float(np.ptp(np.nonzero(r[p].any(1))[0]) + 1) if p in r and r[p].any() else 1.0
                   for t, r in enumerate(region_all) for p in row['players']}
        coded = json.loads(decompress((anchors[wid] / 'keypoints.json.zst').read_bytes()))['decoded']
        decoded = {key: {(name.rsplit('@', 1)[0], int(name.rsplit('@', 1)[1])): np.asarray(v, np.float64)
                         for name, v in values.items()} for key, values in coded.items()}
        base_arrays = {scale: np.load(gpu_base / f'{prefix}-s{scale}-gpu.npy', mmap_mode='r') for scale in (8, 4, 2)}
        ticks = row['judged'][:a.judged or None]
        scores = {m: np.full((6, len(ticks)), np.nan) for m in ('full_lpips', 'foreground_lpips', 'full_dists', 'foreground_dists', 'full_psnr')}
        hashes: dict[str, Any] = {}
        frame_seconds: dict[str, float] = {}
        frame_started = time.monotonic()
        for j, t in enumerate(ticks):
            tick_started = time.monotonic()
            true = np.asarray(truth[t])
            if array_sha(true) != prior_rows[wid]['source_rgb_frame_sha256'][str(t)]:
                raise ValueError('source RGB original control failed')
            regions = region_all[t]
            r = np.logical_or.reduce([*regions.values(), np.zeros((1080, 1920), bool)])
            fill = k6.telea_fill(true, r)
            _, drawn = k6b.k6_oracle(t, native, k6.variant_names()[-9:], truth, region_all, fill,
                kp_meta['compositing_order'], row['scored'], heights, decoded,
                max(1, round(k6.REF_GAP_S * float(Fraction(row['fps'])))))
            s = r.copy()
            for m in drawn.values():
                s |= m
            if sha(np.packbits(r).tobytes()) != row['scoring_masks'][str(t)]['r'] or sha(np.packbits(s).tobytes()) != row['scoring_masks'][str(t)]['s']:
                raise ValueError('original scoring region changed')
            ys, xs = np.nonzero(s)
            crop = (slice(max(0, ys.min()-k6.DISTS_PAD), min(1080, ys.max()+1+k6.DISTS_PAD)),
                    slice(max(0, xs.min()-k6.DISTS_PAD), min(1920, xs.max()+1+k6.DISTS_PAD)))
            order = kp_meta['compositing_order']
            player_crops, player_boxes, player_masks = {}, {}, {}
            for i, unit in enumerate(units):
                player = row['players'][i]
                g = generated[(unit['name'], 'match5')]
                idx = np.flatnonzero(g['t'] == t)
                if len(idx) == 1:
                    player_crops[player] = g['crops'][int(idx[0])]
                if t in boxes[i]:
                    player_boxes[player] = boxes[i][t]
                player_masks[player] = masks_by_window[wid][i][t]
            images = []
            for scale in (8, 4, 2):
                base = np.asarray(base_arrays[scale][t // 3])
                method, _ = compose(base, player_crops, player_boxes, player_masks, order)
                images.extend([base, method])
            all_mask = np.ones((1080, 1920), dtype=bool)
            frame_hashes = []
            for i, image in enumerate(images):
                frame_hashes.append(array_sha(image))
                lp = lpips_reduce(nets['lpips'], [image], true, [[all_mask, s]], 'cuda')[0]
                if P['manifest'] and i == 0 and j == 0:
                    again = lpips_reduce(nets['lpips'], [image], true, [[all_mask, s]], 'cuda')[0]
                    empty, _ = compose(images[0], {}, player_boxes, player_masks, order)
                    controls_ok = identity_controls(images, images[0], empty, images[0], (np.asarray(lp), np.asarray(again)))
                    if not all(controls_ok.values()):
                        raise ValueError(f'in-job control failed for {wid}: {controls_ok}')
                    identity[wid] = controls_ok
                full_d = dists_crops(nets['dists'], [image], true, all_mask, 'cuda')[0]
                fg_d = dists_crops(nets['dists'], [image[crop]], true[crop], s[crop], 'cuda')[0]
                mse = float(np.mean((image.astype(np.float32) - true.astype(np.float32)) ** 2))
                scores['full_lpips'][i, j], scores['foreground_lpips'][i, j] = lp
                scores['full_dists'][i, j], scores['foreground_dists'][i, j] = full_d, fg_d
                scores['full_psnr'][i, j] = 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)
            hashes[str(t)] = frame_hashes
            frame_seconds[str(t)] = time.monotonic() - tick_started
        if any(not np.isfinite(v).all() for v in scores.values()):
            raise ValueError('nonfinite real metric')
        out = publish / safe(wid)
        out.mkdir()
        np.savez_compressed(out / 'scores.npz', **scores, ticks=np.asarray(ticks))
        rates = []
        old_rate = rate_components(a, previous, wid)
        duration = 240 / float(row['fps'])
        kps = (wire / 'packets' / safe(wid) / 'keypoints.kps').read_bytes()
        kps_rate = len(kps) * 8 / duration / 1000
        if abs(kps_rate - old_rate['keypoint_kbps']) > 1e-9:
            raise ValueError('actual KPS packet differs from paid historical rate')
        for scale in (8, 4, 2):
            base_row = next(r for r in diag['base'] if r['window'] == prefix and r['scale'] == scale)
            native_base = next(r for r in source_base_manifest['rows'] if r['window'] == prefix and r['scale'] == scale)
            header = json.dumps({'version': 1, 'window': wid, 'fps': row['fps'], 'scale': scale,
                                 'native_frames': 240, 'saved_ticks': '0:240:3',
                                 'kps_sha256': sha(kps), 'players': [u['name'] for u in units],
                                 'seeds': [k6b.unit_seed(u['name']) for u in units],
                                 'fallback': 'decoded_base', 'base_ivf_sha256': native_base['ivf']['sha256']},
                                sort_keys=True, separators=(',', ':')).encode()
            header_path = out / f's{scale}-bootstrap.json'
            header_path.write_bytes(header)
            if json.loads(header_path.read_bytes())['fallback'] != 'decoded_base':
                raise ValueError('receiver bootstrap corrupt')
            base_rate = native_base['ivf']['bytes'] * 8 / duration / 1000
            header_rate = len(header) * 8 / duration / 1000
            side = kps_rate + old_rate['match_bank_kbps_7200'] + old_rate['adapter_guider_kbps_7200'] + header_rate
            rates.append({'scale': scale, 'base_kbps': base_rate, 'side_kbps': side,
                          'total_kbps': base_rate + side, 'bootstrap_bytes': len(header),
                          'kps_bytes': len(kps), 'base_ivf_bytes': native_base['ivf']['bytes'],
                          'base_rgb_sha256': base_row['sha256']})
            if 'bank_worst_case_kbps' in old_rate:  # manifest windows: worst-case sensitivity row, own crops billed over the window alone
                rates[-1]['bank_worst_case_kbps'] = old_rate['bank_worst_case_kbps']
                rates[-1]['total_kbps_bank_worst_case'] = rates[-1]['total_kbps'] - old_rate['match_bank_kbps_7200'] + old_rate['bank_worst_case_kbps']
        item = {'id': wid, 'ticks': ticks, 'means': {m: v.mean(axis=1).tolist() for m, v in scores.items()},
                'rates': rates, 'composite_hashes': hashes, 'score_seconds': time.monotonic() - frame_started,
                'frame_seconds': frame_seconds,
                'converted_sha256': converted['sha256']}
        write(out / 'window.json', item)
        rows.append(item)
        if a.judged == 0:
            item['comparisons'] = compare_curves(item)
            if early_stop_applies(item, a.no_early_stop):
                item['early_stop'] = 'both metrics lost in one family; three of four impossible'
            write(out / 'window.json', item)
        from experiments.jobs.monitor import publish_progress
        publish_progress(os.environ['PS_STAGE'], len(rows))
        if a.judged == 0 and item.get('early_stop') and not a.no_early_stop:
            del truth
            break
        del truth
    seconds = time.monotonic() - began
    per_frame = sum(r['score_seconds'] for r in rows) / sum(len(r['ticks']) for r in rows)
    calibration_per_frame = 0.0 if P['manifest'] else sum(r['seconds'] for r in calibration) / sum(len(r['ticks']) for r in calibration)
    projected = (control_elapsed + (per_frame + calibration_per_frame) * 160) * 1.35
    summary = {'rows': rows, 'calibration': calibration, 'seconds': seconds,
               'projected_full_seconds': projected if a.cost_model == 'legacy' else None,
               'legacy_projected_full_seconds': projected,
               'cost_model': a.cost_model,
               'device': device, 'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30, 'frames_per_window': a.judged, 'scope': 'optimistic rate floor: fixed research header, 7200s bank/adaptation; tensor bootstrap and full framing unmeasured, no ball update or full semantic wire claim'}
    if P['manifest']:
        summary['in_job_controls'] = identity
    write(publish / 'quality.json', summary)
    write(Path(os.environ['PS_STAGE_DIR']) / 'quality-summary.json', summary)



def early_stop_applies(item: dict[str, Any], disabled: bool) -> bool:
    """The registered arithmetic early stop: both metrics lost in one family at scale 4. W5 scores the second
    window the first one's stop skipped, so it passes `--no-early-stop`; the default is unchanged."""
    if disabled:
        return False
    return any(all(item['comparisons'][family + '_' + metric]['scale4']['won'] is False for metric in ('lpips', 'dists'))
               for family in ('full', 'foreground'))


def compare_curves(row: dict[str, Any]) -> dict[str, Any]:
    """PCHIP on overlapping quality; central scale4 is the frozen point."""
    from scipy.interpolate import PchipInterpolator
    from experiments.background.g5 import envelope, pareto
    from experiments.tennis.k6b_confirm_validate import strict_won

    out: dict[str, Any] = {}
    for metric in ('full_lpips', 'foreground_lpips', 'full_dists', 'foreground_dists'):
        values = row['means'][metric]
        anchors = {str(r['scale']): {'kbps': r['base_kbps'], metric: values[2 * i]} for i, r in enumerate(row['rates'])}
        points = [(r['total_kbps'], -values[2 * i + 1]) for i, r in enumerate(row['rates'])]
        anchor_points = [(r['base_kbps'], -values[2 * i]) for i, r in enumerate(row['rates'])]
        result = strict_won(anchors, metric, row['rates'][1]['total_kbps'], values[3])
        sensitivity = {str(offset): strict_won(anchors, metric, row['rates'][1]['total_kbps'], values[3] + offset) for offset in (-0.001, 0, 0.001)}
        test, hull = pareto(points), envelope(anchor_points)
        bd = None
        if len(test) >= 2 and len(hull) >= 2:
            lo, hi = max(test[0][1], hull[0][1]), min(test[-1][1], hull[-1][1])
            if hi > lo:
                def area(p: list[tuple[float, float]]) -> float:
                    return float(PchipInterpolator([q for _, q in p], [math.log(rate) for rate, _ in p]).integrate(lo, hi))
                bd = math.exp((area(test) - area(hull)) / (hi - lo)) - 1
        out[metric] = {'scale4': result, 'sensitivity': sensitivity, 'bd_rate_fraction': bd,
                       'scope': 'three recovered SVT points only; no extrapolation'}
    return out


def projection_checks(worker_seconds: float, publication_seconds: float,
                      full_reservation_seconds: float) -> tuple[dict[str, bool], float]:
    """Keep the frozen estimator, gated by this request's saved full allowance."""
    projected = worker_seconds + 40 * publication_seconds * 1.35
    valid = all(math.isfinite(value) and value >= 0 for value in
                (worker_seconds, publication_seconds, full_reservation_seconds, projected))
    return {
        'projection_including_publication': valid and projected < full_reservation_seconds,
        'projection_under_full_reservation': valid and worker_seconds < full_reservation_seconds,
    }, projected


def validate() -> None:
    from experiments.tennis.k6b_real_base_cost import requested_model, resource_checks, separated_projection, timing_sequences

    stage = Path(os.environ['PS_STAGE_DIR'])
    summary = json.loads((stage / 'quality-summary.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    smoke_wall = time.time() - dispatch['started']
    publication = max(0.0, smoke_wall - summary['seconds'])
    spec = json.loads((stage.parent / 'spec.json').read_text())
    full_reservation_seconds = spec['full']['seconds']
    model = requested_model(spec)
    projected_with_publication: float | None = None
    cost_checks: dict[str, bool] = {}
    if model == 'legacy':
        cost_checks, projected_with_publication = projection_checks(
            summary['projected_full_seconds'], publication, full_reservation_seconds)
    expected_ticks = [0, 3]
    projection: dict[str, Any] | None = None
    projection_error: str | None = None
    full_stage = os.environ.get('PS_STAGE') == 'full'  # the full stage scores every third frame of the 240-frame window; no projection to check
    if model == 'separated-v1':
        expected_ticks = list(range(0, 240, 3)) if full_stage else [0, 3, 6]
        try:
            execution = json.loads((stage / 'execution.json').read_text())
            staging_record = json.loads((stage / 'staging.json').read_text())
            if not full_stage:
                ids = [r['id'] for r in summary['rows']] if summary.get('in_job_controls') else None
                projection = separated_projection(execution['seconds'], timing_sequences(summary, ids), staging_record['publication_seconds'],
                                                  expected_sequences=len(ids) if ids else 4)
                cost_checks = resource_checks(projection, full_reservation_seconds)
                projected_with_publication = projection['projected_full_seconds']
        except (KeyError, TypeError, ValueError, OSError) as error:
            cost_checks = {'separated_resource_evidence': False}
            projection_error = str(error)
            projected_with_publication = None
    cost_checks['saved_model_matches'] = summary.get('cost_model', 'legacy') == model
    cost_checks['saved_smoke_scale_matches'] = full_stage or spec['scale']['judged']['smoke'] == len(expected_ticks)
    in_job = summary.get('in_job_controls') or None  # set only for manifest windows, which have no prior run to reproduce
    checks = {**cost_checks, 'two_windows': len(summary['rows']) == (len(summary['calibration']) if in_job else 2),
              'both_old_controls': (len(in_job) == len(summary['rows']) and all(all(c.values()) for c in in_job.values())) if in_job else len(summary['calibration']) == 2 and all(r['exact_control'] and r['prediction_control'] for r in summary['calibration']),
              'representative_ticks': summary['frames_per_window'] == len(expected_ticks),
              'qualified_gpu': summary['device']['class_matches'], 'memory_within_limit': summary['peak_gpu_gib'] < 20}
    with tempfile.TemporaryDirectory(prefix='k6b-realbase-validator-') as d:
        root = Path(d)
        extract(stage / 'published.tar', root)
        for row in summary['rows']:
            from experiments.tennis.k4 import safe
            item = root / 'publish' / safe(row['id'])
            with np.load(item / 'scores.npz') as scores:
                checks[row['id'] + ':finite'] = all(np.isfinite(scores[m]).all() and scores[m].shape == (6, len(expected_ticks)) for m in ('full_lpips', 'foreground_lpips', 'full_dists', 'foreground_dists', 'full_psnr'))
                checks[row['id'] + ':ticks'] = scores['ticks'].tolist() == row['ticks'] and row['ticks'] == expected_ticks
            checks[row['id'] + ':rates'] = len(row['rates']) == 3 and all(r['total_kbps'] > r['base_kbps'] > 0 and r['bootstrap_bytes'] > 0 for r in row['rates'])
            checks[row['id'] + ':composites'] = len(row['composite_hashes']) == len(expected_ticks) and all(len(v) == 6 for v in row['composite_hashes'].values())
    projected_worker = projection['projected_worker_seconds'] if projection else (summary['projected_full_seconds'] if model == 'legacy' else None)
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps({'passed': all(checks.values()), 'checks': checks, 'projected_full_seconds': projected_worker, 'projected_with_publication': projected_with_publication, 'full_reservation_seconds': full_reservation_seconds, 'cost_model': model, 'projection': projection, 'projection_error': projection_error}))
    if not all(checks.values()):
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--validate', action='store_true')
    for name in ('anchor', 'control', 'transport', 'masks', 'gpu-base', 'base', 'gen-al', 'gen-med', 'prior', 'al', 'med', 'lpips', 'dists'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--windows-file', type=Path, help='window manifest instead of the frozen Al/Med pair (needs --source and --gen per window, --rates)')
    p.add_argument('--window', action='append')
    p.add_argument('--source', type=Path, action='append')
    p.add_argument('--gen', type=Path, action='append')
    p.add_argument('--rates', type=Path, help='per window id: keypoint-independent bank and adapter kbps (the generalised rate stage)')
    p.add_argument('--judged', type=int, default=2)
    p.add_argument('--no-early-stop', action='store_true', help='score the second window even when the first fails the rule')
    p.add_argument('--cost-model', choices=('legacy', 'separated-v1'), default='legacy')
    a = p.parse_args()
    if a.validate:
        validate()
    else:
        allowed = (0, 3) if a.cost_model == 'separated-v1' else (0, 2)
        required = ('anchor', 'transport', 'masks', 'gpu_base', 'base', 'prior', 'lpips', 'dists') + (('rates',) if a.windows_file else ('control', 'gen_al', 'gen_med', 'al', 'med'))
        if any(getattr(a, n) is None for n in required) or a.judged not in allowed:
            p.error('all inputs and the resource model\'s original 0/2 or separated 0/3 judged frames are required')
        run(a)


if __name__ == '__main__':
    main()

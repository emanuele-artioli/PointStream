"""W3: DCVC-UF streams on Al and Med that decode identically in the encoding job.

Rule, deviation and protocol: docs/experiments/w3-neural-baseline.md. Per window
the job (1) estimates the rate of every (scale, QP) candidate from 25 frames and
picks the points by rate alone, (2) encodes 240 native frames per point to a real
bitstream and decodes the file in two separate processes (two passes each),
(3) re-encodes the three SVT bases of the audited real-base job, and (4) scores
whole-frame and fixed-foreground LPIPS and DISTS on the 80 ticks 0:240:3 with the
real-base judge's code. Development windows, not untouched footage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.tennis import w3_analysis as an

SOURCES: dict[str, dict[str, Any]] = {
    'al': {'wid': 'alcaraz_hurkacz/031@129', 'start': 9, 'fps': 25,
           'source': '054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76',
           'yuv': 'e7b960cd2456b439c5439b337a45e47d2a0a44a2a32025a2de595af3feed4fa2'},
    'med': {'wid': 'medvedev_struff/014@194', 'start': 74, 'fps': 50,
            'source': 'a2898b5cdfb463f59b5084fddfe23a2442cf2ba5feaaddb9d5414873e52dd0be',
            'yuv': '53633ce617b12d779bc8c5746d8d44cbe090ced6c49ba79dbe246cc6abacfb08'},
}
INPUTS = {
    'anchor': '67444a2a8f56c8700fb073a1184ce207ec3ad38cc003dbafa6710bb33836dff9',
    'transport': '6865035e3132b6e07600fdaabc8c9d7152b8b6e3c3260d9f3df21cab33ca33c7',
    'prior_frame': 'b0a55a5c1440463fff97efa975a661c2c65db9ce35e0696fade9d5648c39ccdc',
    'prior_quality': 'e346844d20671e5dfd78828ec61d262aa13109bc1be9a1b421873facdc86daa1',
    'image_ckpt': 'b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302',
    'video_ckpt': '934bde4a12fc0b6b0c679ec6ae21d41cb7722debf8b631ee0dd750d030b1a6b3',
    'lpips': '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
    'dists': '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0',
}
TOOLS = {'SvtAv1EncApp': '4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4',
         'dav1d': 'e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6',
         'ffmpeg': '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2'}
SCALES = {8: (240, 136), 4: (480, 272), 2: (960, 544)}
CAL_QPS = (0, 6, 12, 18, 24, 30, 36, 42, 48, 56, 63)
CAL_FRAMES = 25
NATIVE_FRAMES = 240
# Audited Al base rates of job 20261027T211451Z-f6a754eb (kbps), scale 8, 4, 2.
AUDITED_AL_KBPS = {8: 10.1975, 4: 17.130833, 2: 33.685}
# Row of the audited scores.npz holding each SVT base (hybrid rows are the odd ones).
AUDITED_ROW = {8: 0, 4: 2, 2: 4}
GPU_CLASSES = {'RTX 6000 Ada': ([8, 9], 'sm89'), 'RTX A6000': ([8, 6], 'sm80')}
SCORE_CONTROL_TOLERANCE = 1e-3
PROJECTION_MARGIN = 1.35


class WorkerFailure(RuntimeError):
    pass


def activate_manifest(windows_file: Path, names: list[str] | None) -> list[str]:
    """Replace the frozen Al/Med `SOURCES` by manifest windows (key = window name). Without it nothing here changes. Manifest windows have
    no prior hybrid run, so the audited-SVT controls (Al only) do not apply to them and the anchor/transport/prior archives are not pinned."""
    from experiments.tennis import window_manifest as wm

    chosen = wm.select(wm.load(windows_file), names or None)
    SOURCES.clear()
    for w in chosen:
        SOURCES[w.name] = {'wid': w.id, 'start': w.start, 'fps': w.fps, 'source': w.source_sha256, 'yuv': w.source_yuv_sha256}
    for key in ('anchor', 'transport', 'prior_frame', 'prior_quality'):
        INPUTS.pop(key, None)
    return list(SOURCES)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')


def parse_scales(text: str) -> list[int]:
    scales = [int(x) for x in text.split(',')]
    if not scales or any(s not in SCALES for s in scales):
        raise ValueError('SVT scales are a subset of 8, 4, 2')
    return scales


def parse_point_list(text: str) -> dict[str, list[tuple[int, int]]]:
    """Named points ``window:scale:qp,...`` (W3b), per window in the order given; QP 0..63, no duplicates."""
    plan: dict[str, list[tuple[int, int]]] = {}
    for item in text.split(','):
        parts = item.split(':')
        if len(parts) != 3 or parts[0] not in SOURCES:
            raise ValueError(f'point {item!r} is not window:scale:qp with window in {sorted(SOURCES)}')
        scale, qp = int(parts[1]), int(parts[2])
        if scale not in SCALES or not 0 <= qp <= 63:
            raise ValueError(f'point {item!r}: scale must be one of {sorted(SCALES)} and QP in 0..63')
        if (scale, qp) in plan.setdefault(parts[0], []):
            raise ValueError(f'point {item!r} listed twice')
        plan[parts[0]].append((scale, qp))
    if set(plan) != set(SOURCES):
        raise ValueError('every window needs at least one point')
    return plan


def project_listed_seconds(timings: list[dict[str, float]], setup: float, publication: float, full_points: list[int],
                           svt_factor: float, svt_streams: int) -> float:
    """Full-stage seconds for a named point list, window by window (the windows differ in resolution).

    ``timings[w]`` are the smoke's seconds of window ``w``: ``convert``, ``native_fixed`` and ``native_variable``
    (per point; the smoke codes 25 frames), ``svt_scale8``, ``masks`` (9 ticks) and ``scoring`` (9 ticks, one SVT
    stream and one DCVC point = 2 streams). ``svt_factor`` scales the SVT scale-8 time to the full SVT scales
    (1 for scale 8 only, 21 for 8, 4, 2); ``svt_streams`` is how many SVT streams the full run scores."""
    values = [setup, publication, svt_factor, *(v for t in timings for v in t.values())]
    if any(v < 0 or v != v for v in values) or len(timings) != len(full_points):
        raise ValueError('finite nonnegative timings, one set per window, required')
    total = setup + 4 * publication
    for t, n in zip(timings, full_points):
        total += (t['convert'] + n * t['native_fixed'] + n * (NATIVE_FRAMES / CAL_FRAMES) * t['native_variable']
                  + svt_factor * t['svt_scale8'] + (80 / 9) * t['masks'] + (svt_streams + n) / 2 * (80 / 9) * t['scoring'])
    return PROJECTION_MARGIN * total


def project_full_seconds(t: dict[str, float], publication: float) -> float:
    """Full-stage seconds from smoke timings: one 30 kbps point and one SVT scale at 25 frames.

    Fixed costs (setup, rate calibration) once; DCVC process start-up three times;
    DCVC frame work x 240/25 x 3 points; SVT scale 8 x (1 + 4 + 16) for the three
    scales; foreground masks x 80/9 ticks; scoring x 80/9 ticks x 3 (two images per
    tick in the smoke, six in the full run)."""
    values = [*t.values(), publication]
    if any(v < 0 or v != v for v in values):
        raise ValueError('finite nonnegative timings required')
    return PROJECTION_MARGIN * (
        t['setup'] + t['calibration'] + 3 * t['native_fixed'] + 3 * (NATIVE_FRAMES / CAL_FRAMES) * t['native_variable']
        + 21 * t['svt_scale8'] + (80 / 9) * t['masks'] + 3 * (80 / 9) * t['scoring'] + 4 * publication)


def run_worker(action: str, plan: dict[str, Any], work: Path, tag: str, timeout: float) -> tuple[dict[str, Any], float]:
    from src.codecs.dcvc_uf_worker import dcvc_command

    plan_path, report = work / f'{tag}-plan.json', work / f'{tag}.json'
    write(plan_path, plan)
    command, env, cwd = dcvc_command(Path(sys.prefix), action, plan_path, report)
    started = time.monotonic()
    done = subprocess.run(command, env={**os.environ, **env}, cwd=cwd, capture_output=True, timeout=timeout)
    seconds = time.monotonic() - started
    (work / f'{tag}.log').write_bytes(done.stdout + done.stderr)
    if done.returncode or not report.exists():
        tail = (done.stdout + done.stderr)[-1500:].decode(errors='replace')
        raise WorkerFailure(f'{action} {tag} exited {done.returncode}: {tail}')
    return json.loads(report.read_text()), seconds


def base_plan(a: argparse.Namespace, width: int, height: int, frames: int, source: Path) -> dict[str, Any]:
    return {'structure': 'htl', 'frame_count': frames, 'width': width, 'height': height, 'src_type': 'yuv420',
            'frames_file': str(source.resolve()), 'image_ckpt': str(a.image_ckpt.resolve()),
            'image_sha256': INPUTS['image_ckpt'], 'video_ckpt': str(a.video_ckpt.resolve()),
            'video_sha256': INPUTS['video_ckpt'], 'profile': False, 'deterministic': True, 'stream_policy': 'legacy'}


def calibrate_candidates(a: argparse.Namespace, full_yuv: Path, work: Path, fps: int) -> tuple[list[dict[str, Any]], float]:
    from experiments.tennis.k6 import downscaled_source

    candidates, seconds = [], 0.0
    for scale, (width, height) in SCALES.items():
        raw = downscaled_source(full_yuv, CAL_FRAMES, scale, work / f'cal-s{scale}.yuv')
        plan = {**base_plan(a, width, height, CAL_FRAMES, raw), 'qps': list(CAL_QPS)}
        report, took = run_worker('calibrate', plan, work, f'cal-s{scale}', 600)
        seconds += took
        for row in report['rows']:
            candidates.append({'scale': scale, 'qp': row['qp'], 'native_bytes': row['native_bytes'],
                               'est_kbps': an.estimate_rate_kbps(row['nals'], target_frames=NATIVE_FRAMES, fps=fps)})
        raw.unlink()
    return candidates, seconds


def compact_decode(report: dict[str, Any]) -> dict[str, Any]:
    """A decode report without the per-frame hash lists (kept in the point's reports.json)."""
    return {'deterministic': report['deterministic'], 'container_sha256': report['container_sha256'],
            'environment': report['environment'], 'warnings': report.get('warnings', []),
            'passes': [{'decode_seconds': p['decode_seconds'], 'model_decode_seconds': p['model_decode_seconds'],
                        'frames_sha256_digest': hashlib.sha256(''.join(p['frame_sha256']).encode()).hexdigest(),
                        'first_frame_sha256': p['frame_sha256'][0]} for p in report['passes']]}


def code_point(a: argparse.Namespace, source: Path, work: Path, fps: int, n: int, point: dict[str, Any]) -> dict[str, Any]:
    """Encode one point, decode its file in two separate processes; acceptance (a) and (b) of the deviation."""
    from experiments.tennis.k6 import downscaled_source

    scale, qp = point['scale'], point['qp']
    width, height = SCALES[scale]
    frame_bytes = width * height * 3 // 2
    pdir = work / f'dcvc-s{scale}-q{qp}'
    pdir.mkdir()
    raw = downscaled_source(source, n, scale, pdir / 'input.yuv')
    plan = {**base_plan(a, width, height, n, raw), 'qp': qp, 'container': str(pdir / 'stream.psdc'),
            'i_recon_file': str(pdir / 'i_recon.yuv')}
    encoded, t_encode = run_worker('encode', plan, pdir, 'encode', 900)
    container = pdir / 'stream.psdc'
    record: dict[str, Any] = {
        'codec': 'dcvc-uf-htl', 'scale': scale, 'qp': qp, 'size': [width, height], 'frames': n,
        'target_kbps': point['target_kbps'], 'est_kbps': point['est_kbps'], 'stream': str(container.name),
        'bytes': container.stat().st_size, 'kbps': container.stat().st_size * 8 * fps / n / 1000,
        'container_sha256': digest(container), 'encode_report': encoded, 'decode_reports': {},
        'dpb_reachable': encoded['dpb_reachable'], 'seconds': {'encode_process': t_encode}}
    if record['container_sha256'] != encoded['container_sha256']:
        raise ValueError('container identity changed after the encoder wrote it')
    variable = encoded['encode_seconds']
    process = t_encode
    decodes: dict[str, dict[str, Any]] = {}
    try:
        for tag in ('decode-a', 'decode-b'):
            report, took = run_worker('decode', {**plan, 'out_file': str(pdir / f'{tag}.yuv')}, pdir, tag, 900)
            decodes[tag] = report
            process += took
            variable += sum(p['decode_seconds'] for p in report['passes'])
    except WorkerFailure as error:
        record.update(accepted=False, failure='decode process failed: ' + str(error)[:1500])
        record['decode_reports'] = {tag: compact_decode(r) for tag, r in decodes.items()}
        return record
    write(pdir / 'reports.json', {'encode': encoded, 'decode': decodes})
    record['decode_reports'] = {tag: compact_decode(r) for tag, r in decodes.items()}
    record['seconds'].update(process=process, variable=variable)
    a_file, b_file = pdir / 'decode-a.yuv', pdir / 'decode-b.yuv'
    first = np.fromfile(a_file, np.uint8, count=frame_bytes).astype(np.int16)
    intra = np.fromfile(pdir / 'i_recon.yuv', np.uint8).astype(np.int16)
    if intra.size != frame_bytes:
        raise ValueError('encoder I reconstruction has the wrong size')
    check_a = an.exactness([int(np.abs(first - intra).max())])
    diffs = an.frame_max_diffs(a_file, b_file, frame_bytes, n)
    check_b = an.exactness(diffs, levels=0)
    hashes = [r['passes'][i]['frame_sha256'] for r in decodes.values() for i in (0, 1)]
    identical_hashes = all(h == hashes[0] for h in hashes) and len(hashes[0]) == n
    psnr = an.luma_psnr_per_frame(a_file, raw, width, height, n)
    within = [bool(r['deterministic']) for r in decodes.values()]
    record.update(
        check_a_i_frame=check_a, check_b_two_processes=check_b, identical_frame_hashes=identical_hashes,
        within_process_deterministic=within,
        max_decode_diff=max(check_a['max_abs_diff'], check_b['max_abs_diff']),
        luma_psnr={'min': min(psnr), 'first_group_mean': float(np.mean(psnr[:8])), 'last_group_mean': float(np.mean(psnr[-8:]))},
        accepted=bool(check_a['passed'] and check_b['passed'] and identical_hashes and all(within)),
        decoded=str(a_file))
    raw.unlink()
    (pdir / 'decode-b.yuv').unlink()
    return record


def code_svt(source: Path, work: Path, fps: int, scale: int) -> dict[str, Any]:
    """The audited base: SVT-AV1 preset 4, CRF 63 at 1/scale, 240 frames, decoded by dav1d."""
    from experiments.tennis.k6 import downscaled_source
    from src.codecs import svtav1

    width, height = SCALES[scale]
    pdir = work / f'svt-s{scale}'
    pdir.mkdir()
    raw = downscaled_source(source, NATIVE_FRAMES, scale, pdir / 'input.yuv')
    rec = svtav1.code(raw, pdir, width=width, height=height, fps=Fraction(fps, 1), frames=NATIVE_FRAMES, crf=63,
                      preset=4, threads=4, full_range=False, encoder='SvtAv1EncApp', decoder='dav1d', timeout=1500,
                      cpus=sorted(os.sched_getaffinity(0))[:4])
    stream = Path(rec['stream'])
    if len(svtav1.ivf_frames(stream.read_bytes())) != NATIVE_FRAMES or Path(rec['decoded']).stat().st_size != NATIVE_FRAMES * width * height * 3 // 2:
        raise ValueError('SVT frame count or decoded size is wrong')
    raw.unlink()
    size = stream.stat().st_size
    return {'codec': 'svt-av1', 'scale': scale, 'qp': 63, 'size': [width, height], 'frames': NATIVE_FRAMES,
            'stream': stream.name, 'bytes': size, 'kbps': size * 8 / (NATIVE_FRAMES / fps) / 1000,
            'stream_sha256': rec['stream_sha256'], 'encode_command': rec['encode_command'],
            'seconds': {'encode': rec['encode_seconds'], 'decode': rec['decode_seconds']}, 'decoded': rec['decoded']}


def foreground_masks(wid: str, anchor_dirs: dict[str, Path], wire: Path, truth: Any, ticks: list[int], prior_rows: dict[str, Any]) -> dict[int, np.ndarray]:
    """The fixed foreground S of the real-base judge at each tick, hash-checked against the original scoring masks."""
    from experiments.tennis import k6, k6b
    from experiments.tennis.k6b_rate_streams import decode_keypoints, decode_masks
    from experiments.tennis.k6b_rate_transport import array_sha, sha
    from src.segmentation.masks import decompress

    safe = wid.replace('/', '__')
    native = k6b.k6_window(anchor_dirs[wid])
    row = native['row']
    if row['judged'][:len(ticks)] != ticks:
        raise ValueError('judged ticks differ from 0:240:3')
    kp_meta, _units = decode_keypoints((wire / 'packets' / safe / 'keypoints.kps').read_bytes())
    source_masks = decode_masks((wire / 'packets' / safe / 'exact-rle.matte').read_bytes())[1]
    region_all = [{p: cv2.resize(source_masks[t, i].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
                   for i, p in enumerate(row['players']) if source_masks[t, i].any()} for t in range(240)]
    heights = {(p, t): float(np.ptp(np.nonzero(r[p].any(1))[0]) + 1) if p in r and r[p].any() else 1.0
               for t, r in enumerate(region_all) for p in row['players']}
    coded = json.loads(decompress((anchor_dirs[wid] / 'keypoints.json.zst').read_bytes()))['decoded']
    decoded = {key: {(name.rsplit('@', 1)[0], int(name.rsplit('@', 1)[1])): np.asarray(v, np.float64)
                     for name, v in values.items()} for key, values in coded.items()}
    gap = max(1, round(k6.REF_GAP_S * float(Fraction(row['fps']))))
    masks: dict[int, np.ndarray] = {}
    for t in ticks:
        true = np.asarray(truth[t])
        if array_sha(true) != prior_rows[wid]['source_rgb_frame_sha256'][str(t)]:
            raise ValueError(f'source RGB control failed at tick {t}')
        r = np.logical_or.reduce([*region_all[t].values(), np.zeros((1080, 1920), bool)])
        fill = k6.telea_fill(true, r)
        _, drawn = k6b.k6_oracle(t, native, k6.variant_names()[-9:], truth, region_all, fill,
                                 kp_meta['compositing_order'], row['scored'], heights, decoded, gap)
        s = r.copy()
        for m in drawn.values():
            s |= m
        if sha(np.packbits(r).tobytes()) != row['scoring_masks'][str(t)]['r'] or sha(np.packbits(s).tobytes()) != row['scoring_masks'][str(t)]['s']:
            raise ValueError(f'original scoring region changed at tick {t}')
        masks[t] = s
    return masks


def decoded_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """One limited-range 4:2:0 frame (h*3/2, w) to uint8 RGB on the GPU, as the baselines convert it."""
    import torch
    from src.codecs.quality import yuv420_to_rgb

    plane = torch.from_numpy(frame[None]).cuda()
    return yuv420_to_rgb(plane, width, height, full_range=False)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()


def score_stream(nets: dict[str, Any], decoded: Path, size: list[int], n: int, ticks: list[int], truth: Any,
                 masks: dict[int, np.ndarray]) -> dict[str, list[float]]:
    """Whole-frame and fixed-foreground LPIPS and DISTS of one decoded file, as the real-base judge computes them."""
    from experiments.tennis import k6
    from experiments.visor.h3b import dists_crops, lpips_reduce

    width, height = size
    pixels = np.memmap(decoded, np.uint8, 'r', shape=(n, height * 3 // 2, width))
    scores: dict[str, list[float]] = {m: [] for m in an.METRICS}
    all_mask = np.ones((1080, 1920), bool)
    for t in ticks:
        image = cv2.resize(decoded_rgb(np.array(pixels[t], copy=True), width, height), (1920, 1080), interpolation=cv2.INTER_CUBIC)
        true, s = np.asarray(truth[t]), masks[t]
        ys, xs = np.nonzero(s)
        crop = (slice(max(0, ys.min() - k6.DISTS_PAD), min(1080, ys.max() + 1 + k6.DISTS_PAD)),
                slice(max(0, xs.min() - k6.DISTS_PAD), min(1920, xs.max() + 1 + k6.DISTS_PAD)))
        whole, foreground = lpips_reduce(nets['lpips'], [image], true, [[all_mask, s]], 'cuda')[0]
        scores['full_lpips'].append(float(whole))
        scores['foreground_lpips'].append(float(foreground))
        scores['full_dists'].append(float(dists_crops(nets['dists'], [image], true, all_mask, 'cuda')[0]))
        scores['foreground_dists'].append(float(dists_crops(nets['dists'], [image[crop]], true[crop], s[crop], 'cuda')[0]))
    if not all(np.isfinite(v).all() for v in scores.values()):
        raise ValueError('nonfinite score')
    return scores


def run_window(a: argparse.Namespace, key: str, ctx: dict[str, Any]) -> dict[str, Any]:
    import torch
    from experiments.jobs.monitor import publish_progress
    from experiments.tennis.k4b_features import convert_command
    from experiments.visor.h3 import rgb_memmap
    from experiments.visor.h3b import load_dists
    from src.codecs.quality import load_lpips

    spec = SOURCES[key]
    n, fps = a.frames, spec['fps']
    ticks = list(range(0, n, 3))
    work = ctx['scratch'] / key
    work.mkdir()
    out = ctx['publish'] / key
    out.mkdir()
    timings: dict[str, float] = {}
    started = time.monotonic()
    full = work / 'source-full.yuv'
    subprocess.run(convert_command('ffmpeg', getattr(a, key), full, spec['start'], NATIVE_FRAMES), check=True, capture_output=True, timeout=300)
    if digest(full) != spec['yuv']:
        raise ValueError('native source YUV identity changed: ' + key)
    timings['convert'] = time.monotonic() - started
    started = time.monotonic()
    if a.point_list:
        # W3b: the points are named (chosen on measured rates in the rule entry), so no calibration encodes run.
        candidates: list[dict[str, Any]] = []
        chosen = [{'scale': scale, 'qp': qp, 'target_kbps': None, 'est_kbps': None}
                  for scale, qp in parse_point_list(a.point_list)[key]]
    else:
        candidates, _ = calibrate_candidates(a, full, work, fps)
        chosen = an.select_points(candidates, a.points)
    timings['calibration'] = time.monotonic() - started
    write(out / 'candidates.json', {'candidates': candidates, 'chosen': chosen, 'cal_frames': CAL_FRAMES, 'cal_qps': list(CAL_QPS)})
    result: dict[str, Any] = {'window': key, 'wid': spec['wid'], 'fps': fps, 'frames': n, 'ticks': ticks,
                              'candidates': candidates, 'chosen': chosen, 'dcvc': [], 'svt': [], 'accepted': True,
                              'bracket_ok': None if a.point_list else an.planned_bracket(candidates)}
    source = full if n == NATIVE_FRAMES else work / 'source.yuv'
    if n != NATIVE_FRAMES:
        with full.open('rb') as src, source.open('wb') as dst:
            dst.write(src.read(n * 1920 * 1620))
    # The foreground oracle draws from reference frames anywhere in the window, so the truth covers all 240.
    # Masks and metric networks come first so that every item can be scored, and checkpointed, as it finishes.
    begin = time.monotonic()
    truth = rgb_memmap(full, work / 'truth.npy', NATIVE_FRAMES, 1920, 1080, False, 'cuda')
    masks = foreground_masks(spec['wid'], ctx['anchors'], ctx['wire'], truth, ticks, ctx['prior_rows'])
    np.savez_compressed(out / 'foreground-masks.npz', ticks=np.asarray(ticks), packed=np.stack([np.packbits(masks[t]) for t in ticks]))
    timings['masks'] = time.monotonic() - begin
    nets = {'lpips': load_lpips(ctx['lpips'], 'cuda'), 'dists': load_dists(ctx['dists'], 'cuda')}
    if not all(next(net.parameters()).is_cuda for net in nets.values()):
        raise ValueError('CUDA metric networks required')

    def finish(record: dict[str, Any], tag: str, files: dict[str, Path], item_seconds: float) -> None:
        """Score one item, then publish and checkpoint it (a stop loses only the item in progress)."""
        before = time.monotonic()
        record['scores'] = score_stream(nets, Path(record['decoded']), record['size'], n, ticks, truth, masks)
        record['mean'] = {m: float(np.mean(v)) for m, v in record['scores'].items()}
        record['seconds_item'] = {'code': item_seconds, 'score': time.monotonic() - before}
        record.pop('decoded', None)
        for name, path in files.items():
            shutil.copyfile(path, out / name)
        save_item(ctx, key, tag, record, files)

    native_fixed = native_variable = score_seconds = svt_first = 0.0
    done = 0
    for point in chosen:
        tag = f'dcvc-s{point["scale"]}-q{point["qp"]}'
        loaded = load_item(ctx, key, tag, out)
        if loaded is not None:
            record = loaded
        else:
            before = time.monotonic()
            record = code_point(a, source, work, fps, n, point)
            if not record['accepted']:
                result['dcvc'].append(record)
                result['accepted'] = False
                result['stopped_at'] = {'scale': point['scale'], 'qp': point['qp']}
                write(out / 'result.json', result)
                return result
            finish(record, tag, {f'{tag}.psdc': work / tag / 'stream.psdc', f'{tag}-reports.json': work / tag / 'reports.json'},
                   time.monotonic() - before)
        result['dcvc'].append(record)
        native_fixed += max(0.0, record['seconds']['process'] - record['seconds']['variable'])
        native_variable += record['seconds']['variable']
        score_seconds += record['seconds_item']['score']
        done += 1
        publish_progress(os.environ['PS_STAGE'], done)
    timings['native_fixed'] = native_fixed / max(1, len(chosen))
    timings['native_variable'] = native_variable / max(1, len(chosen))
    for scale in ctx['svt_scales']:
        tag = f'svt-s{scale}'
        item = load_item(ctx, key, tag, out)
        if item is None:
            before = time.monotonic()
            item = code_svt(full, work, fps, scale)
            finish(item, tag, {f'{tag}.ivf': work / tag / item['stream']}, time.monotonic() - before)
        result['svt'].append(item)
        score_seconds += item['seconds_item']['score']
        if not svt_first:
            svt_first = item['seconds_item']['code']
        done += 1
        publish_progress(os.environ['PS_STAGE'], done)
    timings['svt_scale8'] = svt_first
    timings['scoring'] = score_seconds
    nets.clear()
    torch.cuda.empty_cache()
    result['timings'] = timings
    result['audited_controls'] = audited_controls(key, result, ctx, ticks)
    write(out / 'result.json', result)
    return result


def save_item(ctx: dict[str, Any], key: str, tag: str, record: dict[str, Any], files: dict[str, Path]) -> None:
    """Checkpoint one finished item (record with scores, stream, reports) under a dot-name, then rename it into place."""
    checkpoint = ctx.get('checkpoint')
    if checkpoint is None:
        return
    temporary = checkpoint / f'.{key}__{tag}.tmp'
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir()
    write(temporary / 'record.json', record)
    for name, path in files.items():
        shutil.copyfile(path, temporary / name)
    os.replace(temporary, checkpoint / f'{key}__{tag}')


def load_item(ctx: dict[str, Any], key: str, tag: str, out: Path) -> dict[str, Any] | None:
    """A checkpointed item of an earlier attempt: its record, with its published files restored under ``out``."""
    checkpoint = ctx.get('checkpoint')
    saved = checkpoint / f'{key}__{tag}' if checkpoint is not None else None
    if saved is None or not (saved / 'record.json').is_file():
        return None
    for path in saved.iterdir():
        if path.name != 'record.json':
            shutil.copyfile(path, out / path.name)
    record: dict[str, Any] = json.loads((saved / 'record.json').read_text())
    record['resumed_from_checkpoint'] = True
    return record


def audited_controls(key: str, result: dict[str, Any], ctx: dict[str, Any], ticks: list[int]) -> dict[str, Any]:
    """Al's SVT bases against the audited real-base job: rates and per-tick scores."""
    if key != 'al':
        return {'applies': False}
    controls: dict[str, Any] = {'applies': True, 'scales': {}}
    for record in result['svt']:
        scale = record['scale']
        rate_error = abs(record['kbps'] - AUDITED_AL_KBPS[scale])
        entry: dict[str, Any] = {'rate_error_kbps': rate_error, 'bytes_match': rate_error < 1e-4}
        errors = {}
        for metric in an.METRICS:
            prior = ctx['prior_scores'][metric][AUDITED_ROW[scale]][:len(ticks)]
            errors[metric] = float(np.max(np.abs(np.asarray(record['scores'][metric]) - prior)))
        entry['score_max_error'] = errors
        entry['scores_match'] = max(errors.values()) <= SCORE_CONTROL_TOLERANCE
        controls['scales'][str(scale)] = entry
        if entry['bytes_match'] and not entry['scores_match']:
            raise ValueError(f'SVT base scale {scale}: identical bytes but scores differ from the audited job: {errors}')
    return controls


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    envelopes: dict[str, Any] = {}
    for r in results:
        dcvc, svt = r['dcvc'], r['svt']
        if r['accepted'] and len(dcvc) >= 3 and len(svt) >= 3:
            test = {m: [(p['kbps'], p['mean'][m]) for p in dcvc] for m in an.METRICS}
            anchor = {m: [(p['kbps'], p['mean'][m]) for p in svt] for m in an.METRICS}
            envelopes[r['window']] = an.envelope_comparison(test, anchor)
    return {'envelopes': envelopes}


def load_prior(a: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import io

    rows = {r['id']: r for r in json.loads(a.prior_frame.read_text())['rows']}
    if a.prior_quality is None:  # manifest windows: no audited hybrid scores to compare the SVT bases with
        return rows, {}
    with tarfile.open(a.prior_quality) as tar:
        member = tar.extractfile('publish/alcaraz_hurkacz__031@129/scores.npz')
        if member is None:
            raise ValueError('audited scores missing')
        with np.load(io.BytesIO(member.read())) as z:
            scores = {m: np.asarray(z[m]) for m in an.METRICS}
    return rows, scores


def run(a: argparse.Namespace) -> None:
    import torch

    began = time.monotonic()
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('exactly one claimed CUDA device required')
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    name, capability = torch.cuda.get_device_name(0), list(torch.cuda.get_device_capability(0))
    gpu_class = next((c for c, (cap, _) in GPU_CLASSES.items() if c in name and capability == cap), None)
    if gpu_class is None or not uuid.startswith('GPU-'):
        raise ValueError(f'unqualified GPU {name} {capability} {uuid!r}')
    for key, expected in INPUTS.items():
        if digest(getattr(a, key)) != expected:
            raise ValueError('pinned input changed: ' + key)
    for key, spec in SOURCES.items():
        if digest(getattr(a, key)) != spec['source']:
            raise ValueError('source video changed: ' + key)
    from src.codecs import svtav1
    tools = {k: svtav1.tool(k) for k in TOOLS}
    if any(tools[k]['sha256'] != v for k, v in TOOLS.items()):
        raise ValueError('pinned native tool changed')
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch / 'publish'
    publish.mkdir()
    with tarfile.open(a.transport) as tar:
        tar.extractall(scratch / 'wire', filter='data')
    from experiments.tennis.k6b_matte_judge import extract
    from experiments.tennis.k6b_receiver_sam import transport_root
    prior_rows, prior_scores = load_prior(a)
    checkpoint = Path(os.environ.get('PS_CHECKPOINT_DIR', ''))
    ctx = {'checkpoint': checkpoint if checkpoint.name else None, 'scratch': scratch, 'publish': publish, 'anchors': extract([a.anchor], scratch / 'anchors'),
           'wire': transport_root(scratch / 'wire'), 'prior_rows': prior_rows, 'prior_scores': prior_scores,
           'lpips': a.lpips, 'dists': a.dists, 'svt_scales': parse_scales(a.svt_scales)}
    results: list[dict[str, Any]] = []
    setup = time.monotonic() - began
    for key in SOURCES:
        saved = checkpoint / key if checkpoint.name else None
        if saved is not None and saved.is_dir() and (saved / 'result.json').is_file():
            shutil.copytree(saved, publish / key)
            results.append(json.loads((saved / 'result.json').read_text()))
            continue
        result = run_window(a, key, ctx)
        results.append(result)
        if checkpoint.name and result['accepted']:
            temporary = checkpoint / f'.{key}.tmp'
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.copytree(publish / key, temporary)
            os.replace(temporary, checkpoint / key)
        if not result['accepted']:
            break
    seconds = time.monotonic() - began
    summary = {'frames': a.frames, 'points': a.points, 'point_list': a.point_list, 'svt_scales': ctx['svt_scales'], 'gpu': name, 'gpu_class': gpu_class,
               'capability': capability, 'gpu_uuid': uuid, 'tools': tools, 'torch': torch.__version__,
               'accepted': all(r['accepted'] for r in results) and len(results) == len(SOURCES), 'windows': results,
               'setup_seconds': setup, 'seconds': seconds, **summarize(results),
               'inputs': {k: getattr(a, k).name for k in INPUTS},
               'scope': 'Two familiar development windows; real bitstreams; decoded in the encoding job; no timing claim.'}
    if a.windows_file is not None:
        summary['expected_windows'] = list(SOURCES)
    summary = json.loads(json.dumps(summary))  # what the validator reads back: lists, not tuples
    write(publish / 'summary.json', summary)
    write(Path(os.environ['PS_STAGE_DIR']) / 'w3-summary.json', summary)
    if not summary['accepted']:
        raise SystemExit('W3 stop: a point failed the decode acceptance (see w3-summary.json)')


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    summary = json.loads((stage / 'w3-summary.json').read_text())
    spec = json.loads((stage.parent / 'spec.json').read_text())
    manifest_item = next((i for i in spec['staged_inputs'] if i['name'] == 'windows'), None)
    if manifest_item is not None:
        arguments = spec['arguments']
        activate_manifest(Path(manifest_item['path']), [arguments[i + 1] for i, x in enumerate(arguments) if x == '--window'])
    staging = json.loads((stage / 'staging.json').read_text())
    windows = summary['windows']
    listed = summary.get('point_list')
    if listed:
        smoke_plan = parse_point_list(spec['scale']['point_list']['smoke'])
        full_plan = parse_point_list(spec['scale']['point_list']['full'])
        full_svt = parse_scales(str(spec['scale']['svt_scales']['full']))
        projection = project_listed_seconds(
            [{'convert': w['timings']['convert'], 'native_fixed': w['timings']['native_fixed'],
              'native_variable': w['timings']['native_variable'], 'svt_scale8': w['timings']['svt_scale8'],
              'masks': w['timings']['masks'], 'scoring': w['timings']['scoring']} for w in windows],
            summary['setup_seconds'], staging['publication_seconds'], [len(full_plan[w['window']]) for w in windows],
            float(sum({8: 1, 4: 4, 2: 16}[x] for x in full_svt)), len(full_svt))
        selection = {
            'smoke_scale': summary['frames'] == 25 and listed == spec['scale']['point_list']['smoke'],
            'points_as_named': all([(p['scale'], p['qp']) for p in w['chosen']] == smoke_plan[w['window']] for w in windows),
            'smoke_points_in_full_plan': all(pt in full_plan[w] for w, pts in smoke_plan.items() for pt in pts),
            'no_calibration': all(not w['candidates'] and w['bracket_ok'] is None for w in windows)}
    else:
        timings = {'setup': summary['setup_seconds'] + sum(w['timings']['convert'] for w in windows) / 2,
                   'calibration': sum(w['timings']['calibration'] for w in windows),
                   'native_fixed': sum(w['timings']['native_fixed'] for w in windows),
                   'native_variable': sum(w['timings']['native_variable'] for w in windows),
                   'svt_scale8': sum(w['timings']['svt_scale8'] for w in windows),
                   'masks': sum(w['timings']['masks'] for w in windows),
                   'scoring': sum(w['timings']['scoring'] for w in windows)}
        projection = project_full_seconds(timings, staging['publication_seconds'])
        selection = {
            'smoke_scale': summary['frames'] == 25 and summary['points'] == 1,
            'rate_only_selection': all(len(w['candidates']) == len(CAL_QPS) * len(SCALES) and len(w['chosen']) == 1 for w in windows),
            'bracket': all(w['bracket_ok'] and an.planned_bracket(w['candidates']) for w in windows)}
    points = [p for w in windows for p in w['dcvc']]
    checks: dict[str, Any] = {
        'two_windows': [w['window'] for w in windows] == summary.get('expected_windows', ['al', 'med']),
        **selection,
        'accepted': summary['accepted'],
        'gpu': summary['gpu_class'] in GPU_CLASSES and summary['gpu_uuid'].startswith('GPU-'),
        'extension': all(p['encode_report']['environment']['extension']['variant'] == GPU_CLASSES[summary['gpu_class']][1] for p in points),
        'determinism_recorded': all(p['encode_report']['checkpoints']['determinism']['final']['deterministic_algorithms']
                                    and p['encode_report']['checkpoints']['determinism']['final']['cudnn_deterministic']
                                    and not p['encode_report']['checkpoints']['determinism']['final']['cudnn_benchmark'] for p in points),
        'legacy_stream': all(p['encode_report']['checkpoints']['stream_policy'] == 'legacy' for p in points),
        'two_decoders_four_passes': all(len(p['decode_reports']) == 2 and all(len(r['passes']) == 2 for r in p['decode_reports'].values()) for p in points),
        'i_frame_within_one_level': all(p['check_a_i_frame']['passed'] for p in points),
        'decoders_bit_identical': all(p['check_b_two_processes']['max_abs_diff'] == 0 and p['identical_frame_hashes'] for p in points),
        'dpb_status_recorded': all(isinstance(p['dpb_reachable'], bool) for p in points),
        'rate_from_file': all(p['bytes'] == p['encode_report']['container_bytes'] for p in points),
        'scores': all(np.isfinite(v).all() and len(v) == 9 for w in windows for p in [*w['dcvc'], *w['svt']] for v in p['scores'].values()),
        'svt_control_ran': windows[0]['audited_controls']['applies'] and all('score_max_error' in e for e in windows[0]['audited_controls']['scales'].values()),
        'projection': projection < spec['full']['seconds'],
        'budget': spec['budget_seconds'] <= 3300,
        'scale': spec['scale']['frames'] == {'smoke': 25, 'full': 240},
        'inputs': manifest_item is not None or all(item['sha256'] in INPUTS.values() or item['sha256'] in {s['source'] for s in SOURCES.values()} for item in spec['staged_inputs']),
    }
    with tarfile.open(stage / 'published.tar') as tar:
        member = tar.extractfile('publish/summary.json')
        checks['published_summary'] = member is not None and json.load(member) == summary
    write(Path(os.environ['PS_VALIDATION_PATH']), {'passed': all(checks.values()), 'checks': checks, 'projected_full_seconds': projection})
    if not all(checks.values()):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true')
    for key in ('al', 'med', 'anchor', 'transport', 'prior_frame', 'prior_quality', 'image_ckpt', 'video_ckpt', 'lpips', 'dists'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path)
    parser.add_argument('--frames', type=int, choices=(25, 240), default=25)
    parser.add_argument('--points', type=int, choices=(1, 3), default=1)
    parser.add_argument('--svt-scales', default='8')
    parser.add_argument('--point-list', default=None, help='W3b: window:scale:qp,... instead of the calibrated points')
    parser.add_argument('--windows-file', type=Path, help='window manifest instead of the frozen Al/Med pair (then --window/--source per window; '
                        '--anchor, --transport and --prior-frame (the receiver stage result of these windows) are required but not hash-pinned; --prior-quality is not used)')
    parser.add_argument('--window', action='append')
    parser.add_argument('--source', type=Path, action='append')
    args = parser.parse_args()
    if args.windows_file is not None and not args.validate:
        names = activate_manifest(args.windows_file, args.window)
        if len(args.source or []) != len(names):
            parser.error('one --source per selected window, in order')
        for name, path in zip(names, args.source):
            setattr(args, name, path)
    if args.validate:
        validate()
    elif any(getattr(args, k) is None for k in [*INPUTS, *SOURCES, 'anchor', 'transport', 'prior_frame']):
        parser.error('all frozen inputs are required')
    else:
        run(args)


if __name__ == '__main__':
    main()

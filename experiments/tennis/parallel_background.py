"""Paid court reference plus native-frame residuals, with optional paid homographies.

Uncovered content and foreground carried by the background are paid in the
residual. These finite constructions do not bound arbitrary panoramas/codecs.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import struct
import tarfile
import time
from typing import Any

import cv2
import numpy as np

from experiments.tennis.k6b_clean_background_prep import INPUTS, SOURCE_YUV, TOOLS, WINDOW, erase_yuv
from experiments.tennis.k6b_clean_background_score import anchor_at
from experiments.tennis.k6b_ideal_player import TICKS, WEIGHTS, control_error, digest, image_sha, write
from src.codecs import svtav1

CRFS = (45, 55, 63)
POLICIES = ('base-control', 'actual-control', 'reference-residual-q45', 'reference-residual-q55', 'reference-residual-q63')
PREPARED_SHA = '56f54ccdc9db75ce98796597d2f2de0c849ad8041ab57c2def28dc5a4cd25ae0'
WIDTH, HEIGHT = 960, 544


def pack_residual(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map signed [-255,255] residual to 8-bit, with <=1 source-unit error."""
    if source.dtype != np.uint8 or reference.dtype != np.uint8 or source.shape != reference.shape:
        raise ValueError('Matching uint8 planes required')
    return ((source.astype(np.int16) - reference.astype(np.int16) + 256) // 2).clip(0, 255).astype(np.uint8)


def reconstruct(reference: np.ndarray, residual: np.ndarray) -> np.ndarray:
    if reference.dtype != np.uint8 or residual.dtype != np.uint8 or reference.shape != residual.shape:
        raise ValueError('Matching uint8 decoded planes required')
    return (reference.astype(np.int16) + 2 * residual.astype(np.int16) - 256).clip(0, 255).astype(np.uint8)


def inspect_stream(path: Path, frames: int) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 32 or data[:4] != b'DKIF':
        raise ValueError('Invalid IVF')
    if struct.unpack('<HH4sHHIIII', data[4:32])[:8] != (0, 32, b'AV01', WIDTH, HEIGHT, 25, 1, frames):
        raise ValueError('Native IVF geometry/timing changed')
    offset, ticks = 32, []
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError('Truncated native IVF header')
        size, tick = struct.unpack('<IQ', data[offset:offset+12])
        offset += 12
        if size == 0 or offset + size > len(data):
            raise ValueError('Truncated native IVF frame')
        offset += size
        ticks.append(tick)
    if ticks != list(range(frames)):
        raise ValueError('Native IVF ticks missing/reordered')
    return {'bytes': len(data), 'sha256': digest(path), 'frames': frames, 'timestamps': ticks}


def packet_header(reference: Path, residual: Path, frames: int, motion: bytes | None = None) -> bytes:
    header: dict[str, Any] = {'schema': 1, 'width': WIDTH, 'height': HEIGHT, 'frames': frames, 'fps': [25, 1],
                        'transform': 'identity', 'color': 'limited-yuv420', 'residual_color': 'full-range',
                        'residual_decode': {'scale': 2, 'offset': -256, 'clip': [0, 255]},
                        'reference_sha256': digest(reference), 'residual_sha256': digest(residual)}
    if motion is not None:
        import hashlib
        header.update(transform='homography-wire-v1', motion_sha256=hashlib.sha256(motion).hexdigest(),
                      warp={'interpolation': 'linear', 'border_yuv': [16, 128, 128], 'chroma_scale': .5})
    return (json.dumps(header, sort_keys=True, separators=(',', ':')) + '\n').encode()


def header_decode(data: bytes, reference: Path, residual: Path, frames: int, motion: bytes | None = None) -> dict[str, Any]:
    """Decode only the paid schema; reject hidden geometry or transform changes."""
    header = json.loads(data)
    expected = json.loads(packet_header(reference, residual, frames, motion))
    if header != expected or frames not in (24, 240):
        raise ValueError('Paid packet contract or stream identity changed')
    return header


def reconstruct_packet(reference: np.ndarray, residual: np.ndarray, header: dict[str, Any]) -> np.ndarray:
    """Reconstruct from parsed wire parameters rather than encoder-side state."""
    if (reference.dtype != np.uint8 or residual.dtype != np.uint8
            or reference.shape != residual.shape
            or reference.shape != (header['height'] * 3 // 2, header['width'])
            or header['transform'] not in ('identity', 'homography-wire-v1') or header['color'] != 'limited-yuv420'
            or header['residual_color'] != 'full-range'):
        raise ValueError('Decoded packet planes/transform differ from paid header')
    params = header['residual_decode']
    if params != {'scale': 2, 'offset': -256, 'clip': [0, 255]}:
        raise ValueError('Unsupported residual decoder parameters')
    return (reference.astype(np.int16) + params['scale'] * residual.astype(np.int16)
            + params['offset']).clip(*params['clip']).astype(np.uint8)


def compare(means: dict[str, float], curves: dict[str, Any], rate: float) -> dict[str, Any]:
    if set(means) != set(WEIGHTS) or not all(np.isfinite(v) for v in means.values()):
        raise ValueError('Two finite whole-frame metric means required')
    anchor = anchor_at(curves, rate)
    if anchor is None:
        return {'verdict': 'UNRESOLVED/ANCHOR-COVERAGE', 'rate_kbps': rate, 'means': means, 'anchor': None}
    wins = {m: means[m] < anchor[m] - .001 for m in WEIGHTS}
    losses = {m: means[m] > anchor[m] + .001 for m in WEIGHTS}
    return {'verdict': 'PASS' if all(wins.values()) else 'FAIL' if any(losses.values()) else 'UNRESOLVED',
            'rate_kbps': rate, 'means': means, 'anchor': anchor, 'robust_wins': wins,
            'robust_losses': losses, 'sensitivity': .001}


def run(a: argparse.Namespace) -> None:
    import torch
    from experiments.jobs.monitor import publish_progress
    from experiments.tennis.k4b_features import convert_command
    from experiments.tennis.k6 import downscaled_source
    from experiments.visor.h3b import dists_crops, load_dists, lpips_reduce
    from experiments.visor.h3c import downscale_planes
    from src.codecs.quality import load_lpips, yuv420_to_rgb
    from experiments.tennis.parallel_motion import decode_motion, fit_motion, warp_reference

    started = time.monotonic()
    if not torch.cuda.is_available() or 'A6000' not in torch.cuda.get_device_name(0):
        raise ValueError('Qualified A6000 required')
    if digest(a.source) != INPUTS['source'] or digest(a.prepared) != PREPARED_SHA:
        raise ValueError('Frozen source/prepared publication changed')
    if a.motion and (a.transport is None or digest(a.transport) != INPUTS['transport']):
        raise ValueError('Frozen encoder-only source regions required for motion fitting')
    for metric, expected in WEIGHTS.items():
        if digest(getattr(a, metric)) != expected:
            raise ValueError('Metric weight changed')
    tools = {name: svtav1.tool(name) for name in TOOLS}
    if any(tools[name]['sha256'] != expected for name, expected in TOOLS.items()):
        raise ValueError('Native tool changed')
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    memory_limit = int(os.environ.get('PS_GPU_MEMORY_LIMIT_MIB', '5000'))
    if not 4000 <= memory_limit <= 5000:
        raise ValueError('Declare4000–5000MiB measured-class allowance')
    torch.cuda.set_per_process_memory_fraction(memory_limit * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish, inputs = scratch / 'publish', scratch / 'prepared'
    publish.mkdir()
    inputs.mkdir()
    with tarfile.open(a.prepared) as outer:
        stream = outer.extractfile('publish/input.tar')
        if stream is None:
            raise ValueError('Missing prepared input')
        with tarfile.open(fileobj=stream) as inner:
            inner.extractall(inputs, filter='data')
    manifest = json.loads((inputs / 'manifest.json').read_text())
    if manifest['window'] != WINDOW or manifest['ticks'] != TICKS or manifest['source_yuv_sha256'] != SOURCE_YUV:
        raise ValueError('Frozen cohort changed')
    for name, expected in manifest['files'].items():
        if digest(inputs / name) != expected:
            raise ValueError('Prepared member changed: ' + name)
    source = scratch / 'source.yuv'
    subprocess.run(convert_command('ffmpeg', a.source, source, 9, 240), check=True, capture_output=True, timeout=180)
    if source.stat().st_size != 240 * 1620 * 1920 or digest(source) != SOURCE_YUV:
        raise ValueError('Full native source identity changed')
    raw = np.memmap(source, np.uint8, 'r', shape=(240, 1620, 1920))
    with np.load(inputs / 'frames' / '000.npz') as z:
        reference_source = erase_yuv(raw[0], z['region'])
    reference_raw = scratch / 'reference-source.yuv'
    reference_raw.write_bytes(downscale_planes(reference_source, 1920, 1080, WIDTH, HEIGHT).tobytes())
    reference_ivf, reference_yuv = publish / 'reference.ivf', scratch / 'reference.yuv'
    commands = []

    def encode(source_path: Path, stream_path: Path, count: int, crf: int, full_range: bool) -> None:
        command = svtav1.encode_command('SvtAv1EncApp', source_path, stream_path, width=WIDTH, height=HEIGHT,
                                      fps=Fraction(25), frames=count, crf=crf, preset=4, full_range=full_range)
        commands.append(command)
        # Bound native encoder CPU pools while allowing metric-side threads.
        svtav1.run(command, timeout=350, cpus=[min(os.sched_getaffinity(0))])
        inspect_stream(stream_path, count)

    def decode(stream_path: Path, out: Path, count: int) -> None:
        command = svtav1.decode_command('dav1d', stream_path, out, threads=1)
        commands.append(command)
        subprocess.run(command, check=True, capture_output=True, timeout=60)
        again = out.with_suffix('.repeat.yuv')
        subprocess.run(svtav1.decode_command('dav1d', stream_path, again, threads=1), check=True, capture_output=True, timeout=60)
        if out.stat().st_size != count * HEIGHT * WIDTH * 3 // 2 or digest(out) != digest(again):
            raise ValueError('Fresh native decode changed')

    encode(reference_raw, reference_ivf, 1, 35, False)
    decode(reference_ivf, reference_yuv, 1)
    reference = np.fromfile(reference_yuv, np.uint8).reshape(HEIGHT * 3 // 2, WIDTH)
    encode_started = time.monotonic()
    down = downscaled_source(source, a.frames, 2, scratch / 'down.yuv')
    down_raw = np.memmap(down, np.uint8, 'r', shape=(a.frames, HEIGHT * 3 // 2, WIDTH))
    motion = None
    motion_summary = None
    matrices = np.repeat(np.eye(3)[None], a.frames, axis=0)
    if a.motion:
        from experiments.tennis.k6b_rate_streams import decode_masks
        from experiments.tennis.k6b_clean_background_prep import read
        with tarfile.open(a.transport) as tar:
            _, masks = decode_masks(read(tar, 'packets/alcaraz_hurkacz__031@129/exact-rle.matte').getvalue())
        if masks.shape != (240, 2, 540, 960):
            raise ValueError('Full frozen encoder regions required')
        motion, motion_summary = fit_motion(reference, down_raw, masks[:a.frames].any(axis=1))
        (publish / 'motion.bin').write_bytes(motion)
        matrices = decode_motion((publish / 'motion.bin').read_bytes(), a.frames)
        del masks
    residual_source = scratch / 'residual-source.yuv'
    with residual_source.open('wb') as handle:
        for t, frame in enumerate(down_raw):
            predicted = warp_reference(reference, matrices[t]) if a.motion else reference
            handle.write(pack_residual(frame, predicted).tobytes())
    decoded, rates, packet_receipts, contracts = [], [], [], []
    for crf in CRFS:
        stream_path, decoded_path = publish / f'residual-q{crf}.ivf', scratch / f'residual-q{crf}.yuv'
        encode(residual_source, stream_path, a.frames, crf, True)
        decode(stream_path, decoded_path, a.frames)
        header = packet_header(reference_ivf, stream_path, a.frames, motion)
        header_path = publish / f'packet-q{crf}.json'
        header_path.write_bytes(header)
        contract = header_decode(header_path.read_bytes(), reference_ivf, stream_path, a.frames, motion)
        contracts.append(contract)
        count = reference_ivf.stat().st_size + stream_path.stat().st_size + len(header) + (len(motion) if motion else 0)
        # Retain the whole previous optimistic side cost, including its old
        # bootstrap. The new header is an additional paid cost, never a rebate.
        rate = count * 8 / (a.frames / 25) / 1000 + manifest['side_kbps']
        rates.append(rate)
        packet_receipts.append({'crf': crf, 'reference_bytes': reference_ivf.stat().st_size,
                                'residual_bytes': stream_path.stat().st_size, 'header_bytes': len(header),
                                'total_bytes': count, 'rate_kbps': rate, 'header_sha256': digest(publish / f'packet-q{crf}.json'),
                                'residual_sha256': digest(stream_path)})
        decoded.append(np.memmap(decoded_path, np.uint8, 'r', shape=(contract['frames'], contract['height'] * 3 // 2, contract['width'])))
    encoding_seconds = time.monotonic() - encode_started
    nets = {'lpips': load_lpips(a.lpips, 'cuda'), 'dists': load_dists(a.dists, 'cuda')}
    if not all(next(n.parameters()).is_cuda for n in nets.values()):
        raise ValueError('Metric network not CUDA')
    ticks = TICKS[:a.judged or None]
    scores = {m: np.full((len(POLICIES), len(ticks)), np.nan) for m in WEIGHTS}
    all_mask = np.ones((1080, 1920), bool)
    hashes: dict[str, list[str]] = {}
    timings = []
    for j, tick in enumerate(ticks):
        tick_started = time.monotonic()
        with np.load(inputs / 'frames' / f'{tick:03d}.npz') as z:
            original, overlay, written = (z[k].copy() for k in ('images', 'overlay', 'written'))
        if [image_sha(x) for x in original] != manifest['hashes'][str(tick)]['images']:
            raise ValueError('Historical images changed')
        if not np.array_equal(np.where(written[..., None], overlay, original[1]), original[2]):
            raise ValueError('Historical overlay replay changed')

        def rgb(plane: np.ndarray, width: int, height: int) -> np.ndarray:
            tensor = torch.from_numpy(np.array(plane, copy=True)[None]).to('cuda')
            return yuv420_to_rgb(tensor, width, height, full_range=False)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()

        if image_sha(rgb(raw[tick], 1920, 1080)) != manifest['hashes'][str(tick)]['images'][0]:
            raise ValueError('Native CUDA source identity failed')
        images = [original[1], original[2]]
        for residual, contract in zip(decoded, contracts):
            predicted = warp_reference(reference, matrices[tick]) if a.motion else reference
            plane = reconstruct_packet(predicted, residual[tick], contract)
            base = cv2.resize(rgb(plane, contract['width'], contract['height']), (1920, 1080), interpolation=cv2.INTER_CUBIC)
            images.append(np.where(written[..., None], overlay, base))
        hashes[str(tick)] = [image_sha(x) for x in images]
        for i, image in enumerate(images):
            scores['lpips'][i, j] = lpips_reduce(nets['lpips'], [image], original[0], [[all_mask]], 'cuda')[0][0]
            scores['dists'][i, j] = dists_crops(nets['dists'], [image], original[0], all_mask, 'cuda')[0]
        control_error({m: s[:2, :j+1] for m, s in scores.items()}, manifest['controls'], list(range(j+1)))
        if not all(np.isfinite(s[:, :j+1]).all() for s in scores.values()):
            raise ValueError('Nonfinite score')
        np.savez_compressed(publish / 'scores.npz', ticks=np.array(ticks[:j+1]), **{m: s[:, :j+1] for m, s in scores.items()})
        publish_progress(os.environ['PS_STAGE'], j + 1)
        timings.append(time.monotonic() - tick_started)
    assessments = [compare({m: float(s[i+2].mean()) for m, s in scores.items()}, manifest['curves'], rates[i]) for i in range(3)]
    summary = {'motion': a.motion, 'motion_fit': motion_summary, 'motion_bytes': len(motion) if motion else 0,
               'window': WINDOW, 'frames': a.frames, 'ticks': ticks, 'policies': POLICIES, 'crfs': CRFS,
               'assessment': assessments if a.frames == 240 else None, 'smoke_scores_are_not_evidence': a.frames != 240,
               'packets': packet_receipts, 'parsed_contracts': contracts, 'side_kbps': manifest['side_kbps'], 'commands': commands, 'tools': tools,
               'reference_sha256': digest(reference_ivf), 'source_yuv_sha256': SOURCE_YUV,
               'source_sha256': digest(a.source), 'prepared_sha256': digest(a.prepared), 'weight_sha256': WEIGHTS,
               'control_errors': control_error({m: s[:2] for m, s in scores.items()}, manifest['controls'], list(range(len(ticks)))),
               'source_rgb_checks': len(ticks), 'hashes': hashes, 'cuda': True, 'gpu': torch.cuda.get_device_name(0),
               'timing_secondary': True, 'setup_seconds': time.monotonic() - started - encoding_seconds - sum(timings), 'encoding_seconds': encoding_seconds,
               'tick_seconds': timings, 'seconds': time.monotonic() - started,
               'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
               'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20, 'memory_limit_mib': memory_limit,
               'scope': 'Al native240 paid reference/residual pilot with frozen actual overlay; no arbitrary panorama bound, BD-rate or untouched result.'}
    write(publish / 'summary.json', summary)
    write(Path(os.environ['PS_STAGE_DIR']) / 'background-summary.json', summary)


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    s = json.loads((stage / 'background-summary.json').read_text())
    spec = json.loads((stage.parent / 'spec.json').read_text())
    execution = json.loads((stage / 'execution.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    # All native encodes get the conservative full 10x factor. Tick scoring gets
    # 80/3, warm setup once, 35% safety, measured publication also expanded.
    projection = 1.35 * (s['setup_seconds'] + s['encoding_seconds'] * 10
                         + sum(s['tick_seconds']) * 80 / 3) + staging['publication_seconds'] * 10
    checks = {'frames': s['frames'] == 24, 'ticks': s['ticks'] == [0, 3, 6],
              'policies': s['policies'] == list(POLICIES), 'crfs': s['crfs'] == list(CRFS),
              'cuda': s['cuda'] and 'A6000' in s['gpu'], 'source': s['source_sha256'] == INPUTS['source'] and s['source_yuv_sha256'] == SOURCE_YUV,
              'prepared': s['prepared_sha256'] == PREPARED_SHA, 'weights': s['weight_sha256'] == WEIGHTS,
              'source_rgb': s['source_rgb_checks'] == 3,
              'controls': all(0 <= v <= 1e-4 for v in s['control_errors'].values()),
              'memory': 0 < s['peak_allocated_mib'] <= s['peak_reserved_mib'] < s['memory_limit_mib'] <= 5000,
              'native_tools': all(s['tools'][name]['sha256'] == expected for name, expected in TOOLS.items()),
              'finite_projection': bool(np.isfinite(projection)) and 0 < projection < spec['full']['seconds'],
              'budget': spec['budget_seconds'] <= 2700 and execution['seconds'] < spec['smoke']['seconds'],
              'scale': spec['scale'] == {'frames': {'smoke': 24, 'full': 240}, 'judged': {'smoke': 3, 'full': 0}},
              'side_retained': abs(s['side_kbps'] - 5.607128888888889) < 1e-12,
              'timing_secondary': s['timing_secondary']}
    checks['motion_contract'] = s['motion'] == ('--motion' in spec['arguments'])
    if s['motion']:
        checks['encoder_regions'] = any(x['name'] == 'transport' and x['sha256'] == INPUTS['transport'] for x in spec['staged_inputs'])
        checks['motion_coverage'] = (s['motion_fit']['fitted'] + s['motion_fit']['identity_fallbacks'] == 24
                                     and s['motion_fit']['fitted'] >= 22 and s['motion_bytes'] > 8)
    with tarfile.open(stage / 'published.tar') as tar:
        stream = tar.extractfile('publish/scores.npz')
        if stream is None:
            raise ValueError('Missing saved scores')
        with np.load(stream) as z:
            checks['saved_scores'] = z['ticks'].tolist() == [0, 3, 6] and all(z[m].shape == (5, 3) and np.isfinite(z[m]).all() for m in WEIGHTS)
        motion_size = 0
        if s['motion']:
            from experiments.tennis.parallel_motion import decode_motion
            from experiments.tennis.k6b_clean_background_prep import read
            import hashlib
            packet = read(tar, 'publish/motion.bin').getvalue()
            motion_size = len(packet)
            checks['delivered_camera'] = decode_motion(packet, 24).shape == (24, 3, 3)
            checks['camera_hashes'] = all(c['motion_sha256'] == hashlib.sha256(packet).hexdigest() for c in s['parsed_contracts'])
        for receipt in s['packets']:
            count = sum(tar.getmember('publish/' + name).size for name in
                        ('reference.ivf', f'residual-q{receipt["crf"]}.ivf', f'packet-q{receipt["crf"]}.json')) + motion_size
            checks[f'paid_bytes:{receipt["crf"]}'] = count == receipt['total_bytes']
    write(Path(os.environ['PS_VALIDATION_PATH']), {'passed': all(checks.values()), 'checks': checks,
                                                 'projected_full_seconds': projection})
    if not all(checks.values()):
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--source', type=Path)
    p.add_argument('--prepared', type=Path)
    p.add_argument('--transport', type=Path)
    p.add_argument('--motion', action='store_true', help='Paid ORB homography before residual coding')
    p.add_argument('--lpips', type=Path)
    p.add_argument('--dists', type=Path)
    p.add_argument('--frames', type=int, choices=(24, 240), default=24)
    p.add_argument('--judged', type=int, choices=(0, 3), default=3)
    a = p.parse_args()
    if a.validate:
        validate()
    elif any(getattr(a, k) is None for k in ('source', 'prepared', 'lpips', 'dists')):
        p.error('Four frozen inputs required')
    elif (a.frames, a.judged) not in ((24, 3), (240, 0)):
        p.error('Use representative24/3 smoke or native240/80 full pilot')
    else:
        run(a)


if __name__ == '__main__':
    main()

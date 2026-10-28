"""CPU preparation of one paid, encoder-erased background; no model inference."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tarfile
import time
from typing import Any

import cv2
import numpy as np

from experiments.tennis.k6b_ideal_player import TICKS, check_images, digest, image_sha, write

WINDOW = 'alcaraz_hurkacz/031@129'
SAFE = WINDOW.replace('/', '__')
INPUTS = {
    'source': '054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76',
    'transport': '6865035e3132b6e07600fdaabc8c9d7152b8b6e3c3260d9f3df21cab33ca33c7',
    'images': '468250a72a1ce764e9451fc7b4cd51bd29372d3e688b832e1e9c1f002e11bc6c',
    'masks': '610a09038d13a8ad0b5aab84489fdb6128c46361b2853060cb0151b232401768',
    'generated': '6a3f620b394871154290340d7700831aea62f098e2569bf0d130e6735a4a1ef1',
    'prior': 'e346844d20671e5dfd78828ec61d262aa13109bc1be9a1b421873facdc86daa1',
}
TOOLS = {
    'SvtAv1EncApp': '4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4',
    'dav1d': 'e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6',
    'ffmpeg': '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2',
}
SOURCE_YUV = 'e7b960cd2456b439c5439b337a45e47d2a0a44a2a32025a2de595af3feed4fa2'


def read(tar: tarfile.TarFile, name: str) -> io.BytesIO:
    stream = tar.extractfile(name)
    if stream is None:
        raise ValueError('Missing input member: ' + name)
    return io.BytesIO(stream.read())


def erase_yuv(frame: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Telea3 on each native plane; unmasked source samples remain byte-identical."""
    if frame.shape != (1620, 1920) or frame.dtype != np.uint8 or region.shape != (1080, 1920) or region.dtype != bool:
        raise ValueError('Original source YUV and boolean R required')
    result = frame.copy()
    original_chroma = frame[1080:].reshape(2, 540, 960)
    chroma = result[1080:].reshape(2, 540, 960)
    masks = [region, cv2.resize(region.astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST).astype(bool)]
    for original, target, mask in [(frame[:1080], result[:1080], masks[0]),
                                    (original_chroma[0], chroma[0], masks[1]),
                                    (original_chroma[1], chroma[1], masks[1])]:
        if mask.all():
            raise ValueError('Erase region has no background support')
        target[:] = cv2.inpaint(original, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        if not np.array_equal(target[~mask], original[~mask]):
            raise ValueError('Unmasked source samples changed')
    return result


def stream_info(path: Path, frames: int) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 32 or data[:4] != b'DKIF':
        raise ValueError('Invalid IVF')
    fields = struct.unpack('<HH4sHHIIII', data[4:32])
    if fields[:8] != (0, 32, b'AV01', 480, 272, 25, 1, frames):
        raise ValueError('IVF geometry/timing mismatch')
    offset, ticks = 32, []
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError('Truncated IVF framing')
        size, tick = struct.unpack('<IQ', data[offset:offset + 12])
        offset += 12
        if size == 0 or offset + size > len(data):
            raise ValueError('Truncated IVF payload')
        offset += size
        ticks.append(tick)
    if ticks != list(range(frames)):
        raise ValueError('Missing/reordered IVF frames')
    return {'sha256': digest(path), 'bytes': len(data), 'frames': frames,
            'framing_bytes': 32 + 12 * frames, 'timestamps': ticks}


def prepare(a: argparse.Namespace) -> None:
    from experiments.jobs.monitor import publish_progress
    from experiments.tennis import k6b
    from experiments.tennis.k4b_features import convert_command
    from experiments.tennis.k6 import downscaled_source
    from experiments.tennis.k6b_matte_cpu import box_maps
    from experiments.tennis.k6b_rate_streams import decode_keypoints, decode_masks
    from experiments.tennis.k6b_real_base_compose import compose
    from experiments.tennis.k6b_receiver_sam import project
    from experiments.tennis.k6b_side_feasibility import hull
    from src.codecs import svtav1

    began = time.monotonic()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only preparation requires CUDA hidden')
    cv2.setNumThreads(4)
    for name, expected in INPUTS.items():
        if digest(getattr(a, name)) != expected:
            raise ValueError('Frozen input changed: ' + name)
    tools = {name: svtav1.tool(name) for name in TOOLS}
    if any(tools[name]['sha256'] != expected for name, expected in TOOLS.items()):
        raise ValueError('Native tool identity changed')
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    output = scratch / 'prepared'
    output.mkdir()
    (output / 'frames').mkdir()
    publish = scratch / 'publish'
    publish.mkdir()
    source = scratch / 'source.yuv'
    command = convert_command('ffmpeg', a.source, source, 9, 240)
    subprocess.run(command, check=True, capture_output=True, timeout=180)
    if digest(source) != SOURCE_YUV or source.stat().st_size != 240 * 1920 * 1080 * 3 // 2:
        raise ValueError('Original full source YUV identity failed')
    raw = np.memmap(source, np.uint8, 'r', shape=(240, 1620, 1920))
    with tarfile.open(a.transport) as tar:
        meta, units = decode_keypoints(read(tar, f'packets/{SAFE}/keypoints.kps').getvalue())
        mask_meta, regions = decode_masks(read(tar, f'packets/{SAFE}/exact-rle.matte').getvalue())
    if regions.shape[0] != 240 or len(units) != len(meta['compositing_order']):
        raise ValueError('Incomplete original masks or player order')
    boxes = box_maps(units)
    players = []
    predictions = []
    with tarfile.open(a.masks) as tar:
        for unit in units:
            matches = [p for p in meta['compositing_order'] if k6b.unit_name(WINDOW, p) == unit['name']]
            if len(matches) != 1:
                raise ValueError('Player identity/order mismatch')
            players.append(matches[0])
            with np.load(read(tar, 'publish/' + unit['name'] + '-masks.npz')) as z:
                if z['t'].tolist() != TICKS:
                    raise ValueError('Saved receiver support ticks changed')
                predictions.append(z['masks'].copy())
    generated = k6b.load_generated([str(a.generated)], scratch / 'generated')
    with tarfile.open(a.prior) as tar:
        row = json.load(read(tar, 'publish/quality.json'))['rows'][0]
        bootstrap = json.load(read(tar, f'publish/{SAFE}/s4-bootstrap.json'))
        with np.load(read(tar, f'publish/{SAFE}/scores.npz')) as z:
            curves = {m: hull([(r['base_kbps'], -float(z['full_' + m][2*i].mean()))
                               for i, r in enumerate(row['rates'])]) for m in ('lpips', 'dists')}
    setup_seconds = time.monotonic() - began
    hashes: dict[str, Any] = {}
    files = {}
    clean = scratch / 'clean-source.yuv'
    with clean.open('xb') as cleaned, (output / 'source-ticks.yuv').open('xb') as sampled, tarfile.open(a.images) as old:
        old_manifest = json.load(read(old, 'manifest.json'))
        for t in range(a.frames):
            region = np.logical_or.reduce([cv2.resize(m.astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
                                           for m in regions[t]])
            cleaned.write(erase_yuv(np.asarray(raw[t]), region).tobytes())
            if t % 3 == 0:
                sampled.write(raw[t].tobytes())
                payload = read(old, f'frames/{t:03d}.npz')
                if hashlib.sha256(payload.getbuffer()).hexdigest() != old_manifest['files'][f'frames/{t:03d}.npz']:
                    raise ValueError('Original packed image changed')
                with np.load(payload) as z:
                    images = z['images'].copy()
                check_images(images, old_manifest['hashes'][str(t)])
                if image_sha(np.where(region[..., None], images[0], images[1])) != image_sha(images[3]):
                    raise ValueError('Original ideal_R mask/control changed')
                crops, frame_boxes, masks = {}, {}, {}
                for i, (unit, player) in enumerate(zip(units, players)):
                    g = generated[(unit['name'], 'match5')]
                    index = np.flatnonzero(g['t'] == t)
                    if len(index) > 1:
                        raise ValueError('Duplicate generated frame')
                    if len(index):
                        crops[player] = g['crops'][int(index[0])]
                    frame_boxes[player] = boxes[i][t]
                    masks[player] = project(predictions[i][t // 3], frame_boxes[player])
                actual, written = compose(images[1], crops, frame_boxes, masks, meta['compositing_order'])
                overlay, replay_written = compose(np.zeros_like(images[1]), crops, frame_boxes, masks, meta['compositing_order'])
                if not np.array_equal(actual, images[2]) or not np.array_equal(written, replay_written):
                    raise ValueError('Frozen actual composition changed')
                if not np.array_equal(np.where(written[..., None], overlay, images[1]), images[2]):
                    raise ValueError('Overlay/fallback replay failed')
                path = output / 'frames' / f'{t:03d}.npz'
                np.savez_compressed(path, images=images[:3], overlay=overlay, written=written, region=region)
                hashes[str(t)] = {'images': [image_sha(x) for x in images[:3]], 'overlay': image_sha(overlay),
                                  'written': image_sha(written), 'region': image_sha(region),
                                  'source_yuv': image_sha(raw[t])}
                files[path.relative_to(output).as_posix()] = digest(path)
            publish_progress(os.environ['PS_STAGE'], t + 1)
    down = downscaled_source(clean, a.frames, 4, scratch / 'down.yuv')
    stream, decoded = output / 'clean.ivf', output / 'clean.yuv'
    enc = svtav1.encode_command('SvtAv1EncApp', down, stream, width=480, height=272,
                               fps=Fraction(25, 1), frames=a.frames, crf=63, preset=4, full_range=False)
    encoder_cpu = min(os.sched_getaffinity(0))
    subprocess.run(enc, check=True, capture_output=True, timeout=300,
                   preexec_fn=lambda: os.sched_setaffinity(0, {encoder_cpu}))
    info = stream_info(stream, a.frames)
    dec = svtav1.decode_command('dav1d', stream, decoded, threads=1)
    subprocess.run(dec, check=True, capture_output=True, timeout=90)
    again = scratch / 'second.yuv'
    subprocess.run(svtav1.decode_command('dav1d', stream, again, threads=1), check=True, capture_output=True, timeout=90)
    if decoded.stat().st_size != a.frames * 480 * 272 * 3 // 2 or digest(decoded) != digest(again):
        raise ValueError('Native fresh decode changed')
    bootstrap['base_ivf_sha256'] = info['sha256']
    header = json.dumps(bootstrap, sort_keys=True, separators=(',', ':')).encode()
    if len(header) != row['rates'][1]['bootstrap_bytes']:
        raise ValueError('Bootstrap byte allocation changed')
    (output / 'bootstrap.json').write_bytes(header)
    for name in ('clean.ivf', 'clean.yuv', 'source-ticks.yuv', 'bootstrap.json'):
        files[name] = digest(output / name)
    rate = info['bytes'] * 8 / (a.frames / 25) / 1000 + row['rates'][1]['side_kbps']
    manifest = {'schema': 1, 'window': WINDOW, 'frames': a.frames, 'ticks': list(range(0, a.frames, 3)),
                'policies': ['base', 'actual', 'clean_actual', 'clean_ideal_R'],
                'input_sha256': INPUTS, 'tools': tools, 'source_yuv_sha256': SOURCE_YUV,
                'mask_metadata': mask_meta, 'hashes': hashes, 'files': files,
                'controls': old_manifest['controls'], 'curves': curves, 'side_kbps': row['rates'][1]['side_kbps'],
                'rate_kbps': rate, 'ivf': info, 'encode_command': enc, 'decode_command': dec,
                'source_command': command, 'bootstrap_bytes': len(header), 'opencv': cv2.__version__,
                'fill': 'Telea3 independently native YUV420 planes; R nearest; then original area scale4',
                'encoder_affinity_cpus': 1,
                'parent_jobs': ['20261027T211451Z-f6a754eb', '20261028T080453Z-34e50ec2']}
    write(output / 'manifest.json', manifest)
    with tarfile.open(publish / 'input.tar', 'w') as tar:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(output).as_posix())
    elapsed = time.monotonic() - began
    result = {'frames': a.frames, 'cuda_hidden': True, 'setup_seconds': setup_seconds, 'seconds': elapsed,
              'projected_full_seconds': (setup_seconds + (elapsed - setup_seconds) * 10) * 1.35,
              'input_archive_sha256': digest(publish / 'input.tar'), 'rate_kbps': rate,
              'files': len(files), 'ivf': info, 'source_yuv_sha256': SOURCE_YUV, 'tools': tools}
    write(Path(os.environ['PS_STAGE_DIR']) / 'prepare-summary.json', result)
    write(publish / 'prepare-summary.json', result)


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    summary = json.loads((stage / 'prepare-summary.json').read_text())
    spec = json.loads((stage.parent / 'spec.json').read_text())
    checks = {'cpu_only': summary['cuda_hidden'], 'representative': summary['frames'] == 24,
              'source_identity': summary['source_yuv_sha256'] == SOURCE_YUV,
              'tools': all(summary['tools'][n]['sha256'] == s for n, s in TOOLS.items()),
              'projection': 0 < summary['projected_full_seconds'] < spec['full']['seconds'],
              'ivf': summary['ivf']['frames'] == 24, 'file_coverage': summary['files'] == 12}
    # Validate the independently published inner archive, not only scratch.
    with tarfile.open(stage / 'published.tar') as outer:
        payload = read(outer, 'publish/input.tar')
        checks['archive_identity'] = hashlib.sha256(payload.getbuffer()).hexdigest() == summary['input_archive_sha256']
        with tarfile.open(fileobj=payload) as tar:
            manifest = json.load(read(tar, 'manifest.json'))
            checks['member_identities'] = all(hashlib.sha256(read(tar, n).getbuffer()).hexdigest() == s
                                              for n, s in manifest['files'].items())
            checks['tick_coverage'] = manifest['ticks'] == list(range(0, 24, 3))
            checks['fixed_side'] = abs(manifest['side_kbps'] - 5.607128888888889) < 1e-12
    write(Path(os.environ['PS_VALIDATION_PATH']), {'passed': all(checks.values()), 'checks': checks, 'summary': summary})
    if not all(checks.values()):
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--frames', type=int, choices=(24, 240), default=24)
    for name in INPUTS:
        p.add_argument('--' + name, type=Path)
    a = p.parse_args()
    if a.validate:
        validate()
    elif any(getattr(a, name) is None for name in INPUTS):
        p.error('all frozen inputs required')
    else:
        prepare(a)


if __name__ == '__main__':
    main()

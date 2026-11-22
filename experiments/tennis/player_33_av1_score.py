"""Score an independently encoded ordinary AV1 on frozen W7 foreground masks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

import cv2
import numpy as np

from experiments.tennis import k6, w3_neural_baseline as w3
from experiments.tennis.k4b_features import convert_command
from experiments.tennis.k6b_matte_judge import extract
from experiments.tennis.k6b_receiver_sam import transport_root
from experiments.visor.h3 import rgb_memmap
from experiments.visor.h3b import dists_crops, load_dists
from src.codecs import svtav1


def member(archive, name, destination):
    with tarfile.open(archive) as tar:
        stream = tar.extractfile(name)
        if stream is None:
            raise ValueError(f'missing {name}')
        with destination.open('wb') as out:
            while block := stream.read(1024 * 1024):
                out.write(block)


def run(a):
    import torch
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise ValueError('qualified GPU required for parity with W7')
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(min(1., 20000 * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    w3.activate_manifest(a.windows, [a.window])
    spec = w3.SOURCES[a.window]
    if w3.digest(a.source) != spec['source']:
        raise ValueError('source identity')
    root = Path(os.environ['PS_SCRATCH_DIR'])
    publish = root / 'publish'
    publish.mkdir()
    source = root / 'source.yuv'
    subprocess.run(convert_command('ffmpeg', a.source, source, spec['start'], 240), check=True, timeout=300)
    if w3.digest(source) != spec['yuv']:
        raise ValueError('source conversion identity')
    truth = rgb_memmap(source, root / 'truth.npy', 240, 1920, 1080, False, 'cuda')
    anchors = extract([a.anchor], root / 'anchors')
    with tarfile.open(a.transport) as tar:
        tar.extractall(root / 'wire', filter='data')
    prior = json.loads(a.prior.read_text())
    prior_rows = {r['id']: r for r in prior['rows']}
    ticks = list(range(0, 240, 3))[:a.ticks]
    masks = w3.foreground_masks(spec['wid'], anchors, transport_root(root / 'wire'), truth, ticks, prior_rows)
    with tarfile.open(a.candidate) as tar:
        record = json.load(tar.extractfile('publish/result.json'))
    if a.member != 'candidate.ivf':
        record = next(row for row in record['rows'] if row['crf'] == a.crf)
        record.update(width=960, height=544, fps=spec['fps'], source_sha256=spec['source'])
    if record['source_sha256'] != spec['source'] or record['fps'] != spec['fps']:
        raise ValueError('candidate source or cadence mismatch')
    candidate = root / 'candidate.ivf'
    member(a.candidate, 'publish/' + a.member, candidate)
    if w3.digest(candidate) != record['stream_sha256'] or len(svtav1.ivf_frames(candidate.read_bytes())) != 240:
        raise ValueError('candidate stream identity or frame count')
    decoded = root / 'candidate.yuv'
    decoder = svtav1.tool('dav1d')
    subprocess.run(svtav1.decode_command(decoder['path'], candidate, decoded, threads=1), check=True, timeout=600)
    if w3.digest(decoded) != record['decoded_sha256']:
        raise ValueError('candidate decoded identity')
    width, height = record['width'], record['height']
    pixels = np.memmap(decoded, np.uint8, 'r', shape=(240, height * 3 // 2, width))
    net = load_dists(a.dists, 'cuda')
    with tarfile.open(a.control) as tar:
        old = json.load(tar.extractfile(f'publish/{a.window}/result.json'))
    old_record = next(x for x in old['svt'] if x['scale'] == 8)
    old_ivf = root / 'control.ivf'
    member(a.control, f'publish/{a.window}/svt-s8.ivf', old_ivf)
    if w3.digest(old_ivf) != old_record['stream_sha256']:
        raise ValueError('control stream identity')
    old_yuv = root / 'control.yuv'
    subprocess.run(svtav1.decode_command(decoder['path'], old_ivf, old_yuv, threads=1), check=True, timeout=600)
    old_pixels = np.memmap(old_yuv, np.uint8, 'r', shape=(240, 136 * 3 // 2, 240))
    setup_seconds = time.monotonic() - started
    values = []
    control_errors = []
    for t in ticks:
        image = cv2.resize(w3.decoded_rgb(np.array(pixels[t], copy=True), width, height), (1920, 1080), interpolation=cv2.INTER_CUBIC)
        true, mask = np.asarray(truth[t]), masks[t]
        ys, xs = np.nonzero(mask)
        crop = (slice(max(0, ys.min()-k6.DISTS_PAD), min(1080, ys.max()+1+k6.DISTS_PAD)), slice(max(0, xs.min()-k6.DISTS_PAD), min(1920, xs.max()+1+k6.DISTS_PAD)))
        old_rgb = cv2.resize(w3.decoded_rgb(np.array(old_pixels[t], copy=True), 240, 136), (1920, 1080), interpolation=cv2.INTER_CUBIC)
        score, control_score = dists_crops(net, [image[crop], old_rgb[crop]], true[crop], mask[crop], 'cuda')
        values.append(float(score))
        control_errors.append(abs(float(control_score) - old_record['scores']['foreground_dists'][t // 3]))
    if max(control_errors) > 1e-3:
        raise ValueError('historical AV1 DISTS control mismatch')
    result = {'window': a.window, 'source': spec, 'candidate': record, 'ticks': ticks, 'foreground_dists': values, 'control_max_error': max(control_errors), 'mean': float(np.mean(values)), 'elapsed_seconds': time.monotonic()-started, 'setup_seconds': setup_seconds, 'gpu': torch.cuda.get_device_name(0), 'decoder': decoder}
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    (publish/'result.json').write_text(text)
    (Path(os.environ['PS_STAGE_DIR'])/'result.json').write_text(text)


def validate(a):
    root = Path(os.environ['PS_STAGE_DIR'])
    r = json.loads((root/'result.json').read_text())
    expected = 3 if os.environ['PS_STAGE'] == 'smoke' else 80
    checks = {'ticks': len(r['ticks']) == expected, 'finite': bool(np.isfinite(r['foreground_dists']).all()), 'frame_count': r['candidate']['frames'] == 240}
    if expected == 3:
        checks['forecast'] = r['setup_seconds'] + (r['elapsed_seconds'] - r['setup_seconds']) * 80 / 3 * 1.5 + 60 < a.full_limit
    result = {'passed': all(checks.values()), 'checks': checks}
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(result)+'\n')
    if not result['passed']:
        raise SystemExit(str(checks))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('windows', 'source', 'anchor', 'transport', 'prior', 'dists', 'candidate', 'control'):
        p.add_argument('--'+key, type=Path)
    p.add_argument('--window')
    p.add_argument('--member', default='candidate.ivf')
    p.add_argument('--crf', type=int)
    p.add_argument('--ticks', type=int)
    p.add_argument('--full-limit', type=float, default=1400)
    p.add_argument('--validate', action='store_true')
    a = p.parse_args()
    validate(a) if a.validate else run(a)


if __name__ == '__main__':
    main()

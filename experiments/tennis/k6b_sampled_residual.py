"""One framed temporal-XOR diagnostic for the archived sampled masks only."""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import resource
import tarfile
import time
from typing import Any

import numpy as np

from experiments.tennis.k6b_rate_streams import decode_keypoints, decode_masks, seal, unseal
from experiments.tennis.k6b_rate_transport import array_sha, sha, write
from experiments.tennis.k6b_receiver_silhouette import project
from experiments.tennis.k6b_matte_cpu import box_maps


def encode(metadata: dict[str, Any], residuals: list[np.ndarray]) -> bytes:
    previous = np.zeros_like(residuals[0])
    body = bytearray()
    for residual in residuals:
        body.extend(np.packbits(residual ^ previous).tobytes())
        previous = residual
    return seal(b'SXR1', {**metadata, 'shape': list(previous.shape),
                          'frames': len(residuals), 'predictor': 'sampled_temporal_xor'}, bytes(body))


def decode(blob: bytes) -> tuple[dict[str, Any], list[np.ndarray]]:
    meta, body = unseal(blob, b'SXR1')
    shape = tuple(meta['shape'])
    pixels = int(np.prod(shape))
    stride = (pixels + 7) // 8
    if meta['predictor'] != 'sampled_temporal_xor' or len(body) != stride * meta['frames']:
        raise ValueError('Wrong sampled residual framing')
    previous = np.zeros(shape, bool)
    output = []
    for offset in range(0, len(body), stride):
        delta = np.unpackbits(np.frombuffer(body[offset:offset+stride], np.uint8))[:pixels]
        previous = previous ^ delta.reshape(shape).astype(bool)
        output.append(previous)
    return meta, output


def predicted(archive: Path, kps: bytes) -> tuple[dict[str, Any], list[np.ndarray]]:
    meta, units = decode_keypoints(kps)
    boxes = box_maps(units)
    masks = []
    with tarfile.open(archive) as t:
        for i, unit in enumerate(units):
            f = t.extractfile('publish/' + unit['name'] + '-masks.npz')
            if f is None:
                raise ValueError('Missing real predictor masks')
            with np.load(io.BytesIO(f.read()), allow_pickle=False) as z:
                if z['t'].tolist() != list(range(0, 240, 3)):
                    raise ValueError('Sampled ticks changed')
                masks.append({int(tick): project(m, boxes[i][int(tick)])
                              for tick, m in zip(z['t'], z['masks'])})
    return meta, [np.stack([m[tick] for m in masks]) for tick in range(0, 240, 3)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction', type=Path, required=True)
    p.add_argument('--kps', type=Path, required=True)
    p.add_argument('--target', type=Path)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    meta, masks = predicted(a.prediction, a.kps.read_bytes())
    ticks = list(range(0, 240, 3))
    input_hashes = [array_sha(m) for m in masks]
    if a.target:
        target_meta, target = decode_masks(a.target.read_bytes())
        if target_meta['window'] != meta['window']:
            raise ValueError('Target identity differs')
        # Exact R is the native nearest-neighbor x2 source player mask, same as judge.
        exact = [np.repeat(np.repeat(target[tick].astype(bool), 2, axis=1), 2, axis=2)
                 for tick in ticks]
        residuals = [r ^ pred for r, pred in zip(exact, masks)]
        info = {'window': meta['window'], 'units': [u['name'] for u in meta['units']],
                'ticks': ticks, 'predictor_frame_sha256': input_hashes,
                'target_frame_sha256': [array_sha(r) for r in exact],
                'grid': 'original1080p_R', 'scope': '80 archived ticks only',
                'prompt_init': 'recomputable fixed KPS/SAM profile; no added content bits'}
        blob = encode(info, residuals)
        a.packet.write_bytes(blob)
        standalone = encode({**info, 'prompt_init': 'standalone exact sampled R'}, exact)
        write(a.out, {'bytes': len(blob), 'sha256': sha(blob),
                      'standalone_same_coder_bytes': len(standalone),
                      'target_hashes': info['target_frame_sha256'],
                      'window': meta['window'], 'extra_init_identity_framing_included': True})
    else:
        blob = a.packet.read_bytes()
        info, residuals = decode(blob)
        if info['window'] != meta['window'] or info['ticks'] != ticks or info['predictor_frame_sha256'] != input_hashes:
            raise ValueError('Decoder predictor/identity/timing mismatch')
        hashes = [array_sha(pred ^ residual) for pred, residual in zip(masks, residuals)]
        if hashes != info['target_frame_sha256']:
            raise ValueError('Independent correction decode is not exact')
        write(a.out, {'exact': True, 'frames': len(hashes), 'target_hashes': hashes,
                      'packet_sha256': sha(blob), 'source_or_target_read': False})
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write(a.out.with_suffix('.usage.json'), {'cpu_seconds': usage.ru_utime+usage.ru_stime,
                                           'wall_seconds': time.monotonic()-start})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

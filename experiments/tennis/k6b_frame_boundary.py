"""CPU-only frozen 2px boundary diagnostic, after receiver predictions freeze."""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import resource
import tarfile
import time

import numpy as np

from experiments.tennis.k6b_matte_cpu import box_maps
from experiments.tennis.k6b_rate_streams import decode_keypoints, decode_masks
from experiments.tennis.k6b_rate_transport import sha, write
from experiments.tennis.k6b_receiver_silhouette import project


def boundary_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    import cv2

    kernel = np.ones((3, 3), np.uint8)
    p = predicted & ~cv2.erode(predicted.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                                borderValue=0).astype(bool)
    r = target & ~cv2.erode(target.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                            borderValue=0).astype(bool)
    yy, xx = np.mgrid[-2:3, -2:3]
    tolerance = (xx * xx + yy * yy <= 4).astype(np.uint8)
    near_p = cv2.dilate(p.astype(np.uint8), tolerance).astype(bool)
    near_r = cv2.dilate(r.astype(np.uint8), tolerance).astype(bool)
    precision = float((p & near_r).sum() / p.sum()) if p.any() else float(not r.any())
    recall = float((r & near_p).sum() / r.sum()) if r.any() else float(not p.any())
    return {'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'predicted_edge_pixels': int(p.sum()), 'target_edge_pixels': int(r.sum())}


def main() -> int:
    import cv2

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction', type=Path, required=True)
    p.add_argument('--transport', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cv2.setNumThreads(2)
    started = time.monotonic()
    if sha(a.transport.read_bytes()) != '6865035e3132b6e07600fdaabc8c9d7152b8b6e3c3260d9f3df21cab33ca33c7':
        raise ValueError('Frozen transport identity changed')
    rows = []
    with tarfile.open(a.transport) as transport, tarfile.open(a.prediction) as prediction:
        for member in transport:
            if not member.isfile() or not member.name.endswith('keypoints.kps'):
                continue
            stream = transport.extractfile(member)
            assert stream is not None
            meta, units = decode_keypoints(stream.read())
            if meta['window'] not in ('alcaraz_hurkacz/031@129', 'medvedev_struff/014@194'):
                continue
            stream = transport.extractfile(member.name.replace('keypoints.kps', 'exact-rle.matte'))
            assert stream is not None
            _, targets = decode_masks(stream.read())
            cropped = []
            for unit in units:
                stream = prediction.extractfile('publish/' + unit['name'] + '-masks.npz')
                assert stream is not None
                with np.load(io.BytesIO(stream.read()), allow_pickle=False) as z:
                    if z['t'].tolist() != list(range(0, 240, 3)):
                        raise ValueError('Original80 ticks changed')
                    cropped.append(z['masks'])
            boxes = box_maps(units)
            frames = []
            for j, tick in enumerate(range(0, 240, 3)):
                pred = np.logical_or.reduce([project(mask[j], b[tick]) for mask, b in zip(cropped, boxes)])
                truth = np.repeat(np.repeat(targets[tick].any(0), 2, axis=0), 2, axis=1).astype(bool)
                frames.append({'t': tick, **boundary_metrics(pred, truth)})
            rows.append({'window': meta['window'], 'frames': frames,
                         'means': {key: float(np.mean([f[key] for f in frames]))
                                   for key in ('precision', 'recall', 'f1')}})
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write(a.out, {'rows': rows, 'boundary': 'inner3x3 eroded edge; Euclidean2px tolerance',
                  'cpu_seconds': usage.ru_utime + usage.ru_stime,
                  'wall_seconds': time.monotonic() - started,
                  'code_sha256': sha(Path(__file__).read_bytes()),
                  'prediction_archive_sha256': sha(a.prediction.read_bytes())})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

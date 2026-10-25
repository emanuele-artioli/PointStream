"""CPU comparison of sparse reference video, explicit alpha and black keys.

All decoded images are screening diagnostics; generated quality is unmeasured.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import resource
import tarfile
import time
from typing import Any

import numpy as np

from experiments.tennis.k6b_rate_streams import decode_masks, decode_reference, seal, unseal
from experiments.tennis.k6b_rate_transport import array_sha, sha, write
from src.codecs import svtav1
from src.pointstream import mask as mk
from src.pointstream.composite import OBJECT

PILOTS = ('alcaraz_hurkacz__031@129', 'medvedev_struff__014@194')
CRFS = (20, 35, 50)
FLOORS = (1, 8, 32)


def black_key(rgb: np.ndarray, alpha: np.ndarray, floor: int) -> np.ndarray:
    if floor not in FLOORS:
        raise ValueError('Undeclared guard band')
    return np.where(alpha[..., None], np.maximum(rgb, floor), 0).astype(np.uint8)


def recover_key(rgb: np.ndarray, floor: int) -> np.ndarray:
    return rgb.max(axis=-1) >= math.ceil(floor / 2)


def overlap(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    counts = {'intersection': int((a & b).sum()), 'union': int((a | b).sum()),
              'false_positive': int((b & ~a).sum()), 'false_negative': int((a & ~b).sum())}
    return {**counts, 'iou': counts['intersection'] / max(1, counts['union'])}



def mask_accuracy(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    import cv2
    hit_a = hit_b = edge_a = edge_b = 0
    kernel = np.ones((3, 3), np.uint8)
    for aa, bb in zip(a, b):
        ea = aa & ~cv2.erode(aa.astype(np.uint8), kernel).astype(bool)
        eb = bb & ~cv2.erode(bb.astype(np.uint8), kernel).astype(bool)
        edge_a += int(ea.sum())
        edge_b += int(eb.sum())
        hit_a += int((ea & cv2.dilate(eb.astype(np.uint8), kernel).astype(bool)).sum())
        hit_b += int((eb & cv2.dilate(ea.astype(np.uint8), kernel).astype(bool)).sum())
    precision, recall = hit_b/max(1, edge_b), hit_a/max(1, edge_a)
    return {**overlap(a, b), 'boundary_f_2_full_px': 2*precision*recall/max(1e-12, precision+recall)}

def aligned_packet(planes: np.ndarray) -> bytes:
    aligned = np.empty_like(planes)
    shifts = []
    h, w = planes.shape[-2:]
    for i, plane in enumerate(planes):
        y, x = np.nonzero(plane)
        dy, dx = (0, 0) if not len(x) else (h // 2 - int((y.min()+y.max())//2), w // 2 - int((x.min()+x.max())//2))
        aligned[i] = np.roll(plane, (dy, dx), (0, 1))
        shifts.append([dy, dx])
    payload = mk.encode(aligned.astype(np.uint8) * OBJECT, 8)
    return seal(b'ALN1', {'shifts': shifts, 'rule': 'reduced-bbox-centre-circular-roll',
                         'shape': list(planes.shape)}, payload)


def decode_aligned(blob: bytes) -> np.ndarray:
    meta, payload = unseal(blob, b'ALN1')
    values, scale = mk.decode(payload)
    if scale != 8 or list(values.shape) != meta['shape'] or len(meta['shifts']) != len(values):
        raise ValueError('Alignment contract mismatch')
    return np.stack([np.roll(p > 0, (-dy, -dx), (0, 1))
                     for p, (dy, dx) in zip(values, meta['shifts'])])


def command(argv: list[str]) -> dict[str, Any]:
    timing = svtav1.run(argv, 100, cpus=list(os.sched_getaffinity(0)))
    return {'command': argv, **timing}


def ffmpeg_args() -> list[str]:
    return ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-filter_threads', '1', '-y']


def rgb_yuv(rgb: np.ndarray, root: Path) -> tuple[Path, dict[str, Any]]:
    n, h, w, _ = rgb.shape
    raw, out = root / 'rgb.raw', root / 'source.yuv'
    rgb.tofile(raw)
    rec = command([*ffmpeg_args(), '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{w}x{h}',
                   '-framerate', '1', '-i', str(raw), '-frames:v', str(n), '-vf',
                   'scale=in_range=full:out_range=full:out_color_matrix=bt709', '-pix_fmt', 'yuv420p',
                   '-f', 'rawvideo', str(out)])
    raw.unlink()
    return out, rec


def code_yuv(raw: Path, root: Path, n: int, w: int, h: int, crf: int, fps: int = 1) -> dict[str, Any]:
    root.mkdir()
    rec = svtav1.code(raw, root, width=w, height=h, fps=Fraction(fps), frames=n,
                       crf=crf, preset=8, threads=4, full_range=True,
                       encoder='SvtAv1EncApp', decoder='dav1d', timeout=100,
                       cpus=list(os.sched_getaffinity(0)))
    if rec['decoded_frames'] != n or len(svtav1.ivf_frames(Path(rec['stream']).read_bytes())) < 1:
        raise ValueError('Incomplete real video decode')
    return rec


def back_rgb(rec: dict[str, Any], n: int, w: int, h: int, root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    out = root / 'back.rgb'
    conversion = command([*ffmpeg_args(), '-f', 'rawvideo', '-pixel_format', 'yuv420p', '-video_size', f'{w}x{h}',
                          '-framerate', '1', '-i', rec['decoded'], '-frames:v', str(n), '-vf',
                          'scale=in_range=full:out_range=full:in_color_matrix=bt709', '-pix_fmt', 'rgb24',
                          '-f', 'rawvideo', str(out)])
    rgb = np.fromfile(out, np.uint8).reshape(n, h, w, 3)
    return rgb, conversion


def container(records: list[dict[str, Any]], path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    argv = ffmpeg_args()
    for rec in records:
        argv += ['-i', rec['stream']]
    for i in range(len(records)):
        argv += ['-map', f'{i}:v:0', f'-metadata:s:v:{i}', f'title={"color" if i == 0 else "alpha"}']
    argv += ['-c', 'copy', '-metadata', 'POINTSTREAM='+json.dumps(meta, separators=(',', ':')), str(path)]
    mux = command(argv)
    # Decode the delivered container, not only the pre-mux IVF. Explicit per-track selection.
    checks = []
    for i, rec in enumerate(records):
        extracted = path.parent / f'{path.stem}-track{i}.yuv'
        command([*ffmpeg_args(), '-i', str(path), '-map', f'0:v:{i}', '-pix_fmt', 'yuv420p',
                 '-f', 'rawvideo', str(extracted)])
        expected = Path(rec['decoded']).read_bytes()
        if extracted.read_bytes() != expected:
            raise ValueError('Container track decode differs from native dav1d')
        checks.append(sha(expected))
        extracted.unlink()
    return {'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes()),
            'track_decoded_sha256': checks, 'mux': mux, 'contract': meta}


def ref_diagnostics(original: np.ndarray, source_mask: np.ndarray, small_rgb: np.ndarray,
                    small_mask: np.ndarray) -> dict[str, Any]:
    import cv2
    full = np.stack([cv2.resize(im, (512, 784), interpolation=cv2.INTER_CUBIC) for im in small_rgb])
    alpha = np.stack([cv2.resize(m.astype(np.float32), (512, 784), interpolation=cv2.INTER_LINEAR) >= .5 for m in small_mask])
    diff = (full.astype(np.float32) - original.astype(np.float32)) ** 2
    return {**overlap(source_mask, alpha), 'rgb_mse': float(diff.mean()),
            'foreground_rgb_mse': float(diff[source_mask].mean()),
            'decoded_rgb_sha256': array_sha(full), 'decoded_alpha_sha256': array_sha(alpha)}


def reference_banks(packets: Path) -> dict[str, list[dict[str, Any]]]:
    from experiments.tennis.k6b import crop_image
    out: dict[str, list[dict[str, Any]]] = {}
    import cv2
    for root in sorted(packets.iterdir()):
        if not root.is_dir():
            continue
        _, masks = decode_masks((root / 'exact-rle.matte').read_bytes())
        from experiments.tennis.k6b_rate_streams import decode_keypoints
        _, units = decode_keypoints((root / 'keypoints.kps').read_bytes())
        names = [u['name'] for u in units]
        for path in sorted(root.glob('*.png.ref')):
            meta, rgb = decode_reference(path.read_bytes())
            t = meta['reference_frame']
            if t != 0 or rgb.shape != (784, 512, 3):
                raise ValueError('Frozen reference identity/dimensions changed')
            p = names.index(meta['unit'])
            full = np.repeat(np.repeat(masks[t, p], 2, 0), 2, 1)
            crop = crop_image(np.repeat(full[..., None].astype(np.uint8) * 255, 3, 2), np.asarray(meta['reference_box']))[..., 0] >= 128
            match = meta['unit'].split('__')[0]
            out.setdefault(match, []).append({'meta': meta, 'rgb': rgb, 'mask': crop,
                'small': cv2.resize(rgb, (256, 392), interpolation=cv2.INTER_AREA),
                'small_mask': cv2.resize(crop.astype(np.float32), (256, 392), interpolation=cv2.INTER_AREA) >= .5,
                'source_reference_sha256': sha(path.read_bytes()), 'source_mask_sha256': array_sha(full),
                'source_crop_mask_sha256': array_sha(crop)})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--transport', type=Path)
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    if args.validate:
        return validate()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only fleet contract required')
    import cv2
    cv2.setNumThreads(1)
    cpus = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, cpus)
    began = time.monotonic()
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch / 'publish'
    publish.mkdir()
    with tarfile.open(args.transport) as tar:
        tar.extractall(scratch / 'transport', filter='data')
    banks = reference_banks(scratch / 'transport/packets')
    report: dict[str, Any] = {'rows': [], 'alignments': [], 'mode': args.mode, 'gpu_hours': 0,
        'input_sha256': sha(args.transport.read_bytes()), 'code_sha256': sha(Path(__file__).read_bytes()),
        'tools': {n: svtav1.tool(n) for n in ('ffmpeg', 'SvtAv1EncApp', 'dav1d')},
        'numpy': np.__version__, 'opencv': cv2.__version__, 'banks': {}, 'cuda_visible': os.environ['CUDA_VISIBLE_DEVICES']}
    def save() -> None:
        report['wall_seconds'] = time.monotonic() - began
        report['cpu_seconds'] = sum(r.ru_utime+r.ru_stime for r in (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))
        if report['cpu_seconds'] > (1200 if args.mode == 'smoke' else 4800):
            raise RuntimeError('Stage CPU allowance exhausted')
        write(publish / 'reference-video.json', report)
        from experiments.jobs.monitor import publish_progress
        publish_progress('reference-video', len(report['rows']) + len(report['alignments']))
    crfs = (35,) if args.mode == 'smoke' else CRFS
    for match, entries in list(banks.items())[:1 if args.mode == 'smoke' else None]:
        original = np.stack([e['rgb'] for e in entries])
        source_mask = np.stack([e['mask'] for e in entries])
        small = np.stack([e['small'] for e in entries])
        alpha = np.stack([e['small_mask'] for e in entries])
        n, h, w = alpha.shape
        meta: dict[str, Any] = {'units': [e['meta'] for e in entries], 'size': [w, h], 'output_size': [512, 784],
                'order': 'unit-ID-within-match-not-source-chronology', 'rgb_restore': 'bicubic', 'alpha_restore': 'bilinear>=0.5',
                'matrix': 'bt709', 'range': 'full', 'tick': 'reference-index'}
        report['banks'][match] = [{k: v for k, v in e.items() if k not in ('rgb', 'mask', 'small', 'small_mask')} for e in entries]
        alpha_raw = scratch / f'{match}-alpha.yuv'
        y = np.where(alpha, 255, 0).astype(np.uint8).reshape(n, -1)
        np.concatenate([y, np.full((n, h*w//2), 128, np.uint8)], 1).tofile(alpha_raw)
        for crf in crfs:
            alpha_rec = code_yuv(alpha_raw, scratch / f'{match}-alpha-{crf}', n, w, h, crf)
            back_alpha = np.fromfile(alpha_rec['decoded'], np.uint8).reshape(n, -1)[:, :w*h].reshape(n, h, w) >= 128
            for kind, floor in [('full', 0), ('alpha', 0), *[('black', f) for f in FLOORS]]:
                label = f'{match}-{kind}-{floor}-q{crf}'
                root = scratch / label
                root.mkdir()
                rgb = small if kind == 'full' else black_key(small, alpha, floor) if kind == 'black' else np.where(alpha[..., None], small, 0).astype(np.uint8)
                raw, conversion = rgb_yuv(rgb, root)
                rec = code_yuv(raw, root / 'codec', n, w, h, crf)
                decoded, inverse = back_rgb(rec, n, w, h, root)
                back_mask = np.ones_like(alpha) if kind == 'full' else recover_key(decoded, floor) if kind == 'black' else back_alpha
                contract = {**meta, 'kind': kind, 'floor': floor, 'threshold': math.ceil(floor/2) if floor else None,
                            'tracks': ['color', 'alpha'] if kind == 'alpha' else ['color']}
                delivered = container([rec, alpha_rec] if kind == 'alpha' else [rec], publish / f'{label}.mkv', contract)
                row = {'kind': kind, 'floor': floor, 'crf': crf, 'match': match, 'references': n,
                       'container': delivered, 'kbps_7200': delivered['bytes']*8/7200/1000,
                       'codec': rec, 'alpha_codec': alpha_rec if kind == 'alpha' else None,
                       'conversion': conversion, 'inverse_conversion': inverse,
                       'source_rgb_sha256': array_sha(rgb), 'source_small_mask_sha256': array_sha(alpha),
                       'diagnostic_only': ref_diagnostics(original, source_mask, decoded, back_mask)}
                report['rows'].append(row)
                np.savez_compressed(publish / f'{label}-conditioning.npz', rgb=np.stack([cv2.resize(x, (512, 784), interpolation=cv2.INTER_CUBIC) for x in decoded]), alpha=back_mask)
                raw.unlink()
                save()
    for window in PILOTS[:1 if args.mode == 'smoke' else None]:
        packet_root = scratch / 'transport/packets' / window
        _, native = decode_masks((packet_root / 'exact-rle.matte').read_bytes())
        from experiments.tennis.k6b_rate_streams import decode_keypoints
        kmeta, _ = decode_keypoints((packet_root / 'keypoints.kps').read_bytes())
        n = len(native)
        for player in range(2):
            classes = native[:, player].astype(np.uint8) * OBJECT
            small = mk.reduce(classes, 4) > 0
            blob = aligned_packet(small)
            if not np.array_equal(decode_aligned(blob), small):
                raise ValueError('Aligned packet is not reversible')
            name = f'{window}-p{player}-aligned.aln'
            (publish / name).write_bytes(blob)
            plain = mk.encode(small.astype(np.uint8)*OBJECT, 8)
            report['alignments'].append({'window': window, 'player': player, 'file': name, 'bytes': len(blob),
                'plain_h5_bytes': len(plain), 'sha256': sha(blob), 'reduced_sha256': array_sha(small), 'exact': True})
            save()
            for representation in ('binary', 'sdf'):
                planes = mk.video_planes(classes, 4, representation)
                pw, ph = mk.padded_size(960, 540, 4)
                raw = scratch / f'{window}-{player}-{representation}.yuv'
                planes.tofile(raw)
                baseline = mk.upsample(small.astype(np.uint8)*OBJECT, (960, 540)) > 0
                baseline_accuracy = mask_accuracy(classes > 0, baseline)
                for crf in crfs:
                    label = f'{window}-p{player}-{representation}-q{crf}'
                    rec = code_yuv(raw, scratch / label, n, pw, ph, crf, kmeta['fps'][0])
                    decoded = np.fromfile(rec['decoded'], np.uint8).reshape(n, -1)
                    restored = mk.video_classes(decoded, 4, (960, 540)) > 0
                    native_bool = classes > 0
                    meta = {'window': window, 'player': player, 'frames': n, 'coded_size': [pw, ph], 'output_size': [1920, 1080],
                            'scale': 8, 'representation': representation, 'threshold': 128, 'time_base': kmeta['fps']}
                    # IVF + compressed explicit metadata is the delivered mask packet; all bytes counted.
                    payload = seal(b'MVV1', meta, Path(rec['stream']).read_bytes())
                    packet_name = f'{label}.mvv'
                    (publish / packet_name).write_bytes(payload)
                    report['rows'].append({'kind': 'mask-video', 'window': window, 'player': player,
                        'crf': crf, 'representation': representation, 'bytes': len(payload), 'file': packet_name,
                        'sha256': sha(payload), 'frames': n, 'codec': rec,
                        'native_grid_accuracy': mask_accuracy(native_bool, restored), 'baseline_h5_accuracy': baseline_accuracy,
                        'decoded_sha256': array_sha(restored), 'kbps_whole_window': len(payload)*8/(n/kmeta['fps'][0])/1000})
                    save()
                raw.unlink()
    save()
    write(Path(os.environ['PS_STAGE_DIR']) / 'reference-video.json', report)
    print(json.dumps({'rows': len(report['rows']), 'alignments': len(report['alignments']), 'cpu_seconds': report['cpu_seconds'], 'report': str(publish / 'reference-video.json')}))
    return 0


def validate() -> int:
    report = json.loads((Path(os.environ['PS_STAGE_DIR']) / 'reference-video.json').read_text())
    checks = {'cpu_only': report['gpu_hours'] == 0 and report['cuda_visible'] == '',
              'rows_present': len(report['rows']) == (9 if report['mode'] == 'smoke' else 54),
              'alignment_count': len(report['alignments']) == (2 if report['mode'] == 'smoke' else 4)}
    for match, bank in report['banks'].items():
        checks[match+':reference_identity'] = len(bank) == 12 and len({e['meta']['unit'] for e in bank}) == 12 and all(e['meta']['reference_frame'] == 0 and len(e['source_mask_sha256']) == 64 for e in bank)
    toy = np.zeros((1, 4, 4, 3), np.uint8)
    mask = np.zeros((1, 4, 4), bool)
    mask[:, 1:3, 1:3] = True
    checks['precodec_guard_band'] = all(np.array_equal(recover_key(black_key(toy, mask, floor), floor), mask) for floor in FLOORS)
    # Artifacts stay on scratch during smoke validation; validate bytes and actual decoded arrays.
    root = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
    for i, row in enumerate(report['rows']):
        file = row['container'] if 'container' in row else row
        checks[f'{i}:packet'] = sha((root/file['file']).read_bytes()) == file['sha256'] and (root/file['file']).stat().st_size == file['bytes']
        checks[f'{i}:finite'] = all(np.isfinite(v) for v in (row.get('diagnostic_only', row.get('native_grid_accuracy', {}))).values() if isinstance(v, (int, float)))
        checks[f'{i}:frames'] = row['codec']['decoded_frames'] == row.get('references', row.get('frames'))
    for i, row in enumerate(report['alignments']):
        blob = (root/row['file']).read_bytes()
        checks[f'align{i}'] = sha(blob) == row['sha256'] and array_sha(decode_aligned(blob)) == row['reduced_sha256']
    write(Path(os.environ['PS_VALIDATION_PATH']), {'passed': all(checks.values()), 'checks': checks})
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())

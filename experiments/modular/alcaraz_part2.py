"""Per-object Alcaraz appearance refresh, silhouette, and oracle campaign."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3  # noqa: F401  # host C++ runtime before optional pose imports
import struct
import sys
import time
import zlib

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from experiments.modular.appearance_motion_probe import _bbox_affine, _box_payload
from experiments.modular.background_campaign import _timed_vvc
from experiments.modular.foreground_campaign import _alpha_wire, _correct, _residual_signal, _row, _scores
from experiments.modular.foreground_part2_common import load_fixed_alcaraz
from experiments.modular.object_foreground_campaign import _affine_warp, _boxes, split_object_tracks
from src.components.background.sidecar import IntraCodecSidecar
from src.components.codec import tools as codec_tools
from src.pipeline.residual.lossy import residual_clip_fraction

OUT = Path('/home/itec/emanuele/pointstream-data/outputs/modular/foreground-part2/alcaraz')
N_FRAMES = 48
SCHEDULES = (('first_only', 48), ('every_24', 24), ('every_12', 12), ('every_6', 6))


def refresh_indices(presence: np.ndarray, interval: int) -> list[int]:
    """Schedule first and every ``interval``-th visible frame causally."""
    flags = np.asarray(presence)
    if flags.ndim != 1 or flags.size == 0 or flags.dtype != np.bool_:
        raise ValueError('presence must be a nonempty Boolean timeline')
    if interval <= 0:
        raise ValueError('interval must be positive')
    visible = np.flatnonzero(flags)
    if not visible.size:
        raise ValueError('object has no visible frame')
    return [int(i) for i in visible[::interval]]


def _decode_silhouette(wire: bytes) -> tuple[int, np.ndarray]:
    if len(wire) <= 6:
        raise RuntimeError('empty silhouette payload')
    frame, height, width = struct.unpack('<HHH', wire[:6])
    if not height or not width:
        raise RuntimeError('empty silhouette shape')
    packed = np.frombuffer(zlib.decompress(wire[6:]), dtype=np.uint8)
    if packed.size != (height * width + 7) // 8:
        raise RuntimeError('silhouette payload length changed')
    bits = np.unpackbits(packed, bitorder='little')[:height * width]
    return frame, bits.reshape(height, width).astype(bool)


def _pack_silhouette(mask: np.ndarray, box: tuple[int, int, int, int], frame: int) -> bytes:
    """Encode an exact ROI mask with its frame index and dimensions."""
    if not 0 <= frame < N_FRAMES:
        raise ValueError('silhouette frame is outside 48-frame window')
    y1, y2, x1, x2 = box
    roi = np.ascontiguousarray(mask[y1:y2, x1:x2], dtype=np.uint8)
    if not roi.size or not roi.any():
        raise ValueError('empty silhouette ROI')
    return struct.pack('<HHH', frame, *roi.shape) + zlib.compress(np.packbits(roi, bitorder='little').tobytes(), 9)


def _silhouette_wire(mask: np.ndarray, box: tuple[int, int, int, int], frame: int) -> tuple[bytes, np.ndarray]:
    """Charge and decode a frame-indexed ROI mask with shape headers."""
    wire = _pack_silhouette(mask, box, frame)
    restored_frame, restored = _decode_silhouette(wire)
    y1, y2, x1, x2 = box
    if restored_frame != frame or not np.array_equal(restored, mask[y1:y2, x1:x2].astype(bool)):
        raise RuntimeError('silhouette wire round trip failed')
    return wire, restored


def _encode_references(source: np.ndarray, tracks: list[np.ndarray], interval: int, sidecar: IntraCodecSidecar) -> tuple[list[dict], dict]:
    """Code each player's own QP42 references, alphas, boxes, and presence."""
    times: dict[str, float | int] = {'prepare': 0.0, 'encode': 0.0, 'decode': 0.0, 'F': 0, 'M': 0, 'H': 1}
    objects = []
    for obj_id, track in enumerate(tracks):
        start = time.perf_counter()
        boxes, presence, _ = _boxes(track)
        indices = refresh_indices(presence, interval)
        presence_wire = np.packbits(presence.astype(np.uint8), bitorder='little').tobytes()
        bbox_wire = _box_payload(boxes)
        times['M'] += len(presence_wire) + len(bbox_wire)
        times['H'] += 2 * len(indices)  # uint16 frame index per appearance record
        times['prepare'] += time.perf_counter() - start
        refs = []
        for t in indices:
            box = boxes[t]
            start = time.perf_counter()
            alpha_wire, alpha = _alpha_wire(track[t], box)
            times['H'] += len(alpha_wire)
            y1, y2, x1, x2 = box
            crop = np.ascontiguousarray(source[t, y1:y2, x1:x2, ::-1])  # RGB -> BGR on last axis
            if not crop.size:
                raise RuntimeError(f'object {obj_id} frame {t}: empty appearance crop')
            coded_h = max(64, crop.shape[0] + (-crop.shape[0] % 8))
            coded_w = max(64, crop.shape[1] + (-crop.shape[1] % 8))
            padded = np.zeros((coded_h, coded_w, 3), dtype=np.uint8)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            times['prepare'] += time.perf_counter() - start
            start = time.perf_counter()
            wire = sidecar.encode(padded)
            times['encode'] += time.perf_counter() - start
            if not wire:
                raise RuntimeError(f'object {obj_id} frame {t}: empty AV1 crop bitstream')
            times['F'] += len(wire)
            start = time.perf_counter()
            decoded = sidecar.decode(wire)
            times['decode'] += time.perf_counter() - start
            if decoded.shape != padded.shape:
                raise RuntimeError('decoded AV1 crop shape changed')
            refs.append({'frame': t, 'box': box, 'alpha': alpha, 'crop_bgr': decoded[:crop.shape[0], :crop.shape[1]],
                         'crop_bytes': len(wire), 'alpha_bytes': len(alpha_wire), 'index_bytes': 2})
        objects.append({'index': obj_id, 'boxes': boxes, 'presence': presence, 'refs': refs,
                        'bbox_bytes': len(bbox_wire), 'presence_bytes': len(presence_wire)})
    return objects, times


def _render(background: np.ndarray, objects: list[dict], *,
            silhouettes: list[list[np.ndarray | None]] | None = None,
            oracle_tracks: list[np.ndarray] | None = None,
            current_source: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Decode newest causal reference, with an explicitly optional oracle."""
    n, height, width, _ = background.shape
    out = np.empty(background.shape, dtype=np.uint8)
    started = time.perf_counter()
    for t in range(n):
        out[t] = background[t]
        for j, item in enumerate(objects):
            if not item['presence'][t]:
                continue
            ref = item['refs'][0]
            for candidate in item['refs'][1:]:
                if candidate['frame'] > t:
                    break
                ref = candidate
            if ref['frame'] > t:
                raise RuntimeError('future appearance selected')
            target_box = item['boxes'][t]
            if current_source is None:
                crop, alpha, source_box = ref['crop_bgr'], ref['alpha'], ref['box']
            else:
                source_box = target_box
                y1, y2, x1, x2 = target_box
                crop = np.ascontiguousarray(current_source[t, y1:y2, x1:x2, ::-1])
                alpha = np.ones(crop.shape[:2], dtype=bool)
            matrix = _bbox_affine(source_box, target_box)
            patch, cover = _affine_warp(crop, alpha, matrix, source_box, (height, width))
            if silhouettes is not None or oracle_tracks is not None:
                # New alpha can expose pixels outside the old matte, so warp
                # the full crop rectangle for color; apply the new matte below.
                patch, _ = _affine_warp(crop, np.ones(crop.shape[:2], dtype=bool), matrix, source_box, (height, width))
            if silhouettes is not None:
                roi = silhouettes[j][t]
                if roi is None:
                    raise RuntimeError('visible object lacks transmitted silhouette')
                y1, y2, x1, x2 = target_box
                if roi.shape != (y2-y1, x2-x1):
                    raise RuntimeError('silhouette shape disagrees with transmitted bbox')
                cover = np.zeros((height, width), dtype=bool)
                cover[y1:y2, x1:x2] = roi
            if oracle_tracks is not None:
                cover = oracle_tracks[j][t]
            out[t][cover] = patch[cover, ::-1]  # BGR -> RGB on last axis
    return out, time.perf_counter() - started


def _encode_silhouettes(tracks: list[np.ndarray], objects: list[dict]) -> tuple[list[list[np.ndarray | None]], int, float, float]:
    decoded = []
    nbytes = 0
    encode_s = decode_s = 0.0
    for track, item in zip(tracks, objects, strict=True):
        timeline: list[np.ndarray | None] = [None] * len(track)
        for frame in np.flatnonzero(item['presence']):
            t = int(frame)
            start = time.perf_counter()
            wire = _pack_silhouette(track[t], item['boxes'][t], t)
            encode_s += time.perf_counter() - start
            start = time.perf_counter()
            decoded_frame, roi = _decode_silhouette(wire)
            decode_s += time.perf_counter() - start
            if decoded_frame != t:
                raise RuntimeError('silhouette frame index changed')
            y1, y2, x1, x2 = item['boxes'][t]
            if not np.array_equal(roi, track[t, y1:y2, x1:x2]):
                raise RuntimeError('silhouette payload did not round trip')
            timeline[t] = roi
            nbytes += len(wire)
        decoded.append(timeline)
    return decoded, nbytes, encode_s, decode_s


def _summary(objects: list[dict]) -> list[dict]:
    return [{'index': obj['index'], 'visible_frames': int(obj['presence'].sum()),
             'bbox_bytes': obj['bbox_bytes'], 'presence_bytes': obj['presence_bytes'],
             'references': [{k: ref[k] for k in ('frame', 'crop_bytes', 'alpha_bytes', 'index_bytes')}
                            for ref in obj['refs']]} for obj in objects]


def _oracle_scores(source: np.ndarray, delivered: np.ndarray, mask: np.ndarray) -> dict:
    scores = _scores(source, delivered, mask)
    # A current-frame source crop can match all player pixels exactly. The
    # resulting infinite foreground PSNR is an oracle bound, not a claim row;
    # write JSON null rather than a nonstandard Infinity token.
    return {key: value if value is None or np.isfinite(value) else None for key, value in scores.items()}


def _check_actual_row(row: dict) -> None:
    if row['total_bytes'] != sum(int(row[key]) for key in ('B', 'F', 'M', 'R', 'H')):
        raise RuntimeError('wire components do not sum to row total')
    weighted = row['scores']['weighted']
    if weighted is None or not np.isfinite(weighted):
        raise RuntimeError('actual bitstream has nonfinite weighted PSNR')


def run_alcaraz_ladder(out_dir: Path = OUT) -> dict[str, object]:
    """Measure mask/current-appearance oracles and actual charged ladders.

    The loader rejects a moved 24,648 B background. Target masks enter only
    encoder observations, oracle diagnostics, residual gates and scoring.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed = load_fixed_alcaraz(out_dir.parent)
    source, mask = fixed.source_rgb, fixed.mask
    bg_encode_s = float(fixed.background.get('reencode_seconds', fixed.background['encode_seconds']))
    bg_decode_s = float(fixed.background.get('redecode_seconds', fixed.background['decode_seconds']))
    bg_render_s = float(fixed.background.get('rerender_seconds', fixed.background.get('render_seconds', 0.0)))
    if source.shape[0] != N_FRAMES or mask.shape != source.shape[:3]:
        raise RuntimeError('Alcaraz source/mask must be the fixed 48-frame window')
    started = time.perf_counter()
    tracks = split_object_tracks(mask)
    tracking_s = time.perf_counter() - started
    if len(tracks) != 2:
        raise RuntimeError(f'expected two independently coded players, found {len(tracks)}')
    sidecar = IntraCodecSidecar('av1', qp=42)
    crop_path, crop_version = sidecar.probe_encoder()
    decoder = codec_tools.resolve_ffmpeg()
    result: dict[str, object] = {
        'clip_id': 'alcaraz000', 'frames': N_FRAMES, 'source_anchor': fixed.anchor,
        'background': fixed.background, 'plate_seconds': fixed.plate_seconds,
        'tracking_seconds': tracking_s,
        'appearance': {'codec': 'av1', 'qp': 42, 'encoder_path': crop_path, 'encoder_version': crop_version,
                       'decoder_path': decoder.path, 'decoder_version': decoder.version},
        'objects': len(tracks), 'oracles': [], 'reference_schedules': {}, 'rows': [],
        'residual_encodes': [], 'completed': False,
    }
    ledger = out_dir / 'alcaraz000.json'

    def save() -> None:
        ledger.write_text(json.dumps(result, indent=2) + '\n')

    save()
    best = None
    best_objects = None
    candidate_rows: dict[str, dict] = {}
    candidate_objects: dict[str, list[dict]] = {}
    for label, interval in SCHEDULES:
        print(f'alcaraz000: {label} per-object references', flush=True)
        objects, times = _encode_references(source, tracks, interval, sidecar)
        result['reference_schedules'][label] = _summary(objects)
        if label == 'first_only':
            for name, kwargs in (
                ('true_alpha_warped_appearance', {'oracle_tracks': tracks}),
                ('true_alpha_current_uncompressed_appearance', {'oracle_tracks': tracks, 'current_source': source}),
            ):
                delivered, render_s = _render(fixed.background_rgb, objects, **kwargs)
                result['oracles'].append({'name': name, 'target_informed': True, 'claimable': False,
                                          'scores': _oracle_scores(source, delivered, mask),
                                          'exact_foreground': bool(np.array_equal(delivered[mask], source[mask])),
                                          'render_seconds': render_s})
                del delivered
                save()
        base, render_s = _render(fixed.background_rgb, objects)
        row = _row(label, 'neither', B=int(fixed.background['total_bytes']),
                   F=int(times['F']), M=int(times['M']), R=0, H=int(times['H']),
                   source=source, delivered=base, mask=mask,
                   encode_s=bg_encode_s+tracking_s+float(times['prepare'])+float(times['encode']),
                   decode_s=bg_decode_s+float(times['decode']),
                   render_s=bg_render_s+render_s,
                   plate_s=fixed.plate_seconds, source_row=fixed.anchor)
        row['residual_clip_fraction'] = residual_clip_fraction(source.astype(np.int16) - base.astype(np.int16), mask)
        row['foreground_encode_seconds'] = tracking_s+float(times['prepare'])+float(times['encode'])
        row['foreground_decode_seconds'] = float(times['decode'])
        _check_actual_row(row)
        result['rows'].append(row)
        candidate_rows[label] = row
        candidate_objects[label] = objects
        del base
        save()
        print(f"alcaraz000: {label} {row['total_bytes']} B weighted={row['scores']['weighted']:.3f}", flush=True)
        if row['total_bytes'] <= int(fixed.anchor['total_bytes']):
            if best is None or float(row['scores']['weighted']) > float(best['scores']['weighted']):
                best, best_objects = row, objects
    if best is None:
        result['stopped'] = 'all residual-off refresh points exceed anchor'
        save()
        return result

    print('alcaraz000: rate-probe transmitted silhouettes', flush=True)
    silhouettes, silhouette_bytes, silhouette_enc, silhouette_dec = _encode_silhouettes(tracks, candidate_objects['first_only'])
    feasible = {
        label: int(row['total_bytes']) + silhouette_bytes <= int(fixed.anchor['total_bytes'])
        for label, row in candidate_rows.items()
    }
    result['silhouette_probe'] = {
        'payload_bytes': silhouette_bytes,
        'totals_by_arm': {label: int(row['total_bytes']) + silhouette_bytes for label, row in candidate_rows.items()},
        'under_cap_by_arm': feasible,
        'encode_seconds': silhouette_enc, 'decode_seconds': silhouette_dec,
        'format': 'per-object/frame uint16 frame,height,width + zlib9 packed ROI bits',
    }
    save()
    chosen_silhouettes = None
    chosen_silhouette_enc = chosen_silhouette_dec = 0.0
    for label, fits in feasible.items():
        if not fits:
            continue
        parent_row = candidate_rows[label]
        objects = candidate_objects[label]
        base, render_s = _render(fixed.background_rgb, objects, silhouettes=silhouettes)
        row = _row(f'{label}_silhouette', 'neither', B=int(parent_row['B']), F=int(parent_row['F']),
                   M=int(parent_row['M']), R=0, H=int(parent_row['H'])+silhouette_bytes,
                   source=source, delivered=base, mask=mask,
                   encode_s=bg_encode_s+float(parent_row['foreground_encode_seconds'])+silhouette_enc,
                   decode_s=bg_decode_s+float(parent_row['foreground_decode_seconds'])+silhouette_dec,
                   render_s=bg_render_s+render_s,
                   plate_s=fixed.plate_seconds, source_row=fixed.anchor)
        row['residual_clip_fraction'] = residual_clip_fraction(source.astype(np.int16) - base.astype(np.int16), mask)
        row['foreground_encode_seconds'] = parent_row['foreground_encode_seconds']
        row['foreground_decode_seconds'] = parent_row['foreground_decode_seconds']
        _check_actual_row(row)
        result['rows'].append(row)
        save()
        if float(row['scores']['weighted']) > float(best['scores']['weighted']):
            best, best_objects, chosen_silhouettes = row, objects, silhouettes
            chosen_silhouette_enc, chosen_silhouette_dec = silhouette_enc, silhouette_dec
        del base
    result['best_residual_off_arm'] = best['arm']
    save()

    base, render_s = _render(fixed.background_rgb, best_objects, silhouettes=chosen_silhouettes)
    fraction = residual_clip_fraction(source.astype(np.int16) - base.astype(np.int16), mask)
    result['best_residual_clip_fraction'] = fraction
    if fraction > 0.05:
        result['residual_policy'] = 'coarse QP54/62 only; clip fraction exceeds 0.05'
    save()
    signals = {}
    preps = {}
    for region, region_mask in (('fg', mask), ('bg', ~mask)):
        start = time.perf_counter()
        signal = _residual_signal(source, base, region_mask)
        preps[region] = time.perf_counter()-start
        signals[region] = {}
        for qp in (54, 62):
            wire, pixels, enc_s, dec_s, path, version = _timed_vvc(signal, qp)
            if not wire:
                raise RuntimeError(f'empty {region} QP{qp} residual bitstream')
            signals[region][qp] = (len(wire), pixels, enc_s, dec_s)
            result['residual_encodes'].append({'region': region, 'qp': qp, 'bytes': len(wire),
                                               'encode_seconds': enc_s, 'decode_seconds': dec_s,
                                               'encoder_path': path, 'encoder_version': version,
                                               'decoder_path': decoder.path, 'decoder_version': decoder.version})
            save()
        del signal
    common_encode = bg_encode_s+float(best['foreground_encode_seconds'])+chosen_silhouette_enc
    common_decode = bg_decode_s+float(best['foreground_decode_seconds'])+chosen_silhouette_dec
    common_render = bg_render_s+render_s
    components = {key: int(best[key]) for key in ('B', 'F', 'M', 'H')}
    for region in ('fg', 'bg'):
        for qp in (54, 62):
            size, pixels, enc_s, dec_s = signals[region][qp]
            correction_started = time.perf_counter()
            delivered = _correct(base, pixels if region == 'fg' else None, pixels if region == 'bg' else None)
            correction_s = time.perf_counter() - correction_started
            row = _row(str(best['arm']), f'{region}_only_qp{qp}', **components, R=size,
                       source=source, delivered=delivered, mask=mask,
                       encode_s=common_encode+preps[region]+enc_s, decode_s=common_decode+dec_s,
                       render_s=common_render+correction_s, plate_s=fixed.plate_seconds, source_row=fixed.anchor,
                       fg_qp=qp if region == 'fg' else None, bg_qp=qp if region == 'bg' else None)
            row['residual_clip_fraction'] = fraction
            row['correction_render_seconds'] = correction_s
            _check_actual_row(row)
            result['rows'].append(row)
            del delivered
            save()
    for bg_qp in (54, 62):
        for fg_qp in (54, 62):
            f_size, f_pixels, f_enc, f_dec = signals['fg'][fg_qp]
            b_size, b_pixels, b_enc, b_dec = signals['bg'][bg_qp]
            correction_started = time.perf_counter()
            delivered = _correct(base, f_pixels, b_pixels)
            correction_s = time.perf_counter() - correction_started
            row = _row(str(best['arm']), f'both_fg{fg_qp}_bg{bg_qp}', **components, R=f_size+b_size,
                       source=source, delivered=delivered, mask=mask,
                       encode_s=common_encode+preps['fg']+preps['bg']+f_enc+b_enc,
                       decode_s=common_decode+f_dec+b_dec,
                       render_s=common_render+correction_s, plate_s=fixed.plate_seconds, source_row=fixed.anchor,
                       fg_qp=fg_qp, bg_qp=bg_qp)
            row['residual_clip_fraction'] = fraction
            row['correction_render_seconds'] = correction_s
            _check_actual_row(row)
            result['rows'].append(row)
            del delivered
            save()
    result['completed'] = True
    save()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUT)
    run_alcaraz_ladder(parser.parse_args().out_dir)

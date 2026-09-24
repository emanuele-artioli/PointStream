"""Charged temporal object-video arm on the fixed Alcaraz background."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3  # noqa: F401  # load host C++ runtime before optional pose backend
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from experiments.headroom.ladder import resolved_tools
from experiments.modular.appearance_motion_probe import _box_payload
from experiments.modular.foreground_campaign import _row
from experiments.modular.foreground_part2_common import load_fixed_alcaraz
from experiments.modular.measured_ladder import _rgb_to_yuv420, _yuv420_to_rgb
from experiments.modular.object_foreground_campaign import _boxes, split_object_tracks
from src.components.codec.encode import BITSTREAM_SUFFIX, decode, encode
from src.components.codec.y4m import Y4M, read, write
from src.contracts.codecs import EncodeRequest, RateControl
from src.pipeline.residual.lossy import residual_clip_fraction

OUT = Path('/home/itec/emanuele/pointstream-data/outputs/modular/foreground-part2/roi-video')
N_FRAMES = 48
COLOR_QPS = (42, 50)
ALPHA_QPS = (32, 42)


def _build_object_inputs(source: np.ndarray, tracks: list[np.ndarray]) -> tuple[list[dict], float]:
    """Place each visible player in its own fixed, mask-gated RGB canvas."""
    if source.ndim != 4 or source.shape[0] != N_FRAMES or source.shape[-1] != 3:
        raise ValueError('source must be a 48-frame RGB clip')
    if len(tracks) != 2:
        raise ValueError('Alcaraz requires two independent object tracks')
    started = time.perf_counter()
    objects = []
    for object_index, track in enumerate(tracks):
        if track.shape != source.shape[:3] or track.dtype != np.bool_:
            raise ValueError('object track shape or dtype differs from source')
        boxes, presence, _ = _boxes(track)
        height = max(box[1]-box[0] for box in boxes)
        width = max(box[3]-box[2] for box in boxes)
        coded_h = max(64, height + (-height % 8))
        coded_w = max(64, width + (-width % 8))
        color = np.zeros((N_FRAMES, coded_h, coded_w, 3), dtype=np.uint8)
        alpha = np.zeros_like(color)
        for t in np.flatnonzero(presence):
            y1, y2, x1, x2 = boxes[int(t)]
            h, w = y2-y1, x2-x1
            roi_mask = track[t, y1:y2, x1:x2]
            roi_source = source[t, y1:y2, x1:x2]
            color[t, :h, :w][roi_mask] = roi_source[roi_mask]
            alpha[t, :h, :w][roi_mask] = 255
        bbox_wire = _box_payload(boxes)
        presence_wire = np.packbits(presence.astype(np.uint8), bitorder='little').tobytes()
        objects.append({'index': object_index, 'color_rgb': color, 'alpha_rgb': alpha,
                        'boxes': boxes, 'presence': presence, 'bbox_bytes': len(bbox_wire),
                        'presence_bytes': len(presence_wire), 'canvas': (coded_h, coded_w)})
    return objects, time.perf_counter()-started


def _av1_roundtrip(frames_rgb: np.ndarray, qp: int, wire_path: Path) -> dict:
    """Run preset-10 AV1 through the measured RGB/YUV420/RGB path."""
    frames = np.asarray(frames_rgb)
    if frames.ndim != 4 or frames.shape[0] != N_FRAMES or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError('AV1 object input must be a 48-frame uint8 RGB video')
    if frames.shape[1] < 64 or frames.shape[2] < 64 or frames.shape[1] % 8 or frames.shape[2] % 8:
        raise ValueError('AV1 object canvas must be at least 64 and divisible by 8')
    if qp not in (*COLOR_QPS, *ALPHA_QPS):
        raise ValueError('unsupported object video QP')
    wire_path.parent.mkdir(parents=True, exist_ok=True)
    request = EncodeRequest(codec_name='av1', rate_control=RateControl.QP, rate=qp, preset='10', pix_fmt='yuv420p')
    tools = resolved_tools('av1')
    with tempfile.TemporaryDirectory(prefix='ps_object_video_') as directory:
        root = Path(directory)
        prepared = time.perf_counter()
        luma, chroma = _rgb_to_yuv420(frames)
        source_path = root / 'input.y4m'
        write(source_path, Y4M(width=frames.shape[2], height=frames.shape[1], fps=25.0,
                               luma=luma, chroma=chroma))
        prepare_s = time.perf_counter()-prepared
        started = time.perf_counter()
        encode(source_path, wire_path, request, work_dir=root)
        encode_s = time.perf_counter()-started
        if not wire_path.is_file() or wire_path.stat().st_size <= 0:
            raise RuntimeError(f'empty AV1 video: {wire_path}')
        decoded_path = root / 'decoded.y4m'
        started = time.perf_counter()
        decode(wire_path, decoded_path, request)
        decoded = read(decoded_path)
        if decoded.chroma is None:
            raise RuntimeError('AV1 video decoded without chroma')
        decoded_rgb = _yuv420_to_rgb(decoded.luma, decoded.chroma)
        decode_s = time.perf_counter()-started
    if decoded_rgb.shape != frames.shape or decoded_rgb.dtype != np.uint8:
        raise RuntimeError(f'AV1 decode shape/dtype changed: {decoded_rgb.shape}, {decoded_rgb.dtype}')
    return {'pixels_rgb': decoded_rgb, 'bytes': wire_path.stat().st_size,
            'prepare_seconds': prepare_s,
            'encode_seconds': encode_s, 'decode_seconds': decode_s,
            'wire_path': str(wire_path), 'encoder_path': tools['encoder_path'],
            'encoder_version': tools['encoder_version'], 'decoder_path': tools['ffmpeg_path'],
            'decoder_version': tools['ffmpeg_version']}


def _composite(background_rgb: np.ndarray, objects: list[dict],
               color_decodes: list[np.ndarray], alpha_decodes: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Paste decoded object videos using only decoded alpha and wire motion."""
    if len(objects) != len(color_decodes) or len(objects) != len(alpha_decodes):
        raise ValueError('one color and alpha video required per object')
    n, height, width, channels = background_rgb.shape
    if n != N_FRAMES or channels != 3:
        raise ValueError('background must be 48 RGB frames')
    out = np.empty(background_rgb.shape, dtype=np.uint8)
    started = time.perf_counter()
    for t in range(n):
        out[t] = background_rgb[t]
        for obj, color, alpha in zip(objects, color_decodes, alpha_decodes, strict=True):
            if color.shape != alpha.shape or color.shape[0] != n or color.shape[-1] != 3:
                raise ValueError('decoded object videos disagree in shape')
            if not obj['presence'][t]:
                continue
            y1, y2, x1, x2 = obj['boxes'][t]
            h, w = y2-y1, x2-x1
            if h <= 0 or w <= 0 or y1 < 0 or x1 < 0 or y2 > height or x2 > width:
                raise ValueError('decoded bbox is outside background frame')
            if h > color.shape[1] or w > color.shape[2]:
                raise ValueError('decoded bbox exceeds object canvas')
            matte = np.mean(alpha[t, :h, :w].astype(np.uint16), axis=-1) >= 128
            target = out[t, y1:y2, x1:x2]
            target[matte] = color[t, :h, :w][matte]
    return out, time.perf_counter()-started


def _row_checked(*args: object, **kwargs: object) -> dict:
    row = _row(*args, **kwargs)
    if row['total_bytes'] != sum(int(row[key]) for key in ('B', 'F', 'M', 'R', 'H')):
        raise RuntimeError('component bytes do not sum to total')
    weighted = row['scores']['weighted']
    if weighted is None or not np.isfinite(weighted):
        raise RuntimeError('nonfinite weighted score on actual object-video row')
    return row


def run_roi_video(out_dir: Path = OUT) -> dict[str, object]:
    """Measure two separate temporal color and alpha bitstreams per QP pair."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed = load_fixed_alcaraz(out_dir.parent)
    source, mask = fixed.source_rgb, fixed.mask
    if source.shape[0] != N_FRAMES or mask.shape != source.shape[:3]:
        raise RuntimeError('fixed Alcaraz source/mask shape changed')
    started = time.perf_counter()
    tracks = split_object_tracks(mask)
    tracking_s = time.perf_counter()-started
    objects, prep_s = _build_object_inputs(source, tracks)
    B = int(fixed.background['total_bytes'])
    M = sum(obj['bbox_bytes'] + obj['presence_bytes'] for obj in objects)
    H_header = 1  # transmitted object count; AV1 streams contain canvas dimensions
    result = {'clip_id': 'alcaraz000', 'arm': 'separate_temporal_object_video', 'frames': N_FRAMES,
              'source_anchor': fixed.anchor, 'background': fixed.background,
              'offline_plate_seconds': fixed.plate_seconds,
              'tracking_seconds': tracking_s, 'object_video_prepare_seconds': prep_s,
              'object_count': len(objects), 'objects': [
                  {'index': obj['index'], 'visible_frames': int(obj['presence'].sum()),
                   'canvas': obj['canvas'], 'bbox_bytes': obj['bbox_bytes'],
                   'presence_bytes': obj['presence_bytes']} for obj in objects],
              'color_encodes': [], 'alpha_encodes': [], 'rows': [], 'completed': False}
    ledger = out_dir / 'alcaraz000-roi-video.json'

    def save() -> None:
        ledger.write_text(json.dumps(result, indent=2) + '\n')

    save()
    color: dict[tuple[int, int], dict] = {}
    alpha: dict[tuple[int, int], dict] = {}
    for obj in objects:
        j = int(obj['index'])
        for qp in COLOR_QPS:
            print(f'alcaraz000 object {j} color QP{qp}', flush=True)
            path = out_dir / 'wires' / f'object{j}-color-qp{qp}{BITSTREAM_SUFFIX["av1"]}'
            coded = _av1_roundtrip(obj['color_rgb'], qp, path)
            color[j, qp] = coded
            result['color_encodes'].append({k: v for k, v in coded.items() if k != 'pixels_rgb'} | {'object': j, 'qp': qp})
            save()
        for qp in ALPHA_QPS:
            print(f'alcaraz000 object {j} alpha QP{qp}', flush=True)
            path = out_dir / 'wires' / f'object{j}-alpha-qp{qp}{BITSTREAM_SUFFIX["av1"]}'
            coded = _av1_roundtrip(obj['alpha_rgb'], qp, path)
            alpha[j, qp] = coded
            result['alpha_encodes'].append({k: v for k, v in coded.items() if k != 'pixels_rgb'} | {'object': j, 'qp': qp})
            save()
        del obj['color_rgb'], obj['alpha_rgb']
    bg_encode_s = float(fixed.background.get('reencode_seconds', fixed.background['encode_seconds']))
    bg_decode_s = float(fixed.background.get('redecode_seconds', fixed.background['decode_seconds']))
    bg_render_s = float(fixed.background.get('rerender_seconds', fixed.background.get('render_seconds', 0.0)))
    for color_qp in COLOR_QPS:
        for alpha_qp in ALPHA_QPS:
            colors = [color[j, color_qp] for j in range(len(objects))]
            alphas = [alpha[j, alpha_qp] for j in range(len(objects))]
            F = sum(int(item['bytes']) for item in colors)
            H = H_header + sum(int(item['bytes']) for item in alphas)
            decoded, render_s = _composite(fixed.background_rgb, objects,
                                           [item['pixels_rgb'] for item in colors],
                                           [item['pixels_rgb'] for item in alphas])
            row = _row_checked(f'color_qp{color_qp}_alpha_qp{alpha_qp}', 'neither',
                               B=B, F=F, M=M, R=0, H=H, source=source, delivered=decoded, mask=mask,
                               encode_s=bg_encode_s+tracking_s+prep_s+sum(float(item['prepare_seconds'])+float(item['encode_seconds']) for item in colors+alphas),
                               decode_s=bg_decode_s+sum(float(item['decode_seconds']) for item in colors+alphas),
                               render_s=bg_render_s+render_s, plate_s=fixed.plate_seconds,
                               source_row=fixed.anchor)
            row['residual_clip_fraction'] = residual_clip_fraction(source.astype(np.int16)-decoded.astype(np.int16), mask)
            row['object_color_bytes'] = [int(item['bytes']) for item in colors]
            row['object_alpha_bytes'] = [int(item['bytes']) for item in alphas]
            if row['claimable']:
                sender = 'faster' if row['sender_seconds'] < fixed.anchor['encode_seconds'] else 'slower'
                client = 'faster' if row['client_seconds'] < fixed.anchor['decode_seconds'] else 'slower'
                row['latency_sentence'] = f'Sender {sender} and client {client} than the source at a weighted tie or win.'
            result['rows'].append(row)
            del decoded
            save()
            print(f"alcaraz000 temporal color {color_qp} alpha {alpha_qp}: {row['total_bytes']} B, weighted {row['scores']['weighted']:.3f}", flush=True)
    result['completed'] = True
    save()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUT)
    run_roi_video(parser.parse_args().out_dir)

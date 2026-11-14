"""CPU-only source/mask preparation for the prospective codec ladder.

Run through ps-fleet; no encoding, fitting, inference or quality selection.
Outputs source.yuv, masks.npz and a hash-bound preparation.json in published.tar.
The tennis path reproduces the historical K4 limited-range source conversion;
VISOR preserves decoded native 8-bit YUV planes and exact source/mask alignment.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import tarfile
import time
from fractions import Fraction
from typing import Any

import cv2
import numpy as np

from src.segmentation.masks import ClipMasks, decompress


def stream_sha(handle: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1 << 20), b''):
        digest.update(block)
    return digest.hexdigest()


def sha(path: Path) -> str:
    with path.open('rb') as handle:
        return stream_sha(handle)


def selected_masks(blob: bytes, *, kind: str, first: int, frames: int,
                   source_first: int, width: int = 1920, height: int = 1080) -> tuple[np.ndarray, dict[str, Any]]:
    if first < 0 or not 1 <= frames <= 64:
        raise ValueError('invalid selection')
    if kind == 'tennis':
        from experiments.tennis.k6b_rate_streams import decode_masks
        meta, masks = decode_masks(blob)
        if first + frames > len(masks):
            raise ValueError('source matte lacks selected frames')
        union = np.any(masks[first:first + frames], axis=1)
        output = np.stack([cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0 for m in union])
        return output, {'provenance': 'source_model_SAM_region_not_independent_ground_truth',
                        'decoder_metadata': meta, 'source_mask_first': first,
                        'source_mask_geometry': list(union.shape[1:]), 'resize': 'nearest',
                        'all_selected_frames_retained': True}
    if kind != 'visor':
        raise ValueError('unknown mask kind')
    clip = ClipMasks.from_doc(json.loads(decompress(blob)))
    if (clip.width, clip.height) != (width, height) or first + frames > len(clip):
        raise ValueError('VISOR mask geometry/coverage mismatch')
    indices = clip.meta.get('video_indices', [])
    if indices[first:first + frames] != list(range(source_first, source_first + frames)):
        raise ValueError('VISOR video indices differ from selected source')
    if clip.labelled is None or not all(clip.labelled[first:first + frames]):
        raise ValueError('VISOR selected masks are not all labelled')
    drift = clip.meta.get('drift', [])
    if drift[first:first + frames] != [0] * frames:
        raise ValueError('VISOR selected masks have unverified/nonzero drift')
    output = np.stack([clip.foreground(i) for i in range(first, first + frames)])
    return output, {'provenance': 'VISOR_dense_interpolated_not_all_human',
                    'source_mask_first': first, 'source_mask_geometry': [height, width],
                    'video_indices': indices[first:first + frames], 'drift': drift[first:first + frames],
                    'keyframes_selected': [k-first for k in clip.meta.get('keyframes', []) if first <= k < first + frames],
                    'all_selected_frames_retained': True,
                    'limitation': 'Dense masks may omit active objects; sparse human annotations are separate.'}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    from src.codecs import svtav1
    from src.segmentation import visor

    started = time.monotonic()
    if not 1 <= args.frames <= 64 or args.source_first < 0:
        raise ValueError('bounded source selection required')
    if sha(args.video) != args.video_sha256 or sha(args.masks) != args.masks_sha256:
        raise ValueError('source/mask input identity mismatch')
    if args.mask_member:
        with tarfile.open(args.masks) as archive:
            handle = archive.extractfile(args.mask_member)
            if handle is None:
                raise ValueError('mask member unavailable')
            blob = handle.read()
    else:
        blob = args.masks.read_bytes()
    if hashlib.sha256(blob).hexdigest() != args.mask_member_sha256:
        raise ValueError('mask member identity mismatch')
    native_info = visor.video_info(args.video)
    if ((native_info['width'], native_info['height']) != (1920, 1080)
            or Fraction(native_info['r_frame_rate']) != Fraction(args.fps)
            or Fraction(native_info['avg_frame_rate']) != Fraction(args.fps)):
        raise ValueError('native source geometry/cadence differs from preparation')
    masks, mask_record = selected_masks(blob, kind=args.kind, first=args.mask_first, frames=args.frames,
                                        source_first=args.source_first)
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    output = scratch / 'ladder-prepared'
    output.mkdir(exist_ok=False)
    source = output / 'source.yuv'
    source_record: dict[str, Any]
    if args.kind == 'tennis':
        from experiments.tennis.k4b_features import convert_command
        command = convert_command(args.ffmpeg, args.video, source, args.source_first, args.frames)
        command[command.index('-i'):command.index('-i')] = ['-threads', '1']
        command[command.index('-f'):command.index('-f')] = ['-threads', '1', '-filter_threads', '1']
        source_record = {'command': command, 'tool': svtav1.tool(args.ffmpeg),
                         'timing': svtav1.run(command, 600), 'full_range': False,
                         'policy': 'historical_K4_1920x1080_area_limited_yuv420p'}
    else:
        next_index = args.source_first
        formats, ranges, spaces = set(), set(), set()
        with source.open('xb') as handle:
            for index, frame in visor.decoded_frames(args.video, range(args.source_first, args.source_first + args.frames), threads=1):
                if index != next_index or frame.width != 1920 or frame.height != 1080 or frame.format.name not in {'yuv420p', 'yuvj420p'}:
                    raise ValueError('native VISOR decoded index/geometry/format mismatch')
                formats.add(frame.format.name)
                ranges.add(str(frame.color_range))
                spaces.add(str(frame.colorspace))
                handle.write(np.ascontiguousarray(frame.to_ndarray()).tobytes())
                next_index += 1
        if next_index != args.source_first + args.frames:
            raise ValueError('native source frames missing')
        full_range = formats == {'yuvj420p'} or ranges in ({'2'}, {'JPEG'})
        if not full_range or spaces not in ({'1'}, {'ITU709'}):
            raise ValueError('this VISOR preparation expects established full-range BT.709 source')
        source_record = {'decoder': 'PyAV src.segmentation.visor.decoded_frames', 'full_range': True,
                         'formats': sorted(formats), 'ranges': sorted(ranges), 'spaces': sorted(spaces),
                         'policy': 'unaltered_native_YUV420_planes', 'fps': args.fps}
    expected = args.frames * 1920 * 1080 * 3 // 2
    if source.stat().st_size != expected:
        raise ValueError('source byte count mismatch')
    np.savez_compressed(output / 'masks.npz', foreground=masks)
    r = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {'schema': 1, 'status': 'prepared_inputs_not_codec_evidence', 'kind': args.kind,
              'item_id': args.item_id, 'source_group': args.source_group,
              'source_first': args.source_first, 'mask_first': args.mask_first, 'frames': args.frames,
              'width': 1920, 'height': 1080, 'fps': args.fps,
              'native_source_info': native_info,
              'inputs': {'video_sha256': args.video_sha256, 'masks_sha256': args.masks_sha256,
                         'mask_member': args.mask_member, 'mask_member_sha256': args.mask_member_sha256},
              'outputs': {'source.yuv': {'sha256': sha(source), 'bytes': expected},
                          'masks.npz': {'sha256': sha(output / 'masks.npz'), 'bytes': (output / 'masks.npz').stat().st_size}},
              'mask_record': mask_record, 'source_record': source_record,
              'wall_seconds': time.monotonic() - started,
              'scoped_cpu_seconds': r.ru_utime + r.ru_stime + c.ru_utime + c.ru_stime,
              'no_inference_or_quality_selection': True}
    (output / 'preparation.json').write_text(json.dumps(result, indent=2))
    stage = Path(os.environ['PS_STAGE_DIR'])
    preparation_seconds = time.monotonic() - started
    publication_started = time.monotonic()
    with tarfile.open(stage / 'published.tar', 'x') as archive:
        for name in ('source.yuv', 'masks.npz', 'preparation.json'):
            archive.add(output / name, arcname=name)
    publication_seconds = time.monotonic() - publication_started
    total_seconds = time.monotonic() - started
    forecast = 1.5 * (preparation_seconds + publication_seconds) * (args.target_frames / args.frames) + 30
    result['cost_gate'] = {'measured_prepare_seconds': preparation_seconds,
                           'measured_publication_seconds': publication_seconds,
                           'measured_total_seconds': total_seconds, 'actual_frames': args.frames,
                           'target_frames': args.target_frames, 'safety_factor': 1.5, 'overhead_seconds': 30,
                           'forecast_full_seconds': forecast, 'full_cap_seconds': args.full_cap_seconds,
                           'passed': forecast <= args.full_cap_seconds if args.frames < args.target_frames else total_seconds <= args.full_cap_seconds}
    (stage / 'preparation-result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'frames': args.frames, 'kind': args.kind, 'status': result['status'],
                      'wall_seconds': result['wall_seconds'], 'output': str(stage / 'published.tar')}))
    return result


def validate() -> bool:
    stage = Path(os.environ['PS_STAGE_DIR'])
    with tarfile.open(stage / 'published.tar') as archive:
        members = archive.getmembers()
        if {m.name for m in members} != {'source.yuv', 'masks.npz', 'preparation.json'} or any(not m.isfile() for m in members):
            raise ValueError('unexpected publication members')
        reader = archive.extractfile('preparation.json')
        assert reader is not None
        result = json.load(reader)
        for name, expected in result['outputs'].items():
            reader = archive.extractfile(name)
            assert reader is not None
            digest = stream_sha(reader)
            if digest != expected['sha256'] or archive.getmember(name).size != expected['bytes']:
                raise ValueError('published payload identity mismatch')
        reader = archive.extractfile('masks.npz')
        assert reader is not None
        with np.load(io.BytesIO(reader.read()), allow_pickle=False) as mask:
            valid = mask.files == ['foreground'] and mask['foreground'].dtype == bool and mask['foreground'].shape == (result['frames'], 1080, 1920)
        if not valid or result['outputs']['source.yuv']['bytes'] != result['frames'] * 1920 * 1080 * 3 // 2:
            raise ValueError('published source/mask geometry mismatch')
    external = json.loads((stage / 'preparation-result.json').read_text())
    if any(external[k] != result[k] for k in result):
        raise ValueError('external source preparation record differs from publication')
    gate = external['cost_gate']
    forecast = 1.5 * (gate['measured_prepare_seconds'] + gate['measured_publication_seconds']) * (gate['target_frames'] / gate['actual_frames']) + 30
    cost_ok = (gate['actual_frames'] == result['frames'] and gate['target_frames'] == 64
               and abs(forecast - gate['forecast_full_seconds']) <= 1e-6
               and gate['passed'] is True
               and (forecast <= gate['full_cap_seconds'] if gate['actual_frames'] < 64 else gate['measured_total_seconds'] <= gate['full_cap_seconds']))
    Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps({'passed': cost_ok, 'checks': {
        'exact_members': True, 'payload_hashes_and_sizes': True, 'native_frame_bytes': True,
        'all_frame_boolean_masks': True, 'no_quality_or_model_inference': result['no_inference_or_quality_selection'],
        'measured_cost_and_publication_gate': cost_ok}}))
    return cost_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--kind', choices=['tennis', 'visor'])
    parser.add_argument('--video', type=Path)
    parser.add_argument('--video-sha256')
    parser.add_argument('--masks', type=Path)
    parser.add_argument('--masks-sha256')
    parser.add_argument('--mask-member', default='')
    parser.add_argument('--mask-member-sha256')
    parser.add_argument('--source-first', type=int)
    parser.add_argument('--mask-first', type=int)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--fps')
    parser.add_argument('--target-frames', type=int, default=64)
    parser.add_argument('--full-cap-seconds', type=float, default=600)
    parser.add_argument('--item-id')
    parser.add_argument('--source-group')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    args = parser.parse_args()
    if args.validate:
        return 0 if validate() else 1
    if any(getattr(args, name) is None for name in ['kind', 'video', 'video_sha256', 'masks', 'masks_sha256',
                                                   'mask_member_sha256', 'source_first', 'mask_first', 'frames', 'fps', 'item_id', 'source_group']):
        parser.error('all source and selection arguments required')
    if args.target_frames != 64 or not 0 < args.full_cap_seconds <= 600 or args.frames > args.target_frames:
        parser.error('target64 and bounded full cap <=600s required')
    result = prepare(args)
    return 0 if result['cost_gate']['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

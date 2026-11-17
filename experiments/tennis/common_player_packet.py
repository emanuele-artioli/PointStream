"""Versioned paid common-player conditioning; this is not a learned receiver.

The caller pins the complete profile and installed resources. Each contiguous run
has its own paid reference; gaps and cut boundaries reset temporal context.
Coordinates and output boxes use the declared canvas, not the coded IVF size.
The declared RGB resize is a receiver policy; this module does not decode video.
"""
from __future__ import annotations

import hashlib
import json
import struct
from typing import Any

import numpy as np

from experiments.tennis import k6b, k6b_rate_streams as rates, w8_payload as wire

PROFILE = 'common-tennis-player-v1'
RECIPE = {'profile': PROFILE, 'guidance': 3.5, 'steps': 30, 'context': [24, 4],
          'native_frames': 240, 'fps': [50, 1], 'seed_rule': 'sha256(unit)[:7]*1009+run',
          'crop_rule': [1.4, 90, 9], 'generation_size': [512, 784],
          'fallback': 'native-base', 'output': 'all-240', 'reference_codec': 'png',
          'base_to_canvas': 'RGB8-cv2-INTER_CUBIC'}
NATIVE_RECIPE = {'profile': 'common-tennis-native-only-v1', 'native_frames': 240,
                 'fps': [50, 1], 'output': 'all-240', 'fallback': 'native-base',
                 'base_to_canvas': 'RGB8-cv2-INTER_CUBIC'}
RESOURCES = frozenset({'vae_config', 'vae_weights', 'image_encoder_config', 'image_encoder_weights',
                       'denoising_unet', 'reference_unet', 'pose_guider', 'motion_module',
                       'common_lora', 'generator_code', 'generator_env', 'native_decoder'})


def _profile(profile: dict[str, Any]) -> None:
    if set(profile) != {'mode', 'recipe', 'resources', 'coded_size', 'canvas_size', 'cuts', 'runs', 'order'} or profile['mode'] not in ('players', 'native_only'):
        raise ValueError('unsupported profile')
    native = profile['mode'] == 'native_only'
    if wire.canonical(profile['recipe']) != wire.canonical(NATIVE_RECIPE if native else RECIPE):
        raise ValueError('unsupported recipe')
    if not native and ([k6b.GEN_W, k6b.GEN_H] != RECIPE['generation_size'] or
                       [k6b.CROP_MARGIN, k6b.SIZE_PERCENTILE, k6b.CENTRE_SMOOTH] != RECIPE['crop_rule'] or
                       k6b.k4.KP_CONF != 0.5):
        raise ValueError('installed geometry helper differs from v1 recipe')
    resources = profile['resources']
    if not isinstance(resources, dict) or set(resources) != ({'native_decoder'} if native else RESOURCES) or any(
        not isinstance(v, str) or wire.HASH.fullmatch(v) is None for v in resources.values()
    ):
        raise ValueError('invalid installed resource identities')
    for key in ('coded_size', 'canvas_size'):
        size = profile[key]
        if not isinstance(size, list) or len(size) != 2 or any(type(x) is not int or not 1 <= x <= 8192 for x in size):
            raise ValueError('invalid dimensions')
    cuts = profile['cuts']
    if not isinstance(cuts, list) or any(type(x) is not int or not 0 < x < 240 for x in cuts) or cuts != sorted(set(cuts)):
        raise ValueError('invalid cuts')
    if not isinstance(profile['runs'], list) or len(profile['runs']) > 480:
        raise ValueError('invalid runs')
    order = profile['order']
    if not isinstance(order, list) or len(order) > 16 or any(not wire._valid_name(x) for x in order) or len(set(order)) != len(order):
        raise ValueError('invalid compositing order')
    if (native and (order or profile['runs'])) or (not native and not order):
        raise ValueError('invalid mode coverage')


def verify_resources(profile: dict[str, Any], installed: dict[str, str]) -> None:
    """Compare locally measured hashes against the caller-pinned identities."""
    _profile(profile)
    if installed != profile['resources']:
        raise ValueError('installed resource identity mismatch')


def _ivf(blob: bytes, width: int, height: int) -> None:
    """Validate IVF structure, not AV1 decoding or decoded pixels."""
    if len(blob) < 32:
        raise ValueError('truncated IVF')
    magic, version, header, codec, w, h, rate, scale, count, reserved = struct.unpack_from('<4sHH4sHHIIII', blob)
    if (magic, version, header, codec, w, h, rate, scale, count, reserved) != (b'DKIF', 0, 32, b'AV01', width, height, 50, 1, 240, 0):
        raise ValueError('invalid native IVF header')
    pos = 32
    for tick in range(240):
        if pos + 12 > len(blob):
            raise ValueError('truncated IVF frame')
        size, timestamp = struct.unpack_from('<IQ', blob, pos)
        pos += 12
        if not size or size > len(blob) - pos or timestamp != tick:
            raise ValueError('invalid IVF frame/timestamp')
        pos += size
    if pos != len(blob):
        raise ValueError('trailing IVF data')


def _decode(profile: dict[str, Any], sections: dict[str, bytes]) -> dict[str, Any]:
    _profile(profile)
    if 'base' not in sections:
        raise ValueError('missing required section')
    _ivf(sections['base'], *profile['coded_size'])
    if profile['mode'] == 'native_only':
        if set(sections) != {'base'}:
            raise ValueError('native-only packet has extra sections')
        return {}
    if 'keypoints' not in sections:
        raise ValueError('missing required section')
    if sections['keypoints'][:4] != b'KPS2':
        raise ValueError('expected KPS2')
    meta, units = rates.decode_keypoints(sections['keypoints'])
    if meta != {'native_frames': 240, 'names': profile['order'], 'profile': rates.KPS2_PROFILE}:
        raise ValueError('keypoint profile mismatch')
    if [u['name'] for u in units] != profile['order']:
        raise ValueError('track order mismatch')
    output: dict[str, Any] = {}
    expected_names = {'base', 'keypoints'}
    for unit in units:
        name = unit['name']
        t = np.asarray(unit['t'])
        kp = np.asarray(unit['kp'])
        if t.ndim != 1 or not len(t) or kp.shape != (len(t), 17, 3) or not np.isfinite(kp).all() or np.any(np.diff(t) <= 0) or np.any(t < 0) or np.any(t >= 240):
            raise ValueError('invalid track')
        if np.any((kp[:, :, 2] != 0) & (kp[:, :, 2] != 1)):
            raise ValueError('invalid visibility')
        # KPS2 interpolates across contiguous samples, including cuts. Require
        # a gap so no coordinate can depend on a sample from another shot.
        if any(int(t[i]) in profile['cuts'] and t[i] == t[i-1] + 1 for i in range(1, len(t))):
            raise ValueError('KPS2 track crosses a cut without a gap')
        boundaries = [0] + [i for i in range(1, len(t)) if t[i] != t[i-1] + 1] + [len(t)]
        decoded_runs = []
        declared = [r for r in profile['runs'] if isinstance(r, dict) and r.get('unit') == name]
        nonempty = [(a, b) for a, b in zip(boundaries, boundaries[1:]) if a < b]
        if len(declared) != len(nonempty):
            raise ValueError('run coverage mismatch')
        for run_id, ((a, b), run) in enumerate(zip(nonempty, declared)):
            if set(run) != {'unit', 'start', 'end', 'reference'} or type(run['start']) is not int or type(run['end']) is not int or (run['start'], run['end']) != (int(t[a]), int(t[b-1]) + 1):
                raise ValueError('invalid run binding')
            refname = run['reference']
            if not wire._valid_name(refname) or refname in expected_names or refname not in sections:
                raise ValueError('invalid reference binding')
            expected_names.add(refname)
            rm, image = rates.decode_reference(sections[refname])
            if set(rm) != {'unit', 'reference_frame', 'reference_box', 'pixels', 'codec', 'quality', 'color'} or rm['unit'] != name or rm['codec'] != '.png' or rm['quality'] is not None or rm['color'] != 'RGB8' or rm['pixels'] != 'full_prepared_conditioning_crop' or type(rm['reference_frame']) is not int or not run['start'] <= rm['reference_frame'] < run['end']:
                raise ValueError('invalid paid reference')
            box = np.asarray(rm['reference_box'], dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0) or image.shape != (784, 512, 3):
                raise ValueError('invalid reference geometry')
            boxes = k6b.run_boxes(list(kp[a:b]))
            decoded_runs.append({'start': run['start'], 'end': run['end'], 't': t[a:b], 'kp': kp[a:b],
                                 'boxes': boxes, 'reference': image, 'reference_box': box,
                                 'seed': int(hashlib.sha256(name.encode()).hexdigest()[:7], 16) * 1009 + run_id})
        output[name] = decoded_runs
    if len(profile['runs']) != sum(len(v) for v in output.values()) or set(sections) != expected_names:
        raise ValueError('unbound run or section')
    return output


def coverage(decoded: dict[str, Any], order: list[str]) -> list[list[str]]:
    """Per-tick far-to-near active units; an empty list means base only."""
    if set(decoded) != set(order) or len(order) != len(set(order)):
        raise ValueError('coverage order mismatch')
    result: list[list[str]] = [[] for _ in range(240)]
    for name in order:
        for run in decoded[name]:
            for tick in run['t']:
                result[int(tick)].append(name)
    return result


def pose_for_tick(run: dict[str, Any], index: int) -> np.ndarray:
    """Render one conditioning pose on demand from paid geometry."""
    if type(index) is not int or not 0 <= index < len(run['t']):
        raise ValueError('tick outside run')
    return k6b.pose_map(k6b.to_crop(run['kp'][index], run['boxes'][index]))


def build(profile: dict[str, Any], base: bytes, keypoints: bytes | None, references: dict[str, bytes]) -> bytes:
    if {'base', 'keypoints'} & set(references):
        raise ValueError('reserved reference name')
    sections = {'base': base, **references}
    if keypoints is not None:
        sections['keypoints'] = keypoints
    _decode(profile, sections)
    entries = [('base', 'frame', base)]
    if keypoints is not None:
        entries.append(('keypoints', 'frame', keypoints))
    return wire.pack(profile, entries + [(name, 'reference', references[name]) for name in sorted(references)])


def _embedded_profile(blob: bytes) -> dict[str, Any]:
    """Read only the bounded canonical header of this profile's envelope."""
    if len(blob) < 9 or len(blob) > wire.MAX_BODY + wire.MAX_HEADER + 1024 or blob[:5] != wire.MAGIC:
        raise ValueError('invalid common packet envelope')
    size = struct.unpack_from('>I', blob, 5)[0]
    if size > wire.MAX_HEADER or 9 + size > len(blob):
        raise ValueError('invalid header length')
    raw = blob[9:9 + size]
    try:
        header = json.loads(raw)
        if (not isinstance(header, dict) or set(header) != {'version', 'profile', 'sections'} or
                type(header['version']) is not int or header['version'] != 1 or
                not isinstance(header['profile'], dict) or wire.canonical(header) != raw):
            raise ValueError('invalid canonical header')
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, RecursionError) as exc:
        raise ValueError('invalid canonical header') from exc
    return header['profile']


def unpack(blob: bytes, profile: dict[str, Any] | None = None) -> tuple[dict[str, bytes], dict[str, int], dict[str, Any]]:
    embedded = _embedded_profile(blob)
    if profile is not None and wire.canonical(profile) != wire.canonical(embedded):
        raise ValueError('profile identity mismatch')
    _profile(embedded)
    sections, accounting = wire.unpack(blob, embedded)
    decoded = _decode(embedded, sections)
    # Repacking verifies roles, canonical ordering, and the single supported envelope.
    refs = {n: v for n, v in sections.items() if n not in ('base', 'keypoints')}
    if build(embedded, sections['base'], sections.get('keypoints'), refs) != blob:
        raise ValueError('noncanonical sections or accounting roles')
    return sections, accounting, decoded


def inspect(blob: bytes, installed: dict[str, str] | None = None) -> dict[str, Any]:
    """Report paid bytes and coverage, with no claim of native or learned decode."""
    profile = _embedded_profile(blob)
    sections, accounting, decoded = unpack(blob, profile)
    if installed is not None:
        verify_resources(profile, installed)
    ticks = coverage(decoded, profile['order'])
    return {'profile': profile, 'component_bytes': {name: len(data) for name, data in sections.items()},
            'role_bytes': accounting, 'total_bytes': len(blob), 'required_resources': profile['resources'],
            'coverage': {'ticks': len(ticks), 'active_unit_ticks': sum(map(len, ticks)),
                         'base_only_ticks': sum(not tick for tick in ticks), 'units_by_tick': ticks},
            'native_validity': 'container_only', 'learned_receiver': False}

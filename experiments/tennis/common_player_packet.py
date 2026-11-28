"""Versioned paid common-player conditioning; this is not a learned receiver.

The caller pins the complete profile and installed resources. Each contiguous run
has its own paid reference; gaps and cut boundaries reset temporal context.
Coordinates and output boxes use the declared canvas, not the coded IVF size.
The declared RGB resize is a receiver policy; this module does not decode video.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import struct
import zlib
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import numpy as np

from experiments.tennis import k6b, k6b_rate_streams as rates, w8_payload as wire
from experiments.tennis import w2_pose_rate as w2

PROFILE = 'common-tennis-player-v1'
RECIPE = {'profile': PROFILE, 'guidance': 3.5, 'steps': 30, 'context': [24, 4],
          'native_frames': 240, 'fps': [50, 1], 'seed_rule': 'sha256(unit)[:7]*1009+run',
          'crop_rule': [1.4, 90, 9], 'generation_size': [512, 784],
          'fallback': 'native-base', 'output': 'all-240', 'reference_codec': 'png',
          'base_to_canvas': 'RGB8-cv2-INTER_CUBIC'}
# Separately identified native-cadence profile for 25 fps sources.
RECIPE_25 = {**RECIPE, 'profile': 'common-tennis-player-25fps-v1', 'fps': [25, 1]}
RECIPE_2997 = {**RECIPE, 'profile': 'common-tennis-player-30000-1001fps-v1', 'fps': [30000, 1001]}
RECIPE_60 = {**RECIPE, 'profile': 'common-tennis-player-60fps-v1', 'fps': [60, 1]}
# JPEG quality is sender metadata, not a proof of the quantization tables.
JPEG_RECIPES = {q: {**RECIPE, 'profile': 'common-tennis-player-jpeg-v1',
                    'reference_codec': 'jpeg', 'nominal_quality': q,
                    'reference_rgb_hash': 'sha256-numpy-dtype-shape-bytes-v1'}
                for q in (70, 40)}
LOSSLESS_RECIPE = {**RECIPE, 'profile': 'common-tennis-player-lossless-v1',
                   'reference_codec': 'png-or-webp-lossless',
                   'reference_rgb_hash': 'sha256-numpy-dtype-shape-bytes-v1'}
NATIVE_RECIPE = {'profile': 'common-tennis-native-only-v1', 'native_frames': 240,
                 'fps': [50, 1], 'output': 'all-240', 'fallback': 'native-base',
                 'base_to_canvas': 'RGB8-cv2-INTER_CUBIC'}
RESOURCES = frozenset({'vae_config', 'vae_weights', 'image_encoder_config', 'image_encoder_weights',
                       'denoising_unet', 'reference_unet', 'pose_guider', 'motion_module',
                       'common_lora', 'generator_code', 'generator_env', 'native_decoder'})


def _profile(profile: dict[str, Any]) -> None:
    keys = {'mode', 'recipe', 'resources', 'coded_size', 'canvas_size', 'cuts', 'runs', 'order'}
    if not keys <= set(profile) or profile['mode'] not in ('players', 'native_only'):
        raise ValueError('unsupported profile')
    native = profile['mode'] == 'native_only'
    recipe = profile['recipe']
    jpeg = not native and any(wire.canonical(recipe) == wire.canonical(r) for r in JPEG_RECIPES.values())
    lossless = not native and wire.canonical(recipe) == wire.canonical(LOSSLESS_RECIPE)
    bound_reference = jpeg or lossless
    allowed = (NATIVE_RECIPE,) if native else (RECIPE, RECIPE_25, RECIPE_2997, RECIPE_60)
    if not bound_reference and not any(wire.canonical(recipe) == wire.canonical(r) for r in allowed):
        raise ValueError('unsupported recipe')
    if set(profile) != keys | ({'reference_rgb_sha256'} if bound_reference else set()):
        raise ValueError('unsupported profile')
    if not native and ([k6b.GEN_W, k6b.GEN_H] != RECIPE['generation_size'] or
                       [k6b.CROP_MARGIN, k6b.SIZE_PERCENTILE, k6b.CENTRE_SMOOTH] != RECIPE['crop_rule'] or
                       k6b.k4.KP_CONF != 0.5):
        raise ValueError('installed geometry helper differs from v1 recipe')
    resources = profile['resources']
    if not isinstance(resources, dict) or set(resources) != ({'native_decoder'} if native else RESOURCES | ({'reference_decoder'} if bound_reference else set())) or any(
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
    if bound_reference:
        hashes = profile['reference_rgb_sha256']
        refs = [r.get('reference') for r in profile['runs'] if isinstance(r, dict)]
        if (not isinstance(hashes, dict) or len(refs) != len(profile['runs']) or
                any(not wire._valid_name(ref) for ref in refs) or len(set(refs)) != len(refs) or
                set(hashes) != set(refs) or any(not isinstance(v, str) or wire.HASH.fullmatch(v) is None
                                                  for v in hashes.values())):
            raise ValueError('invalid reference RGB hash bindings')
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


def _ivf(blob: bytes, width: int, height: int, fps: tuple[int, int] = (50, 1)) -> None:
    """Validate IVF structure, not AV1 decoding or decoded pixels."""
    if len(blob) < 32:
        raise ValueError('truncated IVF')
    magic, version, header, codec, w, h, rate, scale, count, reserved = struct.unpack_from('<4sHH4sHHIIII', blob)
    if (magic, version, header, codec, w, h, rate, scale, count, reserved) != (b'DKIF', 0, 32, b'AV01', width, height, *fps, 240, 0):
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


# These caps bound both compressed and expanded data before invoking decoders.
MAX_KPS = 4 * 1024 * 1024
MAX_REF = 8 * 1024 * 1024
MAX_TRANSPORT_META = 64 * 1024


def _installed_zstd(coded: bytes, raw_size: int, limit: int) -> bytes:
    """Use the JCV runtime's pinned libzstd 1.5.7 when its Python binding is absent."""
    if not 4 <= raw_size <= limit or not 0 < len(coded) <= limit:
        raise ValueError('invalid installed-decoder Zstandard size')
    try:
        lib = ctypes.CDLL(str(Path(sys.prefix) / 'lib/libzstd.so.1'))
    except OSError as exc:
        raise ValueError('no installed Zstandard decoder') from exc
    lib.ZSTD_versionNumber.restype = ctypes.c_uint
    if lib.ZSTD_versionNumber() != 10507:
        raise ValueError('unsupported installed Zstandard ABI')

    # Official ZSTD_FrameHeader ABI in the pinned environment's include/zstd.h.
    class Header(ctypes.Structure):
        _fields_ = [('content_size', ctypes.c_ulonglong), ('window', ctypes.c_ulonglong),
                    ('block_size', ctypes.c_uint), ('frame_type', ctypes.c_uint),
                    ('header_size', ctypes.c_uint), ('dictionary', ctypes.c_uint),
                    ('checksum', ctypes.c_uint), ('reserved1', ctypes.c_uint),
                    ('reserved2', ctypes.c_uint)]

    lib.ZSTD_getFrameHeader.argtypes = [ctypes.POINTER(Header), ctypes.c_char_p, ctypes.c_size_t]
    lib.ZSTD_getFrameHeader.restype = ctypes.c_size_t
    lib.ZSTD_findFrameCompressedSize.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.ZSTD_findFrameCompressedSize.restype = ctypes.c_size_t
    lib.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
    lib.ZSTD_decompress.restype = ctypes.c_size_t
    header = Header()
    if (lib.ZSTD_getFrameHeader(ctypes.byref(header), coded, len(coded)) != 0
            or header.frame_type != 0 or header.content_size != raw_size or header.window > limit
            or lib.ZSTD_findFrameCompressedSize(coded, len(coded)) != len(coded)):
        raise ValueError('invalid installed-decoder Zstandard frame, size, window or boundary')
    dest = ctypes.create_string_buffer(raw_size)
    if lib.ZSTD_decompress(dest, raw_size, coded, len(coded)) != raw_size:
        raise ValueError('invalid installed-decoder Zstandard payload')
    return dest.raw


def _bounded_transport(blob: bytes, magic: bytes, limit: int) -> tuple[dict[str, Any], bytes]:
    if len(blob) < rates.HEADER.size or len(blob) > rates.HEADER.size + limit:
        raise ValueError('invalid transport size')
    tag, raw_size, coded_size, digest = rates.HEADER.unpack_from(blob)
    if tag != magic or not 4 <= raw_size <= limit or coded_size != len(blob) - rates.HEADER.size:
        raise ValueError('invalid transport identity or size')
    coded = blob[rates.HEADER.size:]
    try:
        if coded.startswith(b'\x28\xb5\x2f\xfd'):
            try:
                import zstandard as zstd
            except ImportError:
                # One lossless 512x784 RGB8 PNG: one filter byte per raster row,
                # with bounded room for image-codec framing and REF1 metadata.
                if magic == b'REF1':
                    limit = min(limit, 784 * (512 * 3 + 1) + 2 * MAX_TRANSPORT_META + 4)
                if raw_size > limit:
                    raise ValueError('reference transport exceeds the bounded RGB raster')
                raw = _installed_zstd(coded, raw_size, limit)
            else:
                # Existing encoder writes a content size; unknown-size frames are unsupported.
                if zstd.frame_content_size(coded) != raw_size:
                    raise ValueError('invalid zstd content size')
                if zstd.get_frame_parameters(coded).window_size > limit:
                    raise ValueError('zstd window exceeds limit')
                try:
                    raw = zstd.ZstdDecompressor(max_window_size=limit // 1024).decompress(
                        coded, max_output_size=limit, allow_extra_data=False)
                except zstd.ZstdError as exc:
                    raise ValueError('invalid zstd stream') from exc
        else:
            dec = zlib.decompressobj()
            raw = dec.decompress(coded, raw_size + 1)
            if not dec.eof or dec.unused_data or dec.unconsumed_tail:
                raise ValueError('invalid zlib boundary')
    except (zlib.error, ImportError) as exc:
        raise ValueError('invalid or unavailable transport compression') from exc
    if len(raw) != raw_size or hashlib.sha256(raw).digest() != digest:
        raise ValueError('invalid transport content')
    length = struct.unpack_from('>I', raw)[0]
    if length > MAX_TRANSPORT_META or length > len(raw) - 4:
        raise ValueError('invalid transport metadata length')
    try:
        meta = json.loads(raw[4:4 + length])
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('invalid transport metadata') from exc
    if not isinstance(meta, dict):
        raise ValueError('invalid transport metadata')
    return meta, raw[4 + length:]


class _BoundedDecoder(w2.Decoder):
    """Limit the W2 Exp-Golomb unary prefix before it can grow without bound."""
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.eg_prefix = 0

    def bit(self, ctx: Any, value: int | None = None) -> int:
        if isinstance(ctx, tuple) and ctx and ctx[0] == 'egk':
            self.eg_prefix += 1
            if self.eg_prefix > 32:
                raise ValueError('oversized Exp-Golomb prefix')
        else:
            self.eg_prefix = 0
        return super().bit(ctx, value)


def _decode_window(raw: bytes, plans: list[list[tuple[int, int]]], pos: int) -> list[dict[str, Any]]:
    setting = w2.Setting(q=4, key=4, predictor='previous', interp='linear', vis='all')
    dec = _BoundedDecoder(raw[pos:])
    units = []
    for runs in plans:
        frames = []
        times: list[int] = []
        for start, length in runs:
            qk, vis = w2._code_run(dec, length, setting, None, None)
            xy = w2.interpolate(qk * 4.0, length, setting)
            frames.append(np.concatenate([xy, w2.expand_vis(vis, length, setting)[:, :, None].astype(np.float64)], 2))
            times.extend(range(start, start + length))
        units.append({'t': np.asarray(times, np.int64), 'kp': np.concatenate(frames)})
    return units


def _bounded_keypoints(blob: bytes, order: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta, raw = _bounded_transport(blob, b'KPS2', MAX_KPS)
    if meta != {'native_frames': 240, 'names': order, 'profile': rates.KPS2_PROFILE}:
        raise ValueError('keypoint profile mismatch')
    if len(raw) < 5 or raw[:4] != bytes([0xa7, 0, 4, 4]) or raw[4] != len(order):
        raise ValueError('invalid W2 coding profile')
    pos = 5
    def varint() -> int:
        nonlocal pos
        value = 0
        for shift in range(0, 35, 7):
            if pos >= len(raw):
                raise ValueError('truncated run plan')
            byte = raw[pos]
            pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError('oversized run varint')
    plans: list[list[tuple[int, int]]] = []
    for _ in order:
        runs: list[tuple[int, int]] = []
        count = varint()
        if not 1 <= count <= 240:
            raise ValueError('invalid run count')
        last_end = total = 0
        for _ in range(count):
            start, length = varint(), varint()
            if not 0 < length <= 240 or (runs and start <= last_end) or start + length > 240:
                raise ValueError('invalid run geometry')
            last_end = start + length
            runs.append((start, length))
            total += length
            if total > 240:
                raise ValueError('too many track frames')
        plans.append(runs)
    decoded = _decode_window(raw, plans, pos)
    return meta, [{'name': name, 't': u['t'], 'kp': u['kp']} for name, u in zip(order, decoded)]


def _jpeg_header(raw: bytes) -> None:
    """Accept one bounded baseline sequential scan before allocating pixels."""
    if not raw.startswith(b'\xff\xd8'):
        raise ValueError('invalid JPEG SOI')
    pos, sof, scan = 2, False, False
    while pos < len(raw):
        if raw[pos] != 0xff:
            raise ValueError('invalid JPEG marker')
        while pos < len(raw) and raw[pos] == 0xff:
            pos += 1
        if pos >= len(raw):
            raise ValueError('truncated JPEG marker')
        marker = raw[pos]
        pos += 1
        if marker == 0xd9:
            if not scan or pos != len(raw):
                raise ValueError('invalid JPEG end')
            return
        if scan:
            raise ValueError('unsupported JPEG marker after scan')
        if marker not in (0xc0, 0xc4, 0xdb, 0xdd, 0xda, 0xfe, *range(0xe0, 0xf0)):
            raise ValueError('unsupported JPEG marker')
        if pos + 2 > len(raw):
            raise ValueError('truncated JPEG length')
        size = int.from_bytes(raw[pos:pos + 2], 'big')
        if size < 2 or pos + size > len(raw):
            raise ValueError('invalid JPEG segment length')
        segment = raw[pos + 2:pos + size]
        pos += size
        if marker == 0xc0:
            if sof or scan or len(segment) != 15 or segment[:6] != b'\x08\x03\x10\x02\x00\x03':
                raise ValueError('invalid JPEG SOF dimensions or components')
            sof = True
        elif marker == 0xda:
            if not sof or scan or len(segment) != 10 or segment[0] != 3 or segment[-3:] != b'\x00\x3f\x00':
                raise ValueError('unsupported JPEG scan')
            scan = True
            # Entropy bytes permit stuffing and restart markers only. The next
            # marker must be EOI; multi-scan and trailing forms are unsupported.
            while pos < len(raw):
                if raw[pos] != 0xff:
                    pos += 1
                    continue
                if pos + 1 >= len(raw):
                    raise ValueError('truncated JPEG scan')
                if raw[pos + 1] == 0x00 or 0xd0 <= raw[pos + 1] <= 0xd7:
                    pos += 2
                    continue
                break
    raise ValueError('truncated JPEG')


def _webp_header(raw: bytes) -> None:
    """Accept one bounded, lossless RGB WebP image without optional chunks."""
    rates.validate_lossless_webp(raw, 512, 784)


def _bounded_reference(blob: bytes, jpeg: bool = False, decoder_hash: str = '',
                       lossless: bool = False) -> tuple[dict[str, Any], np.ndarray]:
    import cv2

    meta, raw = _bounded_transport(blob, b'REF1', MAX_REF)
    if jpeg:
        _jpeg_header(raw)
    elif lossless and meta.get('codec') == '.webp':
        _webp_header(raw)
    elif (len(raw) < 33 or raw[:8] != b'\x89PNG\r\n\x1a\n' or raw[8:12] != b'\x00\x00\x00\x0d' or
            raw[12:16] != b'IHDR' or struct.unpack_from('>II', raw, 16) != (512, 784) or
            raw[24:29] != b'\x08\x02\x00\x00\x00'):
        raise ValueError('invalid RGB8 reference PNG header')
    if jpeg or lossless:
        measured = hashlib.sha256(cv2.getBuildInformation().encode('utf-8')).hexdigest()
        if measured != decoder_hash:
            raise ValueError('reference decoder build identity mismatch')
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('invalid reference image')
    return meta, image[:, :, ::-1]


def _decode(profile: dict[str, Any], sections: dict[str, bytes]) -> dict[str, Any]:
    _profile(profile)
    if 'base' not in sections:
        raise ValueError('missing required section')
    size = profile['coded_size']
    fps = profile['recipe']['fps']
    _ivf(sections['base'], size[0], size[1], (fps[0], fps[1]))
    if profile['mode'] == 'native_only':
        if set(sections) != {'base'}:
            raise ValueError('native-only packet has extra sections')
        return {}
    if 'keypoints' not in sections:
        raise ValueError('missing required section')
    if sections['keypoints'][:4] != b'KPS2':
        raise ValueError('expected KPS2')
    meta, units = _bounded_keypoints(sections['keypoints'], profile['order'])
    if meta != {'native_frames': 240, 'names': profile['order'], 'profile': rates.KPS2_PROFILE}:
        raise ValueError('keypoint profile mismatch')
    if [u['name'] for u in units] != profile['order']:
        raise ValueError('track order mismatch')
    jpeg = profile['recipe'].get('profile') == 'common-tennis-player-jpeg-v1'
    lossless = profile['recipe'].get('profile') == 'common-tennis-player-lossless-v1'
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
            rm, image = _bounded_reference(sections[refname], jpeg, profile['resources'].get('reference_decoder', ''), lossless)
            accepted_codecs = ('.jpg',) if jpeg else (('.png', '.webp') if lossless else ('.png',))
            if set(rm) != {'unit', 'reference_frame', 'reference_box', 'pixels', 'codec', 'quality', 'color'} or rm['unit'] != name or rm['codec'] not in accepted_codecs or rm['quality'] != (profile['recipe']['nominal_quality'] if jpeg else None) or rm['color'] != 'RGB8' or rm['pixels'] != 'full_prepared_conditioning_crop' or type(rm['reference_frame']) is not int or not run['start'] <= rm['reference_frame'] < run['end']:
                raise ValueError('invalid paid reference')
            box = np.asarray(rm['reference_box'], dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0) or image.shape != (784, 512, 3):
                raise ValueError('invalid reference geometry')
            if jpeg or lossless:
                # Match the already frozen preparation manifest's array hash.
                image_hash = hashlib.sha256(f'{image.dtype.str}:{image.shape}:'.encode())
                image_hash.update(image.tobytes())
                if image_hash.hexdigest() != profile['reference_rgb_sha256'][refname]:
                    raise ValueError('reference RGB hash mismatch')
            boxes = k6b.run_boxes(list(kp[a:b]))
            decoded_runs.append({'start': run['start'], 'end': run['end'], 't': t[a:b], 'kp': kp[a:b],
                                 'boxes': boxes, 'reference': image, 'reference_box': box,
                                 'reference_frame': rm['reference_frame'],
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


def optimize_lossless_carrier(blob: bytes) -> bytes:
    """Convert a PNG carrier when an exactly equivalent complete packet is smaller."""
    import cv2

    profile = _embedded_profile(blob)
    sections, _, decoded = unpack(blob, profile)
    if profile['mode'] != 'players' or wire.canonical(profile['recipe']) != wire.canonical(RECIPE):
        raise ValueError('expected PNG player carrier')
    optimized = deepcopy(profile)
    optimized['recipe'] = deepcopy(LOSSLESS_RECIPE)
    optimized['resources']['reference_decoder'] = hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()
    optimized['reference_rgb_sha256'] = {}
    refs = {}
    run_indices: dict[str, int] = {}
    for run in profile['runs']:
        name = run['reference']
        unit = run['unit']
        index = run_indices.get(unit, 0)
        image = decoded[unit][index]['reference']
        run_indices[unit] = index + 1
        digest = hashlib.sha256(f'{image.dtype.str}:{image.shape}:'.encode())
        digest.update(image.tobytes())
        optimized['reference_rgb_sha256'][name] = digest.hexdigest()
        metadata, _ = rates.unseal(sections[name], b'REF1')
        clean = {k: v for k, v in metadata.items() if k not in ('codec', 'quality', 'color')}
        candidate = rates.encode_reference(image, clean, lossless_auto=True)
        refs[name] = min((sections[name], candidate), key=len)
    candidate = build(optimized, sections['base'], sections['keypoints'], refs)
    return candidate if len(candidate) < len(blob) else blob


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

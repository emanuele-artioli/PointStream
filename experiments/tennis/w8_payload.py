"""Bounded W8 wire primitives; payload construction is separate from evidence validation.

The decoder accepts a caller-pinned manifest, never paths or dynamically imported code.
Directory offsets are implicit and consecutive; the whole header is canonical JSON.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import math
import re
import struct
import zlib
from typing import Any

MAGIC = b'PSW8\x01'
WINDOW_MAGIC = b'PSW8\x02'
MAX_HEADER = 4 * 1024 * 1024
MAX_BODY = 512 * 1024 * 1024
MAX_SECTIONS = 10000
NAME = re.compile(r'[A-Za-z0-9_.-]{1,192}\Z')
HASH = re.compile(r'[0-9a-f]{64}\Z')

REQUIRED_MODELS = frozenset({'vae_config', 'vae_weights', 'image_encoder_config', 'image_encoder_weights',
                             'denoising_unet', 'reference_unet', 'pose_guider', 'motion_module', 'tennis_base',
                             'sam', 'generator_env', 'sam_env', 'native_env', 'generator_code', 'native_decoder'})
REQUIRED_RECIPE = frozenset({'profile', 'steps', 'guidance', 'context', 'seed_rule', 'native_frames', 'fps',
                             'saved_ticks', 'fallback', 'scheduler', 'rank', 'compositing_order', 'color_path'})


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def _valid_name(name: Any) -> bool:
    return isinstance(name, str) and NAME.fullmatch(name) is not None and name not in ('.', '..')


def pack(profile: dict[str, Any], sections: list[tuple[str, str, bytes]], *, compact_header: bool = False) -> bytes:
    """Encode named sections. Roles are frame, window or match accounting units."""
    if not sections or len(sections) > MAX_SECTIONS:
        raise ValueError('invalid section count')
    directory = []
    seen: set[str] = set()
    total = 0
    for name, role, body in sections:
        if not _valid_name(name) or name in seen or role not in ('frame', 'window', 'match', 'reference') or not isinstance(body, bytes):
            raise ValueError('invalid section')
        seen.add(name)
        total += len(body)
        if total > MAX_BODY:
            raise ValueError('payload too large')
        directory.append({'name': name, 'role': role, 'length': len(body), 'sha256': sha(body)})
    header = canonical({'version': 1, 'profile': profile, 'sections': directory})
    if len(header) > MAX_HEADER:
        raise ValueError('header too large')
    if compact_header:
        compressed = zlib.compress(header, level=9)
        return WINDOW_MAGIC + struct.pack('>II', len(compressed), len(header)) + compressed + b''.join(body for _, _, body in sections)
    return MAGIC + struct.pack('>I', len(header)) + header + b''.join(body for _, _, body in sections)


def unpack(blob: bytes, expected_profile: dict[str, Any]) -> tuple[dict[str, bytes], dict[str, int]]:
    """Reject alternate encodings, extra data, invalid identities and all size ambiguities."""
    if len(blob) < 9 or len(blob) > MAX_BODY + MAX_HEADER + 1024:
        raise ValueError('invalid envelope')
    if blob[:5] == MAGIC:
        n = struct.unpack('>I', blob[5:9])[0]
        if n > MAX_HEADER or 9 + n > len(blob):
            raise ValueError('invalid header length')
        raw = blob[9:9+n]
        header_end = 9+n
    elif blob[:5] == WINDOW_MAGIC:
        if len(blob) < 13:
            raise ValueError('truncated compressed header')
        n, raw_length = struct.unpack('>II', blob[5:13])
        if n > MAX_HEADER or raw_length > MAX_HEADER or 13+n > len(blob):
            raise ValueError('invalid compressed header length')
        try:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(blob[13:13+n], MAX_HEADER+1)
        except zlib.error as exc:
            raise ValueError('invalid compressed header') from exc
        if len(raw) != raw_length or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('invalid compressed header boundary')
        header_end = 13+n
    else:
        raise ValueError('invalid envelope version')
    try:
        header = json.loads(raw)
        if canonical(header) != raw or set(header) != {'version', 'profile', 'sections'}:
            raise ValueError('noncanonical header')
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, AttributeError, RecursionError) as exc:
        raise ValueError('invalid header') from exc
    if type(header['version']) is not int or header['version'] != 1 or canonical(header['profile']) != canonical(expected_profile):
        raise ValueError('profile identity mismatch')
    entries = header['sections']
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_SECTIONS:
        raise ValueError('invalid directory')
    result: dict[str, bytes] = {}
    accounting = {'frame': 0, 'window': header_end, 'match': 0, 'reference': 0}
    pos = header_end
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'name', 'role', 'length', 'sha256'}:
            raise ValueError('invalid entry')
        name, role, size, digest = entry['name'], entry['role'], entry['length'], entry['sha256']
        if (not _valid_name(name) or name in result or role not in ('frame', 'window', 'match', 'reference')
                or type(size) is not int or size < 0 or size > MAX_BODY or not isinstance(digest, str)
                or HASH.fullmatch(digest) is None or size > len(blob)-pos):
            raise ValueError('invalid section metadata')
        body = blob[pos:pos+size]
        if sha(body) != digest:
            raise ValueError('section checksum mismatch')
        result[name] = body
        accounting[role] += size
        pos += size
    if pos != len(blob) or sum(accounting.values()) != len(blob):
        raise ValueError('unaccounted bytes')
    return result, accounting


def decompress_exact(blob: bytes, length: int) -> bytes:
    """Decode one LZMA stream with an explicit output cap and no trailing stream."""
    if type(length) is not int or not 0 <= length <= MAX_BODY:
        raise ValueError('invalid raw length')
    try:
        decoder = lzma.LZMADecompressor(memlimit=128 * 1024 * 1024)
        raw = decoder.decompress(blob, max_length=length + 1)
    except lzma.LZMAError as exc:
        raise ValueError('invalid compressed body') from exc
    if len(raw) != length or not decoder.eof or decoder.unused_data:
        raise ValueError('compressed body length or boundary mismatch')
    return raw


def quantized_tensor(tensor: Any, bits: int) -> tuple[dict[str, Any], bytes, bytes]:
    """Serialize precisely the existing k6b_fit quantizer's symbols and scales."""
    from experiments.tennis.k6b_fit import QUANT_MIN_VALUES, quantize_channels

    if bits != 4 or not tensor.is_floating_point() or not bool(tensor.isfinite().all()):
        raise ValueError('only finite frozen 4-bit tensors are supported')
    t = tensor.detach().cpu().float().contiguous()
    shape = list(t.shape)
    if t.numel() < QUANT_MIN_VALUES or t.dim() == 0:
        symbols = t.half().numpy().tobytes()
        scales = b''
        kind = 'fp16'
    else:
        import numpy as np

        q, scale, rebuilt = quantize_channels(t, bits)
        # The frozen float quantizer preserves negative zero; int8 alone loses it.
        signs = ((rebuilt == 0) & rebuilt.signbit()).reshape(-1).numpy()
        symbols = q.contiguous().numpy().tobytes() + np.packbits(signs, bitorder='little').tobytes()
        scales, kind = scale.contiguous().numpy().tobytes(), 'int8-channel'
    if not bool(t.half().isfinite().all()):
        raise ValueError('fp16 overflow')
    return {'shape': shape, 'kind': kind, 'bits': bits, 'symbols_bytes': len(symbols), 'scales_bytes': len(scales)}, lzma.compress(symbols, preset=6), lzma.compress(scales, preset=6) if scales else b''


def restore_tensor(meta: dict[str, Any], symbols: bytes, scales: bytes) -> Any:
    """Rebuild float32 using the generator's arithmetic; caller applies adapter cast or guider base."""
    import numpy as np
    import torch

    if set(meta) != {'shape', 'kind', 'bits', 'symbols_bytes', 'scales_bytes'} or meta['bits'] != 4:
        raise ValueError('invalid tensor metadata')
    shape = meta['shape']
    if not isinstance(shape, list) or len(shape) > 8 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError('invalid tensor shape')
    count = math.prod(shape)
    if count > MAX_BODY // 2:
        raise ValueError('tensor too large')
    kind = meta['kind']
    expected = (count * 2, 0) if kind == 'fp16' else (count + (count + 7) // 8, shape[0] * 2) if kind == 'int8-channel' and shape else (-1, -1)
    if (meta['symbols_bytes'], meta['scales_bytes']) != expected:
        raise ValueError('tensor size mismatch')
    raw = decompress_exact(symbols, expected[0])
    if kind == 'fp16':
        if scales:
            raise ValueError('unexpected scales')
        out = torch.from_numpy(np.frombuffer(raw, dtype='<f2').copy()).float().reshape(shape)
    else:
        scale = torch.from_numpy(np.frombuffer(decompress_exact(scales, expected[1]), dtype='<f2').copy()).float()
        q = torch.from_numpy(np.frombuffer(raw[:count], dtype='i1').copy()).reshape(shape[0], -1).float()
        sign_bits = np.unpackbits(np.frombuffer(raw[count:], dtype='u1'), bitorder='little')
        if bool(sign_bits[count:].any()):
            raise ValueError('nonzero sign padding')
        signs = torch.from_numpy(sign_bits[:count].copy()).bool().reshape(shape)
        if not bool(((q >= -7) & (q <= 7)).all()) or not bool(scale.isfinite().all()) or bool((scale < 0).any()):
            raise ValueError('invalid quantized values')
        out = (q * scale[:, None]).reshape(shape)
        if bool((signs & (out != 0)).any()):
            raise ValueError('sign bitmap on nonzero value')
        out = torch.where(signs, -torch.zeros_like(out), out)
    if not bool(out.isfinite().all()):
        raise ValueError('nonfinite reconstructed tensor')
    return out


def encode_checkpoint(state: dict[str, Any], guider_base: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[str, str, bytes]]]:
    """Serialize all frozen fitted adapter and guider tensors, never the installed base."""
    import torch

    parts = ('reference_unet', 'denoising_unet', 'pose_guider')
    if any(not isinstance(state.get(p), dict) or not state[p] for p in parts):
        raise ValueError('checkpoint is missing a required part')
    if set(state['pose_guider']) != set(guider_base):
        raise ValueError('guider/base tensor names differ')
    directory: list[dict[str, Any]] = []
    sections: list[tuple[str, str, bytes]] = []
    for part in parts:
        for name in sorted(state[part]):
            tensor = state[part][name]
            if not _valid_name(name) or not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                raise ValueError('unsupported checkpoint tensor')
            value = tensor.detach().cpu().float()
            if part == 'pose_guider':
                base = guider_base[name]
                if not isinstance(base, torch.Tensor) or not base.is_floating_point() or base.shape != tensor.shape:
                    raise ValueError('wrong installed guider tensor')
                value = value - base.detach().cpu().float()
            meta, symbols, scales = quantized_tensor(value, 4)
            index = len(directory)
            symbol_name = f't{index:06d}.symbols'
            scale_name = f't{index:06d}.scales' if scales else None
            directory.append({'part': part, 'name': name, 'metadata': meta, 'symbols': symbol_name, 'scales': scale_name})
            sections.append((symbol_name, 'match', symbols))
            if scale_name is not None:
                sections.append((scale_name, 'match', scales))
    return directory, sections


def decode_checkpoint(directory: list[dict[str, Any]], sections: dict[str, bytes], guider_base: dict[str, Any], rank: int) -> dict[str, Any]:
    """Reconstruct decoder state with exact generator dtype/arithmetic and schema closure."""
    import torch

    if type(rank) is not int or rank <= 0 or rank > 1024:
        raise ValueError('invalid rank')
    state: dict[str, Any] = {'rank': rank, 'reference_unet': {}, 'denoising_unet': {}, 'pose_guider': {}}
    for entry in directory:
        part, name = entry['part'], entry['name']
        if part not in ('reference_unet', 'denoising_unet', 'pose_guider') or name in state[part]:
            raise ValueError('invalid checkpoint directory')
        scales_name = entry['scales']
        value = restore_tensor(entry['metadata'], sections[entry['symbols']], sections[scales_name] if scales_name else b'')
        if part == 'pose_guider':
            base = guider_base.get(name)
            if not isinstance(base, torch.Tensor) or not base.is_floating_point() or base.shape != value.shape:
                raise ValueError('guider tensor does not match installed model')
            value = base.detach().cpu().float() + value
        else:
            value = value.half()
        state[part][name] = value
    if any(not state[p] for p in ('reference_unet', 'denoising_unet', 'pose_guider')) or set(state['pose_guider']) != set(guider_base):
        raise ValueError('checkpoint directory incomplete')
    return state


def compare_checkpoint(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, int]:
    """Validator-only byte equality against frozen generator quantize_adapters output."""
    counts = {}
    for part in ('reference_unet', 'denoising_unet', 'pose_guider'):
        if set(actual[part]) != set(expected[part]):
            raise ValueError(f'{part}: name mismatch')
        for name in actual[part]:
            a, b = actual[part][name], expected[part][name]
            if a.shape != b.shape or a.dtype != b.dtype or a.contiguous().numpy().tobytes() != b.contiguous().numpy().tobytes():
                raise ValueError(f'{part}.{name}: decoded bytes differ')
        counts[part] = len(actual[part])
    return counts


def pack_match(state: dict[str, Any], guider_base: dict[str, Any], guider_sha256: str) -> bytes:
    """Reusable match object; its tensor directory and framing are paid once per match."""
    if not isinstance(guider_sha256, str) or HASH.fullmatch(guider_sha256) is None:
        raise ValueError('installed guider identity required')
    rank = state.get('rank')
    if type(rank) is not int or not 0 < rank <= 1024:
        raise ValueError('checkpoint rank required')
    directory, sections = encode_checkpoint(state, guider_base)
    profile = {'kind': 'w8-match-v1', 'rank': rank, 'guider_sha256': guider_sha256, 'tensors': directory}
    return pack(profile, sections)


def unpack_match(blob: bytes, expected_sha256: str, guider_sha256: str, guider_base: dict[str, Any]) -> dict[str, Any]:
    """Decode a hash-pinned match object with its embedded authenticated directory."""
    if sha(blob) != expected_sha256 or len(blob) < 9 or blob[:5] != MAGIC:
        raise ValueError('match identity mismatch')
    length = struct.unpack('>I', blob[5:9])[0]
    if length > MAX_HEADER or 9 + length > len(blob):
        raise ValueError('invalid match header')
    try:
        header = json.loads(blob[9:9+length])
        profile = header['profile']
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid match profile') from exc
    if (not isinstance(profile, dict) or set(profile) != {'kind', 'rank', 'guider_sha256', 'tensors'}
            or profile['kind'] != 'w8-match-v1' or profile['guider_sha256'] != guider_sha256
            or not isinstance(profile['tensors'], list) or not profile['tensors']):
        raise ValueError('match profile mismatch')
    sections, _ = unpack(blob, profile)
    names: list[str] = []
    for entry in profile['tensors']:
        if not isinstance(entry, dict) or set(entry) != {'part', 'name', 'metadata', 'symbols', 'scales'}:
            raise ValueError('invalid tensor directory')
        if not _valid_name(entry['name']) or not _valid_name(entry['symbols']):
            raise ValueError('invalid tensor names')
        names.append(entry['symbols'])
        if entry['scales'] is not None:
            if not _valid_name(entry['scales']):
                raise ValueError('invalid tensor scales')
            names.append(entry['scales'])
    if len(set(names)) != len(names) or set(names) != set(sections):
        raise ValueError('match sections mismatch')
    return decode_checkpoint(profile['tensors'], sections, guider_base, profile['rank'])


def unpack_window(blob: bytes, profile: dict[str, Any]) -> tuple[dict[str, bytes], dict[str, int]]:
    """Validate outer window and derive costs from section identities, not claimed roles."""
    keys = {'window', 'models', 'recipe', 'units', 'match', 'ivf', 'kps', 'judged'}
    if set(profile) != keys or not isinstance(profile['window'], str) or not profile['window']:
        raise ValueError('invalid window profile')
    models, recipe, units, match = profile['models'], profile['recipe'], profile['units'], profile['match']
    if (not isinstance(models, dict) or not models or any(not _valid_name(k) or not isinstance(v, str)
            or HASH.fullmatch(v) is None for k, v in models.items()) or not REQUIRED_MODELS <= set(models)
            or not isinstance(recipe, dict) or not REQUIRED_RECIPE <= set(recipe)):
        raise ValueError('model and recipe identities required')
    if (recipe['profile'] != 'match5' or recipe['steps'] != 30 or recipe['guidance'] != 2
            or recipe['context'] != [24, 4] or recipe['native_frames'] != 240
            or recipe['fps'] != [50, 1] or recipe['saved_ticks'] != list(range(0, 240, 3))
            or recipe['seed_rule'] != 'sha256(unit_name)[:7] then seed*1009+run' or recipe['rank'] != 4):
        raise ValueError('wrong frozen generator profile')
    if (not isinstance(match, dict) or set(match) != {'section', 'sha256'} or not _valid_name(match['section'])
            or not isinstance(match['sha256'], str) or HASH.fullmatch(match['sha256']) is None):
        raise ValueError('match identity required')
    if not isinstance(units, list) or len(units) != 2:
        raise ValueError('two unit bindings required')
    refs = []
    unit_ids = set()
    for unit in units:
        if not isinstance(unit, dict) or set(unit) != {'id', 'reference', 'reference_tick', 'box'}:
            raise ValueError('invalid unit binding')
        if (not _valid_name(unit['id']) or unit['id'] in unit_ids or not _valid_name(unit['reference'])
                or type(unit['reference_tick']) is not int or unit['reference_tick'] < 0
                or not isinstance(unit['box'], list) or len(unit['box']) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in unit['box'])):
            raise ValueError('invalid unit data')
        unit_ids.add(unit['id'])
        refs.append(unit['reference'])
    if not all(_valid_name(profile[k]) for k in ('ivf', 'kps', 'judged')):
        raise ValueError('invalid stream name')
    sections, _ = unpack(blob, profile)
    names = [profile['ivf'], profile['kps'], profile['judged'], *refs, match['section']]
    if (len(set(names)) != len(names) or set(names) != set(sections)
            or sha(sections[match['section']]) != match['sha256'] or len(sections[profile['judged']]) != 30):
        raise ValueError('window section closure or match identity mismatch')
    frame = len(sections[profile['ivf']]) + len(sections[profile['kps']])
    reference = sum(len(sections[n]) for n in refs)
    match_bytes = len(sections[match['section']])
    return sections, {'frame': frame, 'reference': reference, 'match': match_bytes,
                      'window': len(blob) - frame - reference - match_bytes}


def cache_identity(profile: dict[str, Any], sections: dict[str, bytes]) -> dict[str, Any]:
    """Object identities required by the conditional warm/reference reuse contract."""
    return {'match_sha256': sha(sections[profile['match']['section']]),
            'reference_sha256_by_unit': {u['id']: sha(sections[u['reference']]) for u in profile['units']}}


def assemble_window(profile: dict[str, Any], ivf: bytes, keypoints: bytes, judged: bytes,
                    references: dict[str, bytes], match: bytes) -> bytes:
    """Assemble a complete window; caller supplies the frozen, independently pinned profile."""
    if set(references) != {unit['id'] for unit in profile['units']}:
        raise ValueError('reference units mismatch')
    if sha(match) != profile['match']['sha256']:
        raise ValueError('match hash mismatch')
    sections = [(profile['ivf'], 'frame', ivf), (profile['kps'], 'frame', keypoints),
                (profile['judged'], 'window', judged)]
    sections.extend((unit['reference'], 'reference', references[unit['id']]) for unit in profile['units'])
    sections.append((profile['match']['section'], 'match', match))
    blob = pack(profile, sections, compact_header=True)
    unpack_window(blob, profile)
    return blob

"""PTP1 tracked 32px pixel patches over native AV1, decoded as RGB24.

Input is explicit raw 8-bit planar YUV420. Placements, container overhead and
both complete IVF streams are serialized; decoder inputs are only the packet,
the declared installed dav1d and NumPy. Placement hints are encoder input, not
an included tracking model. This profile does not establish OS source denial.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from src.codecs import svtav1
from src.pointstream import codec

MAGIC = b'PTP1'
HEADER = struct.Struct('>4sIII')
MAX_METADATA = 2 * 1024 * 1024
FIELDS = {'version', 'profile', 'native', 'numpy_version', 'positions', 'tile_size', 'inner_box'}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate(meta: dict[str, Any], base: bytes, patch: bytes) -> None:
    if (set(meta) != FIELDS or type(meta['version']) is not int or meta['version'] != 1
            or meta['profile'] != 'tracked-pixel-v1' or type(meta['tile_size']) is not int or meta['tile_size'] != 64
            or not isinstance(meta['inner_box'], list) or any(type(x) is not int for x in meta['inner_box'])
            or meta['inner_box'] != [16, 16, 32, 32]):
        raise ValueError('unsupported tracked patch metadata')
    native = meta['native']
    if not isinstance(native, dict):
        raise ValueError('invalid native metadata')
    codec.validate(native)
    if native['width'] * native['height'] * native['frames'] * 3 > codec.MAX_DECODED:
        raise ValueError('decoded RGB exceeds 512 MiB')
    if (not isinstance(meta['numpy_version'], str) or not 1 <= len(meta['numpy_version']) <= 128
            or not meta['numpy_version'].isascii()):
        raise ValueError('invalid NumPy identity')
    coords = meta['positions']
    if not isinstance(coords, list) or len(coords) != native['frames']:
        raise ValueError('placement count differs from frame count')
    for item in coords:
        if item is None:
            continue
        if (not isinstance(item, list) or len(item) != 2 or any(type(x) is not int for x in item)
                or item[0] < 0 or item[1] < 0 or item[0] % 2 or item[1] % 2
                or item[0] + 32 > native['width'] or item[1] + 32 > native['height']):
            raise ValueError('invalid even 32px placement')
    codec.check_native(native, base)
    if any(item is not None for item in coords):
        codec.check_native({**native, 'width': 64, 'height': 64}, patch)
    elif patch:
        raise ValueError('unexpected patch for all-absent placement')


def pack(meta: dict[str, Any], base: bytes, patch: bytes = b'') -> bytes:
    validate(meta, base, patch)
    header = json.dumps(meta, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(header) > MAX_METADATA:
        raise ValueError('metadata too large')
    body = HEADER.pack(MAGIC, len(header), len(base), len(patch)) + header + base + patch
    if len(body) + 32 > codec.MAX_PACKET:
        raise ValueError('packet too large')
    return body + hashlib.sha256(body).digest()


def unpack(data: bytes) -> tuple[dict[str, Any], bytes, bytes]:
    if not HEADER.size + 32 < len(data) <= codec.MAX_PACKET:
        raise ValueError('invalid packet length')
    magic, size, base_size, patch_size = HEADER.unpack_from(data)
    if (magic != MAGIC or not 1 <= size <= MAX_METADATA
            or HEADER.size + size + base_size + patch_size + 32 != len(data)):
        raise ValueError('invalid packet header')
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError('packet checksum mismatch')
    try:
        meta = json.loads(data[HEADER.size:HEADER.size + size])
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('invalid packet metadata') from exc
    if not isinstance(meta, dict):
        raise ValueError('invalid packet metadata')
    offset = HEADER.size + size
    base, patch = data[offset:offset + base_size], data[offset + base_size:-32]
    if pack(meta, base, patch) != data:
        raise ValueError('noncanonical packet')
    return meta, base, patch


def inspect(data: bytes) -> dict[str, Any]:
    meta, base, patch = unpack(data)
    native = meta['native']
    duration = native['frames'] / float(Fraction(*native['fps']))
    return {'metadata': meta, 'packet_bytes': len(data), 'native_base_bytes': len(base),
            'patch_bytes': len(patch), 'placement_and_framing_bytes': len(data) - len(base) - len(patch),
            'duration_seconds': duration, 'wire_kbps': len(data) * 8 / duration / 1000,
            'packet_sha256': digest(data), 'output_format': 'rgb24',
            'resources': {'decoder_sha256': native['decoder_sha256'], 'numpy_version': meta['numpy_version']}}


def origin(position: list[int] | None, width: int, height: int) -> list[int] | None:
    if position is None:
        return None
    if (not isinstance(position, list) or len(position) != 2 or any(type(x) is not int for x in position)
            or width < 32 or height < 32 or not (0 <= position[0] < width and 0 <= position[1] < height)):
        raise ValueError('invalid placement hint')
    return [min(max(2 * round((position[0] - 16) / 2), 0), width - 32),
            min(max(2 * round((position[1] - 16) / 2), 0), height - 32)]


def yuv_to_rgb(frame: np.ndarray, width: int, height: int, full_range: bool) -> np.ndarray:
    """BT.709 nearest chroma, float32 and round; limited matches tennis windows."""
    from src.codecs.quality import KB, KR

    y = frame[:height].astype(np.float32)
    chroma = frame[height:].reshape(2, height // 2, width // 2).astype(np.float32)
    if full_range:
        y /= 255.0
        u = (np.repeat(np.repeat(chroma[0], 2, 0), 2, 1) - 128.0) / 255.0
        v = (np.repeat(np.repeat(chroma[1], 2, 0), 2, 1) - 128.0) / 255.0
    else:
        y = (y - 16.0) / 219.0
        u = (np.repeat(np.repeat(chroma[0], 2, 0), 2, 1) - 128.0) / 224.0
        v = (np.repeat(np.repeat(chroma[1], 2, 0), 2, 1) - 128.0) / 224.0
    r = y + 2.0 * (1.0 - KR) * v
    b = y + 2.0 * (1.0 - KB) * u
    g = (y - KR * r - KB * b) / (1.0 - KR - KB)
    return np.clip(np.round(np.stack([r, g, b], -1) * 255.0), 0, 255).astype(np.uint8)


def decode_native(native: bytes, output: Path, meta: dict[str, Any], decoder: str) -> dict[str, Any]:
    """Decode to Y4M; verify actual geometry, chroma, cadence and every frame."""
    source, y4m = output.with_suffix('.ivf'), output.with_suffix('.y4m')
    source.write_bytes(native)
    command = [decoder, '-i', str(source), '-o', str(y4m), '--muxer', 'yuv4mpeg2', '--threads', '1',
               '--sizelimit', str(meta['width'] * meta['height']), '--quiet']
    timing = svtav1.run(command, 600)
    expected = meta['width'] * meta['height'] * 3 // 2
    with y4m.open('rb') as reader, output.open('xb') as writer:
        header = reader.readline(4096).decode('ascii').split()
        required = ['YUV4MPEG2', f'W{meta["width"]}', f'H{meta["height"]}', f'F{meta["fps"][0]}:{meta["fps"][1]}']
        if not all(token in header for token in required) or not any(t in header for t in ('C420', 'C420jpeg', 'C420mpeg2')):
            raise ValueError('decoded geometry/chroma/cadence mismatch')
        for _ in range(meta['frames']):
            if reader.readline(4096).strip() != b'FRAME':
                raise ValueError('native frame count/format mismatch')
            frame = reader.read(expected)
            if len(frame) != expected:
                raise ValueError('truncated decoded frame')
            writer.write(frame)
        if reader.read(1):
            raise ValueError('unexpected extra decoded frames')
    with output.open('rb') as handle:
        sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'command': command, 'timing': timing, 'y4m_header': header, 'decoded_yuv_sha256': sha}


def _render(data: bytes, *, scratch: Path, decoder: str) -> tuple[bytes, dict[str, Any]]:
    meta, base, patch = unpack(data)
    native = meta['native']
    identity = svtav1.tool(decoder)
    if identity['sha256'] != native['decoder_sha256'] or np.__version__ != meta['numpy_version']:
        raise ValueError('installed decoder or NumPy identity mismatch')
    with tempfile.TemporaryDirectory(prefix='ptp-receiver-', dir=scratch) as temporary:
        root = Path(temporary)
        base_path, patch_path = root / 'base.yuv', root / 'patch.yuv'
        receipts = [decode_native(base, base_path, native, decoder)]
        if patch:
            receipts.append(decode_native(patch, patch_path, {**native, 'width': 64, 'height': 64}, decoder))
        size = native['width'] * native['height'] * 3 // 2
        result = bytearray()
        frame_hashes = []
        with base_path.open('rb') as source:
            patch_handle = patch_path.open('rb') if patch else None
            try:
                for position in meta['positions']:
                    pixels = np.frombuffer(source.read(size), np.uint8).reshape(native['height'] * 3 // 2, native['width'])
                    rgb = yuv_to_rgb(pixels, native['width'], native['height'], native['full_range'])
                    if patch_handle is not None:
                        tile = np.frombuffer(patch_handle.read(64 * 64 * 3 // 2), np.uint8).reshape(96, 64)
                        if position is not None:
                            px = yuv_to_rgb(tile, 64, 64, native['full_range'])
                            x, y = position
                            rgb[y:y + 32, x:x + 32] = px[16:48, 16:48]
                    frame = rgb.tobytes()
                    frame_hashes.append(digest(frame))
                    result.extend(frame)
            finally:
                if patch_handle is not None:
                    patch_handle.close()
        return bytes(result), {'decoder': identity, 'receipts': receipts, 'rgb_frame_sha256': frame_hashes,
                               'decoded_sha256': digest(result), 'decoded_frames': len(frame_hashes), 'numpy_version': np.__version__,
                               'isolation': 'packet and installed decoder only; OS source-denial not established'}


def decode(packet: Path, output: Path, *, scratch: Path, decoder: str = 'dav1d') -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    data = codec.read_packet(packet)
    pixels, details = _render(data, scratch=scratch, decoder=decoder)
    codec.write_new(output, pixels)
    return {**inspect(data), **details, 'operation': 'decode'}


def _patch_video(source: Path, out: Path, native: dict[str, Any], positions: list[list[int] | None]) -> list[str]:
    width, height = native['width'], native['height']
    size = width * height * 3 // 2
    hashes = []
    with source.open('rb') as reader, out.open('xb') as writer:
        for position in positions:
            frame = np.frombuffer(reader.read(size), np.uint8)
            if len(frame) != size:
                raise ValueError('truncated source frame')
            padded = np.empty(64 * 64 * 3 // 2, dtype=np.uint8)
            padded[:4096] = 0 if native['full_range'] else 16
            padded[4096:] = 128
            if position is not None:
                x, y = position
                src_y = frame[:width * height].reshape(height, width)
                src_u = frame[width * height:width * height * 5 // 4].reshape(height // 2, width // 2)
                src_v = frame[width * height * 5 // 4:].reshape(height // 2, width // 2)
                padded[:4096].reshape(64, 64)[16:48, 16:48] = src_y[y:y + 32, x:x + 32]
                padded[4096:5120].reshape(32, 32)[8:24, 8:24] = src_u[y // 2:y // 2 + 16, x // 2:x // 2 + 16]
                padded[5120:].reshape(32, 32)[8:24, 8:24] = src_v[y // 2:y // 2 + 16, x // 2:x // 2 + 16]
            payload = padded.tobytes()
            hashes.append(digest(payload))
            writer.write(payload)
        if reader.read(1):
            raise ValueError('extra source frames')
    return hashes


def encode(source: Path, output: Path, config: dict[str, Any], *, scratch: Path,
           encoder: str = 'SvtAv1EncApp', decoder: str = 'dav1d') -> dict[str, Any]:
    """Encode explicit raw YUV420 and per-frame centre hints into a PTP1 packet."""
    fields = {'profile', 'width', 'height', 'fps', 'frames', 'full_range', 'crf', 'preset', 'input_format', 'positions'}
    if (set(config) not in (fields, fields | {'position_provenance'})
            or config['profile'] != 'tracked-pixel-v1' or config['input_format'] != 'yuv420p'):
        raise ValueError('tracked profile requires explicit raw YUV420 configuration')
    if (type(config['crf']) not in (int, float) or not math.isfinite(config['crf']) or not 0 <= config['crf'] <= 70
            or type(config['preset']) is not int or not 0 <= config['preset'] <= 13):
        raise ValueError('invalid encoder settings')
    if output.exists():
        raise FileExistsError(output)
    encoder_info, decoder_info = svtav1.tool(encoder), svtav1.tool(decoder)
    native = {k: config[k] for k in ('width', 'height', 'fps', 'frames', 'full_range')}
    native.update(version=1, profile='native-av1', decoder_sha256=decoder_info['sha256'])
    codec.validate(native)
    if native['width'] * native['height'] * native['frames'] * 3 > codec.MAX_DECODED:
        raise ValueError('decoded RGB exceeds 512 MiB')
    provenance = config.get('position_provenance', {'kind': 'caller-supplied'})
    if not isinstance(provenance, dict) or len(json.dumps(provenance, allow_nan=False).encode()) > 4096:
        raise ValueError('invalid bounded position provenance')
    hints = config['positions']
    if not isinstance(hints, list) or len(hints) != native['frames']:
        raise ValueError('placement hint count differs from frame count')
    positions = [origin(pos, native['width'], native['height']) for pos in hints]
    expected_size = native['width'] * native['height'] * 3 // 2 * native['frames']
    if source.stat().st_size != expected_size:
        raise ValueError('source size differs from configuration')
    meta = {'version': 1, 'profile': 'tracked-pixel-v1', 'native': native, 'numpy_version': np.__version__,
            'positions': positions, 'tile_size': 64, 'inner_box': [16, 16, 32, 32]}
    with tempfile.TemporaryDirectory(prefix='ptp-encode-', dir=scratch) as temporary:
        root = Path(temporary)
        base_path, patch_path = root / 'base.ivf', root / 'patch.ivf'
        command = svtav1.encode_command(encoder, source, base_path, width=native['width'], height=native['height'],
                                        fps=Fraction(*native['fps']), frames=native['frames'], crf=config['crf'],
                                        preset=config['preset'], full_range=native['full_range'])
        base_time = svtav1.run(command, 600)
        patch_command: list[str] | None = None
        patch_time: dict[str, float] | None = None
        patch_hashes: list[str] = []
        if any(position is not None for position in positions):
            patch_source = root / 'patch.yuv'
            patch_hashes = _patch_video(source, patch_source, native, positions)
            patch_command = svtav1.encode_command(encoder, patch_source, patch_path, width=64, height=64,
                fps=Fraction(*native['fps']), frames=native['frames'], crf=20, preset=4, full_range=native['full_range'])
            patch_time = svtav1.run(patch_command, 600)
        base = base_path.read_bytes()
        patch = patch_path.read_bytes() if patch_command else b''
        data = pack(meta, base, patch)
        pixels, receipt = _render(data, scratch=root, decoder=decoder)
        with source.open('rb') as reader:
            source_sha = hashlib.file_digest(reader, 'sha256').hexdigest()
        codec.write_new(output, data)
    return {**inspect(data), **receipt, 'operation': 'encode', 'encoder': encoder_info,
            'base_encode': {'command': command, 'timing': base_time, 'sha256': digest(base)},
            'patch_encode': None if patch_command is None else {'command': patch_command, 'timing': patch_time,
                             'sha256': digest(patch), 'input_frame_sha256': patch_hashes},
            'source_sha256': source_sha, 'decoded_frames': len(receipt['rgb_frame_sha256']),
            'decoded_bytes': len(pixels), 'config': config,
            'position_provenance': {'caller_asserted': provenance,
                                    'hints_sha256': digest(json.dumps(hints, separators=(',', ':'), allow_nan=False).encode()),
                                    'generation_cost': 'not measured'}}

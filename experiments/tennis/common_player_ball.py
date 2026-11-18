"""Paid 32-pixel ball tiles over a streamed common-player canvas.

The sidecar binds an exact player (or native-only) carrier and costs no second
base. Coordinates are even, top-left origins on the output canvas. Tick order,
50 Hz, limited-range BT.709, 32-in-64 placement and ball-last order are fixed.
"""
from __future__ import annotations

import struct
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np

from experiments.tennis import common_player_packet as player
from experiments.tennis import w8_payload as wire
from src.codecs import svtav1
from src.pointstream import tracked_patch

TICKS = 240
TILE = 64
PATCH = 32
ABSENT = (65535, 65535)
PAIR = struct.Struct('>HH')
MAX_TILE_BYTES = 8 * 1024 * 1024


def _profile(carrier: bytes, decoder_sha256: str) -> tuple[dict, tuple[int, int]]:
    if not wire.HASH.fullmatch(decoder_sha256):
        raise ValueError('invalid tile decoder identity')
    checked = player.inspect(carrier)['profile']
    width, height = checked['canvas_size']
    if min(width, height) < PATCH:
        raise ValueError('canvas smaller than patch')
    return {'profile': 'common-player-ball-sidecar-v1', 'carrier_sha256': wire.sha(carrier),
            'tile_decoder_sha256': decoder_sha256, 'numpy_version': np.__version__}, (width, height)


def _entries(positions: Sequence[tuple[int, int] | None], tile: bytes, size: tuple[int, int]) -> list[tuple[str, str, bytes]]:
    if len(positions) != TICKS or len(tile) > MAX_TILE_BYTES:
        raise ValueError('ball payload size or tick coverage')
    width, height = size
    coords = bytearray()
    for pos in positions:
        if pos is not None:
            if (not isinstance(pos, (list, tuple)) or len(pos) != 2 or any(type(v) is not int or v % 2 for v in pos)
                    or not 0 <= pos[0] <= width - PATCH or not 0 <= pos[1] <= height - PATCH):
                raise ValueError('invalid canvas patch origin')
        coords.extend(PAIR.pack(*(ABSENT if pos is None else pos)))
    if any(pos is not None for pos in positions):
        player._ivf(tile, TILE, TILE)
    elif tile:
        raise ValueError('all-absent sidecar has tile bytes')
    return [('positions', 'frame', bytes(coords)), ('tile', 'frame', tile)]


def pack(carrier: bytes, positions: Sequence[tuple[int, int] | None], tile: bytes, *, decoder_sha256: str) -> bytes:
    profile, size = _profile(carrier, decoder_sha256)
    return wire.pack(profile, _entries(positions, tile, size))


def _open(carrier: bytes, sidecar: bytes, decoder_sha256: str) -> tuple[list[tuple[int, int] | None], bytes, tuple[int, int]]:
    if len(sidecar) > MAX_TILE_BYTES + 8192 or not sidecar.startswith(wire.MAGIC):
        raise ValueError('invalid sidecar length or encoding')
    profile, size = _profile(carrier, decoder_sha256)
    parts, _ = wire.unpack(sidecar, profile)
    if set(parts) != {'positions', 'tile'} or len(parts['positions']) != TICKS * PAIR.size:
        raise ValueError('invalid ball sections')
    positions = [None if pos == ABSENT else pos for pos in PAIR.iter_unpack(parts['positions'])]
    if wire.pack(profile, _entries(positions, parts['tile'], size)) != sidecar:
        raise ValueError('noncanonical sidecar')
    return positions, parts['tile'], size


def inspect(carrier: bytes, sidecar: bytes, *, decoder_sha256: str) -> dict:
    positions, tile, size = _open(carrier, sidecar, decoder_sha256)
    positions_bytes = len(positions) * PAIR.size
    return {'canvas_size': list(size), 'ticks': TICKS, 'present_ticks': sum(x is not None for x in positions),
            'carrier_bytes': len(carrier), 'tile_ivf_bytes': len(tile), 'positions_bytes': positions_bytes,
            'sidecar_framing_bytes': len(sidecar) - len(tile) - positions_bytes,
            'sidecar_bytes': len(sidecar), 'total_bytes': len(carrier) + len(sidecar)}


def render(carrier: bytes, sidecar: bytes, canvases: Iterable[tuple[int, np.ndarray]], *,
           scratch: Path, decoder: str = 'dav1d') -> Iterator[tuple[int, np.ndarray]]:
    """Apply paid tile pixels after player composition. Consume to EOF to check full coverage."""
    identity = svtav1.tool(decoder)
    positions, tile, (width, height) = _open(carrier, sidecar, identity['sha256'])
    with tempfile.TemporaryDirectory(prefix='ball-sidecar-', dir=scratch) as tmp:
        tiles = None
        if tile:
            path = Path(tmp) / 'decoded.yuv'
            tracked_patch.decode_native(tile, path, {'width': TILE, 'height': TILE, 'fps': [50, 1], 'frames': TICKS}, identity['real_path'])
            tiles = np.memmap(path, mode='r', dtype=np.uint8, shape=(TICKS, TILE * 3 // 2, TILE))
        for t, (position, (tick, frame)) in enumerate(zip(positions, canvases, strict=True)):
            if type(tick) is not int or tick != t or frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                raise ValueError('upstream canvas geometry, pixel format or tick order')
            if position is not None:
                if tiles is None:
                    raise ValueError('missing tile stream')
                x, y = position
                rgb = tracked_patch.yuv_to_rgb(tiles[t], TILE, TILE, False)
                frame = frame.copy()
                frame[y:y+PATCH, x:x+PATCH] = rgb[16:48, 16:48]
            yield tick, frame

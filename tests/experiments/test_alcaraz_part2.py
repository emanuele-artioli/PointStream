"""Focused wire and causal-render checks for the Alcaraz part-2 ladder."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.modular.alcaraz_part2 import (
    _check_actual_row,
    _decode_silhouette,
    _encode_references,
    _render,
    _silhouette_wire,
    refresh_indices,
)
from experiments.modular.foreground_campaign import _row


class FakeSidecar:
    """Deterministic 64x64 BGR intra stand-in; no native codec in unit tests."""

    def encode(self, image: np.ndarray) -> bytes:
        self.image = image.copy()
        return b'AV1' + bytes([len(image) % 256])

    def decode(self, wire: bytes) -> np.ndarray:
        assert wire.startswith(b'AV1')
        return self.image.copy()


def _two_tracks() -> tuple[np.ndarray, list[np.ndarray]]:
    source = np.zeros((6, 64, 64, 3), dtype=np.uint8)
    source[:, 8:20, 6:18, 0] = np.arange(6, dtype=np.uint8)[:, None, None] * 30 + 10
    source[:, 25:37, 35:47, 1] = np.arange(6, dtype=np.uint8)[:, None, None] * 20 + 15
    left = np.zeros(source.shape[:3], dtype=bool)
    right = np.zeros_like(left)
    left[[0, 1, 3, 4, 5], 8:20, 6:18] = True
    right[:, 25:37, 35:47] = True
    return source, [left, right]


def test_refresh_indices_count_visible_frames_and_rejects_bad_input() -> None:
    flags = np.array([False, True, True, False, True, True, False, True, True], dtype=bool)
    assert refresh_indices(flags, 2) == [1, 4, 7]
    assert refresh_indices(flags, 20) == [1]
    with pytest.raises(ValueError):
        refresh_indices(flags, 0)
    with pytest.raises(ValueError):
        refresh_indices(np.zeros(3, dtype=bool), 1)
    with pytest.raises(ValueError):
        refresh_indices(np.array([1, 0], dtype=np.uint8), 1)


def test_silhouette_wire_roundtrip_and_rejects_empty_payload() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[9:18, 7:16] = True
    wire, restored = _silhouette_wire(mask, (8, 20, 6, 18), 5)
    assert len(wire) > 6
    assert restored.shape == (12, 12)
    assert np.array_equal(restored, mask[8:20, 6:18])
    frame, decoded = _decode_silhouette(wire)
    assert frame == 5
    assert np.array_equal(decoded, restored)
    with pytest.raises(RuntimeError):
        _decode_silhouette(wire[:6])
    with pytest.raises(ValueError):
        _silhouette_wire(np.zeros_like(mask), (8, 20, 6, 18), 5)


def test_each_player_has_own_references_and_actual_byte_accounting() -> None:
    source, tracks = _two_tracks()
    objects, parts = _encode_references(source, tracks, 2, FakeSidecar())
    assert [ref['frame'] for ref in objects[0]['refs']] == [0, 3, 5]
    assert [ref['frame'] for ref in objects[1]['refs']] == [0, 2, 4]
    assert parts['F'] == sum(ref['crop_bytes'] for obj in objects for ref in obj['refs'])
    assert parts['M'] == sum(obj['bbox_bytes'] + obj['presence_bytes'] for obj in objects)
    assert parts['H'] == 1 + sum(ref['alpha_bytes'] + ref['index_bytes'] for obj in objects for ref in obj['refs'])
    assert all(ref['crop_bgr'].shape[0] <= 64 and ref['crop_bgr'].shape[1] <= 64 for obj in objects for ref in obj['refs'])

    # The actual scored-row helper must charge each component exactly once.
    mask = tracks[0] | tracks[1]
    delivered = source.copy()
    delivered[:] = 1  # finite error in both foreground and background
    row = _row('every_2', 'neither', B=100, F=int(parts['F']), M=int(parts['M']), R=0, H=int(parts['H']),
               source=source, delivered=delivered, mask=mask, encode_s=1, decode_s=2,
               render_s=3, plate_s=4, source_row={'total_bytes': 9999, 'psnr_weighted': 100,
                                                   'encode_seconds': 5, 'decode_seconds': 6})
    assert row['total_bytes'] == sum(row[key] for key in ('B', 'F', 'M', 'R', 'H'))
    assert not row['claimable']
    _check_actual_row(row)
    bad = {**row, 'total_bytes': row['total_bytes'] - 1}
    with pytest.raises(RuntimeError, match='wire components'):
        _check_actual_row(bad)
    bad = {**row, 'scores': {**row['scores'], 'weighted': float('nan')}}
    with pytest.raises(RuntimeError, match='nonfinite'):
        _check_actual_row(bad)


def test_renderer_uses_latest_decoded_reference_without_target_mask() -> None:
    source, tracks = _two_tracks()
    objects, _ = _encode_references(source, tracks, 2, FakeSidecar())
    background = np.full(source.shape, 5, dtype=np.uint8)
    decoded, _ = _render(background, objects)
    assert decoded.shape == source.shape
    # Player 0 is absent at frame 2, and its previous region remains court.
    assert np.array_equal(decoded[2, 10, 10], background[2, 10, 10])
    # At frame 4 player 0 uses frame 3, not the future frame 5 reference.
    assert decoded[4, 12, 10, 0] == source[3, 12, 10, 0]
    # The right player has its own refreshed green appearance at frame 4.
    assert decoded[4, 28, 40, 1] == source[4, 28, 40, 1]

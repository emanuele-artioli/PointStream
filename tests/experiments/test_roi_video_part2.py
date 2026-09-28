"""Object-video wire layout and renderer contract without native encodes."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.modular.roi_video_part2 import (
    _build_object_inputs,
    _composite,
    _rgb_to_yuv420,
    _row_checked,
    _yuv420_to_rgb,
)


def _fixture() -> tuple[np.ndarray, list[np.ndarray]]:
    source = np.zeros((48, 64, 96, 3), dtype=np.uint8)
    source[:, 7:20, 6:18] = (240, 10, 20)  # RGB red-ish left player
    source[:, 30:45, 65:79] = (10, 20, 230)  # RGB blue-ish right player
    left = np.zeros(source.shape[:3], dtype=bool)
    right = np.zeros_like(left)
    left[:, 7:20, 6:18] = True
    right[:, 30:45, 65:79] = True
    return source, [left, right]


def test_two_independent_masked_video_inputs_and_channel_order() -> None:
    source, tracks = _fixture()
    objects, prep_s = _build_object_inputs(source, tracks)
    assert len(objects) == 2
    assert prep_s >= 0
    for j, obj in enumerate(objects):
        color, alpha = obj['color_rgb'], obj['alpha_rgb']
        assert color.shape == alpha.shape
        assert color.shape[0] == 48 and color.shape[1] >= 64 and color.shape[2] >= 64
        assert color.shape[1] % 8 == color.shape[2] % 8 == 0
        assert obj['bbox_bytes'] == 8 * 48
        assert obj['presence_bytes'] == 6
        assert np.array_equal(alpha > 127, np.repeat((color != 0).any(axis=-1)[..., None], 3, axis=-1))
        assert np.all(color[alpha[..., 0] == 0] == 0)
        assert np.array_equal(color[0, 10, 10], (240, 10, 20) if j == 0 else (10, 20, 230))
    assert not np.array_equal(objects[0]['color_rgb'], objects[1]['color_rgb'])
    luma, chroma = _rgb_to_yuv420(objects[0]['color_rgb'])
    assert int(luma[0, 10, 10]) == round(0.299 * 240 + 0.587 * 10 + 0.114 * 20)
    restored = _yuv420_to_rgb(luma, chroma)
    assert restored.shape == objects[0]['color_rgb'].shape
    assert restored[0, 10, 10, 0] > restored[0, 10, 10, 2]


def test_decoder_uses_only_decoded_alpha_color_and_bbox() -> None:
    source, tracks = _fixture()
    objects, _ = _build_object_inputs(source, tracks)
    background = np.full(source.shape, 60, dtype=np.uint8)
    colors = [obj['color_rgb'].copy() for obj in objects]
    alphas = [obj['alpha_rgb'].copy() for obj in objects]
    rendered, seconds = _composite(background, objects, colors, alphas)
    assert seconds >= 0
    assert np.array_equal(rendered[0, 10, 10], source[0, 10, 10])
    assert np.array_equal(rendered[0, 35, 70], source[0, 35, 70])
    assert np.array_equal(rendered[0, 0, 0], background[0, 0, 0])
    # The decoder obeys a different decoded alpha; no source mask is supplied.
    alphas[0][:] = 0
    rendered, _ = _composite(background, objects, colors, alphas)
    assert np.array_equal(rendered[0, 10, 10], background[0, 10, 10])
    with pytest.raises(ValueError, match='one color and alpha'):
        _composite(background, objects, colors, alphas[:1])


def test_actual_row_charges_four_videos_and_motion_once() -> None:
    source, tracks = _fixture()
    objects, _ = _build_object_inputs(source, tracks)
    delivered = np.full_like(source, 1)  # finite FG and BG scores
    mask = tracks[0] | tracks[1]
    color_bytes = [123, 321]
    alpha_bytes = [45, 54]
    row = _row_checked('temporal', 'neither', B=24648, F=sum(color_bytes),
                       M=sum(obj['bbox_bytes'] + obj['presence_bytes'] for obj in objects),
                       R=0, H=1+sum(alpha_bytes), source=source, delivered=delivered,
                       mask=mask, encode_s=1, decode_s=2, render_s=3, plate_s=4,
                       source_row={'total_bytes': 65149, 'psnr_weighted': 100,
                                   'encode_seconds': 5, 'decode_seconds': 6})
    assert row['total_bytes'] == sum(row[key] for key in ('B', 'F', 'M', 'R', 'H'))
    assert row['F'] == 444 and row['H'] == 100
    assert not row['claimable']
    with pytest.raises(ValueError, match='two independent'):
        _build_object_inputs(source, tracks[:1])

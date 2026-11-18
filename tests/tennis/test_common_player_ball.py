"""Contract checks for the common-player canvas sidecar; no quality claims."""
import struct

import numpy as np
import pytest

from experiments.tennis import common_player_ball as ball
from experiments.tennis import common_player_packet as player


DECODER = '7' * 64


def ivf(width, height, payload=b'x'):
    head = struct.pack('<4sHH4sHHIIII', b'DKIF', 0, 32, b'AV01', width, height, 50, 1, 240, 0)
    return head + b''.join(struct.pack('<IQ', len(payload), t) + payload for t in range(240))


def carrier(payload=b'x'):
    profile = {'mode': 'native_only', 'recipe': player.NATIVE_RECIPE, 'resources': {'native_decoder': 'a'*64},
               'coded_size': [32, 32], 'canvas_size': [96, 64], 'cuts': [], 'runs': [], 'order': []}
    return player.build(profile, ivf(32, 32, payload), None, {})


def test_canvas_accounting_and_bindings():
    upstream = carrier()
    positions = [None] * 239 + [(64, 32)]
    tile = ivf(64, 64)
    side = ball.pack(upstream, positions, tile, decoder_sha256=DECODER)
    data = ball.inspect(upstream, side, decoder_sha256=DECODER)
    assert data['present_ticks'] == 1 and data['canvas_size'] == [96, 64]
    assert data['total_bytes'] == len(upstream) + len(side)
    assert len(side) == data['sidecar_framing_bytes'] + 960 + len(tile)
    for bad in (side[:-1], side + b'.', side[:-1] + b'y'):
        with pytest.raises(ValueError):
            ball.inspect(upstream, bad, decoder_sha256=DECODER)
    with pytest.raises(ValueError):
        ball.inspect(carrier(b'y'), side, decoder_sha256=DECODER)
    with pytest.raises(ValueError):
        ball.inspect(upstream, side, decoder_sha256='6'*64)
    for wrong in ([(65, 0)], [(-2, 0)], [(64, 34)], [(0, 0, 0)], [(False, 0)]):
        with pytest.raises(ValueError):
            ball.pack(upstream, [None] * 239 + wrong, tile, decoder_sha256=DECODER)
    with pytest.raises(ValueError):
        ball.pack(upstream, positions, ivf(32, 32), decoder_sha256=DECODER)
    with pytest.raises(ValueError):
        ball.pack(upstream, [None]*240, tile, decoder_sha256=DECODER)


def test_streamed_absence_and_timestamp_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(ball.svtav1, 'tool', lambda name: {'sha256': DECODER})
    upstream = carrier()
    side = ball.pack(upstream, [None]*240, b'', decoder_sha256=DECODER)
    frame = np.zeros((64, 96, 3), np.uint8)

    def render(ticks, canvas=frame):
        return list(ball.render(upstream, side, ((t, canvas) for t in ticks), scratch=tmp_path))

    assert len(render(range(240))) == 240
    assert render(range(240))[-1][1] is frame
    for ticks in ((1,), range(239), range(241)):
        with pytest.raises(ValueError):
            render(ticks)
    with pytest.raises(ValueError):
        render(range(240), frame[:, :32])


def test_overwrite_order_and_untouched_canvas(tmp_path, monkeypatch):
    monkeypatch.setattr(ball.svtav1, 'tool', lambda name: {'sha256': DECODER, 'real_path': 'fixture-decoder'})
    yuv = np.full((96, 64), 128, np.uint8)
    yuv[:64] = 235

    def fixed_tiles(tile, output, geometry, decoder):
        assert geometry == {'frames': 240, 'width': 64, 'height': 64, 'fps': [50, 1]}
        output.write_bytes(yuv.tobytes() * 240)

    monkeypatch.setattr(ball.tracked_patch, 'decode_native', fixed_tiles)
    upstream = carrier()
    side = ball.pack(upstream, [(64, 32), *([None] * 239)], ivf(64, 64), decoder_sha256=DECODER)
    canvas = np.full((64, 96, 3), 48, np.uint8)
    canvas[32:, 32:] = (12, 27, 225)  # Existing receiver pixels inside and outside the tile.
    output = list(ball.render(upstream, side, ((t, canvas) for t in range(240)), scratch=tmp_path))
    expected = canvas.copy()
    expected[32:, 64:] = 255
    np.testing.assert_array_equal(output[0][1], expected)
    assert all(frame is canvas for _, frame in output[1:])
    assert np.all(canvas[32:, 64:] == (12, 27, 225))

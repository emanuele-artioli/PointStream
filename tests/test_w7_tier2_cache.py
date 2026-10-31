from pathlib import Path

from experiments.tennis import w7_tier2_cache as cache


def test_cache_ranges_do_not_fill_gaps_and_command_preserves_frames():
    assert cache.ranges([0, 200, 600]) == [(0, 400), (600, 800)]
    command = cache.decode_command('ffmpeg', Path('/video'), [0, 200, 600], Path('/out'))
    assert command[command.index('-frames:v') + 1] == '600'
    assert command[command.index('-fps_mode') + 1] == 'passthrough'
    assert 'between(n\\,0\\,399)+between(n\\,600\\,799)' in command[command.index('-vf') + 1]


def test_cache_rejects_invalid_chunk_boundaries():
    import pytest

    for starts in ([200, 0], [0, 0], [-200], [201]):
        with pytest.raises(ValueError):
            cache.ranges(starts)


def test_cached_frame_uses_k8b_writable_buffer(tmp_path, monkeypatch):
    import numpy as np

    from experiments.tennis import court, windows

    raw = tmp_path / 'frame.yuv'
    raw.write_bytes(bytes(cache.FRAME_BYTES))
    seen = []

    def convert(yuv, width, height):
        seen.append((yuv.flags.writeable, width, height))
        return np.zeros((1, 1, 3), dtype=np.uint8)

    monkeypatch.setattr(windows, 'yuv_to_rgb', convert)
    monkeypatch.setattr(court, 'register_v2', lambda rgb, previous: {'homography': None, 'fits': False})
    assert cache.register_raw(raw, 0, 200, 201) == [{'frame': 200, 'homography': None, 'fits': False}]
    assert seen == [(True, 1920, 1080)]

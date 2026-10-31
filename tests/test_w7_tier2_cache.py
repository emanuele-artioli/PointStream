from pathlib import Path

from experiments.tennis import w7_tier2_cache as cache


def test_cache_ranges_do_not_fill_gaps_and_command_preserves_frames():
    assert cache.ranges([0, 200, 600]) == [(0, 400), (600, 800)]
    command = cache.decode_command('ffmpeg', Path('/video'), [0, 200, 600], Path('/out'))
    assert command[command.index('-frames:v') + 1] == '600'
    assert command[command.index('-fps_mode') + 1] == 'passthrough'
    assert 'between(n\\,0\\,399)+between(n\\,600\\,799)' in command[command.index('-vf') + 1]

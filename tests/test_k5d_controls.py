from fractions import Fraction

import numpy as np

from experiments.tennis.k5d_controls import variants
from src.pointstream import ball, ball_stream


def test_controls_preserve_appearance_and_shift_positions_and_guard_disk():
    track = [(10, 12), (11, 12), None, (70, 12)]
    appearance = ball.Appearance((130, 150, 170), radius=2, exposure=3, phase=2)
    apps = [appearance, appearance, None, appearance]
    stream = ball_stream.Stream(128, 64, Fraction(30), 4, 'svt-av1-ivf', 'svt', b'native',
                                ball.encode_frames(track, apps, 128, 64))
    native = np.zeros((4, 64, 128, 3), np.uint8)
    arms, disabled = variants(stream, native.copy(), native)
    assert disabled
    assert arms['streak'] == stream
    assert arms['disk'] == arms['native']
    shifted, shifted_apps = ball.decode_frames(arms['wrong_position'].ball_payload, 4, 128, 64)
    assert shifted == [(74, 12), (75, 12), None, (6, 12)]
    assert shifted_apps == apps
    positions, bright_apps = ball.decode_frames(arms['brightness'].ball_payload, 4, 128, 64)
    assert positions == track
    assert bright_apps[0] == ball.Appearance((255, 255, 255), radius=2, exposure=3, phase=2)
    for arm in arms.values():
        assert ball_stream.unpack(ball_stream.pack(arm)) == arm


def test_disk_preserves_parameters_when_psnr_guard_passes():
    track = [(10, 10), (11, 10)]
    a = ball.Appearance((100, 110, 120), radius=2, exposure=4, phase=2)
    stream = ball_stream.Stream(128, 64, Fraction(30), 2, 'svt-av1-ivf', 'svt', b'native',
                                ball.encode_frames(track, [a, a], 128, 64))
    native = np.zeros((2, 64, 128, 3), np.uint8)
    disk = ball_stream.Stream(128, 64, Fraction(30), 2, stream.codec, stream.resource_id, stream.native,
                              ball.encode_frames(track, [ball.Appearance(a.colour, a.radius, 0, a.phase)] * 2, 128, 64))
    source = ball_stream.render_from_native_decoded_frames(disk, native)
    arms, disabled = variants(stream, source, native)
    assert not disabled
    assert arms['disk'] == disk

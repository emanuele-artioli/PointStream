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


def test_cost_forecast_requires_success_and_current_campaign_budget():
    import pytest
    from experiments.tennis.k5d_controls import cost_forecast

    execution = {'exit_code': 0, 'seconds': 40., 'timing_contaminated': False}
    clock = {'started': 100., 'phase_started': 110., 'budget_end': 1000.}
    projected = cost_forecast(execution, clock, 200., 850.)
    assert projected == 40 * 3 * 1.5 + 20
    with pytest.raises(ValueError, match='forecast'):
        cost_forecast(execution, clock, 850., 270.)
    with pytest.raises(ValueError, match='invalid'):
        cost_forecast({**execution, 'timing_contaminated': True}, clock, 200., 850.)
    with pytest.raises(ValueError, match='invalid'):
        cost_forecast({**execution, 'exit_code': 1}, clock, 200., 850.)


def test_bound_calibration_spec_has_frozen_publications_and_budget():
    import json
    from pathlib import Path
    from experiments.jobs.inbox import validate_spec
    from experiments.tennis.k5d_controls import CALIBRATION_SHA

    path = Path('docs/experiments/k5d-inputs/calibration-controls-pilot-spec-20261105.json')
    spec = json.loads(path.read_text())
    validate_spec(spec)
    inputs = {x['name']: x for x in spec['staged_inputs']}
    for name, digest in CALIBRATION_SHA.items():
        item = inputs[name]
        assert item['sha256'] == digest
        assert {'path': item['path'], 'sha256': digest} in spec['inputs']
        assert ('20261105T003756Z-cc450baf' if name.endswith('smoke') else '20261105T000550Z-d8b5c76b') in item['path']
    assert spec['gpu_models'] == ['RTX A6000']
    assert spec['budget_seconds'] == 550 and spec['full']['seconds'] == 400
    assert spec['smoke']['seconds'] == 130 and spec['validator_seconds'] == 15
    assert spec['scale']['frames'] == {'smoke': 16, 'full': 48}


def test_future_full_forecast_fits_fixed_and_marginal_with_slowdown():
    import pytest
    from experiments.tennis.k5d_controls import full188_forecast

    assert full188_forecast(50, 82, 60, reserve=100) == pytest.approx(1.5 * (34 + 188) * 1.2 + 100)
    with pytest.raises(ValueError, match='negative fixed'):
        full188_forecast(10, 50, 10, reserve=100)
    with pytest.raises(ValueError, match='invalid'):
        full188_forecast(50, 49, 50, reserve=100)

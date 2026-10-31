import numpy as np
import pytest

from experiments.tennis.w6b_smoothing import smooth_crop


def test_smoothing_reset_and_validation():
    frame = np.full((64, 64, 3), 100, np.uint8)
    assert np.array_equal(smooth_crop(frame, None, 0.5), frame)
    assert np.array_equal(smooth_crop(frame, frame, 0), frame)
    with pytest.raises(ValueError):
        smooth_crop(frame, frame, 1.1)
    with pytest.raises(ValueError):
        smooth_crop(frame, frame[:-1], 0.5)


def test_smoothing_uses_warped_predecessor(monkeypatch):
    from experiments.visor import h3

    current = np.full((64, 64, 3), 100, np.uint8)
    previous = np.full_like(current, 20)
    seen = []
    monkeypatch.setattr(h3, 'dis_flow', lambda target, source: seen.append((target.copy(), source.copy())) or np.zeros((*target.shape, 2), np.float32))
    result = smooth_crop(current, previous, 0.25)
    assert np.all(result == 80)
    assert np.all(seen[0][0] == 100) and np.all(seen[0][1] == 20)


def test_backward_flow_aligns_predecessor_before_blending(monkeypatch):
    from experiments.visor import h3

    current = np.zeros((64, 64, 3), np.uint8)
    current[:, 20:30] = 200
    previous = np.zeros_like(current)
    previous[:, 23:33] = 200
    flow = np.zeros((64, 64, 2), np.float32)
    flow[..., 0] = 3
    monkeypatch.setattr(h3, 'dis_flow', lambda target, source: flow)
    assert np.array_equal(smooth_crop(current, previous, 0.5), current)

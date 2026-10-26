import numpy as np
from experiments.tennis.k6b_frame_boundary import boundary_metrics


def test_boundary_tolerance_distinguishes_small_offset_from_lost_player() -> None:
    target = np.zeros((80, 80), bool)
    target[20:60, 20:60] = True
    assert boundary_metrics(np.roll(target, 2, axis=1), target)['f1'] == 1.0
    assert boundary_metrics(np.roll(target, 10, axis=1), target)['f1'] < 0.8
    assert boundary_metrics(np.zeros_like(target), target)['f1'] == 0.0

import numpy as np
import pytest

from experiments.tennis import k4b_rate as kr


def _track(n, v=(3.3, -1.7)):
    kps = np.zeros((n, 17, 3))
    for t in range(n):
        kps[t, :, 0] = 500 + np.arange(17) * 4 + v[0] * t
        kps[t, :, 1] = 300 + np.arange(17) * 6 + v[1] * t
    kps[:, :, 2] = .9
    kps[:, 0, 2] = .2
    return kps


def test_closed_loop_quantization_error_stays_within_half_a_step():
    kps = _track(20)
    for q in kr.STEPS_PX:
        decoded, res, vis, start = kr.code(kps, q, "previous", 1)
        assert np.abs(decoded - kps[:, :, :2]).max() <= q / 2 + 1e-9
        assert len(res) == 19 * 34 and len(vis) == 20 * 17 and sum(vis) == 20 * 16
        assert start == pytest.approx(17 * (np.log2(1920 / q) + np.log2(1080 / q)))


def test_velocity_prediction_zeroes_residuals_on_steady_motion_and_interpolation_is_exact():
    kps = _track(13, v=(4., -2.))
    _, res, _, _ = kr.code(kps, 2, "velocity", 1)
    assert all(r == 0 for r in res[34:])
    decoded, res, vis, _ = kr.code(kps, 1, "previous", 4)
    assert len(vis) == 4 * 17 and len(res) == 3 * 34
    assert np.abs(decoded - kps[:, :, :2]).max() <= .5 + 1e-9


def test_tracks_split_at_unfitted_frames():
    def row(t, fitted):
        s = {"fitted": fitted, "kps": _track(1)[0].tolist()} if fitted else {"fitted": False}
        return {"t": t, "track": "p", "role": "far", "area": 100, "height": .1, "settings": {"crop1": s}}
    out = kr.tracks([row(t, t != 3) for t in range(7)])
    assert [tr["t"] for tr in out] == [[0, 1, 2], [4, 5, 6]] and out[0]["height"][0] == pytest.approx(108)


def test_entropy_of_a_fair_coin_is_one_bit_per_symbol():
    from collections import Counter
    assert kr.entropy_bits(Counter({0: 5, 1: 5})) == pytest.approx(10)

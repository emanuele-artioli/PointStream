import numpy as np

from experiments.visor import pose_coding as pc


def basis(seed: int = 0) -> pc.Basis:
    rng = np.random.default_rng(seed)
    return pc.Basis(mean=rng.normal(0, 0.1, 45), components=np.linalg.qr(rng.normal(size=(45, 45)))[0].T)


def test_combination_grid():
    combos = pc.combinations()
    assert len(combos) == 3080
    assert len({pc.combo_key(c) for c in combos}) == 3080
    assert not any(c["send_hz"] is None and c["fill"] == "linear" for c in combos)


def test_schedule_fill_and_latency():
    sent = pc.schedule(10, 3)
    assert sent.tolist() == [0, 3, 6, 9]
    assert pc.schedule(11, 3).tolist() == [0, 3, 6, 9, 10]
    values = np.arange(4, dtype=float)[:, None] * 3
    assert pc.fill(values, sent, 10, "hold")[:, 0].tolist() == [0, 0, 0, 3, 3, 3, 6, 6, 6, 9]
    assert np.allclose(pc.fill(values, sent, 10, "linear")[:, 0], np.arange(10))
    combo = {"smoother": ("gauss", 133.3), "send_hz": 10.0, "fill": "linear"}
    assert pc.lookahead_frames(combo["smoother"], 30.0) == 4
    assert pc.send_step(10.0, 30.0) == 3
    assert np.isclose(pc.latency_ms(combo, 30.0), 6 * 1000 / 30)
    assert pc.latency_ms({"smoother": ("one_euro", 1.0, 0.0), "send_hz": 10.0, "fill": "hold"}, 50.0) == 0.0


def test_smoothers():
    t = np.arange(40, dtype=float)
    ramp = np.stack([t, 2 * t], 1)
    g = pc.gaussian(ramp, 4)
    assert np.allclose(g[4:-4], ramp[4:-4])  # a symmetric window keeps a line inside the track
    const = np.ones((30, 3))
    assert np.allclose(pc.one_euro(const, 30.0, 1.0, 0.5), const)
    rng = np.random.default_rng(0)
    noisy = rng.normal(0, 1, (300, 2))
    assert pc.one_euro(noisy, 30.0, 0.5, 0.0).std() < 0.6 * noisy.std()


def test_residuals_and_lossless_round_trip():
    s = np.array([[0], [2], [4], [7]])
    assert pc.residuals(s, "previous")[:, 0].tolist() == [2, 2, 3]
    assert pc.residuals(s, "velocity")[:, 0].tolist() == [2, 0, 1]
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.3, (25, 51))
    x[:, 48:50] = rng.uniform(0, 1000, (25, 2))
    b = basis()
    combo = {"smoother": ("none",), "send_hz": None, "fill": "hold", "subspace": None, "step_scale": 1.0,
             "prediction": "previous"}
    symbols, sent = pc.encode(x, combo, 30.0, b)
    back = pc.decode(symbols, sent, len(x), combo, b)
    assert np.all(np.abs(back - x) <= pc.steps(None, 1.0) / 2 + 1e-12)
    full = dict(combo, subspace=45)
    symbols, sent = pc.encode(x, full, 30.0, b)
    err = np.abs(pc.decode(symbols, sent, len(x), full, b) - x)
    # A complete orthonormal basis loses only its quantization: at most sqrt(45) half-steps per angle.
    assert err[:, :48].max() <= np.sqrt(45) * pc.ANGLE_STEP / 2
    assert np.all(err[:, 48:] <= pc.steps(45, 1.0)[-3:] / 2 + 1e-12)


def test_root_round_trip_and_entropy():
    t = np.array([[0.1, -0.05, 0.5], [0.0, 0.02, 0.8]])
    p = pc.root_params(t, 600.0, (704.0, 704.0))
    assert np.allclose(pc.root_transl(p, 600.0, (704.0, 704.0)), t)
    assert pc.entropy_bits([1, 1, 1, 1]) == 0.0
    assert np.isclose(pc.entropy_bits([0, 1, 0, 1]), 1.0)
    assert np.isclose(pc.pooled_bits([np.array([[0, 5], [1, 5]]), np.array([[0, 5], [1, 5]])]), 2.0)

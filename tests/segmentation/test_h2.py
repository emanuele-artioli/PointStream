import numpy as np

from experiments.visor import h2, mano_np
from experiments.visor import pose_coding as pc
from tests.segmentation.test_mano_np import synthetic_arrays


def test_expanded_size_follows_the_models_aspect():
    assert h2.expanded_size(np.array([0, 0, 30, 40])) == 40  # already 3:4
    assert h2.expanded_size(np.array([0, 0, 60, 40])) == 80  # wide: height grows to 4/3 of the width
    assert h2.expanded_size(np.array([0, 0, 10, 40])) == 40  # tall: width grows, the longer side stays


def test_pck_is_per_joint_then_mean():
    dist = np.array([[0.01, 0.2], [0.2, 0.2], [0.01, 0.01]])
    mask = np.array([[True, True], [True, False], [True, True]])
    # joint 0: 2 of 3 correct; joint 1: 1 of 2 correct
    assert np.isclose(h2.pck(dist, mask, 0.05), 100 * (2 / 3 + 1 / 2) / 2)


def test_procrustes_removes_similarity():
    rng = np.random.default_rng(0)
    gt = rng.normal(0, 1, (21, 3))
    r = mano_np.rodrigues(np.array([0.3, 1.0, -0.4]))
    pred = 2.5 * gt @ r.T + np.array([1.0, -2.0, 0.5])
    assert h2.procrustes_error(pred, gt) < 1e-9
    assert h2.procrustes_error(gt + rng.normal(0, 0.1, gt.shape), gt) > 0.01


def test_mask_window_round_trip():
    mask = np.zeros((100, 200), bool)
    mask[40:60, 90:130] = True
    window, data = h2.pack_window(mask, np.array([90, 40, 130, 60]))
    back = h2.unpack_window(window, data)
    x0, y0, x1, y1 = window
    assert back.shape == (y1 - y0, x1 - x0)
    assert back.sum() == mask.sum()
    assert np.array_equal(back, mask[y0:y1, x0:x1])


def test_paired_bootstrap_interval_contains_a_clear_difference():
    rng = np.random.default_rng(1)
    groups = [(rng.normal(1.0, 0.1, 20), rng.normal(0.0, 0.1, 20)) for _ in range(30)]
    out = h2.paired_bootstrap(groups, lambda gs: (float(np.mean([g[0].mean() for g in gs])),
                                                  float(np.mean([g[1].mean() for g in gs]))), rng, 200)
    assert out["low"] > 0.8 and out["high"] < 1.2


def test_accel_errors_zero_for_exact_motion():
    t = np.arange(5)
    gt = np.random.default_rng(2).normal(0, 0.05, (5, 21, 3))
    a = {"t": t, "right": np.ones(5, bool), "gt_world": gt, "hamer_rel_world": gt - gt[:, :1]}
    errors = h2.accel_errors({}, a, "hamer")
    assert len(errors) == 3 and np.allclose(errors, 0)


def test_coding_choice_picks_cheapest_within_the_error_limit():
    def row(key, kbps, mp, px, lat):
        return {"key": key, "kbps": kbps, "mpjpe_vs_truth_mm": mp, "err2d_vs_truth_px": px, "latency_ms": lat,
                "bits_per_sent_frame": 1.0, "err2d_vs_uncoded_px": 0.0}
    hot = [row("a", 10, 20.0, 5.0, 0), row("b", 2, 20.5, 5.1, 0), row("c", 1, 25.0, 5.0, 0), row("d", 0.5, 20.0, 5.0, 100)]
    vis = [dict(r) for r in hot]
    choices = h2.coding_choice({"hot3d": {"wilor": hot}, "visor": {"wilor": vis}},
                               {"hot3d": {"wilor": {"mpjpe_vs_truth_mm": 20.0, "err2d_vs_truth_px": 5.0}}}, "wilor")
    assert choices["0"]["key"] == "b"  # c exceeds 5% on MPJPE; d needs 100 ms
    assert choices["100"]["key"] == "d"


def visor_units(model_mano: mano_np.Mano, frames: int = 30):
    rng = np.random.default_rng(3)
    t = np.concatenate([np.arange(frames), np.arange(frames)])
    right = np.concatenate([np.ones(frames, bool), np.zeros(frames, bool)])
    n = len(t)
    rotvec = np.cumsum(rng.normal(0, 0.02, (n, 16, 3)), 0) + np.array([np.pi / 2, 0, 0])
    transl = np.stack([np.linspace(-0.01, 0.01, n), np.linspace(0.0, 0.02, n), np.full(n, 2.0)], -1)
    a = {"t": t, "right": right, "box": np.tile([800.0, 400.0, 1000.0, 600.0], (n, 1))}
    for k in h2.MODELS:
        a[f"{k}_rotvec"] = rotvec
        a[f"{k}_betas"] = np.zeros((n, 10))
        a[f"{k}_transl"] = transl
    meta = {"unit": "item", "hands": n, "fps": 50.0, "frames": frames, "focal": 37500.0, "principal": [960.0, 540.0],
            "image_shape": [1080, 1920]}
    return [(meta, a)]


def test_coding_grid_on_synthetic_tracks():
    mano = mano_np.Mano(synthetic_arrays())
    data = h2.CodingSet("visor", "wilor", visor_units(mano), mano)
    assert len(data.tracks) == 2 and len(data.order) == 60
    h2._SETS[("visor", "wilor")] = data
    rows = h2.evaluate_smoother(("visor", "wilor"), ("none",), (mano.hands_mean, mano.hands_components))
    assert len(rows) == 7 * 4 * 5 * 2  # send modes x subspaces x steps x predictions
    by_key = {r["key"]: r for r in rows}
    base = by_key[h2.H1_BASELINE]
    coarse = by_key["none|all|full|q8|previous"]
    assert base["kbps"] > coarse["kbps"] > 0
    assert base["mpjpe_vs_uncoded_mm"] < coarse["mpjpe_vs_uncoded_mm"]
    half = by_key["none|15Hz-hold|full|q1|previous"]
    assert half["sent_share"] < 0.5 and half["kbps"] < base["kbps"]
    assert by_key["none|15Hz-linear|full|q1|previous"]["latency_ms"] == (pc.send_step(15.0, 50.0) - 1) * 20.0


def test_keypoint_norm_uses_labelled_joints_only():
    gt = np.zeros((2, 21, 2))
    gt[0, :, 0] = np.linspace(0, 30, 21)
    gt[0, :, 1] = np.linspace(0, 40, 21)
    gt[0, 20] = [1000, 1000]  # outside the frame: not labelled
    existence = np.ones((2, 21), bool)
    existence[0, 20] = False
    existence[1, 1:] = False  # one labelled joint: no extent
    norm = h2.keypoint_norm(gt, existence)
    assert np.isclose(norm[0], h2.expanded_size(np.array([0, 0, 28.5, 38.0])))
    assert np.isnan(norm[1])

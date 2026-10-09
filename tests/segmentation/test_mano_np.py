import numpy as np
import pytest

from experiments.visor import mano_np


def synthetic_arrays(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
    kintree = np.array([[2**32 - 1] + parents[1:], list(range(16))], np.int64)
    weights = rng.random((778, 16)) ** 4
    weights /= weights.sum(1, keepdims=True)
    regressor = rng.random((16, 778))
    regressor /= regressor.sum(1, keepdims=True)
    return {"J_regressor": regressor, "v_template": rng.normal(0, 0.05, (778, 3)),
            "shapedirs": rng.normal(0, 0.002, (778, 3, 10)), "posedirs": rng.normal(0, 0.002, (778, 3, 135)),
            "weights": weights, "kintree_table": kintree, "f": np.zeros((4, 3), np.uint32),
            "hands_mean": rng.normal(0, 0.1, 45), "hands_components": np.linalg.qr(rng.normal(size=(45, 45)))[0]}


def test_rodrigues_round_trip_and_unwrap():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 1, (50, 3))
    m = mano_np.rodrigues(r)
    assert np.allclose(mano_np.rodrigues(mano_np.rotvec_of(m)), m, atol=1e-9)
    inside = r / np.linalg.norm(r, axis=1, keepdims=True) * rng.uniform(0, np.pi - 1e-3, (50, 1))
    assert np.allclose(mano_np.rotvec_of(mano_np.rodrigues(inside)), inside, atol=1e-9)
    # A rotation crossing pi flips its axis-angle; unwrapping keeps the sequence continuous.
    axis = np.array([0.0, 0.0, 1.0])
    angles = np.linspace(2.9, 3.4, 11)
    wrapped = mano_np.rotvec_of(mano_np.rodrigues(angles[:, None] * axis))
    assert np.abs(np.diff(wrapped, axis=0)).max() > 1.0
    unwrapped = mano_np.unwrap_rotvecs(wrapped)
    assert np.allclose(unwrapped[:, 2], angles, atol=1e-9)


def test_rest_pose_gives_rest_joints_and_tips():
    model = mano_np.Mano(synthetic_arrays())
    betas = np.zeros(10)
    rot = np.broadcast_to(np.eye(3), (1, 16, 3, 3))
    joints, verts = model.forward(rot, betas, vertices=True)
    v_shaped, rest = model.shaped(betas)
    assert np.allclose(verts[0], v_shaped)
    inverse = np.argsort(mano_np.MANO_TO_OPENPOSE)
    raw = joints[0][inverse]
    assert np.allclose(raw[:16], rest)
    assert np.allclose(raw[16:], v_shaped[list(mano_np.TIPS)])


def test_tips_only_path_matches_full_skinning():
    model = mano_np.Mano(synthetic_arrays())
    rng = np.random.default_rng(2)
    rot = mano_np.rodrigues(rng.normal(0, 0.4, (7, 16, 3)))
    betas = rng.normal(0, 1, 10)
    fast, _ = model.forward(rot, betas, transl=rng.normal(0, 0.1, (7, 3)), left=True)
    full, verts = model.forward(rot, betas, transl=None, left=True, vertices=True)
    assert verts is not None
    shift = fast - full
    assert np.allclose(shift, shift[:, :1])  # same skeleton, translation only


@pytest.mark.parametrize("left", [False, True])
def test_change_frame_matches_transformed_points(left):
    model = mano_np.Mano(synthetic_arrays())
    rng = np.random.default_rng(3)
    rot = mano_np.rodrigues(rng.normal(0, 0.5, (4, 16, 3)))
    betas = rng.normal(0, 1, 10)
    transl = rng.normal(0, 0.2, (4, 3))
    joints, verts = model.forward(rot, betas, transl, left=left, vertices=True)
    assert verts is not None
    rotation = mano_np.rodrigues(np.array([0.3, -1.1, 0.7]))
    offset = np.array([0.1, -0.4, 0.9])
    _, rest = model.shaped(betas)
    new_rot = rot.copy()
    new_t = np.zeros_like(transl)
    for n in range(4):
        new_rot[n, 0], new_t[n] = mano_np.change_frame(rotation, offset, rot[n, 0], transl[n], rest[0], left)
    moved, moved_verts = model.forward(new_rot, betas, new_t, left=left, vertices=True)
    assert np.allclose(moved, joints @ rotation.T + offset, atol=1e-10)
    assert np.allclose(moved_verts, verts @ rotation.T + offset, atol=1e-10)

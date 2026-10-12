import numpy as np
import pytest

from experiments.visor import h3c
from experiments.visor import h3e
from experiments.visor import h3e_generate as gen
from experiments.visor import mano_np
from experiments.visor.h2 import pinhole


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720), (1440, 1080)])
def test_canvas_intrinsics_project_where_the_canvas_shows_the_point(size):
    width, height = size
    geometry = h3e.canvas_geometry(width, height)
    focal, centre = 1400.0, (width / 2 + 7.0, height / 2 - 3.0)
    K = h3e.canvas_intrinsics(focal, centre, geometry)
    points = np.array([[0.05, -0.02, 0.4], [-0.1, 0.08, 0.6]])
    source = pinhole(points, focal, centre)
    expected = source * geometry["scale"] - np.asarray(geometry["offset"])
    projected = (K @ points.T).T
    assert np.allclose(projected[:, :2] / projected[:, 2:], expected, atol=1e-3)
    frame = np.zeros((height, width, 3), np.uint8)
    canvas = h3e.to_canvas(frame, geometry)
    assert canvas.shape == (h3e.CANVAS[1], h3e.CANVAS[0], 3)
    assert h3e.from_canvas(canvas, geometry).shape == (height, width, 3)


def test_canvas_round_trip_keeps_a_smooth_image():
    width, height = 1920, 1080
    geometry = h3e.canvas_geometry(width, height)
    xs = np.linspace(0, 255, width)[None, :, None]
    frame = np.broadcast_to(xs, (height, width, 3)).astype(np.uint8)
    back = h3e.from_canvas(h3e.to_canvas(frame, geometry), geometry)
    inner = (slice(None), slice(40, width - 40))  # the cropped margins are edge copies
    assert np.abs(back[inner].astype(int) - frame[inner].astype(int)).max() <= 2


def test_generated_times_cover_the_window_at_stride_three():
    times = h3e.generated_times(240)
    assert len(times) == h3e.GENERATED == 81
    assert times[:3] == [0, 3, 6] and times[-2:] == [237, 239]
    judged = [t for t in times[1:] if t % h3e.JUDGE_EVERY == 0]
    assert judged[0] == 15 and judged[-1] == 225 and all(t % 5 == 0 for t in judged)


def test_joints_at_puts_left_then_right_and_zeroes_absent_hands():
    coded = np.arange(3 * 21 * 3, dtype=float).reshape(3, 21, 3) + 1
    t = np.array([0, 0, 3])
    right = np.array([False, True, True])
    out = h3e.joints_at([0, 3, 6], coded, t, right)
    assert out.shape == (3, 42, 3)
    assert np.array_equal(out[0, :21], coded[0]) and np.array_equal(out[0, 21:], coded[1])
    assert not out[1, :21].any() and np.array_equal(out[1, 21:], coded[2])
    assert not out[2].any()


def test_joint_orders_invert_mano_np():
    assert gen.MANO_TO_OPENPOSE == mano_np.MANO_TO_OPENPOSE
    mano_order = np.arange(21)
    openpose = mano_order[list(mano_np.MANO_TO_OPENPOSE)]
    joints = np.zeros((1, 42, 3))
    joints[0, :21, 0] = openpose
    joints[0, 21:, 0] = openpose + 100
    back = gen.reorder(joints, "mano")
    assert np.array_equal(back[0, :21, 0], mano_order) and np.array_equal(back[0, 21:, 0], mano_order + 100)
    assert gen.reorder(joints, "openpose") is joints


def test_silhouette_fills_the_projected_triangles():
    points = np.array([[10.0, 10.0], [50.0, 10.0], [10.0, 50.0]])
    mask = h3e.silhouette(points, np.array([[0, 1, 2]]), (64, 64))
    assert mask[15, 15] and not mask[45, 45] and not mask[5, 5]


FRONT = [{"kbps": 2.0, "pose": 0.04}, {"kbps": 8.0, "pose": 0.02}, {"kbps": 32.0, "pose": 0.01}]


def test_saving_on_the_front_interpolates_in_log_rate():
    saving, where = h3e.saving_against(FRONT, 2.0, 0.03)
    assert where == "on the front" and np.isclose(saving, 1 - 2.0 / 4.0)


def test_saving_beyond_the_front_takes_its_ends_as_bounds():
    low, where = h3e.saving_against(FRONT, 4.0, 0.005)
    assert np.isclose(low, 1 - 4.0 / 32.0) and "lower bound" in where
    high, where = h3e.saving_against(FRONT, 4.0, 0.05)
    assert np.isclose(high, 1 - 4.0 / 2.0) and "upper bound" in where


def items_with(v1, v2):
    return {i: {"v1_statistic": {"saving": a}, "v2": {"saving": b}} for i, a, b in zip(h3c.PILOT, v1, v2)}


@pytest.mark.parametrize("v1,v2,expected", [
    ([0.8, 0.75, 0.85, 0.7], [0.9] * 4, "passes"),
    ([0.2, 0.3, 0.1, 0.4], [0.3] * 4, "fails"),
    ([0.2, 0.3, 0.1, 0.4], [0.9] * 4, "unclear"),  # only V2, with its granted outline, clears 50%
    ([0.9, 0.2, 0.8, 0.5], [0.9] * 4, "unclear"),
])
def test_decision_rule(v1, v2, expected):
    assert h3e.decide(items_with(v1, v2))["decision"] == expected


def test_decision_waits_for_every_pilot_item():
    assert h3e.decide(dict(list(items_with([0.9] * 4, [0.9] * 4).items())[:3]))["decision"] == "incomplete"


def test_generator_joints_keep_the_projection_and_move_the_hand_near():
    rng = np.random.default_rng(1)
    focal_w, centre = 37500.0, (960.0, 540.0)
    hand = np.c_[rng.normal(0, 0.04, (21, 2)) + [0.3, 0.2], 20.0 + rng.normal(0, 0.03, 21)]
    joints = np.zeros((2, 42, 3))
    joints[0, 21:] = hand
    focal = h3e.generator_focal(1920)
    assert np.isclose(focal, 960.0)
    out = h3e.generator_joints(joints, focal_w, centre, focal)
    assert not out[0, :21].any() and not out[1].any()
    assert np.allclose(pinhole(out[0, 21:], focal, centre), pinhole(hand, focal_w, centre), atol=1e-3)
    assert np.isclose(out[0, 21, 2], hand[0, 2] * focal / focal_w) and 0.3 < out[0, 21, 2] < 0.8
    assert np.allclose(out[0, 21:, 2] - out[0, 21, 2], hand[:, 2] - hand[0, 2], atol=1e-6)


def test_hull_drops_points_a_mix_of_two_encodes_beats():
    # (4, 0.032) lies above the chord between (2, 0.04) and (8, 0.01): time-sharing those two reaches 0.03 at 4.
    points = [{"kbps": 2.0, "pose": 0.04}, {"kbps": 4.0, "pose": 0.032}, {"kbps": 8.0, "pose": 0.01}, {"kbps": 16.0, "pose": 0.012}]
    assert [(q["kbps"], q["pose"]) for q in h3e.hull(points)] == [(2.0, 0.04), (8.0, 0.01)]


def test_bd_rate_on_pose_of_half_the_rate_is_minus_half():
    anchor = [{"kbps": r, "pose": e} for r, e in ((2.0, 0.04), (4.0, 0.025), (8.0, 0.015), (16.0, 0.01))]
    half = [{"kbps": q["kbps"] / 2, "pose": q["pose"]} for q in anchor]
    assert h3e.bd_rate_pose(half, anchor) == pytest.approx(-0.5, abs=1e-9)
    assert h3e.bd_rate_pose([{"kbps": 1.0, "pose": 0.001}, {"kbps": 2.0, "pose": 0.0005}], anchor) is None

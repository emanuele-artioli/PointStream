import numpy as np

from experiments.tennis import k6


def _body(x0=900., y0=400., h=200.):
    """A COCO-17 stick figure, every keypoint confident."""
    kp = np.zeros((17, 3))
    pts = {0: (0, 0), 1: (-5, -5), 2: (5, -5), 3: (-10, 0), 4: (10, 0), 5: (-25, 30), 6: (25, 30), 7: (-35, 70),
           8: (35, 70), 9: (-40, 105), 10: (40, 105), 11: (-18, 110), 12: (18, 110), 13: (-20, 155), 14: (20, 155),
           15: (-22, 195), 16: (22, 195)}
    for i, (dx, dy) in pts.items():
        kp[i] = (x0 + dx * h / 200, y0 + dy * h / 200, .9)
    return kp


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    region = np.zeros((1080, 1920), bool)
    region[390:600, 850:950] = True
    return image, region


def test_warp_at_the_uncoded_keypoints_reproduces_the_source():
    image, region = _scene()
    kp = _body()
    canvas = np.zeros_like(image)
    drawn = np.zeros(region.shape, bool)
    mine = k6.draw(canvas, drawn, image, region, kp, kp, 16.)
    assert mine.sum() == region.sum() and (mine == region).all()
    assert (canvas[mine] == image[mine]).all()


def test_a_rigid_shift_of_the_keypoints_moves_the_drawing_by_the_shift():
    image, region = _scene(1)
    kp = _body()
    moved = kp.copy()
    moved[:, 0] += 7
    moved[:, 1] -= 3
    canvas = np.zeros_like(image)
    drawn = np.zeros(region.shape, bool)
    mine = k6.draw(canvas, drawn, image, region, kp, moved, 16.)
    expected = np.roll(np.roll(region, 7, 1), -3, 0)
    assert (mine == expected).all()
    assert (canvas[mine] == np.roll(np.roll(image, 7, 1), -3, 0)[mine]).all()


def test_backward_map_without_bones_translates_and_without_keypoints_draws_nothing():
    src, dst = _body(), _body()
    dst[:, :2] += 5
    src[[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4], 2] = 0  # only the nose: no bone
    px, py = np.meshgrid(np.arange(3.), np.arange(2.))
    mx, my = k6.backward_map(px, py, dst, src, 10.)
    assert np.allclose(mx, px - 5) and np.allclose(my, py - 5)
    src[:, 2] = 0
    assert k6.backward_map(px, py, dst, src, 10.) is None


def test_reference_is_at_least_the_gap_away_and_nearest_in_pose():
    near_pose = _body(h=200.)
    other = _body(h=200.)
    other[9:11, 1] -= 60  # arms raised
    poses = {"p": [{"fitted": True, "kps": (near_pose if t == 30 else other).tolist()} for t in range(60)]}
    heights = {("p", t): 200. for t in range(60)}
    target = near_pose.copy()
    assert k6.choose_reference(0, "p", target, poses, list(range(60)), heights, 25) == 30
    assert k6.choose_reference(10, "p", target, poses, list(range(60)), heights, 25) == 35  # 30 is too close; ties -> nearer


def test_tracks_break_at_unfitted_or_unscored_frames_and_codes_cover_every_tracked_frame():
    kp = _body()
    seq = [{"fitted": True, "kps": kp.tolist()} for _ in range(10)]
    seq[4] = {"fitted": False}
    tracks = k6.code_tracks({"p": seq}, [t for t in range(10) if t != 7])
    assert [tr["t"] for tr in tracks] == [[0, 1, 2, 3], [5, 6], [8, 9]]
    decoded, symbols = k6.coded_keypoints(tracks)
    for c in k6.CODES:
        name = k6.code_name(c)
        assert sorted(t for _, t in decoded[name]) == [0, 1, 2, 3, 5, 6, 8, 9]
        assert symbols[name]["player_frames"] == 8
        assert np.abs(decoded[name][("p", 0)][:, :2] - kp[:, :2]).max() <= c[0] / 2 + 1e-9


def test_variant_names_and_decision_are_fixed():
    names = k6.variant_names()
    assert len(names) == len(k6.ANCHORS) + 2 * len(k6.CODES) + 1 and len(set(names)) == len(names)
    assert k6.DECISION["bound"]["ci_low_above"] == .5 and k6.DECISION["charged"]["ci_low_above"] == .25
    assert k6.CHOSEN == (4, "velocity", 4)

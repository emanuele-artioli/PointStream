"""A wide, short landmark span is the bottom-edge false positive."""

from demo.evaluation.pose_backends import is_border_strip


def test_full_width_bottom_strip_is_rejected() -> None:
    xs = [0.0, 1920.0]
    ys = [1010.0, 1050.0]
    assert is_border_strip(xs, ys, frame_w=1920, frame_h=1080)


def test_hand_sized_box_is_kept() -> None:
    xs = [800.0, 1200.0]
    ys = [400.0, 900.0]
    assert not is_border_strip(xs, ys, frame_w=1920, frame_h=1080)


def test_same_fractions_at_another_resolution() -> None:
    assert is_border_strip([0.0, 640.0], [336.0, 350.0], frame_w=640, frame_h=360)
    assert not is_border_strip([100.0, 200.0], [200.0, 300.0], frame_w=640, frame_h=360)

import numpy as np

from demo.experiments.compare_pose_on_sam_crops import choose_hand, expand_box, inside_fraction, summarize


def test_expand_box_stays_inside_the_frame():
    box = expand_box({"x": 0, "y": 10, "w": 40, "h": 80}, 100, 100, margin=0.5)
    assert box[0] == 0
    assert box[2] <= 100
    assert box[3] <= 100


def test_inside_fraction_counts_mask_hits():
    mask = np.zeros((10, 10), np.uint8)
    mask[2:5, 2:5] = 255
    points = np.array([[3, 3], [0, 0], [4, 4]])
    assert inside_fraction(points, mask) == 2 / 3


def test_choose_hand_keeps_the_low_score_that_lands_in_the_mask():
    mask = np.zeros((20, 20), np.uint8)
    mask[:, :] = 255
    good = np.stack([np.full(21, 5.0), np.full(21, 5.0)], axis=1)
    bad = np.stack([np.full(21, 100.0), np.full(21, 100.0)], axis=1)
    chosen = choose_hand(
        [("Left", good, np.full(21, 0.1)), ("Right", bad, np.full(21, 0.9))],
        mask,
    )
    assert chosen["side"] == "Left"
    assert chosen["confidence"] < 0.25


def test_ranking_prefers_joints_inside_the_crop():
    rows = [
        {"look": False, "models": {"a": {"inside": 0.9, "confidence": 0.4, "span_px": 30}, "b": {"inside": 0.1, "confidence": 0.9, "span_px": 30}}},
        {"look": True, "models": {"a": {"inside": 0.0, "confidence": 0.1, "span_px": 1}, "b": {"inside": 1.0, "confidence": 0.9, "span_px": 30}}},
    ]
    press = summarize(rows, press_only=True)
    assert press["ranking"][0] == "a"
    assert press["crops"] == 1

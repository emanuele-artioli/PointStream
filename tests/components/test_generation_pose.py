"""Pose-rescale: one shared letterbox, one hand-computable case."""

from __future__ import annotations

import numpy as np

from src.components.generation.controlnet import ControlNetGenerator
from src.components.generation.pix2pix import Pix2PixGenerator
from src.components.generation.pose import fit_to_canvas, rescale_keypoints
from src.components.generation.spade import Spade4TennisGenerator
from src.contracts.conditioning import ConditioningBundle, GenerationParams


def test_fit_to_canvas_keeps_the_mapped_box_inside_when_aspects_disagree():
    """Canvas 100x50, source 50x100: the old max/max scale overflowed.

    min(100/50, 50/100) = 0.5, so the fitted box is 25x50, centred at x=37.
    Offsets are non-negative and the paste region fits the canvas. The
    pre-rewrite copies used max(100,50)/max(50,100) = 1, producing a 50x100
    box and offset_y = -25.
    """
    box = fit_to_canvas(source_width=50, source_height=100, canvas_width=100, canvas_height=50)
    assert box.scaled_width == 25
    assert box.scaled_height == 50
    assert box.offset_x == 37
    assert box.offset_y == 0
    assert box.offset_x >= 0 and box.offset_y >= 0
    assert box.offset_x + box.scaled_width <= box.canvas_width
    assert box.offset_y + box.scaled_height <= box.canvas_height
    assert box.scale == 0.5


def test_rescale_keypoints_matches_a_hand_computed_letterbox():
    """BBox (10, 20, 60, 120) onto a 100x50 canvas.

    Origin (10, 20) → (37, 0); far corner (60, 120) → (62, 50);
    midpoint (35, 70) → (49.5, 25). Confidence is passed through.
    """
    box = fit_to_canvas(50, 100, 100, 50)
    bbox = (10, 20, 60, 120)
    keypoints = np.array(
        [
            [10.0, 20.0, 1.0],
            [60.0, 120.0, 0.7],
            [35.0, 70.0, 0.4],
        ]
    )
    mapped = rescale_keypoints(keypoints, box, bbox=bbox)
    np.testing.assert_allclose(mapped[0, :2], (37.0, 0.0))
    np.testing.assert_allclose(mapped[1, :2], (62.0, 50.0))
    np.testing.assert_allclose(mapped[2, :2], (49.5, 25.0))
    np.testing.assert_allclose(mapped[:, 2], keypoints[:, 2])


def test_rescale_keypoints_maps_every_frame_of_a_sequence_not_just_the_last():
    """The copy-pasted block took ``pose_tensor[-1]`` after mutating all frames."""
    box = fit_to_canvas(50, 100, 100, 50)
    frames = np.array(
        [
            [[10.0, 20.0, 1.0]],
            [[60.0, 120.0, 1.0]],
        ]
    )
    mapped = rescale_keypoints(frames, box, bbox=(10, 20, 60, 120))
    np.testing.assert_allclose(mapped[0, 0, :2], (37.0, 0.0))
    np.testing.assert_allclose(mapped[1, 0, :2], (62.0, 50.0))


def test_controlnet_pix2pix_and_spade_share_the_same_letterbox():
    pose = np.full((3, 100, 50), 255, dtype=np.uint8)
    appearance = np.full((3, 100, 50), 128, dtype=np.uint8)
    bundle = ConditioningBundle(appearance=appearance, pose=pose)
    params = GenerationParams(width=100, height=50)
    expected = fit_to_canvas(50, 100, 100, 50)
    prepared_cn = ControlNetGenerator(variant="pose", width=100, height=50).prepare(
        bundle, params
    )
    prepared_p2p = Pix2PixGenerator(width=100, height=50).prepare(bundle, params)
    prepared_spade = Spade4TennisGenerator(width=100, height=50).prepare(bundle, params)
    assert prepared_cn["letterbox"] == expected
    assert prepared_p2p["letterbox"] == expected
    assert prepared_spade["letterbox"] == expected
    canvas = prepared_cn["pose"]
    assert canvas[:, :37].sum() == 0
    assert canvas[:, 37 + 25 :].sum() == 0
    assert canvas[:, 37:62].sum() > 0

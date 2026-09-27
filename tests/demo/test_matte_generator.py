"""Matte head, soft-edge alpha, box hold, and DWB2 pose roundtrip."""

from __future__ import annotations

import numpy as np
import torch

from demo.models.hand_objective import composite_hand_metrics, hand_step_loss, selection_min, smooth_max
from demo.models.matte import (
    dwb2_roundtrip,
    hand_alpha_bgr,
    interpolate_hand_alphas,
    letterbox_alpha,
    soft_edge_alpha,
    stabilize_hand_boxes,
    steep_alpha,
)
from demo.models.unet_generator import HandSPADEUNet
from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand
from demo.pipeline.maps.av1_crf import AV1_LADDER, av1_output_args


def test_smooth_max_tracks_the_worse_loss() -> None:
    worse = torch.tensor(0.8)
    better = torch.tensor(0.1)
    value = smooth_max([worse, better])
    assert float(value) >= 0.8
    assert float(value) < 0.9


def test_step_loss_ignores_rgb_outside_the_hand() -> None:
    outputs = torch.zeros(1, 4, 8, 8)
    targets = torch.zeros(1, 4, 8, 8)
    targets[:, 3, 2:6, 2:6] = 1.0
    outputs[:, 0, 0, 0] = 1.0  # RGB error outside the mask
    outputs[:, 0, 3, 3] = 1.0  # RGB error on the hand
    outputs[:, 3, 0, 0] = 0.5  # alpha leak
    loss, appearance, matte = hand_step_loss(outputs, targets, lpips_fn=None)
    assert abs(float(appearance) - (1.0 / 16.0)) < 1e-5
    assert float(matte) > 0.0
    assert float(loss) >= max(float(appearance), float(matte))


def test_selection_min_rejects_a_single_regression() -> None:
    baseline = {"appearance": 0.2, "matte": 0.1, "jitter": 0.05}
    current = {"appearance": 0.1, "matte": 0.1, "jitter": 0.10}
    score, bottleneck = selection_min(current, baseline)
    assert bottleneck == "jitter"
    assert score == 0.5


def test_composite_metrics_see_leak_and_jitter() -> None:
    hand = np.zeros((8, 8), dtype=np.uint8)
    hand[2:6, 2:6] = 255
    ref = np.zeros((8, 8, 3), dtype=np.uint8)
    rec0 = ref.copy()
    rec1 = ref.copy()
    rec1[3, 3] = 255
    rec1[0, 0] = 255
    bg = np.zeros_like(ref)
    metrics = composite_hand_metrics([ref, ref], [rec0, rec1], [bg, bg], [hand, hand])
    assert metrics["appearance"] > 0.0
    assert metrics["matte"] > 0.0
    assert metrics["jitter"] > 0.0


def test_spade_emits_rgb_and_alpha() -> None:
    model = HandSPADEUNet(in_channels=6, out_channels=4)
    out = model(torch.zeros(1, 6, 256, 256))
    assert out.shape == (1, 4, 256, 256)
    assert float(out[:, :3].min()) >= -1.0
    assert float(out[:, :3].max()) <= 1.0
    assert float(out[:, 3].min()) >= 0.0
    assert float(out[:, 3].max()) <= 1.0


def test_hand_alpha_keeps_red_and_drops_green() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :4] = (40, 40, 255)  # BGR red hand
    frame[:, 4:] = (40, 210, 70)  # BGR green tool
    alpha = hand_alpha_bgr(frame)
    assert int(alpha[:, :4].min()) == 255
    assert int(alpha[:, 4:].max()) == 0


def test_letterbox_alpha_stays_binary() -> None:
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[5:15, 10:30] = 255
    boxed = letterbox_alpha(mask, [10, 5, 30, 15], target_size=32)
    assert boxed.shape == (32, 32)
    assert set(np.unique(boxed).tolist()) <= {0, 255}


def test_soft_edge_has_falloff_and_zero_outside() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    soft = soft_edge_alpha(mask, falloff_px=2.0)
    assert soft.shape == (32, 32)
    assert float(soft[0, 0]) == 0.0
    assert float(soft[16, 16]) == 1.0
    # Boundary pixels are partial.
    assert 0.0 < float(soft[8, 16]) < 1.0 or 0.0 < float(soft[9, 16]) < 1.0


def test_steep_alpha_crushes_speckles() -> None:
    a = np.array([0.0, 0.2, 0.5, 1.0], dtype=np.float32)
    steep = steep_alpha(a, gamma=4.0)
    assert float(steep[0]) == 0.0
    assert float(steep[1]) < 0.01
    assert float(steep[3]) == 1.0


def test_interpolate_fills_short_gap() -> None:
    a = np.zeros((8, 8), dtype=np.uint8)
    a[2:6, 2:6] = 255
    empty = np.zeros((8, 8), dtype=np.uint8)
    seq = [a, empty, empty, a]
    filled = interpolate_hand_alphas(seq, min_hand_px=10, max_gap=4)
    assert int(np.count_nonzero(filled[1])) > 0
    assert int(np.count_nonzero(filled[2])) > 0


def test_stabilize_holds_size_and_follows_center() -> None:
    def hand(bbox):
        return SingleHand("Right", 0.9, bbox, [[0.1, 0.2, 0.0]] * 21, [[120.0, 220.0]] * 21)

    wobble = [
        FrameHandPose(0, [hand([100, 200, 180, 320])]),
        FrameHandPose(1, [hand([101, 200, 181, 320])]),
    ]
    held = stabilize_hand_boxes(wobble, 1920, 1080)
    assert held[0].hands[0].bbox == held[1].hands[0].bbox

    # A tenth-or-more change in span updates the size. 80 px wide -> 100 px is +25%.
    grown = [
        FrameHandPose(0, [hand([100, 200, 180, 320])]),
        FrameHandPose(1, [hand([90, 200, 190, 320])]),
    ]
    grown_held = stabilize_hand_boxes(grown, 1920, 1080)
    assert grown_held[1].hands[0].bbox[2] - grown_held[1].hands[0].bbox[0] == 100

    # Size stays fixed while the center follows a translation.
    moved = [
        FrameHandPose(0, [hand([100, 200, 180, 320])]),
        FrameHandPose(1, [hand([140, 200, 220, 320])]),
    ]
    moved_held = stabilize_hand_boxes(moved, 1920, 1080, smooth=1.0)
    box = moved_held[1].hands[0].bbox
    assert box[2] - box[0] == 80
    assert box[0] == 140


def test_dwb2_roundtrip_keeps_a_hand() -> None:
    hand = SingleHand(
        handedness="Right",
        confidence=0.9,
        bbox=[100, 200, 180, 320],
        landmarks_norm=[[0.1, 0.2, 0.0]] * 21,
        landmarks_pixel=[[120.0 + i, 220.0 + i] for i in range(21)],
    )
    poses = [FrameHandPose(frame_idx=0, hands=[hand]), FrameHandPose(frame_idx=1, hands=[hand])]
    decoded, nbytes = dwb2_roundtrip(poses, 1920, 1080)
    assert nbytes > 0
    assert len(decoded) == 2
    assert decoded[0].hands[0].handedness == "Right"
    assert len(decoded[0].hands[0].landmarks_pixel) == 21


def test_av1_ladder_covers_180_to_1080() -> None:
    names = [n for n, _ in AV1_LADDER]
    assert names == ["180p", "240p", "360p", "540p", "720p", "1080p"]
    native = av1_output_args(None)
    assert "-crf" in native and "63" in native
    assert "-vf" not in native
    scaled = av1_output_args("320:180")
    assert "scale=320:180" in scaled

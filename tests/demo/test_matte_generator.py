"""Matte head and DWB2 pose roundtrip used by the hand generator."""

from __future__ import annotations

import numpy as np
import torch

from demo.models.matte import dwb2_roundtrip, hand_alpha_bgr, letterbox_alpha
from demo.models.unet_generator import HandSPADEUNet
from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand


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

from __future__ import annotations

from pathlib import Path

import pytest

from src.encoder.video_io import decode_video_to_tensor
from src.shared.lpips_metric import compute_lpips_from_frames, default_weights_path

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _require_vgg_weights() -> None:
    if not default_weights_path().exists():
        pytest.skip(f"VGG-19-bn weights not found at {default_weights_path()}")


def test_lpips_between_identical_videos_is_near_zero(real_tennis_10f_video: Path) -> None:
    _require_vgg_weights()

    decoded = decode_video_to_tensor(real_tennis_10f_video)
    frames_rgb01 = decoded.tensor.flip(dims=[1])  # BGR -> RGB, Shape: [Frames, 3, H, W]

    result = compute_lpips_from_frames(frames_rgb01, frames_rgb01)

    assert result["lpips_vgg_uncalibrated"] is not None
    assert result["lpips_vgg_uncalibrated"] == pytest.approx(0.0, abs=1e-5)


def test_lpips_between_real_and_shuffled_frames_is_nonzero(real_tennis_10f_video: Path) -> None:
    _require_vgg_weights()

    decoded = decode_video_to_tensor(real_tennis_10f_video)
    frames_rgb01 = decoded.tensor.flip(dims=[1])
    if frames_rgb01.shape[0] < 2:
        pytest.skip("need at least 2 frames to reorder")
    reversed_frames = frames_rgb01.flip(dims=[0])

    result = compute_lpips_from_frames(frames_rgb01, reversed_frames)

    assert result["lpips_vgg_uncalibrated"] is not None
    assert result["lpips_vgg_uncalibrated"] > 0.0

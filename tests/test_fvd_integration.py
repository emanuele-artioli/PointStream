from __future__ import annotations

from pathlib import Path

import pytest

from src.encoder.video_io import decode_video_to_tensor, encode_video_frames_ffmpeg, iter_video_frames_ffmpeg, probe_video_metadata
from src.shared.fvd import compute_fvd_from_frames, default_weights_path


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _require_i3d_weights() -> None:
    if not default_weights_path().exists():
        pytest.skip(f"I3D weights not found at {default_weights_path()}")


def test_fvd_between_identical_videos_is_near_zero(real_tennis_10f_video: Path) -> None:
    _require_i3d_weights()

    decoded = decode_video_to_tensor(real_tennis_10f_video)

    result = compute_fvd_from_frames(decoded.tensor, decoded.tensor)

    assert result["fvd"] is not None
    assert result["fvd"] == pytest.approx(0.0, abs=1e-3)


def test_fvd_between_real_and_degraded_video_is_clearly_higher(
    real_tennis_10f_video: Path,
    test_run_artifacts_dir: Path,
) -> None:
    _require_i3d_weights()

    metadata = probe_video_metadata(real_tennis_10f_video)
    degraded_path = test_run_artifacts_dir / "fvd_degraded.mp4"
    degraded_path.unlink(missing_ok=True)

    def _degrade(frame):
        import cv2
        import numpy as np

        small = cv2.resize(frame, (max(1, metadata.width // 16), max(1, metadata.height // 16)))
        blurred = cv2.resize(small, (metadata.width, metadata.height), interpolation=cv2.INTER_NEAREST)
        noise = np.random.default_rng(0).integers(0, 60, size=blurred.shape, dtype=np.uint8)
        return np.clip(blurred.astype(np.int16) + noise.astype(np.int16), 0, 255).astype("uint8")

    degraded_frames = [
        _degrade(frame)
        for frame in iter_video_frames_ffmpeg(real_tennis_10f_video, width=metadata.width, height=metadata.height)
    ]
    encode_video_frames_ffmpeg(
        degraded_path,
        degraded_frames,
        fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
        codec="libx264",
        crf=40,
    )

    reference_decoded = decode_video_to_tensor(real_tennis_10f_video)
    predicted_decoded = decode_video_to_tensor(degraded_path)

    self_result = compute_fvd_from_frames(reference_decoded.tensor, reference_decoded.tensor)
    degraded_result = compute_fvd_from_frames(reference_decoded.tensor, predicted_decoded.tensor)

    assert self_result["fvd"] is not None
    assert degraded_result["fvd"] is not None
    assert degraded_result["fvd"] > self_result["fvd"]
    # Non-degenerate: a heavily-blurred, noised copy should register a clearly
    # nonzero perceptual distance, not a rounding-error-scale value.
    assert degraded_result["fvd"] > 1.0

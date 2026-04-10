from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.decoder.genai_compositor import DiffusersCompositor
from src.encoder.video_io import encode_video_frames_ffmpeg


pytestmark = pytest.mark.skipif(
    os.environ.get("POINTSTREAM_ENABLE_GENAI", "0").strip() != "1",
    reason="GenAI node test requires POINTSTREAM_ENABLE_GENAI=1",
)


def test_diffusers_compositor_two_frame_smoke() -> None:
    compositor = DiffusersCompositor(seed=2026)

    frame_h, frame_w = 180, 320
    reference_crop = torch.full((3, 128, 64), 180, dtype=torch.uint8)

    dense_pose_sequence = torch.zeros((2, 18, 3), dtype=torch.float32)
    for frame_idx in range(2):
        dense_pose_sequence[frame_idx, :, 0] = torch.linspace(130.0 + frame_idx * 4.0, 180.0 + frame_idx * 4.0, 18)
        dense_pose_sequence[frame_idx, :, 1] = torch.linspace(45.0 + frame_idx * 2.0, 150.0 + frame_idx * 2.0, 18)
        dense_pose_sequence[frame_idx, :, 2] = 0.95

    out_frames: list[np.ndarray] = []
    for frame_idx in range(2):
        background = torch.zeros((3, frame_h, frame_w), dtype=torch.uint8)
        output = compositor.process(
            reference_crop_tensor=reference_crop,
            dense_dwpose_tensor=dense_pose_sequence[frame_idx : frame_idx + 1],
            warped_background_frame=background,
        )
        out_frames.append(np.asarray(output.permute(1, 2, 0).cpu().numpy(), dtype=np.uint8))

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "assets" / "debug_genai_composite.mp4"
    output_path.unlink(missing_ok=True)
    encode_video_frames_ffmpeg(
        output_path=output_path,
        frames_bgr=out_frames,
        fps=10.0,
        width=frame_w,
        height=frame_h,
        codec="libx264",
        pix_fmt="yuv420p",
        crf=18,
        preset="veryfast",
    )

    assert output_path.exists()

from __future__ import annotations

from pathlib import Path

import torch

import src.decoder.genai_compositor as gc
from src.encoder.residual_calculator import ResidualCalculator
from tests.video_utils import create_dummy_video


def test_residual_calculator_uses_genai_compositor_when_enabled(
    monkeypatch,
    mock_encoder_pipeline,
    test_run_artifacts_dir: Path,
) -> None:
    monkeypatch.setenv("POINTSTREAM_ENABLE_GENAI", "1")
    monkeypatch.setenv("POINTSTREAM_GENAI_BACKEND", "animate-anyone")

    calls = {"count": 0}

    def _fake_process(
        self,
        reference_crop_tensor,
        dense_dwpose_tensor,
        warped_background_frame,
        actor_identity=None,
        metadata_mask=None,
        metadata_bbox=None,
    ):
        _ = (reference_crop_tensor, dense_dwpose_tensor, actor_identity, metadata_mask, metadata_bbox)
        calls["count"] += 1
        out = warped_background_frame.to(torch.int16) + 1
        return torch.clamp(out, 0, 255).to(torch.uint8)

    monkeypatch.setattr(gc.DiffusersCompositor, "process", _fake_process)

    video_path = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "residual_genai_sync.mp4",
        num_frames=6,
        width=160,
        height=96,
        fps=24.0,
    )

    payload, _decoded, frame_states = mock_encoder_pipeline.encode_video_file_with_states(
        video_path=video_path,
        chunk_id="residual_genai_sync_0001",
        start_frame_id=0,
    )

    calculator = ResidualCalculator(device="cpu")
    compositor = calculator._synthesis_engine.get_genai_compositor()
    assert isinstance(compositor, gc.DiffusersCompositor)
    assert compositor.uses_temporal_pose_sequence() is True

    output_path = test_run_artifacts_dir / "debug_residual_genai_sync.mp4"
    output_path.unlink(missing_ok=True)

    residual_packet = calculator.process(
        chunk=payload.chunk,
        payload=payload,
        frame_states=frame_states,
        debug_output_path=output_path,
    )

    assert residual_packet.residual_video_uri == str(output_path)
    assert output_path.exists()
    assert calls["count"] > 0

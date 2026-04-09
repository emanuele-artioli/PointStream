from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.residual_calculator import BinaryActorImportanceMapper, ResidualCalculator
from src.encoder.video_io import iter_video_frames_ffmpeg, probe_video_metadata
from src.transport.disk import DiskTransport


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_weighted_residual_debug_video_has_sparse_actor_regions(
    real_encoder_pipeline,
    real_tennis_10f_video: Path,
    test_run_artifacts_dir: Path,
) -> None:
    debug_output = test_run_artifacts_dir / "debug_residual.mp4"
    debug_output.unlink(missing_ok=True)

    payload, _decoded, frame_states = real_encoder_pipeline.encode_video_file_with_states(
        video_path=real_tennis_10f_video,
        chunk_id="residual_real_0001",
        max_frames=10,
    )

    calculator = ResidualCalculator(importance_mapper=BinaryActorImportanceMapper())
    residual_packet = calculator.process(
        chunk=payload.chunk,
        payload=payload,
        frame_states=frame_states,
        debug_output_path=debug_output,
    )

    assert residual_packet.residual_video_uri == str(debug_output)
    assert debug_output.exists()

    metadata = probe_video_metadata(debug_output)
    assert metadata.num_frames == payload.chunk.num_frames
    assert metadata.width == payload.chunk.width
    assert metadata.height == payload.chunk.height

    frames = list(
        iter_video_frames_ffmpeg(
            debug_output,
            width=metadata.width,
            height=metadata.height,
        )
    )
    assert len(frames) == payload.chunk.num_frames

    stacked = np.stack(frames, axis=0)
    assert int(np.max(stacked)) > 0

    # Signed residual uses neutral gray (128) in ignored regions and sparse offsets in ROI.
    neutral_deviation = np.abs(stacked.astype(np.int16) - 128)
    active_ratio = float(np.count_nonzero(neutral_deviation > 8)) / float(neutral_deviation.size)
    assert active_ratio > 0.001
    assert active_ratio < 0.05


def test_full_codec_loop_composites_signed_residual_into_final_reconstruction(
    real_encoder_pipeline,
    real_tennis_10f_video: Path,
    test_run_artifacts_dir: Path,
) -> None:
    final_output = test_run_artifacts_dir / "debug_final_reconstruction.mp4"
    final_output.unlink(missing_ok=True)

    payload, _decoded, _frame_states = real_encoder_pipeline.encode_video_file_with_states(
        video_path=real_tennis_10f_video,
        chunk_id="debug_final_reconstruction",
        max_frames=10,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        transport = DiskTransport(root_dir=tmp_dir)
        transport.send(payload)
        recovered_payload = transport.receive("debug_final_reconstruction")

        decoder = DecoderRenderer(output_root=test_run_artifacts_dir)
        decoded = decoder.process(recovered_payload, output_path=final_output)

    assert decoded.output_uri == str(final_output)
    assert final_output.exists()

    recon_meta = probe_video_metadata(final_output)
    assert recon_meta.num_frames == 10
    assert recon_meta.width == payload.chunk.width
    assert recon_meta.height == payload.chunk.height

    source_frames = list(
        iter_video_frames_ffmpeg(
            real_tennis_10f_video,
            width=payload.chunk.width,
            height=payload.chunk.height,
        )
    )
    recon_frames = list(
        iter_video_frames_ffmpeg(
            final_output,
            width=payload.chunk.width,
            height=payload.chunk.height,
        )
    )

    assert len(source_frames) == len(recon_frames) == 10
    source_np = np.stack(source_frames, axis=0).astype(np.float32)
    recon_np = np.stack(recon_frames, axis=0).astype(np.float32)
    mae = float(np.mean(np.abs(source_np - recon_np)))
    assert mae < 35.0

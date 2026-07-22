from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

from src.decoder.decoder_renderer import DecoderRenderer
from src.encoder.residual_calculator import ResidualCalculator
from src.shared.config import PointstreamConfig
from src.shared.schemas import ResidualMode
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

    calculator = ResidualCalculator(
        config=PointstreamConfig(),
        device="cpu",
        residual_mode=ResidualMode.PLAYERS_ONLY,
    )
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
    # Real-model detections and H.265 quantization can vary across environments.
    # Keep a non-zero lower bound without making the test flaky.
    assert active_ratio > 5e-05
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

    decoded_dir = Path(decoded.output_uri)
    assert decoded_dir == final_output.with_suffix("")
    assert decoded_dir.is_dir()

    frame_paths = sorted(decoded_dir.glob("frame_*.png"))
    assert len(frame_paths) == 10

    source_frames = list(
        iter_video_frames_ffmpeg(
            real_tennis_10f_video,
            width=payload.chunk.width,
            height=payload.chunk.height,
        )
    )
    recon_frames = [cv2.imread(str(frame_path), cv2.IMREAD_COLOR) for frame_path in frame_paths]

    assert len(source_frames) == len(recon_frames) == 10
    source_np = np.stack(source_frames, axis=0).astype(np.float32)
    recon_np = np.stack(recon_frames, axis=0).astype(np.float32)
    mae = float(np.mean(np.abs(source_np - recon_np)))
    assert mae < 35.0


def test_residual_calculator_does_not_seek_into_source_by_start_frame_id(
    real_encoder_pipeline,
    real_tennis_20f_video: Path,
    test_run_artifacts_dir: Path,
) -> None:
    """chunk.source_uri is a self-contained per-chunk file (frame 0 of the file is the
    chunk's first frame); start_frame_id only labels emitted frame_id numbers, it must
    never be used to seek into source_uri. Two chunks pointed at the SAME 20-frame source
    file but carrying different start_frame_id must therefore read/align to the identical
    underlying frames -- exactly as ActorExtractor already does when detecting actors.
    A seek-offset regression here silently detects actors on one frame slice while
    computing the residual against a different, shifted slice.
    """
    metadata = probe_video_metadata(real_tennis_20f_video)
    frames = list(
        iter_video_frames_ffmpeg(real_tennis_20f_video, width=metadata.width, height=metadata.height)
    )
    assert len(frames) == 20

    # Sanity check: frames 8 apart in this clip are genuinely different, not a uniform dummy.
    frame_gap_diff = float(np.mean(np.abs(frames[0].astype(np.float32) - frames[8].astype(np.float32))))
    assert frame_gap_diff > 5.0

    payload_a, _decoded_a, frame_states_a = real_encoder_pipeline.encode_video_file_with_states(
        video_path=real_tennis_20f_video,
        chunk_id="align_start0",
        start_frame_id=0,
        max_frames=10,
    )
    payload_b, _decoded_b, frame_states_b = real_encoder_pipeline.encode_video_file_with_states(
        video_path=real_tennis_20f_video,
        chunk_id="align_start8",
        start_frame_id=8,
        max_frames=10,
    )

    calculator = ResidualCalculator(
        config=PointstreamConfig(),
        device="cpu",
        background_block_downscale_factor=None,
    )

    out_a = test_run_artifacts_dir / "debug_residual_align_start0.mp4"
    out_b = test_run_artifacts_dir / "debug_residual_align_start8.mp4"
    out_a.unlink(missing_ok=True)
    out_b.unlink(missing_ok=True)

    calculator.process(chunk=payload_a.chunk, payload=payload_a, frame_states=frame_states_a, debug_output_path=out_a)
    calculator.process(chunk=payload_b.chunk, payload=payload_b, frame_states=frame_states_b, debug_output_path=out_b)

    frames_a = list(iter_video_frames_ffmpeg(out_a, width=payload_a.chunk.width, height=payload_a.chunk.height))
    frames_b = list(iter_video_frames_ffmpeg(out_b, width=payload_b.chunk.width, height=payload_b.chunk.height))
    assert len(frames_a) == len(frames_b) == 10

    stacked_a = np.stack(frames_a, axis=0).astype(np.float32)
    stacked_b = np.stack(frames_b, axis=0).astype(np.float32)

    # Both chunks share the exact same source file and num_frames, so a correct
    # implementation reads source frames 0..9 in both cases regardless of start_frame_id,
    # producing near-identical residuals (only lossy-codec noise between the two runs).
    # A seek-by-start_frame_id regression instead reads frames 8..17 for the second chunk,
    # producing a residual dominated by the genuine 8-frame content gap asserted above.
    mean_abs_diff = float(np.mean(np.abs(stacked_a - stacked_b)))
    assert mean_abs_diff < 3.0

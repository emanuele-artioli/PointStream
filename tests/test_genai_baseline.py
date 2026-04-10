from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.mock_extractors import MockActorExtractor
from src.encoder.reference_extractor import ReferenceExtractor
from src.encoder.video_io import iter_video_frames_ffmpeg, probe_video_metadata
from src.shared.schemas import VideoChunk
from src.transport.disk import DiskTransport
from tests.video_utils import create_dummy_video


def test_reference_extractor_emits_jpeg_per_player_track(test_run_artifacts_dir: Path) -> None:
    video_path = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "genai_ref_extract.mp4",
        num_frames=10,
        width=320,
        height=180,
        fps=30.0,
    )
    metadata = probe_video_metadata(video_path)
    chunk = VideoChunk(
        chunk_id="genai_ref_extract_0001",
        source_uri=str(video_path),
        start_frame_id=0,
        fps=metadata.fps,
        num_frames=metadata.num_frames,
        width=metadata.width,
        height=metadata.height,
    )

    actor_bundle = MockActorExtractor().process_with_states(chunk)
    references = ReferenceExtractor().process(chunk=chunk, frame_states=actor_bundle.frame_states)

    assert len(references) == 2
    for reference in references:
        assert isinstance(reference.track_id, int)
        assert len(reference.reference_crop_jpeg) > 64

        decoded = cv2.imdecode(np.frombuffer(reference.reference_crop_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert int(decoded.size) > 0


def test_actor_reference_jpegs_roundtrip_over_disk_transport(
    mock_encoder_pipeline,
    test_run_artifacts_dir: Path,
) -> None:
    video_path = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "genai_ref_transport.mp4",
        num_frames=8,
        width=320,
        height=180,
        fps=30.0,
    )

    payload, _decoded = mock_encoder_pipeline.encode_video_file(
        video_path=video_path,
        chunk_id="genai_ref_transport_0001",
        start_frame_id=0,
    )
    assert len(payload.actor_references) == 2

    before = {ref.track_id: ref.reference_crop_jpeg for ref in payload.actor_references}

    with tempfile.TemporaryDirectory() as tmp_dir:
        transport = DiskTransport(root_dir=tmp_dir)
        transport.send(payload)
        recovered = transport.receive("genai_ref_transport_0001")

    after = {ref.track_id: ref.reference_crop_jpeg for ref in recovered.actor_references}
    assert sorted(after.keys()) == sorted(before.keys())
    for track_id, jpeg_bytes in before.items():
        assert after[track_id] == jpeg_bytes


def test_decoder_mock_genai_stage_uses_transmitted_reference_crops(
    mock_encoder_pipeline,
    test_run_artifacts_dir: Path,
) -> None:
    video_path = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "genai_ref_decode.mp4",
        num_frames=12,
        width=320,
        height=180,
        fps=30.0,
    )

    payload, _decoded = mock_encoder_pipeline.encode_video_file(
        video_path=video_path,
        chunk_id="genai_ref_decode_0001",
        start_frame_id=0,
    )
    assert len(payload.actor_references) == 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        transport = DiskTransport(root_dir=tmp_dir)
        transport.send(payload)
        recovered_payload = transport.receive("genai_ref_decode_0001")

    renderer = DecoderRenderer(output_root=test_run_artifacts_dir)
    with_refs_output = test_run_artifacts_dir / "debug_final_reconstruction.mp4"
    without_refs_output = test_run_artifacts_dir / "debug_final_reconstruction_no_refs.mp4"
    with_refs_output.unlink(missing_ok=True)
    without_refs_output.unlink(missing_ok=True)

    decoded_with = renderer.process(recovered_payload, output_path=with_refs_output)
    payload_without_refs = recovered_payload.model_copy(update={"actor_references": []})
    decoded_without = renderer.process(payload_without_refs, output_path=without_refs_output)

    assert decoded_with.output_uri == str(with_refs_output)
    assert decoded_without.output_uri == str(without_refs_output)
    assert with_refs_output.exists()
    assert without_refs_output.exists()

    frames_with = list(
        iter_video_frames_ffmpeg(
            with_refs_output,
            width=payload.chunk.width,
            height=payload.chunk.height,
        )
    )
    frames_without = list(
        iter_video_frames_ffmpeg(
            without_refs_output,
            width=payload.chunk.width,
            height=payload.chunk.height,
        )
    )

    assert len(frames_with) == len(frames_without) == int(payload.chunk.num_frames)

    changed_pixel_counts: list[int] = []
    for frame_idx in range(min(3, len(frames_with))):
        frame_diff = np.abs(
            frames_with[frame_idx].astype(np.int16) - frames_without[frame_idx].astype(np.int16)
        )
        per_pixel_delta = np.sum(frame_diff, axis=2)
        changed_pixel_counts.append(int(np.count_nonzero(per_pixel_delta > 25)))

    # Use a sparse high-delta metric so codec noise does not make this flaky on CI.
    assert max(changed_pixel_counts) >= 300

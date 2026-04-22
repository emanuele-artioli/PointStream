from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.mock_extractors import MockActorExtractor
from src.encoder.reference_extractor import ReferenceExtractor
from src.encoder.video_io import encode_video_frames_ffmpeg, iter_video_frames_ffmpeg, probe_video_metadata
from src.shared.schemas import FrameState, SceneActor, VideoChunk
from src.transport.disk import DiskTransport
from tests.video_utils import create_dummy_video


def test_reference_extractor_prefers_first_confident_observation(test_run_artifacts_dir: Path) -> None:
    video_path = test_run_artifacts_dir / "test_chunks" / "first_confident_reference.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    frame0 = np.zeros((72, 96, 3), dtype=np.uint8)
    frame1 = np.zeros((72, 96, 3), dtype=np.uint8)
    frame2 = np.zeros((72, 96, 3), dtype=np.uint8)

    frame0[10:38, 10:28] = np.array([255, 0, 0], dtype=np.uint8)
    frame1[8:48, 8:40] = np.array([0, 255, 0], dtype=np.uint8)
    frame2[12:42, 10:30] = np.array([0, 0, 255], dtype=np.uint8)

    encode_video_frames_ffmpeg(
        output_path=video_path,
        frames_bgr=[frame0, frame1, frame2],
        fps=24.0,
        width=96,
        height=72,
        codec="libx264",
        pix_fmt="yuv420p",
        crf=18,
        preset="veryfast",
    )

    chunk = VideoChunk(
        chunk_id="first_confident_reference",
        source_uri=str(video_path),
        start_frame_id=0,
        fps=24.0,
        num_frames=3,
        width=96,
        height=72,
    )
    mask = np.ones((24, 16), dtype=np.uint8).tolist()
    frame_states = [
        FrameState(
            frame_id=0,
            actors=[SceneActor(track_id="player_1", class_name="player", bbox=[10, 10, 28, 38], mask=mask)],
        ),
        FrameState(
            frame_id=1,
            actors=[SceneActor(track_id="player_1", class_name="player", bbox=[8, 8, 40, 48], mask=mask)],
        ),
        FrameState(
            frame_id=2,
            actors=[SceneActor(track_id="player_1", class_name="player", bbox=[10, 12, 30, 42], mask=mask)],
        ),
    ]

    references = ReferenceExtractor().process(chunk=chunk, frame_states=frame_states)
    assert len(references) == 1
    assert references[0].reference_crop_jpeg is not None

    decoded = cv2.imdecode(
        np.frombuffer(references[0].reference_crop_jpeg, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None
    mean_bgr = np.mean(decoded, axis=(0, 1))
    assert float(mean_bgr[0]) > float(mean_bgr[1])


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
        assert reference.reference_crop_jpeg is not None
        jpeg_bytes = reference.reference_crop_jpeg
        assert len(jpeg_bytes) > 64

        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
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

    before: dict[int, bytes] = {}
    for ref in payload.actor_references:
        assert ref.reference_crop_jpeg is not None
        before[ref.track_id] = ref.reference_crop_jpeg

    with tempfile.TemporaryDirectory() as tmp_dir:
        transport = DiskTransport(root_dir=tmp_dir)
        transport.send(payload)
        recovered = transport.receive("genai_ref_transport_0001")

        after = {ref.track_id: ref.reference_crop_uri for ref in recovered.actor_references}
        assert sorted(after.keys()) == sorted(before.keys())
        for track_id, jpeg_bytes in before.items():
            reference_uri = after[track_id]
            assert reference_uri is not None
            reference_path = Path(str(reference_uri))
            assert reference_path.exists()
            assert reference_path.read_bytes() == jpeg_bytes


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
        assert len(frames_with) == int(payload.chunk.num_frames)

        # Compare pre-encode tensors to avoid FFmpeg codec variability across CI runners.
        renderer_compare = DecoderRenderer(output_root=test_run_artifacts_dir)
        actor_state_with = renderer_compare._build_actor_state(recovered_payload)
        actor_state_without = renderer_compare._build_actor_state(payload_without_refs)

        assert actor_state_with
        assert not actor_state_without

        synthesis_with = renderer_compare._synthesis_engine.synthesize(
            recovered_payload,
            include_guidance_overlays=False,
        )
        base_with = renderer_compare._compositor.composite(
            predicted_frames=synthesis_with.frames_bgr,
            residual_video_uri=recovered_payload.residual.residual_video_uri,
            width=int(recovered_payload.chunk.width),
            height=int(recovered_payload.chunk.height),
        )
        renderer_compare._actor_state = actor_state_with
        final_with = renderer_compare._render_genai_baseline(base_with)

        synthesis_without = renderer_compare._synthesis_engine.synthesize(
            payload_without_refs,
            include_guidance_overlays=False,
        )
        base_without = renderer_compare._compositor.composite(
            predicted_frames=synthesis_without.frames_bgr,
            residual_video_uri=payload_without_refs.residual.residual_video_uri,
            width=int(payload_without_refs.chunk.width),
            height=int(payload_without_refs.chunk.height),
        )
        renderer_compare._actor_state = actor_state_without
        final_without = renderer_compare._render_genai_baseline(base_without)

        assert final_with.shape == final_without.shape
        assert not torch.equal(final_with, final_without)

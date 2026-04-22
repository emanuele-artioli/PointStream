from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.decoder.mock_renderer import DecoderRenderer, _ClientActorState
from src.encoder.video_io import encode_video_frames_ffmpeg, probe_video_metadata
from src.shared.schemas import ActorMaskFrame
from src.shared.synthesis_engine import SynthesisEngine
from tests.video_utils import create_dummy_video


def test_decoder_output_matches_chunk_dimensions(mock_encoder_pipeline, test_run_artifacts_dir: Path) -> None:
    reconstruction_path = test_run_artifacts_dir / "mock_reconstruction.mp4"
    reconstruction_path.unlink(missing_ok=True)

    video_path = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "dec001.mp4",
        num_frames=6,
        width=640,
        height=360,
        fps=30.0,
    )

    payload, _decoded = mock_encoder_pipeline.encode_video_file(
        video_path=video_path,
        chunk_id="dec001",
        start_frame_id=0,
    )

    engine = SynthesisEngine(seed=2026)
    synthesis = engine.synthesize(payload)
    synthesis_repeat = engine.synthesize(payload)
    assert np.array_equal(
        synthesis.frames_bgr.cpu().numpy(),
        synthesis_repeat.frames_bgr.cpu().numpy(),
    )
    frames = [
        np.asarray(frame.permute(1, 2, 0).cpu().numpy(), dtype=np.uint8)
        for frame in synthesis.frames_bgr
    ]
    encode_video_frames_ffmpeg(
        output_path=reconstruction_path,
        frames_bgr=frames,
        fps=payload.chunk.fps,
        width=payload.chunk.width,
        height=payload.chunk.height,
        codec="libx264",
        pix_fmt="yuv420p",
        crf=18,
        preset="veryfast",
    )

    rendered_meta = probe_video_metadata(reconstruction_path)
    assert rendered_meta.num_frames == payload.chunk.num_frames
    assert rendered_meta.width == payload.chunk.width
    assert rendered_meta.height == payload.chunk.height

    decoded = DecoderRenderer(output_root=test_run_artifacts_dir).process(payload)

    assert decoded.chunk_id == "dec001"
    assert decoded.num_frames == 6
    assert decoded.width == 640
    assert decoded.height == 360
    assert decoded.output_uri.endswith("dec001.mp4")
    assert Path(decoded.output_uri).parent == test_run_artifacts_dir
    assert reconstruction_path.exists()


class _TemporalSpyCompositor:
    def __init__(self) -> None:
        self.window_lengths: list[int] = []

    def uses_temporal_pose_sequence(self) -> bool:
        return True

    def process(
        self,
        reference_crop_tensor,
        dense_dwpose_tensor,
        warped_background_frame,
        actor_identity=None,
        metadata_mask=None,
    ):
        _ = reference_crop_tensor
        _ = actor_identity
        _ = metadata_mask
        self.window_lengths.append(int(dense_dwpose_tensor.shape[0]))
        return warped_background_frame


def test_render_genai_baseline_uses_preroll_and_fixed_temporal_window(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("POINTSTREAM_ANIMATE_ANYONE_WINDOW", "4")
    monkeypatch.setenv("POINTSTREAM_GENAI_PREROLL_FRAMES", "1")

    renderer = DecoderRenderer(output_root=tmp_path)
    spy = _TemporalSpyCompositor()
    setattr(renderer, "_genai_compositor", spy)

    reference_crop = torch.full((3, 24, 16), 128, dtype=torch.uint8)
    dense_pose = torch.zeros((3, 18, 3), dtype=torch.float32)
    dense_pose[:, :, 0] = torch.linspace(30.0, 70.0, 18)
    dense_pose[:, :, 1] = torch.linspace(20.0, 90.0, 18)
    dense_pose[:, :, 2] = 0.9

    renderer._actor_state = {
        1: _ClientActorState(
            track_id=1,
            object_id="person_1",
            reference_crop_tensor=reference_crop,
            dense_pose_tensor=dense_pose,
        )
    }

    frame_tensor = torch.zeros((3, 3, 96, 160), dtype=torch.uint8)
    out = renderer._render_genai_baseline(frame_tensor)

    assert torch.equal(out, frame_tensor)
    assert spy.window_lengths == [4, 4]


def test_decoder_actor_mask_decode_supports_legacy_png_payload(mock_encoder_pipeline, test_run_artifacts_dir: Path) -> None:
    video_path = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "legacy_mask_decode.mp4",
        num_frames=4,
        width=160,
        height=96,
        fps=24.0,
    )
    payload, _decoded = mock_encoder_pipeline.encode_video_file(
        video_path=video_path,
        chunk_id="legacy_mask_decode",
        start_frame_id=0,
    )

    assert payload.actors
    actor_packet = payload.actors[0]

    legacy_mask = np.zeros((24, 16), dtype=np.uint8)
    legacy_mask[4:20, 3:13] = 255
    ok, encoded = cv2.imencode(".png", legacy_mask)
    assert ok

    actor_packet.mask_frames = [
        ActorMaskFrame(
            frame_id=int(payload.chunk.start_frame_id),
            bbox=[0, 0, 16, 24],
            mask_codec="png",
            mask_payload=None,
            mask_height=None,
            mask_width=None,
            mask_png=encoded.tobytes(),
            source="source",
        )
    ]

    renderer = DecoderRenderer(output_root=test_run_artifacts_dir)
    decoded_masks = renderer._decode_actor_masks(actor_packet)

    assert int(payload.chunk.start_frame_id) in decoded_masks
    decoded_entry = decoded_masks[int(payload.chunk.start_frame_id)]
    assert decoded_entry.bbox == (0, 0, 16, 24)
    decoded_mask = decoded_entry.mask_gray
    assert decoded_mask.shape == (24, 16)
    assert int(np.count_nonzero(decoded_mask)) > 0

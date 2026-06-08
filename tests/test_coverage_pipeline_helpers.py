from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import types
from typing import Any, cast

import cv2
import numpy as np
import pytest
import torch

from src.encoder import orchestrator
from src.encoder import residual_calculator as residual_calc
from src.shared.config import PointstreamConfig
from src.shared.schemas import (
    ActorMaskFrame,
    ActorPacket,
    CameraPose,
    EncodedChunkPayload,
    FrameState,
    KeyframeEvent,
    ObjectClass,
    PanoramaPacket,
    ResidualMode,
    ResidualPacket,
    RigidObjectPacket,
    SceneActor,
    SceneActorReference,
    TensorSpec,
    VideoChunk,
    BallPacket,
    BallState,
)
from src.shared.synthesis_engine import SynthesisEngine


def _build_minimal_payload(tmp_path: Path) -> tuple[VideoChunk, EncodedChunkPayload, list[FrameState]]:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"x")

    chunk = VideoChunk(
        chunk_id="cov_0001",
        source_uri=str(source_path),
        start_frame_id=0,
        fps=30.0,
        num_frames=2,
        width=4,
        height=4,
    )
    panorama = PanoramaPacket(
        chunk_id=chunk.chunk_id,
        panorama_uri="memory://panorama/cov.jpg",
        frame_width=4,
        frame_height=4,
        camera_poses=[
            CameraPose(frame_id=0, tx=0.0, ty=0.0, tz=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
            CameraPose(frame_id=1, tx=0.0, ty=0.0, tz=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        ],
        panorama_image=None,
        homography_matrices=[np.eye(3, dtype=np.float64).tolist(), np.eye(3, dtype=np.float64).tolist()],
        selected_frame_indices=[0, 1],
    )
    actor_mask_png = cv2.imencode(".png", np.full((2, 2), 255, dtype=np.uint8))[1].tobytes()
    actor_packet = ActorPacket(
        chunk_id=chunk.chunk_id,
        object_id="person_1",
        appearance_embedding_spec=TensorSpec(name="appearance", shape=[1, 2], dtype="torch.float32"),
        pose_tensor_spec=TensorSpec(name="pose", shape=[1, 18, 3], dtype="torch.float32"),
        events=[
            KeyframeEvent(
                frame_id=0,
                object_id="person_1",
                object_class=ObjectClass.PERSON,
                coordinates=[0.0, 0.0, 0.0],
            )
        ],
        mask_frames=[
            ActorMaskFrame(
                frame_id=0,
                bbox=[0, 0, 2, 2],
                mask_codec="png",
                mask_png=actor_mask_png,
                mask_height=2,
                mask_width=2,
            )
        ],
    )
    reference_image = np.zeros((2, 2, 3), dtype=np.uint8)
    reference_image[:, :, 2] = 255
    reference_jpeg = cv2.imencode(".jpg", reference_image)[1].tobytes()
    payload = EncodedChunkPayload(
        chunk=chunk,
        panorama=panorama,
        actors=[actor_packet],
        actor_references=[SceneActorReference(track_id=1, reference_crop_jpeg=reference_jpeg)],
        rigid_objects=[
            RigidObjectPacket(
                chunk_id=chunk.chunk_id,
                object_id="rigid_1",
                trajectory_spec=TensorSpec(name="rigid", shape=[1, 4], dtype="torch.float32"),
                events=[],
            )
        ],
        ball=BallPacket(
            chunk_id=chunk.chunk_id,
            object_id="ball_0",
            trajectory_spec=TensorSpec(name="ball", shape=[1, 4], dtype="torch.float32"),
            events=[],
            states=[
                BallState(frame_id=0, ball_x=1.0, ball_y=1.0, velocity_x=0.0, velocity_y=0.0, is_visible=True),
                BallState(frame_id=1, ball_x=1.5, ball_y=1.5, velocity_x=0.5, velocity_y=0.5, is_visible=True),
            ],
        ),
        residual=ResidualPacket(chunk_id=chunk.chunk_id, codec="placeholder", residual_video_uri="memory://residual.mp4"),
    )

    frame_states = [
        FrameState(
            frame_id=0,
            actors=[
                SceneActor(
                    track_id="person_1",
                    class_name="player",
                    bbox=[0.0, 0.0, 4.0, 4.0],
                    mask=[[1, 1], [1, 1]],
                    pose_dw=[[0.0, 0.0, 1.0] for _ in range(18)],
                )
            ],
        ),
        FrameState(
            frame_id=1,
            actors=[
                SceneActor(
                    track_id="person_1",
                    class_name="player",
                    bbox=[0.0, 0.0, 4.0, 4.0],
                    mask=[[1, 1], [1, 1]],
                    pose_dw=[[0.0, 0.0, 1.0] for _ in range(18)],
                )
            ],
        ),
    ]
    return chunk, payload, frame_states


def test_residual_calculator_covers_players_only_and_full_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    chunk, payload, frame_states = _build_minimal_payload(tmp_path)
    original_frames = [
        np.full((4, 4, 3), 10, dtype=np.uint8),
        np.full((4, 4, 3), 20, dtype=np.uint8),
    ]

    monkeypatch.setattr(residual_calc, "probe_video_metadata", lambda _uri: SimpleNamespace(num_frames=2, fps=30.0, width=4, height=4))
    monkeypatch.setattr(residual_calc, "iter_video_frames_ffmpeg", lambda *args, **kwargs: iter(original_frames))

    encoded_outputs: list[np.ndarray] = []

    def _fake_encode_video_frames_ffmpeg(**kwargs):
        frames_bgr = kwargs["frames_bgr"]
        encoded_outputs.extend(list(frames_bgr))
        return Path(kwargs["output_path"])

    monkeypatch.setattr(residual_calc, "encode_video_frames_ffmpeg", _fake_encode_video_frames_ffmpeg)

    class _FakeDiffusersCompositorBase:
        pass

    class _FakeCompositor(_FakeDiffusersCompositorBase):
        def __init__(self) -> None:
            self.seen_shapes: list[tuple[int, ...]] = []
            self.seen_metadata: list[tuple[np.ndarray | None, tuple[int, ...] | None]] = []

        def uses_temporal_pose_sequence(self) -> bool:
            return True

        def process(self, reference_crop_tensor, dense_dwpose_tensor, warped_background_frame, actor_identity=None, metadata_mask=None, metadata_bbox=None):
            _ = (reference_crop_tensor, actor_identity)
            self.seen_shapes.append(tuple(dense_dwpose_tensor.shape))
            self.seen_metadata.append((metadata_mask, None if metadata_bbox is None else tuple(metadata_bbox)))
            return (warped_background_frame + 1).to(torch.float32)

    class _FakeSynthesisEngine(SynthesisEngine):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if "config" not in kwargs:
                kwargs["config"] = PointstreamConfig()
            super().__init__(*args, **kwargs)
            self.seed = 7
            self.device: Any = "cpu"
            self._compositor = _FakeCompositor()

        def synthesize(self, payload, include_guidance_overlays=False):
            _ = (payload, include_guidance_overlays)
            frames = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
            return SimpleNamespace(frames_bgr=frames)

        def get_genai_compositor(self):
            return self._compositor

        def _unroll_sparse_actor_poses(self, payload):
            _ = payload
            return {"person_1": torch.ones((2, 18, 3), dtype=torch.float32)}

    monkeypatch.setattr(residual_calc, "DiffusersCompositor", _FakeDiffusersCompositorBase)
    monkeypatch.setenv("POINTSTREAM_ENABLE_GENAI", "1")
    monkeypatch.setenv("POINTSTREAM_ANIMATE_ANYONE_WINDOW", "2")
    monkeypatch.setenv("POINTSTREAM_GENAI_PREROLL_FRAMES", "0")
    monkeypatch.setenv("POINTSTREAM_FFMPEG_CODEC", "libsvtav1")

    for mode in (ResidualMode.PLAYERS_ONLY, ResidualMode.FULL_VIDEO):
        cfg = PointstreamConfig(genai_backend="animate-anyone")
        calculator = residual_calc.ResidualCalculator(
            config=cfg,
            synthesis_engine=_FakeSynthesisEngine(config=cfg),
            importance_mapper=residual_calc.BinaryActorImportanceMapper(),
            residual_mode=mode,
        )
        residual_packet = calculator.process(
            chunk=chunk,
            payload=payload,
            frame_states=frame_states,
            debug_output_path=tmp_path / f"{mode.value}.mp4",
        )

        assert residual_packet.mode == mode
        assert residual_packet.codec == "libsvtav1"
        assert residual_packet.residual_video_uri.endswith(".mp4")

        compositor = calculator._synthesis_engine.get_genai_compositor()
        assert getattr(compositor, "seen_shapes") == [(1, 18, 3), (2, 18, 3)]
        assert getattr(compositor, "seen_metadata")[0][1] == (0, 0, 2, 2)

    assert len(encoded_outputs) == 4


def test_residual_calculator_adaptive_background_downscale_preserves_player_region(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chunk, payload, frame_states = _build_minimal_payload(tmp_path)

    # Keep a compact player ROI in the top-left; background occupies the remaining area.
    frame_states = [
        FrameState(
            frame_id=0,
            actors=[
                SceneActor(
                    track_id="person_1",
                    class_name="player",
                    bbox=[0.0, 0.0, 2.0, 2.0],
                    mask=[[1, 1], [1, 1]],
                    pose_dw=[[0.0, 0.0, 1.0] for _ in range(18)],
                )
            ],
        )
    ]

    # Shape: [Height, Width, Channels]
    original_frame = np.array(
        [
            [[10, 10, 10], [30, 30, 30], [180, 180, 180], [20, 20, 20]],
            [[40, 40, 40], [60, 60, 60], [40, 40, 40], [220, 220, 220]],
            [[200, 200, 200], [30, 30, 30], [250, 250, 250], [20, 20, 20]],
            [[15, 15, 15], [210, 210, 210], [35, 35, 35], [240, 240, 240]],
        ],
        dtype=np.uint8,
    )

    monkeypatch.setattr(
        residual_calc,
        "probe_video_metadata",
        lambda _uri: SimpleNamespace(num_frames=1, fps=30.0, width=4, height=4),
    )
    monkeypatch.setattr(
        residual_calc,
        "iter_video_frames_ffmpeg",
        lambda *args, **kwargs: iter([original_frame]),
    )

    encoded_outputs: list[np.ndarray] = []

    def _fake_encode_video_frames_ffmpeg(**kwargs):
        encoded_outputs.extend(list(kwargs["frames_bgr"]))
        return Path(kwargs["output_path"])

    monkeypatch.setattr(residual_calc, "encode_video_frames_ffmpeg", _fake_encode_video_frames_ffmpeg)
    monkeypatch.setenv("POINTSTREAM_ENABLE_GENAI", "0")

    class _FakeSynthesisEngine(SynthesisEngine):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if "config" not in kwargs:
                kwargs["config"] = PointstreamConfig()
            super().__init__(*args, **kwargs)
            self.seed = 7
            self.device: Any = "cpu"

        def synthesize(self, payload, include_guidance_overlays=False):
            _ = (payload, include_guidance_overlays)
            # Shape: [Frames, Channels, Height, Width]
            return SimpleNamespace(frames_bgr=torch.zeros((1, 3, 4, 4), dtype=torch.float32))

    cfg = PointstreamConfig(genai_backend=None)
    calculator = residual_calc.ResidualCalculator(
        config=cfg,
        synthesis_engine=_FakeSynthesisEngine(config=cfg),
        importance_mapper=residual_calc.BinaryActorImportanceMapper(),
        residual_mode=ResidualMode.FULL_VIDEO,
        background_block_downscale_factor=2,
    )

    calculator.process(
        chunk=chunk.model_copy(update={"num_frames": 1}),
        payload=payload,
        frame_states=frame_states,
        debug_output_path=tmp_path / "adaptive_full_video.mp4",
    )

    assert len(encoded_outputs) == 1
    encoded = encoded_outputs[0]

    # Player ROI must remain full-resolution signed residual (original - predicted + 128).
    expected_player = np.clip(original_frame[:2, :2].astype(np.int16) + 128, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(encoded[:2, :2], expected_player)

    # Background is adaptively downscaled/upscaled, so high-frequency pixels should be altered.
    expected_raw = np.clip(original_frame.astype(np.int16) + 128, 0, 255).astype(np.uint8)
    assert not np.array_equal(encoded[2:, 2:], expected_raw[2:, 2:])


def test_encoder_pipeline_streams_actor_states_and_uses_shifted_ball(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    chunk = VideoChunk(
        chunk_id="pipe_cov",
        source_uri=str(tmp_path / "source.mp4"),
        start_frame_id=0,
        fps=30.0,
        num_frames=2,
        width=4,
        height=4,
    )

    class _FakeActorExtractor:
        def process_with_states_streaming(self, chunk, on_frame_state=None):
            _ = chunk
            states = [
                FrameState(frame_id=0, actors=[]),
                FrameState(frame_id=1, actors=[]),
            ]
            for state in states:
                if on_frame_state is not None:
                    on_frame_state(state)
            packet = ActorPacket(
                chunk_id=chunk.chunk_id,
                object_id="person_1",
                appearance_embedding_spec=TensorSpec(name="appearance", shape=[1, 2], dtype="torch.float32"),
                pose_tensor_spec=TensorSpec(name="pose", shape=[1, 18, 3], dtype="torch.float32"),
                events=[],
                mask_frames=[],
            )
            return orchestrator.ActorExtractionResult(frame_states=states, actor_packets=[packet])

        def process_with_states(self, chunk):
            return self.process_with_states_streaming(chunk)

    class _FakeBallExtractor:
        def process_shifted(self, chunk, panorama, actor_bundle):
            _ = (chunk, panorama)
            first = actor_bundle.wait_for_frame_state(0)
            second = actor_bundle.wait_for_frame_state(1)
            assert first is not None
            assert second is not None
            return BallPacket(
                chunk_id=chunk.chunk_id,
                object_id="ball_0",
                trajectory_spec=TensorSpec(name="ball", shape=[1, 4], dtype="torch.float32"),
                events=[],
                states=[],
            )

        def process(self, chunk, panorama, frame_states):
            return self.process_shifted(chunk, panorama, frame_states)

    class _FakeReferenceExtractor:
        def process(self, chunk, frame_states):
            _ = (chunk, frame_states)
            return [SceneActorReference(track_id=1, reference_crop_jpeg=b"abc")]

    class _FakeBackgroundModeler:
        def process(self, chunk, decoded_video_tensor=None, frame_states=None):
            _ = (chunk, decoded_video_tensor, frame_states)
            return PanoramaPacket(
                chunk_id=chunk.chunk_id,
                panorama_uri="memory://pano.jpg",
                frame_width=4,
                frame_height=4,
                camera_poses=[CameraPose(frame_id=0, tx=0.0, ty=0.0, tz=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)],
                panorama_image=[[[0, 0, 0]]],
                homography_matrices=[np.eye(3, dtype=np.float64).tolist()],
                selected_frame_indices=[0],
            )

    class _FakeResidualCalculator:
        def __init__(self, **kwargs) -> None:
            pass
        def process(self, chunk, payload, frame_states, debug_output_path=None):
            _ = (chunk, payload, frame_states, debug_output_path)
            return ResidualPacket(chunk_id=chunk.chunk_id, codec="libsvtav1", residual_video_uri=str(tmp_path / "residual.mp4"))

    pipeline = orchestrator.EncoderPipeline(
        config=PointstreamConfig(enable_shifted_ball=True),
        actor_extractor=_FakeActorExtractor(),
        ball_extractor=_FakeBallExtractor(),
        reference_extractor=_FakeReferenceExtractor(),
        residual_calculator=cast(Any, _FakeResidualCalculator(config=PointstreamConfig())),
    )
    pipeline._background_modeler = cast(Any, _FakeBackgroundModeler())
    pipeline._object_tracker = cast(Any, types.SimpleNamespace(process=lambda chunk: [RigidObjectPacket(chunk_id=chunk.chunk_id, object_id="rigid_1", trajectory_spec=TensorSpec(name="rigid", shape=[1, 4], dtype="torch.float32"), events=[])]))

    monkeypatch.setenv("POINTSTREAM_ENABLE_SHIFTED_BALL", "1")

    payload = pipeline.encode_chunk(chunk)

    assert payload.chunk.chunk_id == "pipe_cov"
    assert payload.ball.object_id == "ball_0"
    assert len(payload.actor_references) == 1
    assert payload.residual.codec == "libsvtav1"


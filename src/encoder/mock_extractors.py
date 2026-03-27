from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.encoder.video_io import iter_video_frames_ffmpeg, probe_video_metadata
from src.shared.schemas import (
    ActorPacket,
    BallPacket,
    InterpolateCommandEvent,
    KeyframeEvent,
    ObjectClass,
    RigidObjectPacket,
    SemanticEvent,
    TensorSpec,
    VideoChunk,
)
from src.shared.tags import gpu_bound


class ActorExtractor:
    def __init__(
        self,
        detector_model: Any | None = None,
        segmenter_model: Any | None = None,
        pose_model: Any | None = None,
        render_debug_keyframes: bool = True,
    ) -> None:
        from src.encoder.actor_components import (
            PayloadEncoder,
            PipelineBuilder,
            StandardTennisHeuristic,
            Yolo26Detector,
            YoloPoseEstimator,
            YoloSegmenter,
        )

        self._render_debug_keyframes = render_debug_keyframes
        # Models are loaded once in component initialization and reused frame-by-frame.
        self._pipeline = PipelineBuilder(
            detector=Yolo26Detector(model_name="yolo26n.pt", model=detector_model),
            heuristic=StandardTennisHeuristic(),
            segmenter=YoloSegmenter(model_name="yolo26n-seg.pt", model=segmenter_model),
            pose_estimator=YoloPoseEstimator(model_name="yolo26n-pose.pt", model=pose_model),
            payload_encoder=PayloadEncoder(pose_delta_threshold=20.0),
        )

    @gpu_bound
    def process(self, chunk: VideoChunk) -> list[ActorPacket]:
        frames_bgr = self._load_frames(chunk)
        frame_states, packets = self._pipeline.run(chunk=chunk, frames_bgr=frames_bgr)
        if self._render_debug_keyframes:
            self._pipeline.render_debug_keyframes(
                chunk=chunk,
                frames_bgr=frames_bgr,
                frame_states=frame_states,
                actor_packets=packets,
                out_dir=Path(__file__).resolve().parents[2] / "assets" / "debug_actors",
            )
        return packets

    def _load_frames(self, chunk: VideoChunk) -> list[np.ndarray]:
        source = Path(chunk.source_uri)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"ActorExtractor source video not found: {source}")

        metadata = probe_video_metadata(source)
        frames: list[np.ndarray] = []
        for frame in iter_video_frames_ffmpeg(
            source,
            width=metadata.width,
            height=metadata.height,
        ):
            frames.append(frame)
            if len(frames) >= chunk.num_frames:
                break
        if not frames:
            raise ValueError(f"ActorExtractor decoded zero frames from source: {source}")

        return frames


class MockActorExtractor:
    @gpu_bound
    def process(self, chunk: VideoChunk) -> list[ActorPacket]:
        batch = 1

        appearance = torch.zeros(batch, 2, 256, dtype=torch.float32)
        poses = torch.zeros(batch, chunk.num_frames, 18, 3, dtype=torch.float32)

        events_a: list[SemanticEvent] = [
            KeyframeEvent(
                frame_id=chunk.start_frame_id,
                object_id="person_0",
                object_class=ObjectClass.PERSON,
                coordinates=[0.1, 0.2, 0.9, 0.95],
            ),
            InterpolateCommandEvent(
                frame_id=chunk.start_frame_id + 1,
                object_id="person_0",
                object_class=ObjectClass.PERSON,
                target_frame_id=chunk.start_frame_id + min(5, chunk.num_frames - 1),
                method="linear",
            ),
        ]
        events_b: list[SemanticEvent] = [
            KeyframeEvent(
                frame_id=chunk.start_frame_id,
                object_id="person_1",
                object_class=ObjectClass.PERSON,
                coordinates=[0.05, 0.1, 0.85, 0.9],
            ),
            InterpolateCommandEvent(
                frame_id=chunk.start_frame_id + 1,
                object_id="person_1",
                object_class=ObjectClass.PERSON,
                target_frame_id=chunk.start_frame_id + min(5, chunk.num_frames - 1),
                method="linear",
            ),
        ]

        return [
            ActorPacket(
                chunk_id=chunk.chunk_id,
                object_id="person_0",
                appearance_embedding_spec=TensorSpec(
                    name="actor_appearance",
                    shape=list(appearance.shape),
                    dtype=str(appearance.dtype),
                ),
                pose_tensor_spec=TensorSpec(
                    name="actor_pose_dw",
                    shape=list(poses.shape),
                    dtype=str(poses.dtype),
                ),
                events=events_a,
            ),
            ActorPacket(
                chunk_id=chunk.chunk_id,
                object_id="person_1",
                appearance_embedding_spec=TensorSpec(
                    name="actor_appearance",
                    shape=list(appearance.shape),
                    dtype=str(appearance.dtype),
                ),
                pose_tensor_spec=TensorSpec(
                    name="actor_pose_dw",
                    shape=list(poses.shape),
                    dtype=str(poses.dtype),
                ),
                events=events_b,
            ),
        ]


class ObjectTracker:
    @gpu_bound
    def process(self, chunk: VideoChunk) -> list[RigidObjectPacket]:
        batch = 1

        # Shape: [Batch, Frames, Points, Coords]
        tracks = torch.zeros(batch, chunk.num_frames, 32, 2, dtype=torch.float32)

        racket_events: list[SemanticEvent] = [
            KeyframeEvent(
                frame_id=chunk.start_frame_id,
                object_id="racket_0",
                object_class=ObjectClass.RACKET,
                coordinates=[0.4, 0.5],
            ),
            InterpolateCommandEvent(
                frame_id=chunk.start_frame_id + 1,
                object_id="racket_0",
                object_class=ObjectClass.RACKET,
                target_frame_id=chunk.start_frame_id + min(6, chunk.num_frames - 1),
                method="linear",
            ),
        ]

        return [
            RigidObjectPacket(
                chunk_id=chunk.chunk_id,
                object_id="racket_0",
                trajectory_spec=TensorSpec(
                    name="rigid_trajectory",
                    shape=list(tracks.shape),
                    dtype=str(tracks.dtype),
                ),
                events=racket_events,
            )
        ]


class BallTracker:
    @gpu_bound
    def process(self, chunk: VideoChunk) -> BallPacket:
        batch = 1

        # Shape: [Batch, Frames, Params]
        trajectory = torch.zeros(batch, chunk.num_frames, 4, dtype=torch.float32)

        ball_events: list[SemanticEvent] = [
            KeyframeEvent(
                frame_id=chunk.start_frame_id,
                object_id="ball_0",
                object_class=ObjectClass.BALL,
                coordinates=[0.52, 0.45, 0.01, -0.02],
            ),
            InterpolateCommandEvent(
                frame_id=chunk.start_frame_id + 1,
                object_id="ball_0",
                object_class=ObjectClass.BALL,
                target_frame_id=chunk.start_frame_id + min(4, chunk.num_frames - 1),
                method="spline",
            ),
        ]

        return BallPacket(
            chunk_id=chunk.chunk_id,
            object_id="ball_0",
            trajectory_spec=TensorSpec(
                name="ball_trajectory",
                shape=list(trajectory.shape),
                dtype=str(trajectory.dtype),
            ),
            events=ball_events,
        )

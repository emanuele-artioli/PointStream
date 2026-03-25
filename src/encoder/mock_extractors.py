from __future__ import annotations

import torch

from src.shared.schemas import (
    ActorPacket,
    BallPacket,
    InterpolateCommandEvent,
    KeyframeEvent,
    ObjectClass,
    RigidObjectPacket,
    StaticCommandEvent,
    SemanticEvent,
    TensorSpec,
    VideoChunk,
)
from src.shared.tags import gpu_bound


class ActorExtractor:
    @gpu_bound
    def process(self, chunk: VideoChunk) -> list[ActorPacket]:
        batch = 1

        # Shape: [Batch, Objects, EmbedDim]
        appearance = torch.zeros(batch, 2, 256, dtype=torch.float32)
        # Shape: [Batch, Frames, Keypoints, Coords]
        poses = torch.zeros(batch, chunk.num_frames, 17, 3, dtype=torch.float32)

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
            StaticCommandEvent(
                frame_id=chunk.start_frame_id + min(2, chunk.num_frames - 1),
                object_id="person_1",
                object_class=ObjectClass.PERSON,
                hold_until_frame_id=chunk.start_frame_id + min(8, chunk.num_frames - 1),
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
                    name="actor_pose",
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
                    name="actor_pose",
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

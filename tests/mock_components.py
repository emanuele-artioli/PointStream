import cv2
import numpy as np
import torch
from pathlib import Path

from src.decoder.genai_compositor import BaseCompositor
from src.encoder.actor_pipeline import ActorExtractionResult
from src.shared.schemas import (
    ActorPacket,
    BallPacket,
    FrameState,
    InterpolateCommandEvent,
    KeyframeEvent,
    ObjectClass,
    SceneActor,
    SemanticEvent,
    TensorSpec,
    VideoChunk,
)
from src.shared.tags import gpu_bound


class MockActorExtractor:
    @gpu_bound
    def process(self, chunk: VideoChunk) -> list[ActorPacket]:
        return self.process_with_states(chunk).actor_packets

    @gpu_bound
    def process_with_states(self, chunk: VideoChunk) -> ActorExtractionResult:
        batch = 1

        appearance = torch.zeros(batch, 2, 256, dtype=torch.float32)
        poses = torch.zeros(batch, chunk.num_frames, 18, 3, dtype=torch.float32)
        pose_a = poses[0, 0].clone()
        pose_b = poses[0, 0].clone()
        pose_a[:, 0] = torch.linspace(100.0, 220.0, 18)
        pose_a[:, 1] = torch.linspace(80.0, 260.0, 18)
        pose_a[:, 2] = 0.9
        pose_b[:, 0] = torch.linspace(420.0, 560.0, 18)
        pose_b[:, 1] = torch.linspace(90.0, 280.0, 18)
        pose_b[:, 2] = 0.9

        events_a: list[SemanticEvent] = [
            KeyframeEvent(
                frame_id=chunk.start_frame_id,
                object_id="person_0",
                object_class=ObjectClass.PERSON,
                coordinates=pose_a.reshape(-1).tolist(),
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
                coordinates=pose_b.reshape(-1).tolist(),
            ),
            InterpolateCommandEvent(
                frame_id=chunk.start_frame_id + 1,
                object_id="person_1",
                object_class=ObjectClass.PERSON,
                target_frame_id=chunk.start_frame_id + min(5, chunk.num_frames - 1),
                method="linear",
            ),
        ]

        packets = [
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
        frame_states = self._build_frame_states(chunk)
        return ActorExtractionResult(frame_states=frame_states, actor_packets=packets)

    def _build_frame_states(self, chunk: VideoChunk) -> list[FrameState]:
        frame_states: list[FrameState] = []
        width = int(chunk.width)
        height = int(chunk.height)

        for frame_idx in range(int(chunk.num_frames)):
            shift = float((frame_idx * 7) % max(1, width // 3))

            p0_bbox = [
                40.0 + shift,
                80.0,
                min(width - 1.0, 120.0 + shift),
                min(height - 1.0, 260.0),
            ]
            p1_bbox = [
                max(0.0, width - 160.0 - shift),
                90.0,
                max(1.0, width - 80.0 - shift),
                min(height - 1.0, 280.0),
            ]
            r0_bbox = [
                min(width - 2.0, p0_bbox[2] - 12.0),
                min(height - 2.0, p0_bbox[1] + 48.0),
                min(width - 1.0, p0_bbox[2] + 20.0),
                min(height - 1.0, p0_bbox[1] + 96.0),
            ]
            r1_bbox = [
                max(0.0, p1_bbox[0] - 24.0),
                min(height - 2.0, p1_bbox[1] + 56.0),
                max(1.0, p1_bbox[0] + 8.0),
                min(height - 1.0, p1_bbox[1] + 102.0),
            ]

            actors = [
                SceneActor(
                    track_id="person_0",
                    class_name="player",
                    bbox=p0_bbox,
                    mask=self._solid_mask(mask_h=48, mask_w=28),
                ),
                SceneActor(
                    track_id="person_1",
                    class_name="player",
                    bbox=p1_bbox,
                    mask=self._solid_mask(mask_h=52, mask_w=28),
                ),
                SceneActor(
                    track_id="racket_0",
                    class_name="racket",
                    bbox=r0_bbox,
                    mask=self._solid_mask(mask_h=20, mask_w=10),
                ),
                SceneActor(
                    track_id="racket_1",
                    class_name="racket",
                    bbox=r1_bbox,
                    mask=self._solid_mask(mask_h=20, mask_w=10),
                ),
            ]
            frame_states.append(FrameState(frame_id=frame_idx, actors=actors))

        return frame_states

    def _solid_mask(self, mask_h: int, mask_w: int) -> list[list[int]]:
        return np.ones((mask_h, mask_w), dtype=np.uint8).tolist()


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


class MockCompositor(BaseCompositor):
    """Lightweight fallback compositor used when GenAI is disabled in tests."""

    @gpu_bound
    def process(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        warped_background_frame: torch.Tensor,
        actor_identity: str | None = None,
        metadata_mask: np.ndarray | None = None,
        metadata_bbox: tuple[int, int, int, int] | None = None,
        debug_dir: str | Path | None = None,
        frame_idx: int | None = None,
    ) -> torch.Tensor:
        _ = actor_identity
        _ = metadata_mask
        _ = metadata_bbox
        frame_np = self._to_frame_numpy(warped_background_frame)
        pose_np = self._to_pose_numpy(dense_dwpose_tensor)
        crop_np = self._to_crop_numpy(reference_crop_tensor)

        x1, y1, x2, y2 = self._estimate_bbox_from_pose(pose_np=pose_np, frame_height=frame_np.shape[0], frame_width=frame_np.shape[1])

        # Draw a filled placeholder silhouette to prove compositor wiring end-to-end.
        cv2.rectangle(frame_np, (x1, y1), (x2, y2), color=(35, 80, 210), thickness=-1)

        crop_h = max(1, y2 - y1)
        crop_w = max(1, x2 - x1)
        resized_crop = cv2.resize(crop_np, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        frame_np[y1:y2, x1:x2] = resized_crop

        return torch.from_numpy(frame_np).permute(2, 0, 1).contiguous().to(torch.uint8)

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.encoder.video_io import iter_video_frames_ffmpeg, probe_video_metadata
from src.shared.schemas import (
    ActorPacket,
    BallPacket,
    FrameState,
    InterpolateCommandEvent,
    KeyframeEvent,
    ObjectClass,
    RigidObjectPacket,
    SceneActor,
    SemanticEvent,
    TensorSpec,
    VideoChunk,
)
from src.shared.tags import gpu_bound


@dataclass(frozen=True)
class ActorExtractionResult:
    frame_states: list[FrameState]
    actor_packets: list[ActorPacket]


class ActorExtractor:
    def __init__(
        self,
        detector_model: Any | None = None,
        segmenter_model: Any | None = None,
        pose_model: Any | None = None,
        render_debug_keyframes: bool = True,
        detector_backend: str = "yolo26",
        detector_caption: str = "tennis player",
        pose_backend: str = "yolo",
        segmenter_backend: str = "yolo",
        segmenter_caption: str = "tennis player",
        pose_delta_threshold: float = 20.0,
        include_mask_metadata: bool = False,
        metadata_mask_codec: str = "auto",
        dwpose_device: str = "cuda",
    ) -> None:
        from src.encoder.actor_components import (
            BaseDetector,
            BasePoseEstimator,
            BaseSegmenter,
            DwposeEstimator,
            NoOpSegmenter,
            PayloadEncoder,
            PipelineBuilder,
            SamSegmenter,
            StandardTennisHeuristic,
            YoloEDetector,
            Yolo26Detector,
            YoloPoseEstimator,
            YoloeSegmenter,
            YoloSegmenter,
        )

        self._render_debug_keyframes = render_debug_keyframes

        caption_list = [part.strip() for part in str(detector_caption).split(",") if part.strip()]
        if not caption_list:
            caption_list = ["tennis player"]

        detector: BaseDetector
        normalized_detector_backend = detector_backend.strip().lower()
        if normalized_detector_backend in {"yolo", "yolo26", "yolo26n"}:
            detector = Yolo26Detector(model_name="yolo26n.pt", model=detector_model)
        elif normalized_detector_backend in {"yoloe", "yolo-e", "yolo_e"}:
            detector = YoloEDetector(
                model_name="yoloe-26n-seg.pt",
                model=detector_model,
                captions=caption_list,
            )
        else:
            raise ValueError(f"Unsupported detector backend: {detector_backend}")

        pose_estimator: BasePoseEstimator
        normalized_pose_backend = pose_backend.strip().lower()
        if normalized_pose_backend in {"yolo", "yolo-pose", "yolo_pose"}:
            pose_estimator = YoloPoseEstimator(model_name="yolo26n-pose.pt", model=pose_model)
        elif normalized_pose_backend in {"dwpose", "dw-pose", "dw_pose"}:
            pose_estimator = DwposeEstimator(torchscript_device=dwpose_device)
        else:
            raise ValueError(f"Unsupported pose backend: {pose_backend}")

        segmenter: BaseSegmenter
        normalized_segmenter_backend = segmenter_backend.strip().lower()
        if normalized_segmenter_backend in {"yolo", "yolo-seg", "yolo_seg"}:
            segmenter = YoloSegmenter(model_name="yolo26n-seg.pt", model=segmenter_model)
        elif normalized_segmenter_backend in {"yoloe", "yolo-e", "yolo_e"}:
            segmenter_caption_list = [part.strip() for part in str(segmenter_caption).split(",") if part.strip()]
            if not segmenter_caption_list:
                segmenter_caption_list = ["tennis player"]
            segmenter = YoloeSegmenter(
                model_name="yoloe-26n-seg.pt",
                model=segmenter_model,
                captions=segmenter_caption_list,
            )
        elif normalized_segmenter_backend in {"sam3", "sam-3"}:
            segmenter = SamSegmenter(model_name="sam3.pt", model=segmenter_model)
        elif normalized_segmenter_backend in {"sam", "sam2", "sam-2"}:
            segmenter = SamSegmenter(model_name="sam2_b.pt", model=segmenter_model)
        elif normalized_segmenter_backend in {"none", "off", "noop", "no-op"}:
            segmenter = NoOpSegmenter()
        else:
            raise ValueError(f"Unsupported segmenter backend: {segmenter_backend}")

        # Models are loaded once in component initialization and reused frame-by-frame.
        self._pipeline = PipelineBuilder(
            detector=detector,
            heuristic=StandardTennisHeuristic(),
            segmenter=segmenter,
            pose_estimator=pose_estimator,
            payload_encoder=PayloadEncoder(
                pose_delta_threshold=float(pose_delta_threshold),
                include_mask_metadata=bool(include_mask_metadata),
                metadata_mask_codec=str(metadata_mask_codec),
            ),
        )

    @gpu_bound
    def process(self, chunk: VideoChunk) -> list[ActorPacket]:
        return self.process_with_states(chunk).actor_packets

    @gpu_bound
    def process_with_states(self, chunk: VideoChunk) -> ActorExtractionResult:
        frames_bgr = self._load_frames(chunk)
        frame_states, packets = self._pipeline.run(chunk=chunk, frames_bgr=frames_bgr)
        if self._render_debug_keyframes:
            self._pipeline.render_debug_keyframes(
                chunk=chunk,
                frames_bgr=frames_bgr,
                frame_states=frame_states,
                actor_packets=packets,
                out_dir=self._resolve_debug_keyframes_dir(),
            )
        return ActorExtractionResult(frame_states=frame_states, actor_packets=packets)

    def _resolve_debug_keyframes_dir(self) -> Path:
        override = os.environ.get("POINTSTREAM_DEBUG_ARTIFACT_DIR")
        if override:
            return Path(override) / "debug_actors"

        project_root = Path(__file__).resolve().parents[2]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return project_root / "outputs" / timestamp / "debug" / "debug_actors"

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

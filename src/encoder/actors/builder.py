"""Assembling an actor pipeline from config.

Backends are selected by name string, so the mapping here is part of the
config contract: renaming a key silently changes which backend runs."""

from __future__ import annotations
import logging
from pathlib import Path
import cv2
import numpy as np
from src.shared.schemas import ActorPacket, FrameState, SceneActor, VideoChunk
from src.shared.dwpose_draw import draw_dwpose_canvas
from src.shared.profiling import PipelineProfiler
from src.encoder.actors.detection import BaseDetector
from src.encoder.actors.heuristics import BaseHeuristic
from src.encoder.actors.pose import BasePoseEstimator
from src.encoder.actors.segmentation import BaseSegmenter
from src.encoder.actors.payload import PayloadEncoder
_COCO_PERSON_CLASS_ID = 0
_COCO_TENNIS_RACKET_CLASS_ID = 38
_LOGGER = logging.getLogger(__name__)


class PipelineBuilder:
    def __init__(
        self,
        detector: BaseDetector,
        heuristic: BaseHeuristic,
        segmenter: BaseSegmenter,
        pose_estimator: BasePoseEstimator,
        payload_encoder: PayloadEncoder,
    ) -> None:
        self.detector = detector
        self.heuristic = heuristic
        self.segmenter = segmenter
        self.pose_estimator = pose_estimator
        self.payload_encoder = payload_encoder
        self.profiler = PipelineProfiler()

    def get_timings(self) -> dict[str, float]:
        return self.profiler.get_timings()

    def iter_filtered_states(self, chunk: VideoChunk, frames_bgr: list[np.ndarray]):
        iterator = self.detector.iter_track(frames_bgr)
        while True:
            with self.profiler.stage("detection"):
                try:
                    state = next(iterator)
                except StopIteration:
                    break

            if state.frame_id >= len(frames_bgr):
                continue

            frame = frames_bgr[state.frame_id]
            h, w = frame.shape[:2]
            with self.profiler.stage("heuristic"):
                selected = self.heuristic.select(state, frame_shape=(h, w))

            updated_actors: list[SceneActor] = []
            for actor in selected.actors:
                with self.profiler.stage("segmentation"):
                    segmented = self.segmenter.segment(frame, actor) if self.segmenter else actor
                with self.profiler.stage("pose_estimation"):
                    with_pose = self.pose_estimator.estimate(frame, segmented) if self.pose_estimator else segmented
                updated_actors.append(with_pose)

            yield FrameState(frame_id=state.frame_id, actors=updated_actors)

    def run(self, chunk: VideoChunk, frames_bgr: list[np.ndarray]) -> tuple[list[FrameState], list[ActorPacket]]:
        filtered_states = list(self.iter_filtered_states(chunk=chunk, frames_bgr=frames_bgr))

        with self.profiler.stage("metadata_generation"):
            packets = self.payload_encoder.encode(chunk=chunk, frame_states=filtered_states)
        return filtered_states, packets

    def render_debug_keyframes(
        self,
        chunk: VideoChunk,
        frames_bgr: list[np.ndarray],
        frame_states: list[FrameState],
        actor_packets: list[ActorPacket],
        out_dir: str | Path,
    ) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        keyframe_ids: set[int] = set()
        for packet in actor_packets:
            for event in packet.events:
                if event.event_type == "keyframe":
                    keyframe_ids.add(int(event.frame_id))
            for mask_frame in packet.mask_frames:
                keyframe_ids.add(int(mask_frame.frame_id))

        frame_state_by_id = {chunk.start_frame_id + s.frame_id: s for s in frame_states}
        for global_frame_id in sorted(keyframe_ids):
            local_idx = global_frame_id - chunk.start_frame_id
            if local_idx < 0 or local_idx >= len(frames_bgr):
                continue

            frame = frames_bgr[local_idx]
            state = frame_state_by_id.get(global_frame_id)
            if state is None:
                continue

            stem = f"{chunk.chunk_id}_frame_{global_frame_id:05d}"
            
            people_dw = []
            for actor in state.actors:
                if actor.class_name != "player":
                    continue
                if actor.mask is not None:
                    mask_img = np.clip(np.asarray(actor.mask, dtype=np.float32) * 255, 0, 255).astype(np.uint8)
                    cv2.imwrite(str(out_path / f"{stem}_{actor.track_id}_mask.png"), mask_img)
                if actor.pose_dw is None:
                    continue
                pose = np.asarray(actor.pose_dw, dtype=np.float32)
                if pose.shape == (18, 3):
                    people_dw.append(pose)

            if not people_dw:
                continue

            try:
                skeleton_canvas = draw_dwpose_canvas(
                    height=int(frame.shape[0]),
                    width=int(frame.shape[1]),
                    people_dw=np.stack(people_dw, axis=0),
                    confidence_threshold=0.2,
                )
            except ModuleNotFoundError as exc:
                _LOGGER.warning("Skipping debug skeleton rendering because dwpose is unavailable: %s", exc)
                continue
            overlay = frame.copy()
            mask = np.any(skeleton_canvas > 0, axis=2)
            overlay[mask] = skeleton_canvas[mask]

            cv2.imwrite(str(out_path / f"{stem}_skeleton_black.png"), skeleton_canvas)
            cv2.imwrite(str(out_path / f"{stem}_skeleton_overlay.png"), overlay)

from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Any

import numpy as np
import torch

from src.encoder.video_io import iter_video_frames_ffmpeg
from src.shared.schemas import (
    BallPacket,
    BallState,
    FrameState,
    InterpolateCommandEvent,
    KeyframeEvent,
    ObjectClass,
    PanoramaPacket,
    SemanticEvent,
    TensorSpec,
    VideoChunk,
)
from src.shared.tags import gpu_bound


class SegmentationBallExtractor:
    """Detect ball via an object detector (YOLO) inside an ROI between players.

    This extractor is best-effort: if detector or weights missing, it raises a
    RuntimeError. It outputs the same BallPacket semantics as the residual
    extractor so it can be swapped in the pipeline.
    """

    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        confidence: float = 0.25,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("ultralytics YOLO is required for SegmentationBallExtractor") from exc

        weight_ref = model_name
        # Allow override via env or local assets resolution in actor components may be preferable
        self._model = YOLO(weight_ref)
        self._conf = float(confidence)
        self._device = device
        self._class_id = int(os.environ.get("POINTSTREAM_BALL_CLASS_ID", "32"))

    @gpu_bound
    def process(self, chunk: VideoChunk, panorama: PanoramaPacket, frame_states: list[FrameState]) -> BallPacket:
        frame_count = int(chunk.num_frames)
        if frame_count <= 0:
            raise ValueError("SegmentationBallExtractor received zero configured frames")

        ball_states: list[BallState] = []
        previous_visible = False
        previous_x = 0.0
        previous_y = 0.0

        frame_iterator = self._iter_source_frames(chunk)
        for frame_idx, frame_bgr in enumerate(frame_iterator):
            if frame_idx >= frame_count:
                break
            frame_state = self._resolve_frame_state(frame_states=frame_states, frame_idx=frame_idx)

            roi = self._compute_roi_from_frame_state(frame_state=frame_state, frame_bgr=frame_bgr)
            found_x = 0.0
            found_y = 0.0
            is_visible = False

            if roi is not None:
                x1, y1, x2, y2 = roi
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size != 0:
                    try:
                        results = self._model.predict(source=crop, conf=self._conf, verbose=False)
                    except Exception:
                        results = None

                    if results:
                        res = results[0]
                        boxes = getattr(res, "boxes", None)
                        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
                            cls = boxes.cls.cpu().numpy().astype(np.int32) if getattr(boxes, "cls", None) is not None else np.zeros((xyxy.shape[0],), dtype=np.int32)
                            # filter by desired class id
                            matches = [i for i, c in enumerate(cls) if int(c) == self._class_id]
                            if matches:
                                best = matches[0]
                                bx1, by1, bx2, by2 = xyxy[best].tolist()
                                cx = (bx1 + bx2) * 0.5 + x1
                                cy = (by1 + by2) * 0.5 + y1
                                found_x = float(cx)
                                found_y = float(cy)
                                is_visible = True

            if is_visible:
                if previous_visible:
                    vx = found_x - previous_x
                    vy = found_y - previous_y
                else:
                    vx = 0.0
                    vy = 0.0
                previous_x = found_x
                previous_y = found_y
            else:
                found_x = previous_x
                found_y = previous_y
                vx = 0.0
                vy = 0.0

            previous_visible = is_visible
            state = BallState(
                frame_id=chunk.start_frame_id + frame_idx,
                ball_x=float(found_x),
                ball_y=float(found_y),
                velocity_x=float(vx),
                velocity_y=float(vy),
                is_visible=bool(is_visible),
            )
            ball_states.append(state)
            if frame_idx < len(frame_states):
                frame_states[frame_idx] = frame_states[frame_idx].model_copy(update={"ball_state": state})

        if not ball_states:
            raise ValueError("SegmentationBallExtractor decoded zero source frames")

        trajectory = torch.zeros((1, len(ball_states), 4), dtype=torch.float32)
        events: list[SemanticEvent] = []
        for idx, state in enumerate(ball_states):
            trajectory[0, idx] = torch.tensor(
                [state.ball_x, state.ball_y, state.velocity_x, state.velocity_y],
                dtype=torch.float32,
            )
            if state.is_visible:
                events.append(
                    KeyframeEvent(
                        frame_id=state.frame_id,
                        object_id="ball_0",
                        object_class=ObjectClass.BALL,
                        coordinates=[state.ball_x, state.ball_y, state.velocity_x, state.velocity_y],
                    )
                )
            else:
                events.append(
                    InterpolateCommandEvent(
                        frame_id=state.frame_id,
                        object_id="ball_0",
                        object_class=ObjectClass.BALL,
                        target_frame_id=state.frame_id,
                        method="linear",
                    )
                )

        return BallPacket(
            chunk_id=chunk.chunk_id,
            object_id="ball_0",
            trajectory_spec=TensorSpec(
                name="ball_trajectory",
                shape=list(trajectory.shape),
                dtype=str(trajectory.dtype),
            ),
            events=events,
            states=ball_states,
        )

    @gpu_bound
    def process_shifted(self, chunk: VideoChunk, panorama: PanoramaPacket, actor_bundle: Any) -> BallPacket:
        _ = panorama
        frame_count = int(chunk.num_frames)
        if frame_count <= 0:
            raise ValueError("SegmentationBallExtractor received zero configured frames")

        ball_states: list[BallState] = []
        previous_visible = False
        previous_x = 0.0
        previous_y = 0.0

        frame_iterator = self._iter_source_frames(chunk)
        for frame_idx, frame_bgr in enumerate(frame_iterator):
            if frame_idx >= frame_count:
                break

            if frame_idx <= 0:
                frame_state = FrameState(frame_id=frame_idx, actors=[])
            else:
                previous_state = actor_bundle.wait_for_frame_state(frame_idx - 1)
                frame_state = previous_state if previous_state is not None else FrameState(frame_id=frame_idx, actors=[])

            roi = self._compute_roi_from_frame_state(frame_state=frame_state, frame_bgr=frame_bgr)
            found_x = 0.0
            found_y = 0.0
            is_visible = False

            if roi is not None:
                x1, y1, x2, y2 = roi
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size != 0:
                    try:
                        results = self._model.predict(source=crop, conf=self._conf, verbose=False)
                    except Exception:
                        results = None

                    if results:
                        res = results[0]
                        boxes = getattr(res, "boxes", None)
                        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
                            cls = boxes.cls.cpu().numpy().astype(np.int32) if getattr(boxes, "cls", None) is not None else np.zeros((xyxy.shape[0],), dtype=np.int32)
                            matches = [i for i, c in enumerate(cls) if int(c) == self._class_id]
                            if matches:
                                best = matches[0]
                                bx1, by1, bx2, by2 = xyxy[best].tolist()
                                cx = (bx1 + bx2) * 0.5 + x1
                                cy = (by1 + by2) * 0.5 + y1
                                found_x = float(cx)
                                found_y = float(cy)
                                is_visible = True

            if is_visible:
                if previous_visible:
                    vx = found_x - previous_x
                    vy = found_y - previous_y
                else:
                    vx = 0.0
                    vy = 0.0
                previous_x = found_x
                previous_y = found_y
            else:
                found_x = previous_x
                found_y = previous_y
                vx = 0.0
                vy = 0.0

            previous_visible = is_visible
            state = BallState(
                frame_id=chunk.start_frame_id + frame_idx,
                ball_x=float(found_x),
                ball_y=float(found_y),
                velocity_x=float(vx),
                velocity_y=float(vy),
                is_visible=bool(is_visible),
            )
            ball_states.append(state)

        if not ball_states:
            raise ValueError("SegmentationBallExtractor decoded zero source frames")

        trajectory = torch.zeros((1, len(ball_states), 4), dtype=torch.float32)
        events: list[SemanticEvent] = []
        for idx, state in enumerate(ball_states):
            trajectory[0, idx] = torch.tensor(
                [state.ball_x, state.ball_y, state.velocity_x, state.velocity_y],
                dtype=torch.float32,
            )
            if state.is_visible:
                events.append(
                    KeyframeEvent(
                        frame_id=state.frame_id,
                        object_id="ball_0",
                        object_class=ObjectClass.BALL,
                        coordinates=[state.ball_x, state.ball_y, state.velocity_x, state.velocity_y],
                    )
                )
            else:
                events.append(
                    InterpolateCommandEvent(
                        frame_id=state.frame_id,
                        object_id="ball_0",
                        object_class=ObjectClass.BALL,
                        target_frame_id=state.frame_id,
                        method="linear",
                    )
                )

        return BallPacket(
            chunk_id=chunk.chunk_id,
            object_id="ball_0",
            trajectory_spec=TensorSpec(
                name="ball_trajectory",
                shape=list(trajectory.shape),
                dtype=str(trajectory.dtype),
            ),
            events=events,
            states=ball_states,
        )

    def _iter_source_frames(self, chunk: VideoChunk) -> Iterator[np.ndarray]:
        return iter_video_frames_ffmpeg(chunk.source_uri, width=int(chunk.width), height=int(chunk.height))

    def _resolve_frame_state(self, frame_states: list[FrameState], frame_idx: int) -> FrameState:
        if frame_idx < len(frame_states):
            return frame_states[frame_idx]
        return FrameState(frame_id=frame_idx, actors=[])

    def _compute_roi_from_frame_state(self, frame_state: FrameState, frame_bgr: np.ndarray, pad_ratio: float = 0.10):
        h, w = frame_bgr.shape[:2]
        players = [a for a in frame_state.actors if a.class_name == "player"]
        if len(players) < 2:
            return None

        centers = [(p, (p.bbox[1] + p.bbox[3]) * 0.5) for p in players]
        centers.sort(key=lambda t: t[1])
        top_player = centers[0][0]
        bottom_player = centers[-1][0]

        def _clip(bbox):
            x1 = int(max(0, min(w - 1, int(np.floor(bbox[0])))))
            y1 = int(max(0, min(h - 1, int(np.floor(bbox[1])))))
            x2 = int(max(0, min(w, int(np.ceil(bbox[2])))))
            y2 = int(max(0, min(h, int(np.ceil(bbox[3])))))
            return x1, y1, x2, y2

        t_x1, t_y1, t_x2, t_y2 = _clip(top_player.bbox)
        b_x1, b_y1, b_x2, b_y2 = _clip(bottom_player.bbox)

        x1 = min(t_x1, b_x1)
        y1 = min(t_y1, b_y1)
        x2 = max(t_x2, b_x2)
        y2 = max(t_y2, b_y2)

        # pad
        rw = max(1, x2 - x1)
        rh = max(1, y2 - y1)
        px = int(round(rw * pad_ratio))
        py = int(round(rh * pad_ratio))
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

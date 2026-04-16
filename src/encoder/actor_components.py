from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.shared.schemas import (
    ActorMaskFrame,
    ActorPacket,
    FrameState,
    InterpolateCommandEvent,
    KeyframeEvent,
    ObjectClass,
    SceneActor,
    SemanticEvent,
    TensorSpec,
    VideoChunk,
)
from src.shared.mask_codec import encode_binary_mask, encode_polygon_mask, normalize_mask_codec
from src.shared.dwpose_draw import draw_dwpose_canvas


_COCO_PERSON_CLASS_ID = 0
_COCO_TENNIS_RACKET_CLASS_ID = 38
_LOGGER = logging.getLogger(__name__)


def _resolve_local_weight_path(model_name: str) -> Path | None:
    candidate = Path(model_name)
    if candidate.exists():
        return candidate

    project_root = Path(__file__).resolve().parents[2]
    assets_candidate = project_root / "assets" / "weights" / model_name
    if assets_candidate.exists():
        return assets_candidate

    return None


def _require_local_or_optin_weight(model_name: str) -> str:
    local_path = _resolve_local_weight_path(model_name)
    if local_path is not None:
        return str(local_path)

    if os.environ.get("POINTSTREAM_ALLOW_AUTO_MODEL_DOWNLOAD", "0") == "1":
        return model_name

    raise FileNotFoundError(
        f"Required model weights not found for '{model_name}'. "
        "Place weights in assets/weights/ or set POINTSTREAM_ALLOW_AUTO_MODEL_DOWNLOAD=1."
    )


def _clip_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    x1 = float(np.clip(x1, 0.0, width - 1.0))
    y1 = float(np.clip(y1, 0.0, height - 1.0))
    x2 = float(np.clip(x2, x1 + 1.0, width))
    y2 = float(np.clip(y2, y1 + 1.0, height))
    return [x1, y1, x2, y2]


def _bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _intersection_area(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _build_crop_with_padding(frame_bgr: np.ndarray, bbox: list[float], pad_ratio: float = 0.15) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio

    cx1 = int(max(0, np.floor(x1 - pad_x)))
    cy1 = int(max(0, np.floor(y1 - pad_y)))
    cx2 = int(min(w, np.ceil(x2 + pad_x)))
    cy2 = int(min(h, np.ceil(y2 + pad_y)))
    if cx2 <= cx1 or cy2 <= cy1:
        return np.empty((0, 0, 3), dtype=np.uint8), (0, 0)

    return frame_bgr[cy1:cy2, cx1:cx2].copy(), (cx1, cy1)


def coco17_to_dwpose18(coco17: np.ndarray, confidence_threshold: float = 0.2) -> np.ndarray:
    """Convert a COCO-17 pose [17,3] into a DWPose/OpenPose-18 pose [18,3]."""
    if coco17.shape != (17, 3):
        raise ValueError(f"Expected COCO pose shape [17,3], got: {coco17.shape}")

    out = np.zeros((18, 3), dtype=np.float32)
    # openpose18 index -> coco17 index, neck is synthesized
    op18_from_coco17 = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

    for op_idx, coco_idx in enumerate(op18_from_coco17):
        if coco_idx is None:
            continue
        out[op_idx] = coco17[coco_idx]

    # Synthesize neck if both shoulders are confident.
    lsho = coco17[5]
    rsho = coco17[6]
    if float(lsho[2]) >= confidence_threshold and float(rsho[2]) >= confidence_threshold:
        neck_xy = 0.5 * (lsho[:2] + rsho[:2])
        neck_conf = min(float(lsho[2]), float(rsho[2]))
        out[1] = np.array([neck_xy[0], neck_xy[1], neck_conf], dtype=np.float32)

    return out


class BaseDetector(ABC):
    @abstractmethod
    def track(self, frames_bgr: list[np.ndarray]) -> list[FrameState]:
        raise NotImplementedError


class Yolo26Detector(BaseDetector):
    def __init__(self, model_name: str = "yolo26n.pt", model: Any | None = None) -> None:
        self.model_name = model_name
        self._model: Any = model if model is not None else self._load_model()

    def _load_model(self) -> Any:
        from ultralytics import YOLO

        weight_ref = _require_local_or_optin_weight(self.model_name)
        try:
            return YOLO(weight_ref)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize detector model '{self.model_name}'") from exc

    def track(self, frames_bgr: list[np.ndarray]) -> list[FrameState]:
        if not frames_bgr:
            return []

        states: list[FrameState] = []
        previous_state: FrameState | None = None

        for frame_idx, frame in enumerate(frames_bgr):
            actors = self._detect_frame(frame, frame_idx)
            state = FrameState(frame_id=frame_idx, actors=actors)
            repaired = self._enforce_minimum_candidates(
                frame_bgr=frame,
                state=state,
                previous_state=previous_state,
            )
            states.append(repaired)
            previous_state = repaired

        return states

    def _detect_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> list[SceneActor]:
        detections: list[SceneActor] = []

        results = self._model.track(
            source=frame_bgr,
            persist=True,
            verbose=False,
            conf=0.1,
        )

        if not results:
            return detections

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        cls = boxes.cls.cpu().numpy().astype(np.int32) if getattr(boxes, "cls", None) is not None else np.zeros((xyxy.shape[0],), dtype=np.int32)

        if getattr(boxes, "id", None) is not None:
            tid_np = boxes.id.cpu().numpy().astype(np.int32) if hasattr(boxes.id, "cpu") else np.asarray(boxes.id, dtype=np.int32)
        else:
            tid_np = -np.ones((xyxy.shape[0],), dtype=np.int32)

        h, w = frame_bgr.shape[:2]
        for det_idx in range(xyxy.shape[0]):
            class_id = int(cls[det_idx])
            if class_id not in (_COCO_PERSON_CLASS_ID, _COCO_TENNIS_RACKET_CLASS_ID):
                continue

            class_name = "player" if class_id == _COCO_PERSON_CLASS_ID else "racket"
            track_id_int = int(tid_np[det_idx]) if det_idx < tid_np.shape[0] else -1
            if track_id_int < 0:
                track_id = f"{class_name}_{frame_idx}_{det_idx}"
            else:
                track_id = f"{class_name}_{track_id_int}"

            bbox = _clip_bbox(xyxy[det_idx].tolist(), width=w, height=h)
            detections.append(
                SceneActor(
                    track_id=track_id,
                    class_name=class_name,
                    bbox=bbox,
                )
            )

        return detections

    def _enforce_minimum_candidates(
        self,
        frame_bgr: np.ndarray,
        state: FrameState,
        previous_state: FrameState | None,
    ) -> FrameState:
        players = [a for a in state.actors if a.class_name == "player"]
        rackets = [a for a in state.actors if a.class_name == "racket"]

        if len(players) < 2:
            recovered_players = self.recover_missing_tracks(
                frame_bgr=frame_bgr,
                frame_id=state.frame_id,
                class_name="player",
                missing_count=2 - len(players),
                previous_state=previous_state,
            )
            players = players + recovered_players

        if len(rackets) < 2:
            recovered_rackets = self.recover_missing_tracks(
                frame_bgr=frame_bgr,
                frame_id=state.frame_id,
                class_name="racket",
                missing_count=2 - len(rackets),
                previous_state=previous_state,
            )
            rackets = rackets + recovered_rackets

        return FrameState(frame_id=state.frame_id, actors=(players[:2] + rackets[:2]))

    def recover_missing_tracks(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
        class_name: str,
        missing_count: int,
        previous_state: FrameState | None,
        max_retries: int = 1,
    ) -> list[SceneActor]:
        if missing_count <= 0:
            return []

        recovered: list[SceneActor] = []
        previous_candidates: list[SceneActor] = []
        if previous_state is not None:
            previous_candidates = [a for a in previous_state.actors if a.class_name == class_name]

        for miss_idx in range(missing_count):
            if miss_idx < len(previous_candidates):
                seed = previous_candidates[miss_idx]
                recovered_actor: SceneActor | None = None
                retry_budget = max(1, max_retries)
                for _ in range(retry_budget):
                    recovered_actor = self._recover_from_roi(
                        frame_bgr=frame_bgr,
                        class_name=class_name,
                        seed_actor=seed,
                    )
                    if recovered_actor is not None:
                        break
                if recovered_actor is not None:
                    recovered.append(recovered_actor)
                    continue

                recovered.append(
                    SceneActor(
                        track_id=seed.track_id,
                        class_name=class_name,
                        bbox=seed.bbox,
                    )
                )
                continue

            # Fallback synthetic interpolation if no previous track is available.
            h, w = frame_bgr.shape[:2]
            if class_name == "player":
                bbox = [0.05 * w, 0.15 * h, 0.2 * w, 0.55 * h] if miss_idx == 0 else [0.75 * w, 0.45 * h, 0.95 * w, 0.95 * h]
            else:
                bbox = [0.18 * w, 0.25 * h, 0.24 * w, 0.35 * h] if miss_idx == 0 else [0.70 * w, 0.55 * h, 0.76 * w, 0.65 * h]

            recovered.append(
                SceneActor(
                    track_id=f"{class_name}_interp_{frame_id}_{miss_idx}",
                    class_name=class_name,
                    bbox=_clip_bbox(bbox, width=w, height=h),
                )
            )

        return recovered

    def _recover_from_roi(self, frame_bgr: np.ndarray, class_name: str, seed_actor: SceneActor) -> SceneActor | None:
        crop, (ox, oy) = _build_crop_with_padding(frame_bgr, seed_actor.bbox, pad_ratio=0.20)
        if crop.size == 0:
            return None

        class_id = _COCO_PERSON_CLASS_ID if class_name == "player" else _COCO_TENNIS_RACKET_CLASS_ID
        results = self._model.predict(source=crop, classes=[class_id], verbose=False, conf=0.05)

        if not results:
            return None

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None or len(boxes.xyxy) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        best = xyxy[0]

        h, w = frame_bgr.shape[:2]
        mapped = [float(best[0] + ox), float(best[1] + oy), float(best[2] + ox), float(best[3] + oy)]
        return SceneActor(
            track_id=seed_actor.track_id,
            class_name=class_name,
            bbox=_clip_bbox(mapped, width=w, height=h),
        )


class BaseHeuristic(ABC):
    @abstractmethod
    def select(self, frame_state: FrameState, frame_shape: tuple[int, int]) -> FrameState:
        raise NotImplementedError


class StandardTennisHeuristic(BaseHeuristic):
    def select(self, frame_state: FrameState, frame_shape: tuple[int, int]) -> FrameState:
        h, w = frame_shape
        players = [a for a in frame_state.actors if a.class_name == "player"]
        rackets = [a for a in frame_state.actors if a.class_name == "racket"]

        selected_players = self._select_players(players=players, rackets=rackets, frame_width=w, frame_height=h)
        selected_rackets = self._select_rackets(rackets=rackets, selected_players=selected_players)

        return FrameState(frame_id=frame_state.frame_id, actors=selected_players + selected_rackets)

    def _select_players(
        self,
        players: list[SceneActor],
        rackets: list[SceneActor],
        frame_width: int,
        frame_height: int,
    ) -> list[SceneActor]:
        if len(players) <= 2:
            return players[:2]

        scored: list[tuple[float, SceneActor]] = []
        for actor in players:
            overlap_score = sum(_intersection_area(actor.bbox, racket.bbox) for racket in rackets)
            area_score = _bbox_area(actor.bbox)
            cx, _cy = _bbox_center(actor.bbox)
            center_score = 1.0 - abs(cx - frame_width * 0.5) / max(1.0, frame_width * 0.5)
            score = overlap_score * 1.0 + area_score * 0.01 + center_score * 100.0
            scored.append((score, actor))

        top_half = [item for item in scored if _bbox_center(item[1].bbox)[1] < frame_height * 0.5]
        bottom_half = [item for item in scored if _bbox_center(item[1].bbox)[1] >= frame_height * 0.5]

        selected: list[SceneActor] = []
        if top_half:
            selected.append(sorted(top_half, key=lambda p: p[0], reverse=True)[0][1])
        if bottom_half:
            selected.append(sorted(bottom_half, key=lambda p: p[0], reverse=True)[0][1])

        if len(selected) < 2:
            selected_ids = {s.track_id for s in selected}
            remaining = [item for item in scored if item[1].track_id not in selected_ids]
            remaining_sorted = sorted(remaining, key=lambda p: p[0], reverse=True)
            for _, actor in remaining_sorted:
                if len(selected) >= 2:
                    break
                selected.append(actor)

        return selected[:2]

    def _select_rackets(self, rackets: list[SceneActor], selected_players: list[SceneActor]) -> list[SceneActor]:
        if len(rackets) <= 2:
            return rackets[:2]

        scored: list[tuple[float, SceneActor]] = []
        for racket in rackets:
            area_score = _bbox_area(racket.bbox)
            proximity = 0.0
            rx, ry = _bbox_center(racket.bbox)
            for player in selected_players:
                px, py = _bbox_center(player.bbox)
                distance = float(np.hypot(rx - px, ry - py))
                proximity += 1.0 / max(distance, 1.0)
            score = area_score * 0.02 + proximity * 1000.0
            scored.append((score, racket))

        scored_sorted = sorted(scored, key=lambda p: p[0], reverse=True)
        return [actor for _, actor in scored_sorted[:2]]


class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, frame_bgr: np.ndarray, actor: SceneActor) -> SceneActor:
        raise NotImplementedError


class NoOpSegmenter(BaseSegmenter):
    def segment(self, frame_bgr: np.ndarray, actor: SceneActor) -> SceneActor:
        _ = frame_bgr
        return actor


class YoloSegmenter(BaseSegmenter):
    def __init__(self, model_name: str = "yolo26n-seg.pt", model: Any | None = None) -> None:
        self.model_name = model_name
        self._model: Any = model if model is not None else self._load_model()

    def _load_model(self) -> Any:
        from ultralytics import YOLO

        weight_ref = _require_local_or_optin_weight(self.model_name)
        try:
            return YOLO(weight_ref)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize segmenter model '{self.model_name}'") from exc

    def segment(self, frame_bgr: np.ndarray, actor: SceneActor) -> SceneActor:
        if actor.mask is not None:
            return actor

        crop, _ = _build_crop_with_padding(frame_bgr, actor.bbox, pad_ratio=0.0)
        if crop.size == 0:
            return actor

        try:
            results = self._model.predict(source=crop, verbose=False, conf=0.2)
        except Exception as exc:
            _LOGGER.warning("Segmenter inference failed for actor %s on current frame: %s", actor.track_id, exc)
            return actor

        if not results:
            return actor

        masks = getattr(results[0], "masks", None)
        if masks is None or getattr(masks, "data", None) is None or len(masks.data) == 0:
            return actor

        mask_np = masks.data[0]
        if hasattr(mask_np, "cpu"):
            mask_np = mask_np.cpu().numpy()
        mask_bin = (np.asarray(mask_np) > 0.5).astype(np.uint8).tolist()

        mask_polygons: list[list[list[float]]] | None = None
        segments_xy = getattr(masks, "xy", None)
        if segments_xy is not None and len(segments_xy) > 0:
            try:
                first_segment = np.asarray(segments_xy[0], dtype=np.float32)
                if first_segment.ndim == 2 and first_segment.shape[1] == 2 and first_segment.shape[0] >= 3:
                    mask_polygons = [first_segment.tolist()]
            except Exception:
                mask_polygons = None

        return actor.model_copy(update={"mask": mask_bin, "mask_polygons": mask_polygons})


class BasePoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frame_bgr: np.ndarray, actor: SceneActor) -> SceneActor:
        raise NotImplementedError


class YoloPoseEstimator(BasePoseEstimator):
    def __init__(self, model_name: str = "yolo26n-pose.pt", model: Any | None = None) -> None:
        self.model_name = model_name
        self._model: Any = model if model is not None else self._load_model()

    def _load_model(self) -> Any:
        from ultralytics import YOLO

        weight_ref = _require_local_or_optin_weight(self.model_name)
        try:
            return YOLO(weight_ref)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize pose model '{self.model_name}'") from exc

    def estimate(self, frame_bgr: np.ndarray, actor: SceneActor) -> SceneActor:
        if actor.class_name != "player":
            return actor

        crop, (ox, oy) = _build_crop_with_padding(frame_bgr, actor.bbox, pad_ratio=0.10)
        if crop.size == 0:
            return actor

        results = self._model.predict(source=crop, verbose=False, conf=0.2)

        if not results:
            return actor

        keypoints = getattr(results[0], "keypoints", None)
        if keypoints is None or getattr(keypoints, "xy", None) is None or len(keypoints.xy) == 0:
            return actor

        xy = keypoints.xy[0]
        conf = keypoints.conf[0] if getattr(keypoints, "conf", None) is not None else None

        if hasattr(xy, "cpu"):
            xy_np = xy.cpu().numpy()
        else:
            xy_np = np.asarray(xy)

        if conf is None:
            conf_np = np.ones((xy_np.shape[0],), dtype=np.float32)
        elif hasattr(conf, "cpu"):
            conf_np = conf.cpu().numpy().astype(np.float32)
        else:
            conf_np = np.asarray(conf, dtype=np.float32)

        if xy_np.shape[0] != 17:
            raise ValueError(f"YOLO pose keypoint shape mismatch for actor {actor.track_id}: {xy_np.shape}")

        coco17 = np.concatenate([xy_np, conf_np[:, None]], axis=-1).astype(np.float32)
        coco17[:, 0] += float(ox)
        coco17[:, 1] += float(oy)

        pose_dw = coco17_to_dwpose18(coco17)
        return actor.model_copy(update={"pose_dw": pose_dw.tolist()})


class DwposeEstimator(BasePoseEstimator):
    def __init__(self, torchscript_device: str = "cuda") -> None:
        self.torchscript_device = torchscript_device
        self._model: Any = self._load_model()

    def _load_model(self) -> Any:
        from dwpose import DwposeDetector

        try:
            return DwposeDetector.from_pretrained_default(torchscript_device=self.torchscript_device)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize DWPose estimator on device '{self.torchscript_device}'") from exc

    def estimate(self, frame_bgr: np.ndarray, actor: SceneActor) -> SceneActor:
        if actor.class_name != "player":
            return actor

        crop, (ox, oy) = _build_crop_with_padding(frame_bgr, actor.bbox, pad_ratio=0.10)
        if crop.size == 0:
            return actor

        _canvas, pose_json, _ = self._model(
            crop,
            output_type="np",
            image_and_json=True,
            include_body=True,
            include_hand=False,
            include_face=False,
        )

        people = pose_json.get("people", []) if isinstance(pose_json, dict) else []
        if not people:
            return actor

        raw = people[0].get("pose_keypoints_2d")
        if not raw:
            return actor

        pts = np.asarray(raw, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 17:
            dw = coco17_to_dwpose18(pts)
        elif pts.shape[0] >= 18:
            dw = pts[:18].copy()
        else:
            return actor

        dw[:, 0] += float(ox)
        dw[:, 1] += float(oy)
        return actor.model_copy(update={"pose_dw": dw.tolist()})


@dataclass
class PayloadEncoder:
    pose_delta_threshold: float = 20.0
    include_mask_metadata: bool = False
    metadata_mask_codec: str = "auto"

    def __post_init__(self) -> None:
        self._last_transmitted_pose_coords: dict[str, np.ndarray] = {}
        codec_raw = str(self.metadata_mask_codec).strip()
        if codec_raw == "auto":
            codec_raw = os.environ.get("POINTSTREAM_METADATA_MASK_CODEC", codec_raw)

        codec_normalized = codec_raw.strip().lower().replace("_", "-")
        if codec_normalized in {"segmenter-native", "yolo-native"}:
            self.metadata_mask_codec = "segmenter-native"
        else:
            self.metadata_mask_codec = normalize_mask_codec(codec_normalized)

    def encode(self, chunk: VideoChunk, frame_states: list[FrameState]) -> list[ActorPacket]:
        actor_events: dict[str, list[SemanticEvent]] = {}
        actor_masks: dict[str, list[ActorMaskFrame]] = {}

        for state in frame_states:
            frame_id = chunk.start_frame_id + state.frame_id
            for actor in state.actors:
                if actor.class_name != "player":
                    continue

                if self.include_mask_metadata:
                    maybe_mask = self._encode_mask_frame(chunk=chunk, actor=actor, frame_id=frame_id)
                    if maybe_mask is not None:
                        actor_masks.setdefault(actor.track_id, []).append(maybe_mask)

                if actor.pose_dw is None:
                    continue

                coords = np.asarray(actor.pose_dw, dtype=np.float32)
                if coords.shape != (18, 3):
                    continue

                key = actor.track_id
                events = actor_events.setdefault(key, [])
                flattened = coords.reshape(-1)
                last = self._last_transmitted_pose_coords.get(key)

                if last is None:
                    events.append(
                        KeyframeEvent(
                            frame_id=frame_id,
                            object_id=actor.track_id,
                            object_class=ObjectClass.PERSON,
                            coordinates=flattened.tolist(),
                        )
                    )
                    self._last_transmitted_pose_coords[key] = flattened
                    continue

                delta = float(np.linalg.norm(flattened - last, ord=2))
                if delta >= self.pose_delta_threshold:
                    events.append(
                        KeyframeEvent(
                            frame_id=frame_id,
                            object_id=actor.track_id,
                            object_class=ObjectClass.PERSON,
                            coordinates=flattened.tolist(),
                        )
                    )
                    self._last_transmitted_pose_coords[key] = flattened
                else:
                    events.append(
                        InterpolateCommandEvent(
                            frame_id=frame_id,
                            object_id=actor.track_id,
                            object_class=ObjectClass.PERSON,
                            target_frame_id=min(frame_id + 1, chunk.start_frame_id + chunk.num_frames - 1),
                            method="linear",
                        )
                    )

        actor_ids = sorted(actor_events.keys())

        if not actor_ids:
            return []

        packets: list[ActorPacket] = []
        for actor_id in sorted(actor_events.keys())[:2]:
            packets.append(
                ActorPacket(
                    chunk_id=chunk.chunk_id,
                    object_id=actor_id,
                    appearance_embedding_spec=TensorSpec(
                        name="actor_appearance",
                        shape=[1, 2, 256],
                        dtype=str(torch.float32),
                    ),
                    pose_tensor_spec=TensorSpec(
                        name="actor_pose_dw",
                        shape=[1, chunk.num_frames, 18, 3],
                        dtype=str(torch.float32),
                    ),
                    events=actor_events[actor_id],
                    mask_frames=actor_masks.get(actor_id, []),
                )
            )

        return packets

    def _encode_mask_frame(self, chunk: VideoChunk, actor: SceneActor, frame_id: int) -> ActorMaskFrame | None:
        if actor.mask is None:
            return None

        raw_mask = np.asarray(actor.mask, dtype=np.float32)
        if raw_mask.ndim != 2 or raw_mask.size == 0:
            return None

        if self.metadata_mask_codec == "segmenter-native":
            native_mask_frame = self._encode_segmenter_native_mask(
                chunk=chunk,
                actor=actor,
                frame_id=frame_id,
                mask_shape=(int(raw_mask.shape[0]), int(raw_mask.shape[1])),
            )
            if native_mask_frame is not None:
                return native_mask_frame

        try:
            codec_name, encoded_payload, mask_height, mask_width = encode_binary_mask(
                raw_mask,
                codec="auto" if self.metadata_mask_codec == "segmenter-native" else self.metadata_mask_codec,
            )
        except Exception:
            return None

        clipped_bbox = _clip_bbox(actor.bbox, width=int(chunk.width), height=int(chunk.height))
        x1 = int(np.floor(clipped_bbox[0]))
        y1 = int(np.floor(clipped_bbox[1]))
        x2 = int(np.ceil(clipped_bbox[2]))
        y2 = int(np.ceil(clipped_bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None

        return ActorMaskFrame(
            frame_id=int(frame_id),
            bbox=[x1, y1, x2, y2],
            mask_codec=codec_name,
            mask_payload=encoded_payload,
            mask_height=int(mask_height),
            mask_width=int(mask_width),
            source="source",
        )

    def _encode_segmenter_native_mask(
        self,
        chunk: VideoChunk,
        actor: SceneActor,
        frame_id: int,
        mask_shape: tuple[int, int],
    ) -> ActorMaskFrame | None:
        if not actor.mask_polygons:
            return None

        mask_height, mask_width = mask_shape
        if mask_height <= 0 or mask_width <= 0:
            return None

        try:
            codec_name, encoded_payload, encoded_h, encoded_w = encode_polygon_mask(
                actor.mask_polygons,
                height=int(mask_height),
                width=int(mask_width),
                codec="poly-v1",
            )
        except Exception:
            return None

        clipped_bbox = _clip_bbox(actor.bbox, width=int(chunk.width), height=int(chunk.height))
        x1 = int(np.floor(clipped_bbox[0]))
        y1 = int(np.floor(clipped_bbox[1]))
        x2 = int(np.ceil(clipped_bbox[2]))
        y2 = int(np.ceil(clipped_bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None

        return ActorMaskFrame(
            frame_id=int(frame_id),
            bbox=[x1, y1, x2, y2],
            mask_codec=codec_name,
            mask_payload=encoded_payload,
            mask_height=int(encoded_h),
            mask_width=int(encoded_w),
            source="source",
        )


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

    def run(self, chunk: VideoChunk, frames_bgr: list[np.ndarray]) -> tuple[list[FrameState], list[ActorPacket]]:
        frame_states = self.detector.track(frames_bgr)

        filtered_states: list[FrameState] = []
        for state in frame_states:
            if state.frame_id >= len(frames_bgr):
                continue

            frame = frames_bgr[state.frame_id]
            h, w = frame.shape[:2]
            selected = self.heuristic.select(state, frame_shape=(h, w))

            updated_actors: list[SceneActor] = []
            for actor in selected.actors:
                segmented = self.segmenter.segment(frame, actor)
                with_pose = self.pose_estimator.estimate(frame, segmented)
                updated_actors.append(with_pose)

            filtered_states.append(FrameState(frame_id=state.frame_id, actors=updated_actors))

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

        frame_state_by_id = {chunk.start_frame_id + s.frame_id: s for s in frame_states}
        for global_frame_id in sorted(keyframe_ids):
            local_idx = global_frame_id - chunk.start_frame_id
            if local_idx < 0 or local_idx >= len(frames_bgr):
                continue

            frame = frames_bgr[local_idx]
            state = frame_state_by_id.get(global_frame_id)
            if state is None:
                continue

            people_dw = []
            for actor in state.actors:
                if actor.class_name != "player" or actor.pose_dw is None:
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
                return
            overlay = frame.copy()
            mask = np.any(skeleton_canvas > 0, axis=2)
            overlay[mask] = skeleton_canvas[mask]

            stem = f"{chunk.chunk_id}_frame_{global_frame_id:05d}"
            cv2.imwrite(str(out_path / f"{stem}_skeleton_black.png"), skeleton_canvas)
            cv2.imwrite(str(out_path / f"{stem}_skeleton_overlay.png"), overlay)


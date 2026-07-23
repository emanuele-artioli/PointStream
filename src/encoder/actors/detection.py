"""Actor detectors: the YOLO backends that find people and rackets."""

from __future__ import annotations
from src.shared.player_extraction import build_crop_with_padding
from abc import ABC, abstractmethod
import logging
from typing import Any
import numpy as np
from src.shared.schemas import FrameState, SceneActor
from src.encoder.actors.weights import _clip_bbox, _configure_ultralytics_weights_dir, _require_local_or_optin_weight, _resolve_local_weight_path
_COCO_PERSON_CLASS_ID = 0
_COCO_TENNIS_RACKET_CLASS_ID = 38
_LOGGER = logging.getLogger(__name__)


class BaseDetector(ABC):
    @abstractmethod
    def track(self, frames_bgr: list[np.ndarray]) -> list[FrameState]:
        raise NotImplementedError

    def iter_track(self, frames_bgr: list[np.ndarray]):
        for state in self.track(frames_bgr):
            yield state
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

    def iter_track(self, frames_bgr: list[np.ndarray]):
        if not frames_bgr:
            return

        previous_state: FrameState | None = None

        for frame_idx, frame in enumerate(frames_bgr):
            actors = self._detect_frame(frame, frame_idx)
            state = FrameState(frame_id=frame_idx, actors=actors)
            repaired = self._enforce_minimum_candidates(
                frame_bgr=frame,
                state=state,
                previous_state=previous_state,
            )
            previous_state = repaired
            yield repaired

    def track(self, frames_bgr: list[np.ndarray]) -> list[FrameState]:
        return list(self.iter_track(frames_bgr))

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
        crop, (ox, oy) = build_crop_with_padding(frame_bgr, seed_actor.bbox, pad_ratio=0.20)
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
class YoloEDetector(Yolo26Detector):
    def __init__(
        self,
        model_name: str = "yoloe-26n-seg.pt",
        model: Any | None = None,
        captions: list[str] | None = None,
    ) -> None:
        self._captions = [caption.strip() for caption in (captions or ["tennis player"]) if caption.strip()]
        super().__init__(model_name=model_name, model=model)
        self._configure_classes()

    def _load_model(self) -> Any:
        from ultralytics import YOLOE

        _configure_ultralytics_weights_dir()
        weight_ref = _require_local_or_optin_weight(self.model_name)
        try:
            return YOLOE(weight_ref)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize YOLOE detector model '{self.model_name}'") from exc

    def _configure_classes(self) -> None:
        if not self._captions:
            return

        # YOLOE text prompting needs MobileCLIP TorchScript weights. We require
        # this file to be present locally to keep runtime artifacts deterministic.
        text_encoder_name = "mobileclip2_b.ts"
        if _resolve_local_weight_path(text_encoder_name) is None:
            raise FileNotFoundError(
                "Required YOLOE text encoder weights not found for 'mobileclip2_b.ts'. "
                "Place it in assets/weights/ before using detector/segmenter captions."
            )

        _configure_ultralytics_weights_dir()
        try:
            self._model.set_classes(self._captions)
        except Exception as exc:
            _LOGGER.warning("YOLOE detector class prompt setup failed; continuing without prompts: %s", exc)

    def _detect_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> list[SceneActor]:
        detections: list[SceneActor] = []

        try:
            results = self._model.track(
                source=frame_bgr,
                persist=True,
                verbose=False,
                conf=0.1,
            )
        except Exception as exc:
            _LOGGER.warning("YOLOE detector inference failed on frame %s: %s", frame_idx, exc)
            return detections

        if not results:
            return detections

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        cls = (
            boxes.cls.cpu().numpy().astype(np.int32)
            if getattr(boxes, "cls", None) is not None
            else np.zeros((xyxy.shape[0],), dtype=np.int32)
        )

        if getattr(boxes, "id", None) is not None:
            tid_np = (
                boxes.id.cpu().numpy().astype(np.int32)
                if hasattr(boxes.id, "cpu")
                else np.asarray(boxes.id, dtype=np.int32)
            )
        else:
            tid_np = -np.ones((xyxy.shape[0],), dtype=np.int32)

        names = getattr(result, "names", {}) or {}

        h, w = frame_bgr.shape[:2]
        for det_idx in range(xyxy.shape[0]):
            class_id = int(cls[det_idx]) if det_idx < cls.shape[0] else 0
            class_name_raw = str(names.get(class_id, "player")).strip().lower()
            class_name = "racket" if "racket" in class_name_raw else "player"

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

    def _recover_from_roi(self, frame_bgr: np.ndarray, class_name: str, seed_actor: SceneActor) -> SceneActor | None:
        crop, (ox, oy) = build_crop_with_padding(frame_bgr, seed_actor.bbox, pad_ratio=0.20)
        if crop.size == 0:
            return None

        try:
            results = self._model.predict(source=crop, verbose=False, conf=0.05)
        except Exception:
            return None

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

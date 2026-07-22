"""Segmenters producing the actor masks that drive compositing."""

from __future__ import annotations
from src.shared.player_extraction import build_crop_with_padding
from abc import ABC, abstractmethod
import logging
from typing import Any
import cv2
import numpy as np
from src.shared.schemas import SceneActor
from src.encoder.actors.weights import _clip_bbox, _configure_ultralytics_weights_dir, _require_local_or_optin_weight, _resolve_local_weight_path
_COCO_PERSON_CLASS_ID = 0
_COCO_TENNIS_RACKET_CLASS_ID = 38
_LOGGER = logging.getLogger(__name__)


class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, frame_bgr: np.ndarray, actor: SceneActor) -> SceneActor:
        raise NotImplementedError
class NoOpSegmenter(BaseSegmenter):
    def segment(self, frame_bgr: np.ndarray, actor: SceneActor) -> SceneActor:
        _ = frame_bgr
        return actor
class CannySegmenter(BaseSegmenter):
    def __init__(self, base_segmenter: BaseSegmenter | None = None, lower_threshold: str | int = "auto", upper_threshold: str | int = "auto") -> None:
        self.base_segmenter = base_segmenter
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def segment(self, frame_bgr: np.ndarray, actor: SceneActor) -> SceneActor:
        if actor.mask is not None:
            return actor

        # Optional: run semantic segmentation first
        semantic_mask = None
        if self.base_segmenter is not None:
            actor = self.base_segmenter.segment(frame_bgr, actor)
            if actor.mask is not None:
                semantic_mask = np.asarray(actor.mask, dtype=np.uint8)

        crop, _ = build_crop_with_padding(frame_bgr, actor.bbox, pad_ratio=0.0)
        if crop.size == 0:
            return actor

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        low = self.lower_threshold
        high = self.upper_threshold
        
        if str(low).lower() == "auto" or str(high).lower() == "auto":
            v = float(np.median(gray))
            sigma = 0.33
            if str(low).lower() == "auto":
                low = int(max(0.0, (1.0 - sigma) * v))
            if str(high).lower() == "auto":
                high = int(min(255.0, (1.0 + sigma) * v))
                
        edges = cv2.Canny(gray, int(low), int(high))
        
        if semantic_mask is not None:
            if semantic_mask.shape != edges.shape:
                semantic_mask = np.asarray(
                    cv2.resize(semantic_mask, (edges.shape[1], edges.shape[0]), interpolation=cv2.INTER_NEAREST),
                    dtype=np.uint8,
                )
            
            # Dilate the semantic mask to ensure it covers the Canny edges of the player boundary
            # Use a kernel size relative to the crop size, e.g., 5x5 or 7x7
            kernel = np.ones((5, 5), np.uint8)
            semantic_mask_dilated = cv2.dilate(semantic_mask, kernel, iterations=1)
            
            # Mask the Canny edges with the dilated semantic mask
            edges = cv2.bitwise_and(edges, edges, mask=semantic_mask_dilated)
            
        mask_bin = ((edges > 127).astype(np.uint8) * 255).tolist()
        return actor.model_copy(update={"mask": mask_bin})
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

        crop, _ = build_crop_with_padding(frame_bgr, actor.bbox, pad_ratio=0.0)
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
class YoloeSegmenter(YoloSegmenter):
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
            raise RuntimeError(f"Failed to initialize YOLOE segmenter model '{self.model_name}'") from exc

    def _configure_classes(self) -> None:
        if not self._captions:
            return

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
            _LOGGER.warning("YOLOE segmenter class prompt setup failed; continuing without prompts: %s", exc)
class SamSegmenter(BaseSegmenter):
    def __init__(self, model_name: str = "sam3.pt", model: Any | None = None) -> None:
        self.model_name = model_name
        self._model: Any = model if model is not None else self._load_model()

    def _load_model(self) -> Any:
        from ultralytics import SAM

        weight_ref = _require_local_or_optin_weight(self.model_name)
        try:
            return SAM(weight_ref)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize SAM segmenter model '{self.model_name}'") from exc

    def segment(self, frame_bgr: np.ndarray, actor: SceneActor) -> SceneActor:
        if actor.mask is not None:
            return actor

        h, w = frame_bgr.shape[:2]
        clipped_bbox = _clip_bbox(actor.bbox, width=w, height=h)
        x1 = int(np.floor(clipped_bbox[0]))
        y1 = int(np.floor(clipped_bbox[1]))
        x2 = int(np.ceil(clipped_bbox[2]))
        y2 = int(np.ceil(clipped_bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return actor

        bbox_prompt = np.asarray([[x1, y1, x2, y2]], dtype=np.float32)
        try:
            results = self._model.predict(source=frame_bgr, bboxes=bbox_prompt, verbose=False)
        except Exception as exc:
            _LOGGER.warning("SAM segmenter inference failed for actor %s on current frame: %s", actor.track_id, exc)
            return actor

        if not results:
            return actor

        masks = getattr(results[0], "masks", None)
        if masks is None or getattr(masks, "data", None) is None or len(masks.data) == 0:
            return actor

        mask_np = masks.data[0]
        if hasattr(mask_np, "cpu"):
            mask_np = mask_np.cpu().numpy()
        mask_bin_global = (np.asarray(mask_np) > 0.5).astype(np.uint8)
        if mask_bin_global.ndim != 2:
            return actor

        local_mask = mask_bin_global[y1:y2, x1:x2]
        if local_mask.size == 0:
            return actor

        mask_polygons: list[list[list[float]]] | None = None
        segments_xy = getattr(masks, "xy", None)
        if segments_xy is not None and len(segments_xy) > 0:
            try:
                first_segment = np.asarray(segments_xy[0], dtype=np.float32)
                if first_segment.ndim == 2 and first_segment.shape[1] == 2 and first_segment.shape[0] >= 3:
                    first_segment[:, 0] -= float(x1)
                    first_segment[:, 1] -= float(y1)
                    mask_polygons = [first_segment.tolist()]
            except Exception:
                mask_polygons = None

        return actor.model_copy(update={"mask": local_mask.tolist(), "mask_polygons": mask_polygons})

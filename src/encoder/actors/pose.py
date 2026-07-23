"""Pose estimators producing the keypoints used as generative conditioning."""

from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import Any
import numpy as np
from src.shared.schemas import SceneActor
from src.shared.player_extraction import extract_pose_dwpose18
from src.encoder.actors.weights import _require_local_or_optin_weight
_COCO_PERSON_CLASS_ID = 0
_COCO_TENNIS_RACKET_CLASS_ID = 38
_LOGGER = logging.getLogger(__name__)


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

        pose_dw = extract_pose_dwpose18(frame_bgr, actor.bbox, self._model, model_type="yolo", pad_ratio=0.10)
        if pose_dw is not None:
            return actor.model_copy(update={"pose_dw": pose_dw.tolist()})
        return actor
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

        pose_dw = extract_pose_dwpose18(frame_bgr, actor.bbox, self._model, model_type="dwpose", pad_ratio=0.10)
        if pose_dw is not None:
            return actor.model_copy(update={"pose_dw": pose_dw.tolist()})
        return actor

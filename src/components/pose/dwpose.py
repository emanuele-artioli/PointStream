"""Registered DWPose WholeBody estimator with explicit missing-joint semantics."""

from __future__ import annotations

import os
import hashlib
import importlib.metadata
import json
from typing import Any

import cv2
import numpy as np

from src.components.detection.geometry import Box
from src.components.detection.types import Detection, is_person
from src.components.detection.weights import resolve_weight
from src.components.perception.coordinates import CropTransform, crop_and_mask, make_crop_transform
from src.components.pose.wire import Pose
from src.contracts.keypoints import COCO_WHOLEBODY_133
from src.contracts.observation import EstimatorProvenance


class DWPoseEstimator:
    """Estimate one WholeBody-133 pose from a detection crop.

    Confidence is preserved joint by joint. Scores above ``visible_threshold``
    are marked visible, lower positive scores are marked present with low
    visibility, and absent joints have coordinates and confidence set to zero.
    DWPose ONNX weights are local files; this class never downloads models.
    """

    def __init__(
        self,
        model_name: str = "dw-ll_ucoco_384.onnx",
        det_model_name: str = "yolox_l.onnx",
        model: Any | None = None,
        device: str = "cuda",
        min_confidence: float = 0.05,
        visible_threshold: float = 0.5,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("DWPose device must be 'cpu' or 'cuda'")
        if not 0.0 <= min_confidence <= visible_threshold <= 1.0:
            raise ValueError("DWPose confidence thresholds must satisfy 0 <= min <= visible <= 1")
        self.model_name = model_name
        self.det_model_name = det_model_name
        self.device = device
        self.min_confidence = float(min_confidence)
        self.visible_threshold = float(visible_threshold)
        self.emits = COCO_WHOLEBODY_133
        self._model_revision = "injected-model"
        self._checkpoint_sha256: str | None = None
        self._model = model if model is not None else self._load_model()

    def _load_model(self) -> Any:
        det_path = resolve_dwpose_weight(self.det_model_name, "PS_DWPOSE_DET")
        pose_path = resolve_dwpose_weight(self.model_name, "PS_DWPOSE_POSE")
        detector_hash = _sha256(det_path)
        pose_hash = _sha256(pose_path)
        self._checkpoint_sha256 = hashlib.sha256(
            f"det={detector_hash};pose={pose_hash}".encode("ascii")
        ).hexdigest()
        try:
            rtmlib_version = importlib.metadata.version("rtmlib")
        except importlib.metadata.PackageNotFoundError:
            rtmlib_version = "unknown"
        self._model_revision = f"rtmlib-{rtmlib_version}"
        try:
            from rtmlib import Wholebody
        except ImportError as exc:
            raise RuntimeError("DWPose requires rtmlib and local ONNX weights.") from exc
        return Wholebody(
            det=str(det_path),
            det_input_size=(640, 640),
            pose=str(pose_path),
            pose_input_size=(288, 384),
            to_openpose=False,
            backend="onnxruntime",
            device=self.device,
        )

    def provenance(self, *, policy: str = "offline_bidirectional") -> EstimatorProvenance:
        """Return the exact model, weights, configuration, and execution policy."""
        config = {
            "det_model": self.det_model_name,
            "pose_model": self.model_name,
            "det_input_size": [640, 640],
            "pose_input_size": [288, 384],
            "to_openpose": False,
            "backend": "onnxruntime",
            "device": self.device,
            "min_confidence": self.min_confidence,
            "visible_threshold": self.visible_threshold,
        }
        config_sha256 = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        checkpoint_hash = self._checkpoint_sha256 or hashlib.sha256(
            b"injected-test-dwpose-model"
        ).hexdigest()
        return EstimatorProvenance(
            name="dwpose-wholebody-133",
            model_revision=self._model_revision,
            checkpoint_sha256=checkpoint_hash,
            config_sha256=config_sha256,
            policy=policy,
        )

    def estimate(
        self,
        frame: np.ndarray,
        detection: Detection,
        *,
        bbox: Box | None = None,
        mask: np.ndarray | None = None,
    ) -> Pose | None:
        """Estimate a single person in an object crop and restore frame pixels."""
        pose, _ = self.estimate_with_transform(frame, detection, bbox=bbox, mask=mask)
        return pose

    def estimate_with_transform(
        self,
        frame: np.ndarray,
        detection: Detection,
        *,
        bbox: Box | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[Pose | None, CropTransform | None]:
        """Estimate a pose and return the reversible crop transform used by DWPose."""
        if not is_person(detection.class_name):
            return None, None
        image = np.asarray(frame)
        height, width = image.shape[:2]
        box = (bbox or detection.bbox).clip(width, height)
        x0, y0, x1, y1 = (
            int(np.floor(box.x1)),
            int(np.floor(box.y1)),
            int(np.ceil(box.x2)),
            int(np.ceil(box.y2)),
        )
        if x1 <= x0 or y1 <= y0:
            return None, None
        transform = make_crop_transform(
            (x0, y0, x1, y1),
            source_size=(width, height),
            target_size=(x1 - x0, y1 - y0),
        )
        if mask is not None:
            binary = np.asarray(mask) != 0
            if binary.shape == (y1 - y0, x1 - x0):
                full_frame_mask = np.zeros(image.shape[:2], dtype=bool)
                full_frame_mask[y0:y1, x0:x1] = binary
                binary = full_frame_mask
            elif binary.shape != image.shape[:2]:
                raise ValueError("DWPose mask must match either the full frame or its detection crop")
        else:
            binary = np.ones(image.shape[:2], dtype=np.uint8)
        crop, _ = crop_and_mask(image, binary, transform)
        if crop.size == 0:
            return None, transform

        keypoints, scores = self._model(crop)
        candidates = _normalise_outputs(keypoints, scores)
        if not candidates:
            return None, transform
        candidate = _select_person(candidates, crop.shape[1], crop.shape[0])
        values = candidate.copy()
        absent = values[:, 2] < self.min_confidence
        values[~absent, :2] = transform.canvas_to_source(values[~absent, :2])
        values[absent] = 0.0
        present = ~absent
        visibility = np.where(
            absent,
            0,
            np.where(values[:, 2] >= self.visible_threshold, 2, 1),
        ).astype(np.uint8)
        return (
            Pose(
                schema=COCO_WHOLEBODY_133,
                values=values,
                present=present,
                visibility=visibility,
            ),
            transform,
        )

    def estimate_to_schema(
        self,
        frame: np.ndarray,
        detection: Detection,
        consumer: Any,
        *,
        bbox: Box | None = None,
        mask: np.ndarray | None = None,
    ) -> Pose | None:
        from src.components.pose.wire import to_wire

        pose = self.estimate(frame, detection, bbox=bbox, mask=mask)
        return None if pose is None else to_wire(pose, consumer)


def resolve_dwpose_weight(name: str, env_name: str):
    """Resolve a local DWPose file from an environment override or weight slot."""
    explicit = os.environ.get(env_name, "").strip()
    return resolve_weight(explicit or name)


def _sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_outputs(keypoints: Any, scores: Any) -> list[np.ndarray]:
    if keypoints is None:
        return []
    points = np.asarray(keypoints, dtype=np.float32)
    if points.size == 0:
        return []
    if points.ndim == 2:
        points = points[None, ...]
    if points.ndim != 3 or points.shape[1] < len(COCO_WHOLEBODY_133) or points.shape[2] < 2:
        return []
    score_array = np.asarray(scores, dtype=np.float32) if scores is not None else None
    if score_array is not None:
        if score_array.ndim == 1:
            score_array = score_array[None, ...]
        if score_array.ndim != 2 or score_array.shape[0] != points.shape[0]:
            return []
    result: list[np.ndarray] = []
    for index, person in enumerate(points):
        values = np.zeros((len(COCO_WHOLEBODY_133), 3), dtype=np.float32)
        values[:, :2] = person[: len(COCO_WHOLEBODY_133), :2]
        if score_array is None:
            values[:, 2] = 1.0
        elif score_array.shape[1] >= len(COCO_WHOLEBODY_133):
            values[:, 2] = score_array[index, : len(COCO_WHOLEBODY_133)]
        else:
            continue
        if np.all(np.isfinite(values)):
            result.append(values)
    return result


def _select_person(candidates: list[np.ndarray], width: int, height: int) -> np.ndarray:
    """Choose the crop-centered person when a detector returns several poses."""
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    normalizer = max(1.0, float(np.hypot(width, height)))

    def score(values: np.ndarray) -> tuple[float, float]:
        visible = values[:, 2] >= 0.05
        if not bool(visible.any()):
            return (-1.0, -np.inf)
        points = values[visible, :2]
        box_center = 0.5 * (points.min(axis=0) + points.max(axis=0))
        distance = float(np.linalg.norm(box_center - center)) / normalizer
        confidence = float(np.mean(values[visible, 2]))
        return (confidence - distance, confidence)

    return max(candidates, key=score)


__all__ = ["DWPoseEstimator"]

"""Object Keypoint Similarity (OKS) metric using a large YOLO pose oracle.

Evaluates human anatomical fidelity between reference and predicted frames.
Rather than thresholded binary hit/miss (PCK), this measures continuous
Gaussian similarity (OKS) and continuous normalized Euclidean keypoint drift.

Default model: yolo26x-pose.pt (Extra-Large YOLO pose), with yolov8x-pose.pt
as supported alternative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.components.detection.weights import resolve_weight
from src.components.metrics.frames import paired

#: Standard COCO-17 keypoint standard deviations (sigmas) for OKS calculation.
COCO_17_SIGMAS: np.ndarray = np.array(
    [
        0.026,  # 0: nose
        0.025,  # 1: left eye
        0.025,  # 2: right eye
        0.035,  # 3: left ear
        0.035,  # 4: right ear
        0.079,  # 5: left shoulder
        0.079,  # 6: right shoulder
        0.072,  # 7: left elbow
        0.072,  # 8: right elbow
        0.062,  # 9: left wrist
        0.062,  # 10: right wrist
        0.107,  # 11: left hip
        0.107,  # 12: right hip
        0.087,  # 13: left knee
        0.087,  # 14: right knee
        0.089,  # 15: left ankle
        0.089,  # 16: right ankle
    ],
    dtype=np.float64,
)

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _load_yolo_pose(model_name: str, device: str) -> Any:
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        from ultralytics import YOLO

        path = resolve_weight(model_name)
        model = YOLO(str(path))
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


@dataclass(frozen=True)
class PoseDetailedResult:
    """Detailed anatomical evaluation for one frame or clip."""

    oks: float
    mean_drift_norm: float
    ref_detected: bool
    pred_detected: bool
    dropout_count: int
    num_visible_gt: int
    mean_conf_diff: float


def compute_oks(
    pts_pred: np.ndarray,
    pts_gt: np.ndarray,
    *,
    bbox_gt: tuple[float, float, float, float] | None = None,
    conf_pred: np.ndarray | None = None,
    conf_gt: np.ndarray | None = None,
    conf_thresh: float = 0.25,
    sigmas: np.ndarray = COCO_17_SIGMAS,
) -> tuple[float, float, int, int, float]:
    """Compute Object Keypoint Similarity (OKS) and normalized drift.

    Args:
        pts_pred: (K, 2) or (K, 3) predicted keypoint coordinates.
        pts_gt: (K, 2) or (K, 3) ground truth keypoint coordinates.
        bbox_gt: (x1, y1, x2, y2) bounding box for actor scale computation.
        conf_pred: (K,) confidence scores for predicted keypoints.
        conf_gt: (K,) confidence scores for GT keypoints.
        conf_thresh: Confidence threshold for visibility.
        sigmas: (K,) per-joint standard deviations.

    Returns:
        tuple of (oks, mean_drift_norm, dropout_count, num_visible_gt, mean_conf_diff)
    """
    pts_p = np.asarray(pts_pred, dtype=np.float64)[:, :2]
    pts_g = np.asarray(pts_gt, dtype=np.float64)[:, :2]
    k = len(pts_g)

    # Determine visibility
    if conf_gt is not None:
        vis_gt = np.asarray(conf_gt) >= conf_thresh
    elif pts_pred.shape[-1] >= 3:
        vis_gt = np.asarray(pts_gt)[:, 2] > 0
    else:
        # If coordinates are non-zero, treat as visible
        vis_gt = np.any(pts_g > 0, axis=-1)

    if conf_pred is not None:
        vis_pred = np.asarray(conf_pred) >= conf_thresh
    elif pts_pred.shape[-1] >= 3:
        vis_pred = np.asarray(pts_pred)[:, 2] > 0
    else:
        vis_pred = np.any(pts_p > 0, axis=-1)

    num_visible = int(np.sum(vis_gt))
    if num_visible == 0:
        # No visible keypoints in GT
        if not np.any(vis_pred):
            return 1.0, 0.0, 0, 0, 0.0
        return 0.0, 1.0, 0, 0, 0.0

    # Determine scale s
    if bbox_gt is not None:
        x1, y1, x2, y2 = bbox_gt
        area = max(float(x2 - x1) * float(y2 - y1), 1.0)
    else:
        vis_coords = pts_g[vis_gt]
        min_xy = vis_coords.min(axis=0)
        max_xy = vis_coords.max(axis=0)
        area = max(float((max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])), 1.0)
    scale = np.sqrt(area)

    variances = 2.0 * (scale**2) * (sigmas[:k] ** 2)

    # Distance
    diffs = pts_p - pts_g
    dists_sq = np.sum(diffs**2, axis=-1)
    dists = np.sqrt(dists_sq)

    oks_terms = []
    drifts = []
    dropouts = 0

    for i in range(k):
        if not vis_gt[i]:
            continue
        if not vis_pred[i]:
            # Keypoint was visible in GT but lost in prediction (dropout)
            dropouts += 1
            oks_terms.append(0.0)
            drifts.append(1.0)  # normalized drift penalty equal to full object scale
        else:
            e = float(np.exp(-dists_sq[i] / max(variances[i], 1e-6)))
            oks_terms.append(e)
            drifts.append(float(dists[i] / max(scale, 1e-6)))

    mean_oks = float(np.mean(oks_terms)) if oks_terms else 1.0
    mean_drift = float(np.mean(drifts)) if drifts else 0.0

    conf_diff = 0.0
    if conf_gt is not None and conf_pred is not None:
        c_gt = np.asarray(conf_gt)[vis_gt]
        c_pr = np.asarray(conf_pred)[vis_gt]
        conf_diff = float(np.mean(c_pr - c_gt))

    return mean_oks, mean_drift, dropouts, num_visible, conf_diff


class PoseMetric:
    """Object Keypoint Similarity (OKS) metric using a heavy YOLO pose estimator.

    Higher is better; 1.0 is exact anatomical alignment, 0.0 is total pose failure.
    """

    name = "pose_oks"

    def __init__(
        self,
        *,
        model_name: str = "yolo26x-pose.pt",
        conf: float = 0.25,
        device: str = "cuda:0",
        estimator: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, tuple[float, float, float, float] | None] | None] | None = None,
    ) -> None:
        self.model_name = model_name
        self.conf = conf
        self.device = device
        self._estimator = estimator

    def _extract_pose(
        self, frame_rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float] | None] | None:
        """Extract primary actor keypoints, confidences, and bbox.

        Returns:
            (keypoints: (K, 2), confidences: (K,), bbox: (x1, y1, x2, y2)) or None
        """
        if self._estimator is not None:
            return self._estimator(frame_rgb)

        model = _load_yolo_pose(self.model_name, self.device)
        u8_frame = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        results = model.predict(source=u8_frame, verbose=False, conf=self.conf)
        if not results or not hasattr(results[0], "keypoints") or results[0].keypoints is None:
            return None

        kpts_obj = results[0].keypoints
        if len(kpts_obj.xy) == 0:
            return None

        # Primary detection (first detection / highest conf)
        kpts = kpts_obj.xy[0].detach().cpu().numpy()
        confs = kpts_obj.conf[0].detach().cpu().numpy() if kpts_obj.conf is not None else np.ones(len(kpts))

        bbox = None
        if hasattr(results[0], "boxes") and results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
            b = results[0].boxes.xyxy[0].detach().cpu().numpy()
            bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

        return kpts, confs, bbox

    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate mean OKS across frames in the clip."""
        ref_clip, pred_clip = paired(reference, predicted)
        t_len = ref_clip.shape[0]
        oks_scores = []

        for t in range(t_len):
            ref_pose = self._extract_pose(ref_clip[t])
            pred_pose = self._extract_pose(pred_clip[t])

            if ref_pose is None and pred_pose is None:
                oks_scores.append(1.0)
                continue
            if ref_pose is None or pred_pose is None:
                oks_scores.append(0.0)
                continue

            ref_kpts, ref_confs, ref_bbox = ref_pose
            pred_kpts, pred_confs, _ = pred_pose

            oks, _, _, _, _ = compute_oks(
                pts_pred=pred_kpts,
                pts_gt=ref_kpts,
                bbox_gt=ref_bbox,
                conf_pred=pred_confs,
                conf_gt=ref_confs,
                conf_thresh=self.conf,
            )
            oks_scores.append(oks)

        return float(np.mean(oks_scores)) if oks_scores else 1.0

    def score_detailed(
        self, reference: np.ndarray, predicted: np.ndarray
    ) -> tuple[PoseDetailedResult, ...]:
        """Detailed frame-by-frame anatomical evaluation."""
        ref_clip, pred_clip = paired(reference, predicted)
        t_len = ref_clip.shape[0]
        results = []

        for t in range(t_len):
            ref_pose = self._extract_pose(ref_clip[t])
            pred_pose = self._extract_pose(pred_clip[t])

            if ref_pose is None and pred_pose is None:
                results.append(
                    PoseDetailedResult(
                        oks=1.0,
                        mean_drift_norm=0.0,
                        ref_detected=False,
                        pred_detected=False,
                        dropout_count=0,
                        num_visible_gt=0,
                        mean_conf_diff=0.0,
                    )
                )
                continue

            if ref_pose is None:
                results.append(
                    PoseDetailedResult(
                        oks=0.0,
                        mean_drift_norm=1.0,
                        ref_detected=False,
                        pred_detected=True,
                        dropout_count=0,
                        num_visible_gt=0,
                        mean_conf_diff=0.0,
                    )
                )
                continue

            if pred_pose is None:
                ref_kpts, ref_confs, _ = ref_pose
                vis_count = int(np.sum(ref_confs >= self.conf))
                results.append(
                    PoseDetailedResult(
                        oks=0.0,
                        mean_drift_norm=1.0,
                        ref_detected=True,
                        pred_detected=False,
                        dropout_count=vis_count,
                        num_visible_gt=vis_count,
                        mean_conf_diff=-1.0,
                    )
                )
                continue

            ref_kpts, ref_confs, ref_bbox = ref_pose
            pred_kpts, pred_confs, _ = pred_pose

            oks, drift, dropouts, num_vis, conf_diff = compute_oks(
                pts_pred=pred_kpts,
                pts_gt=ref_kpts,
                bbox_gt=ref_bbox,
                conf_pred=pred_confs,
                conf_gt=ref_confs,
                conf_thresh=self.conf,
            )
            results.append(
                PoseDetailedResult(
                    oks=oks,
                    mean_drift_norm=drift,
                    ref_detected=True,
                    pred_detected=True,
                    dropout_count=dropouts,
                    num_visible_gt=num_vis,
                    mean_conf_diff=conf_diff,
                )
            )

        return tuple(results)


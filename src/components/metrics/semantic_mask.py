"""Silhouette mask Intersection-over-Union (IoU) using SAM 3.1 or high-capacity segmentation oracle.

Evaluates silhouette fidelity and boundary containment between reference and
predicted frames to detect background leakage, ghosting, or severed limbs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sys
from types import ModuleType
from typing import Any, Callable

import numpy as np

from src.components.detection.weights import resolve_weight
from src.components.metrics.frames import paired

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _ensure_sam3_compat() -> None:
    """Ensure PyTorch 2.2 compatibility shims for SAM3 imports."""
    if "torch.nn.attention" not in sys.modules:

        class SDPBackend:
            MATH = 1
            EFFICIENT_ATTENTION = 2
            FLASH_ATTENTION = 3

        @contextmanager
        def sdpa_kernel(*args: Any, **kwargs: Any):
            yield

        mod = ModuleType("torch.nn.attention")
        mod.SDPBackend = SDPBackend  # type: ignore[attr-defined]
        mod.sdpa_kernel = sdpa_kernel  # type: ignore[attr-defined]
        sys.modules["torch.nn.attention"] = mod

    if "timm.layers" not in sys.modules:
        try:
            import timm.models.layers as layers

            sys.modules["timm.layers"] = layers
        except ImportError:
            pass


def _load_segmenter(model_name: str, device: str) -> Any:
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        path = resolve_weight(model_name)
        if "sam" in model_name.lower():
            _ensure_sam3_compat()
            from ultralytics import SAM

            _MODEL_CACHE[key] = SAM(str(path))
        else:
            from ultralytics import YOLO

            _MODEL_CACHE[key] = YOLO(str(path))
    return _MODEL_CACHE[key]


@dataclass(frozen=True)
class MaskDetailedResult:
    """Detailed silhouette evaluation for one frame."""

    iou: float
    precision: float
    recall: float
    leakage_area_px: int
    ref_area_px: int
    pred_area_px: int


def compute_mask_iou(
    mask_pred: np.ndarray, mask_gt: np.ndarray
) -> tuple[float, float, float, int, int, int]:
    """Calculate IoU, precision, recall, and leakage between binary masks.

    Returns:
        (iou, precision, recall, leakage_px, ref_area_px, pred_area_px)
    """
    m_p = np.asarray(mask_pred, dtype=bool)
    m_g = np.asarray(mask_gt, dtype=bool)

    if m_p.shape != m_g.shape:
        raise ValueError(f"Mask shapes must match; got {m_p.shape} vs {m_g.shape}")

    area_p = int(np.sum(m_p))
    area_g = int(np.sum(m_g))

    if area_g == 0 and area_p == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    if area_g == 0 or area_p == 0:
        return 0.0, 0.0, 0.0, area_p, area_g, area_p

    intersection = int(np.sum(m_p & m_g))
    union = int(np.sum(m_p | m_g))
    leakage = int(np.sum(m_p & (~m_g)))

    iou = float(intersection / max(union, 1))
    precision = float(intersection / max(area_p, 1))
    recall = float(intersection / max(area_g, 1))

    return iou, precision, recall, leakage, area_g, area_p


class SamIouMetric:
    """Silhouette mask IoU evaluated using SAM 3.1 with fallback to yolo26x-seg.

    Higher is better; 1.0 is exact pixel-level mask overlap, 0.0 is total failure.
    """

    name = "sam_iou"

    def __init__(
        self,
        *,
        model_name: str = "sam3.pt",
        fallback_model: str = "yolo26x-seg.pt",
        conf: float = 0.25,
        device: str = "cuda:0",
        segmenter: Callable[[np.ndarray], np.ndarray | None] | None = None,
    ) -> None:
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.conf = conf
        self.device = device
        self._segmenter = segmenter

    def _extract_mask(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Extract primary actor silhouette mask."""
        if self._segmenter is not None:
            return self._segmenter(frame_rgb)

        u8_frame = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        h, w = u8_frame.shape[:2]

        try:
            model = _load_segmenter(self.model_name, self.device)
            results = model.predict(source=u8_frame, verbose=False, conf=self.conf)
        except Exception:
            # Graceful fallback to high-capacity YOLO segmenter if SAM3 fails
            model = _load_segmenter(self.fallback_model, self.device)
            results = model.predict(source=u8_frame, verbose=False, conf=self.conf)

        if not results or not hasattr(results[0], "masks") or results[0].masks is None:
            return None

        masks_obj = results[0].masks
        if len(masks_obj.data) == 0:
            return None

        # Primary mask (first actor detection)
        mask_t = masks_obj.data[0].detach().cpu().numpy()
        if mask_t.shape != (h, w):
            import cv2

            mask_t = cv2.resize(mask_t.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

        return mask_t > 0.5

    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate mean silhouette IoU across frames."""
        ref_clip, pred_clip = paired(reference, predicted)
        t_len = ref_clip.shape[0]
        iou_scores = []

        for t in range(t_len):
            m_ref = self._extract_mask(ref_clip[t])
            m_pred = self._extract_mask(pred_clip[t])

            if m_ref is None and m_pred is None:
                iou_scores.append(1.0)
                continue
            if m_ref is None or m_pred is None:
                iou_scores.append(0.0)
                continue

            iou, _, _, _, _, _ = compute_mask_iou(m_pred, m_ref)
            iou_scores.append(iou)

        return float(np.mean(iou_scores)) if iou_scores else 1.0

    def score_detailed(
        self, reference: np.ndarray, predicted: np.ndarray
    ) -> tuple[MaskDetailedResult, ...]:
        """Detailed frame-by-frame silhouette evaluation."""
        ref_clip, pred_clip = paired(reference, predicted)
        t_len = ref_clip.shape[0]
        results = []

        for t in range(t_len):
            m_ref = self._extract_mask(ref_clip[t])
            m_pred = self._extract_mask(pred_clip[t])

            if m_ref is None and m_pred is None:
                results.append(
                    MaskDetailedResult(
                        iou=1.0,
                        precision=1.0,
                        recall=1.0,
                        leakage_area_px=0,
                        ref_area_px=0,
                        pred_area_px=0,
                    )
                )
                continue
            if m_ref is None:
                area_p = int(np.sum(m_pred)) if m_pred is not None else 0
                results.append(
                    MaskDetailedResult(
                        iou=0.0,
                        precision=0.0,
                        recall=0.0,
                        leakage_area_px=area_p,
                        ref_area_px=0,
                        pred_area_px=area_p,
                    )
                )
                continue
            if m_pred is None:
                area_g = int(np.sum(m_ref))
                results.append(
                    MaskDetailedResult(
                        iou=0.0,
                        precision=0.0,
                        recall=0.0,
                        leakage_area_px=0,
                        ref_area_px=area_g,
                        pred_area_px=0,
                    )
                )
                continue

            iou, prec, rec, leak, a_g, a_p = compute_mask_iou(m_pred, m_ref)
            results.append(
                MaskDetailedResult(
                    iou=iou,
                    precision=prec,
                    recall=rec,
                    leakage_area_px=leak,
                    ref_area_px=a_g,
                    pred_area_px=a_p,
                )
            )

        return tuple(results)


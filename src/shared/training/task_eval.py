"""Score a handful of coding-task crops for the training stop rule.

PSNR is region-scoped on the object mask. LPIPS is on the bounding box of
that mask, because LPIPS is a patch metric and cannot take a mask. Both
scopes are named on the record. This is a stopping signal, not a result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.components.metrics.lpips import LpipsMetric
from src.components.metrics.psnr import masked_psnr


@dataclass(frozen=True)
class ItemScore:
    key: str
    lpips: float
    psnr: float
    n_mask_pixels: int


@dataclass(frozen=True)
class TaskScores:
    lpips: float
    psnr: float
    n: int
    items: tuple[ItemScore, ...]
    lpips_scope: str = "bbox of object mask"
    psnr_scope: str = "object mask"


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive-exclusive (y0, y1, x0, x1) of True pixels."""
    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 2:
        raise ValueError(f"mask must be (H, W), got {selected.shape}")
    rows = np.where(selected.any(axis=1))[0]
    cols = np.where(selected.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise ValueError("empty object mask; cannot score a region")
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def crop_to_mask_bbox(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    y0, y1, x0, x1 = bbox_from_mask(mask)
    return np.asarray(image)[y0:y1, x0:x1]


def score_item(
    reference: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    *,
    lpips: LpipsMetric,
) -> tuple[float, float, int]:
    """Return (lpips_bbox, psnr_mask, n_mask_pixels)."""
    ref = np.asarray(reference)
    pred = np.asarray(predicted)
    binary = np.asarray(mask, dtype=bool)
    if ref.shape != pred.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape} vs predicted {pred.shape}")
    n_pixels = int(binary.sum())
    psnr = float(masked_psnr(ref[None], pred[None], binary[None]))
    lpips_value = float(
        lpips.score(crop_to_mask_bbox(ref, binary)[None], crop_to_mask_bbox(pred, binary)[None])
    )
    return lpips_value, psnr, n_pixels


def mean_scores(items: list[ItemScore]) -> TaskScores:
    if not items:
        raise ValueError("cannot average zero scored items")
    return TaskScores(
        lpips=float(sum(item.lpips for item in items) / len(items)),
        psnr=float(sum(item.psnr for item in items) / len(items)),
        n=len(items),
        items=tuple(items),
    )


def static_copy_scores(
    appearance: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    key: str,
    lpips: LpipsMetric,
) -> ItemScore:
    """Paste the keyframe (appearance) as the prediction. The floor."""
    lpips_value, psnr, n_pixels = score_item(target, appearance, mask, lpips=lpips)
    return ItemScore(key=key, lpips=lpips_value, psnr=psnr, n_mask_pixels=n_pixels)

"""Region-scoped PSNR and calibrated LPIPS for a generated crop against a reference.

Two scopes, always both: the **player region** and the whole generation canvas.

PSNR is scoped by the letterboxed object mask when one exists. LPIPS cannot be
computed over a mask — it is a patch metric — so it is scoped to the *bounding
box of that mask*, and the box actually used travels with the score.

**LPIPS is the ranking key** (`PLAN.md` §2.5: the subfield rejects PSNR for
generatively reconstructed content, and the usable PSNR range on this task is
~11-21 dB against a ~2 dB per-clip sd). PSNR is reported alongside, never
instead. The LPIPS backend here is the published ``lpips`` package with its
learned calibration; the uncalibrated VGG-MSE that used to wear the name could
not tell a good reconstruction from an unrelated image (§2.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.components.generation._numpy import as_hwc, prepare_letterboxed
from src.components.generation.pose import Letterbox
from src.components.metrics.evaluator import triage
from src.components.metrics.region import Region

#: AlexNet's deepest stage sees a 2x2 map at 64px. Below that LPIPS is not a
#: number worth ranking on, so a smaller player box is padded out to this and
#: the padding is recorded rather than hidden.
LPIPS_MIN_SIDE = 64


@dataclass(frozen=True)
class ProbeScore:
    object_psnr_db: float
    frame_psnr_db: float
    n_object_pixels: int
    n_frame_pixels: int
    differs_from_input: bool
    differs_from_reference: bool
    region_kind: str
    object_lpips: float | None = None
    frame_lpips: float | None = None
    lpips_box: tuple[int, int, int, int] | None = None
    lpips_box_padded: bool = False


def _letterbox_mask(mask: np.ndarray, box: Letterbox) -> np.ndarray:
    from src.components.generation.pose import letterbox_image

    binary = np.asarray(mask > 0, dtype=np.uint8) * 255
    pasted = letterbox_image(binary, box)
    return pasted > 0


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight ``(x1, y1, x2, y2)`` around the True pixels, or None if empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y_indices = np.flatnonzero(rows)
    x_indices = np.flatnonzero(cols)
    return (
        int(x_indices[0]),
        int(y_indices[0]),
        int(x_indices[-1]) + 1,
        int(y_indices[-1]) + 1,
    )


def _pad_box(
    box: tuple[int, int, int, int], *, width: int, height: int, min_side: int = LPIPS_MIN_SIDE
) -> tuple[tuple[int, int, int, int], bool]:
    """Grow ``box`` about its centre until each side reaches ``min_side``.

    Returns the box and whether it had to grow. A padded box measures the player
    plus some surrounding canvas; the caller records that rather than pretending
    the scope was unchanged.
    """
    x1, y1, x2, y2 = box
    padded = False
    if x2 - x1 < min_side:
        padded = True
        centre = (x1 + x2) // 2
        x1 = max(0, centre - min_side // 2)
        x2 = min(width, x1 + min_side)
        x1 = max(0, x2 - min_side)
    if y2 - y1 < min_side:
        padded = True
        centre = (y1 + y2) // 2
        y1 = max(0, centre - min_side // 2)
        y2 = min(height, y1 + min_side)
        y1 = max(0, y2 - min_side)
    return (int(x1), int(y1), int(x2), int(y2)), padded


def _lpips_of(
    metric: Any, reference: np.ndarray, predicted: np.ndarray
) -> float:
    """``reference``/``predicted`` are HWC uint8. Returns a calibrated distance."""
    return float(metric.score(reference[None, ...], predicted[None, ...]))


def score_generation(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    object_mask: np.ndarray | None,
    canvas_width: int,
    canvas_height: int,
    appearance: np.ndarray | None = None,
    lpips_metric: Any = None,
) -> ProbeScore:
    """Score ``predicted`` against letterboxed ``reference``.

    Object scope is the letterboxed alpha/mask of the reference when supplied,
    otherwise the letterbox content box. Whole-frame is the entire generation
    canvas. Both are always returned.

    ``appearance`` is the keyframe that was handed to the engine. When given,
    ``differs_from_input`` compares the prediction to that image, not to the
    reference. That is the diagnostic the old harness mixed up with the score.

    ``lpips_metric`` is any object with ``score(reference, predicted)`` over a
    ``(T, H, W, C)`` clip. Omit it and the LPIPS fields stay ``None`` — a probe
    that could not build the metric reports that rather than a zero.
    """
    prepared = prepare_letterboxed(reference, None, canvas_width, canvas_height)
    reference_hwc = as_hwc(prepared["appearance"])[..., :3]
    predicted_hwc = as_hwc(predicted)[..., :3]
    if predicted_hwc.shape[:2] != reference_hwc.shape[:2]:
        import cv2

        predicted_hwc = cv2.resize(
            predicted_hwc,
            (reference_hwc.shape[1], reference_hwc.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    box: Letterbox = prepared["letterbox"]
    content_box = (
        box.offset_x,
        box.offset_y,
        box.offset_x + box.scaled_width,
        box.offset_y + box.scaled_height,
    )
    if object_mask is not None:
        letterboxed = _letterbox_mask(object_mask, box)
        region = Region.object(mask=letterboxed, name="player")
        region_kind = "mask"
        lpips_source = _mask_bbox(letterboxed) or content_box
    else:
        region = Region.object(box=content_box, name="player")
        region_kind = "box"
        lpips_source = content_box

    record = triage(reference_hwc[None, ...], predicted_hwc[None, ...], regions=[region])
    object_scores = record.for_role("object")
    frame_scores = record.for_role("whole-frame")
    if not object_scores or not frame_scores:
        raise RuntimeError("triage did not return object and whole-frame PSNR")
    object_psnr = float(object_scores[0].value)
    frame_psnr = float(frame_scores[0].value)

    object_lpips: float | None = None
    frame_lpips: float | None = None
    lpips_box: tuple[int, int, int, int] | None = None
    padded = False
    if lpips_metric is not None:
        height, width = reference_hwc.shape[:2]
        lpips_box, padded = _pad_box(lpips_source, width=width, height=height)
        x1, y1, x2, y2 = lpips_box
        object_lpips = _lpips_of(
            lpips_metric, reference_hwc[y1:y2, x1:x2], predicted_hwc[y1:y2, x1:x2]
        )
        frame_lpips = _lpips_of(lpips_metric, reference_hwc, predicted_hwc)

    differs_from_reference = not np.array_equal(predicted_hwc, reference_hwc)
    if appearance is None:
        differs_from_input = differs_from_reference
    else:
        init = as_hwc(prepare_letterboxed(appearance, None, canvas_width, canvas_height)["appearance"])
        differs_from_input = not np.array_equal(predicted_hwc, init[..., :3])
    return ProbeScore(
        object_psnr_db=object_psnr,
        frame_psnr_db=frame_psnr,
        n_object_pixels=int(object_scores[0].n_pixels),
        n_frame_pixels=int(frame_scores[0].n_pixels),
        differs_from_input=differs_from_input,
        differs_from_reference=differs_from_reference,
        region_kind=region_kind,
        object_lpips=object_lpips,
        frame_lpips=frame_lpips,
        lpips_box=lpips_box,
        lpips_box_padded=padded,
    )

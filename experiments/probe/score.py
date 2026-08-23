"""Region-scoped PSNR for a generated crop against a later reference frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.components.generation._numpy import as_hwc, prepare_letterboxed
from src.components.generation.pose import Letterbox
from src.components.metrics.evaluator import triage
from src.components.metrics.region import Region


@dataclass(frozen=True)
class ProbeScore:
    object_psnr_db: float
    frame_psnr_db: float
    n_object_pixels: int
    n_frame_pixels: int
    differs_from_input: bool
    differs_from_reference: bool
    region_kind: str


def _letterbox_mask(mask: np.ndarray, box: Letterbox) -> np.ndarray:
    from src.components.generation.pose import letterbox_image

    binary = np.asarray(mask > 0, dtype=np.uint8) * 255
    pasted = letterbox_image(binary, box)
    return pasted > 0


def score_generation(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    object_mask: np.ndarray | None,
    canvas_width: int,
    canvas_height: int,
    appearance: np.ndarray | None = None,
) -> ProbeScore:
    """Score ``predicted`` against letterboxed ``reference``.

    Object scope is the letterboxed alpha/mask of the reference when supplied,
    otherwise the letterbox content box. Whole-frame is the entire generation
    canvas. Both are always returned.

    ``appearance`` is the keyframe that was handed to the engine. When given,
    ``differs_from_input`` compares the prediction to that image, not to the
    reference. That is the diagnostic the old harness mixed up with the score.
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
    if object_mask is not None:
        region = Region.object(mask=_letterbox_mask(object_mask, box), name="player")
        region_kind = "mask"
    else:
        region = Region.object(
            box=(
                box.offset_x,
                box.offset_y,
                box.offset_x + box.scaled_width,
                box.offset_y + box.scaled_height,
            ),
            name="player",
        )
        region_kind = "box"

    record = triage(reference_hwc[None, ...], predicted_hwc[None, ...], regions=[region])
    object_scores = record.for_role("object")
    frame_scores = record.for_role("whole-frame")
    if not object_scores or not frame_scores:
        raise RuntimeError("triage did not return object and whole-frame PSNR")
    object_psnr = float(object_scores[0].value)
    frame_psnr = float(frame_scores[0].value)
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
    )

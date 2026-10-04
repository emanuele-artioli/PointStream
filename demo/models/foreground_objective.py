"""Smoke-only hand objective. The campaign loss in hand_objective.py is unchanged.

RGB is in [-1, 1], alpha is in [0, 1], and the display background is -1:

    shown = alpha * rgb + (1 - alpha) * (-1)

The diagnostic loss is the sum of masked RGB MAE, balanced alpha MAE, and
full-crop composite MAE, each with weight 1.
"""

from __future__ import annotations

import numpy as np
import torch


def compose(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Decoder-side composite. No source mask is required."""
    return alpha * rgb + (1.0 - alpha) * -1.0


def bgr_uint8_to_rgb_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert one BGR uint8 image to RGB in [-1, 1], channel-first."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an HxWx3 BGR image")
    rgb = np.ascontiguousarray(image[..., ::-1]).astype(np.float32)
    tensor = torch.from_numpy(rgb / 127.5 - 1.0).permute(2, 0, 1)
    return tensor


def _check_ranges(outputs: torch.Tensor, source_rgb: torch.Tensor, target_alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if outputs.ndim != 4 or outputs.shape[1] != 4:
        raise ValueError("outputs must have shape [B,4,H,W] (RGB plus alpha)")
    if source_rgb.shape != outputs[:, :3].shape:
        raise ValueError("source_rgb must match the output RGB shape")
    if target_alpha.shape != outputs[:, 3:4].shape:
        raise ValueError("target_alpha must have shape [B,1,H,W]")
    if not torch.isfinite(outputs).all() or not torch.isfinite(source_rgb).all() or not torch.isfinite(target_alpha).all():
        raise ValueError("objective inputs must be finite")
    if bool(((target_alpha < 0) | (target_alpha > 1)).any()):
        raise ValueError("target_alpha must be in [0,1]")
    if bool(((source_rgb < -1) | (source_rgb > 1)).any()) or bool(((outputs[:, :3] < -1) | (outputs[:, :3] > 1)).any()):
        raise ValueError("source and predicted RGB must be in [-1,1]")
    if bool(((outputs[:, 3:4] < 0) | (outputs[:, 3:4] > 1)).any()):
        raise ValueError("predicted alpha must be in [0,1]")
    return outputs[:, :3], outputs[:, 3:4]


def masked_rgb_mae(pred_rgb: torch.Tensor, source_rgb: torch.Tensor, target_alpha: torch.Tensor) -> torch.Tensor:
    """Global MAE over 3 * sum(target alpha), with an empty-mask guard."""
    weight = target_alpha.expand_as(pred_rgb)
    numerator = ((pred_rgb - source_rgb).abs() * weight).sum()
    alpha_mass = target_alpha.sum()
    if float(alpha_mass.detach()) == 0.0:
        return numerator * 0.0
    return numerator / (3.0 * alpha_mass)


def balanced_alpha_mae(pred_alpha: torch.Tensor, target_alpha: torch.Tensor) -> torch.Tensor:
    """Mean of the foreground and background MAEs for regions that exist.

    Foreground is target alpha >= 0.5 and background is the complement. The
    error is against the soft target value. A missing region is omitted from
    the average instead of contributing a zero term over a fixed denominator.
    """
    error = (pred_alpha - target_alpha).abs()
    regions = []
    foreground = target_alpha >= 0.5
    background = ~foreground
    if bool(foreground.any()):
        regions.append(error[foreground].mean())
    if bool(background.any()):
        regions.append(error[background].mean())
    if not regions:
        return error.sum() * 0.0
    return torch.stack(regions).mean()


def composite_mae(pred_rgb: torch.Tensor, pred_alpha: torch.Tensor, source_rgb: torch.Tensor, target_alpha: torch.Tensor) -> torch.Tensor:
    return (compose(pred_rgb, pred_alpha) - compose(source_rgb, target_alpha)).abs().mean()


def smoke_hand_objective(
    outputs: torch.Tensor,
    source_rgb: torch.Tensor,
    target_alpha: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Diagnostic objective on the original unmatted RGB crop."""
    pred_rgb, pred_alpha = _check_ranges(outputs, source_rgb, target_alpha)
    parts = {
        "masked_rgb_mae": masked_rgb_mae(pred_rgb, source_rgb, target_alpha),
        "balanced_alpha_mae": balanced_alpha_mae(pred_alpha, target_alpha),
        "composite_mae": composite_mae(pred_rgb, pred_alpha, source_rgb, target_alpha),
    }
    return sum(parts.values()), parts


def evaluation_metrics(
    pred_rgb: torch.Tensor,
    pred_alpha: torch.Tensor,
    source_rgb: torch.Tensor,
    target_alpha: torch.Tensor,
) -> dict[str, float]:
    """Corrected metrics shared by every smoke arm. LPIPS is not included."""
    inside = target_alpha >= 0.5
    outside = target_alpha < 0.5
    pred_mask = pred_alpha >= 0.5
    intersection = (pred_mask & inside).sum().to(torch.float32)
    union = (pred_mask | inside).sum().to(torch.float32)
    inside_rgb = (pred_rgb - source_rgb).abs()
    if bool(inside.any()):
        inside_rgb_mae = float((inside_rgb * inside).sum() / (3.0 * inside.sum()))
        foreground_opacity_mae = float((pred_alpha - target_alpha).abs()[inside].mean())
    else:
        inside_rgb_mae = 0.0
        foreground_opacity_mae = 0.0
    outside_alpha_mae = float((pred_alpha - target_alpha).abs()[outside].mean()) if bool(outside.any()) else 0.0
    iou = float(intersection / union) if float(union) > 0 else 1.0
    shown = compose(pred_rgb, pred_alpha)
    target = compose(source_rgb, target_alpha)
    return {
        "masked_rgb_mae": float(masked_rgb_mae(pred_rgb, source_rgb, target_alpha).detach()),
        "composite_mae": float((shown - target).abs().mean().detach()),
        "inside_rgb_mae": inside_rgb_mae,
        "foreground_opacity_mae": foreground_opacity_mae,
        "outside_alpha_mae": outside_alpha_mae,
        "mask_iou_0_5": iou,
    }

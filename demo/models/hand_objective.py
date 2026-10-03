"""Hand training loss and the ship gate.

The step loss is a smooth maximum of two per-frame terms: appearance inside
the SAM mask (and its soft edge), and predicted alpha outside that edge.
Jitter is not part of the step. It is measured on an exported clip and enters
only the selection score.

The selection score is the minimum of appearance, matte leak, and jitter,
each divided by the same number on the stream already shipped. A value below
1 means one of the three got worse.
"""

from __future__ import annotations

import numpy as np
import torch

# A fully speckled outside (mean alpha 0.2) should cost about as much as a
# clearly wrong hand (masked L1 around 0.4).
MATTE_LOSS_SCALE = 2.0
SMOOTH_MAX_TAU = 0.05


def smooth_max(losses: list[torch.Tensor], tau: float = SMOOTH_MAX_TAU) -> torch.Tensor:
    """Differentiable maximum. tau is small so the worst term dominates."""
    stacked = torch.stack([loss.reshape(()) for loss in losses])
    return torch.logsumexp(stacked / tau, dim=0) * tau


def hand_step_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    lpips_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (smooth_max, appearance, scaled matte) for a 4-channel batch.

    Appearance compares RGB only where the target alpha is on the hand or in
    the soft edge. The matte term is the mean predicted alpha outside that edge.
    """
    pred_rgb = outputs[:, :3]
    pred_a = outputs[:, 3:4]
    tgt_rgb = targets[:, :3]
    tgt_a = targets[:, 3:4]
    region = (tgt_a >= 0.05).to(pred_rgb.dtype)
    outside = tgt_a < 0.05
    denom = region.sum().clamp(min=1.0)
    loss_l1 = ((pred_rgb - tgt_rgb).abs() * region).sum() / denom
    shown = pred_rgb * region + (-1.0) * (1.0 - region)
    tgt_shown = tgt_rgb * region + (-1.0) * (1.0 - region)
    loss_app = loss_l1
    if lpips_fn is not None:
        loss_app = loss_app + 0.8 * lpips_fn(shown, tgt_shown).mean()
    if bool(outside.any()):
        loss_matte = pred_a[outside].abs().mean() * MATTE_LOSS_SCALE
    else:
        loss_matte = pred_a.sum() * 0.0
    return smooth_max([loss_app, loss_matte]), loss_app, loss_matte


def smoke_hand_objective(
    outputs: torch.Tensor,
    source_rgb: torch.Tensor,
    target_alpha: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Diagnostic RGB+alpha objective with explicit black compositing.

    RGB and source_rgb are RGB tensors normalized to [-1, 1]. Output alpha
    and target_alpha are in [0, 1]. ``source_rgb`` must be the original
    unmatted crop; applying target alpha to a black-matted target a second
    time changes the reference.
    """
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

    pred_rgb, pred_alpha = outputs[:, :3], outputs[:, 3:4]
    alpha_weight = target_alpha.expand_as(pred_rgb)
    alpha_mass = target_alpha.sum()
    masked_denom = torch.where(alpha_mass > 0, 3.0 * alpha_mass, torch.ones_like(alpha_mass))
    masked_rgb = ((pred_rgb - source_rgb).abs() * alpha_weight).sum() / masked_denom

    foreground = target_alpha >= 0.5
    background = ~foreground
    alpha_error = (pred_alpha - target_alpha).abs()
    region_errors = [alpha_error[region].mean() for region in (foreground, background) if bool(region.any())]
    balanced_alpha = torch.stack(region_errors).mean() if region_errors else alpha_error.sum() * 0.0

    pred_composite = pred_alpha * pred_rgb + (1.0 - pred_alpha) * -1.0
    target_composite = target_alpha * source_rgb + (1.0 - target_alpha) * -1.0
    composite = (pred_composite - target_composite).abs().mean()
    parts = {"masked_rgb_mae": masked_rgb, "balanced_alpha_mae": balanced_alpha, "composite_mae": composite}
    return sum(parts.values()), parts


def selection_min(
    current: dict[str, float],
    baseline: dict[str, float],
) -> tuple[float, str]:
    """Min of baseline/current for appearance, matte, and jitter (lower raw is better)."""
    scores: dict[str, float] = {}
    for key in ("appearance", "matte", "jitter"):
        if key not in current or key not in baseline:
            continue
        scores[key] = float(baseline[key]) / max(float(current[key]), 1e-8)
    if not scores:
        raise ValueError("selection_min needs at least one shared metric")
    bottleneck = min(scores, key=scores.get)
    return scores[bottleneck], bottleneck


def composite_hand_metrics(
    ref_frames: list[np.ndarray],
    rec_frames: list[np.ndarray],
    bg_frames: list[np.ndarray],
    alphas: list[np.ndarray],
) -> dict[str, float]:
    """Raw errors on a composited clip. Lower is better. Jitter is on the hand only."""
    n = min(len(ref_frames), len(rec_frames), len(bg_frames), len(alphas))
    app = []
    matte = []
    jitter_rec = []
    prev = None
    prev_mask = None
    for i in range(n):
        mask = np.asarray(alphas[i])
        if mask.ndim == 3:
            mask = mask[..., 0]
        hand = mask >= 127
        outside = ~hand
        ref = ref_frames[i].astype(np.float32)
        rec = rec_frames[i].astype(np.float32)
        bg = bg_frames[i].astype(np.float32)
        if hand.any():
            app.append(float(np.abs(ref[hand] - rec[hand]).mean()) / 255.0)
        if outside.any():
            matte.append(float(np.abs(rec[outside] - bg[outside]).mean()) / 255.0)
        if prev is not None and hand.any() and prev_mask is not None:
            both = hand & prev_mask
            if both.any():
                jitter_rec.append(float(np.abs(rec[both] - prev[both]).mean()) / 255.0)
        prev = rec
        prev_mask = hand
    return {
        "appearance": float(np.mean(app)) if app else 0.0,
        "matte": float(np.mean(matte)) if matte else 0.0,
        "jitter": float(np.mean(jitter_rec)) if jitter_rec else 0.0,
    }

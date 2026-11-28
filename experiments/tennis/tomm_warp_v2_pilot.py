"""Track A Stage 1: 48-frame Shanghai match102 pilot for real-time CPU k6_ref_warp v2.

Evaluates 5 variants on the frozen 48-frame Shanghai match102 wire payload:
  1. v1_gauss_sam:       Baseline Gaussian keypoint warp + SAM 3.1 mask (reproduction check)
  2. v2_lbs_sam:         Bone-aligned Linear Blend Skinning (LBS) + SAM 3.1 mask
  3. v2_mls_sam:         Moving Least Squares (MLS) Similarity warp + SAM 3.1 mask
  4. v2_lbs_warped_mask: Bone-aligned LBS + zero-SAM warped PSR reference alpha mask (100% source-denied)
  5. v2_mls_warped_mask: MLS Similarity + zero-SAM warped PSR reference alpha mask (100% source-denied)

All rendering and compositing run strictly on 1 CPU thread (0 GPU-hours).
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys
import tarfile
import tempfile
import time
from typing import Any

import cv2
import numpy as np
import torch

from experiments.tennis import common_player_packet as packet
from experiments.tennis import k6b
from experiments.tennis.player_requalification import sha256
from experiments.tennis.tomm_multi_backbone_generate import _warp_crop_gaussian
from experiments.tennis.tomm_multi_backbone_score import (
    DDIM30_48F_SECONDS,
    YOLO_POSE_SHA256,
    best_yolo_oks,
)
from experiments.tennis.tomm_v5_pareto_pilot import CARRIER_SHA256, smooth_keypoints
from experiments.tennis.tomm_v5_score import (
    ALEXNET_SHA256,
    FEATURES_SHA256,
    PIXELS_SHA256,
    VGG_SHA256,
    WITNESS_SHA256,
    is_seam_tick,
    load_base_frames,
    load_source_frames,
    mask_960,
    mask_full_res,
)
from experiments.visor.h3 import dis_flow, warp
from experiments.visor.h3b import luma
from src.codecs import quality

CARD = Path(__file__).resolve().parents[2] / "docs/experiments/test-crop-joint-20261122"
sys.path.insert(0, str(CARD))
from score_native import get_support, tensor  # noqa: E402

WARP_MAP_STEP = 4

# COCO-17 bones: (joint_a, joint_b, sigma_fraction_of_height)
# Torso bones get a slightly wider influence radius; extremities stay tighter so limbs rotate cleanly.
COCO_BONES: tuple[tuple[int, int, float], ...] = (
    (5, 6, 0.085),   # shoulders
    (11, 12, 0.085), # hips
    (5, 11, 0.095),  # left torso
    (6, 12, 0.095),  # right torso
    (5, 7, 0.055),   # left upper arm
    (7, 9, 0.050),   # left forearm
    (6, 8, 0.055),   # right upper arm
    (8, 10, 0.050),  # right forearm
    (11, 13, 0.065), # left thigh
    (13, 15, 0.055), # left calf
    (12, 14, 0.065), # right thigh
    (14, 16, 0.055), # right calf
    (0, 5, 0.065),   # head-left shoulder
    (0, 6, 0.065),   # head-right shoulder
)


def _crop_from_canvas(canvas: np.ndarray, box: np.ndarray, out_w: int = 512, out_h: int = 784) -> np.ndarray:
    """Extract a crop matching k6b.to_crop geometry from a full-res canvas."""
    x0, y0, bw, bh = [float(v) for v in box]
    h_img, w_img = canvas.shape[:2]
    # Build affine map from crop pixel (u, v) to canvas pixel (x, y)
    sx = bw / float(out_w)
    sy = bh / float(out_h)
    m = np.array([[sx, 0.0, x0], [0.0, sy, y0]], dtype=np.float32)
    return cv2.warpAffine(
        canvas,
        m,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def extract_ref_alpha_from_base(
    ref_crop: np.ndarray,
    base_frame_0: np.ndarray,
    box_0: np.ndarray,
    ref_kp: np.ndarray,
    conf_thr: float = 0.5,
) -> np.ndarray:
    """Extract a 100% source-denied float32 alpha matte in [0, 1] for ref_crop at t=0.

    Uses ONLY receiver-available data: the transmitted PSR reference crop, the decoded
    cleaned SVT-AV1 base frame at t=0, and the decoded t=0 pose keypoints.
    """
    h, w = ref_crop.shape[:2]
    # Work at 1/4 resolution (196 x 128) for fast CPU GrabCut (~5-10 ms once per GOP)
    sh, sw = h // 4, w // 4
    ref_small = cv2.resize(ref_crop, (sw, sh), interpolation=cv2.INTER_AREA)
    base_crop = _crop_from_canvas(base_frame_0, box_0, out_w=w, out_h=h)
    base_small = cv2.resize(base_crop, (sw, sh), interpolation=cv2.INTER_AREA)

    # Lab perceptual difference between player reference crop and clean court background
    ref_lab = cv2.cvtColor(ref_small, cv2.COLOR_RGB2LAB).astype(np.float32)
    base_lab = cv2.cvtColor(base_small, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = np.linalg.norm(ref_lab - base_lab, axis=-1)

    # Skeleton prior mask at 1/4 resolution
    skel_fg = np.zeros((sh, sw), dtype=np.uint8)
    skel_pr = np.zeros((sh, sw), dtype=np.uint8)
    valid = ref_kp[:, 2] >= conf_thr
    pts = ref_kp[:, :2] / 4.0

    # Draw thick torso & limbs for probable FG, thinner core for definite FG
    for ja, jb, _ in COCO_BONES:
        if valid[ja] and valid[jb]:
            pa = (int(round(pts[ja, 0])), int(round(pts[ja, 1])))
            pb = (int(round(pts[jb, 0])), int(round(pts[jb, 1])))
            cv2.line(skel_fg, pa, pb, 255, thickness=max(2, sw // 32))
            cv2.line(skel_pr, pa, pb, 255, thickness=max(10, sw // 6))
    # Fill torso polygon if shoulders & hips are visible
    torso_idx = [5, 6, 12, 11]
    if all(valid[i] for i in torso_idx):
        poly = np.round(pts[torso_idx]).astype(np.int32)
        cv2.fillConvexPoly(skel_fg, poly, 255)
        cv2.fillConvexPoly(skel_pr, poly, 255)
    # Head circle
    if valid[0]:
        p0 = (int(round(pts[0, 0])), int(round(pts[0, 1])))
        cv2.circle(skel_fg, p0, max(3, sw // 24), 255, -1)
        cv2.circle(skel_pr, p0, max(10, sw // 7), 255, -1)

    # Build GrabCut trimap using background subtraction + skeleton prior
    gc_mask = np.full((sh, sw), cv2.GC_BGD, dtype=np.uint8)
    # Probable background everywhere inside dilated skeleton or moderate diff
    gc_mask[(skel_pr > 0) | (diff > 12.0)] = cv2.GC_PR_BGD
    # Probable foreground where both near skeleton and distinct from court base
    gc_mask[(skel_pr > 0) & (diff > 12.0)] = cv2.GC_PR_FGD
    # Definite foreground along the thin skeleton core (prevents chopping low-contrast limbs/shoes)
    gc_mask[skel_fg > 0] = cv2.GC_FGD

    # Ensure both FG and BG pixels exist before calling GrabCut
    has_fg = np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))
    has_bg = np.any((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD))
    if has_fg and has_bg:
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.setRNGSeed(0)
            cv2.grabCut(
                ref_small,
                gc_mask,
                (0, 0, sw, sh),
                bgd_model,
                fgd_model,
                3,
                cv2.GC_INIT_WITH_MASK,
            )
            fg = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).astype(np.uint8) * 255
        except cv2.error:
            fg = ((skel_pr > 0) & (diff > 16.0)).astype(np.uint8) * 255
    else:
        fg = skel_pr.copy()

    # Keep only components overlapping the skeleton prior to suppress detached court noise
    num_labels, labels = cv2.connectedComponents(fg)
    if num_labels > 1:
        keep = np.zeros_like(fg)
        for lbl in range(1, num_labels):
            comp = labels == lbl
            if np.any(comp & (skel_pr > 0)):
                keep[comp] = 255
        if keep.any():
            fg = keep

    # Smooth morphology and upsample with slight edge feathering
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k3)
    alpha_full = cv2.resize(fg.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha_full = cv2.GaussianBlur(alpha_full, (5, 5), 1.2)
    return np.clip(alpha_full, 0.0, 1.0).astype(np.float32)


def _compute_lbs_maps(
    h: int,
    w: int,
    src_kp: np.ndarray,
    dst_kp: np.ndarray,
    conf_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute bone-aligned Linear Blend Skinning (LBS) inverse warp maps (map_x, map_y)."""
    valid = (src_kp[:, 2] >= conf_thr) & (dst_kp[:, 2] >= conf_thr)
    bones: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for ja, jb, sig_frac in COCO_BONES:
        if valid[ja] and valid[jb]:
            pa, pb = src_kp[ja, :2].astype(np.float32), src_kp[jb, :2].astype(np.float32)
            qa, qb = dst_kp[ja, :2].astype(np.float32), dst_kp[jb, :2].astype(np.float32)
            if float(np.hypot(*(qb - qa))) >= 2.0 and float(np.hypot(*(pb - pa))) >= 2.0:
                bones.append((pa, pb, qa, qb, max(8.0, sig_frac * float(h))))
    if len(bones) < 2:
        return None

    gy = np.arange(0, h, WARP_MAP_STEP, dtype=np.float32)
    gx = np.arange(0, w, WARP_MAP_STEP, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(gx, gy)  # (gh, gw)

    acc_x = np.zeros_like(grid_x, dtype=np.float32)
    acc_y = np.zeros_like(grid_y, dtype=np.float32)
    acc_w = np.full_like(grid_x, 1e-6, dtype=np.float32)

    for pa, pb, qa, qb, sigma in bones:
        # Bone similarity transform mapping dst segment (qa -> qb) to src segment (pa -> pb)
        vd = qb - qa
        vs = pb - pa
        len_d2 = float(vd[0] * vd[0] + vd[1] * vd[1])
        len_d = math.sqrt(len_d2)
        len_s = float(math.hypot(vs[0], vs[1]))
        # Clamp scale factor to avoid extreme stretching on noisy/foreshortened joints
        scale = max(0.55, min(1.80, len_s / max(len_d, 1e-4)))
        # Unit tangent vectors
        ud = vd / len_d
        us = vs / max(len_s, 1e-4)
        # 2D rotation from ud to us: cos = ud·us, sin = ud_x*us_y - ud_y*us_x
        c = float(ud[0] * us[0] + ud[1] * us[1])
        s = float(ud[0] * us[1] - ud[1] * us[0])

        # Vector from qa to grid pixel
        rx = grid_x - qa[0]
        ry = grid_y - qa[1]

        # Distance from grid pixel to segment [qa, qb]
        t_proj = np.clip((rx * vd[0] + ry * vd[1]) / len_d2, 0.0, 1.0)
        dx = rx - t_proj * vd[0]
        dy = ry - t_proj * vd[1]
        dist2 = dx * dx + dy * dy

        w_b = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)

        # Transformed source coordinates under bone similarity
        src_x = pa[0] + scale * (c * rx - s * ry)
        src_y = pa[1] + scale * (s * rx + c * ry)

        acc_x += w_b * src_x
        acc_y += w_b * src_y
        acc_w += w_b

    map_x_small = (acc_x / acc_w).astype(np.float32)
    map_y_small = (acc_y / acc_w).astype(np.float32)
    map_x = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return map_x, map_y


def _compute_mls_similarity_maps(
    h: int,
    w: int,
    src_kp: np.ndarray,
    dst_kp: np.ndarray,
    conf_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Closed-form Moving Least Squares (MLS) Similarity warp (Schaefer et al. 2006).

    Maps dst pixels back to src pixels using keypoints + bone midpoints as control handles.
    """
    valid = (src_kp[:, 2] >= conf_thr) & (dst_kp[:, 2] >= conf_thr)
    src_pts: list[np.ndarray] = [src_kp[i, :2].astype(np.float32) for i in range(len(src_kp)) if valid[i]]
    dst_pts: list[np.ndarray] = [dst_kp[i, :2].astype(np.float32) for i in range(len(dst_kp)) if valid[i]]
    # Add bone midpoints so long limbs stay rigid along their length
    for ja, jb, _ in COCO_BONES:
        if valid[ja] and valid[jb]:
            src_pts.append(0.5 * (src_kp[ja, :2] + src_kp[jb, :2]).astype(np.float32))
            dst_pts.append(0.5 * (dst_kp[ja, :2] + dst_kp[jb, :2]).astype(np.float32))
    if len(src_pts) < 4:
        return None

    q = np.stack(src_pts, axis=0)  # target in inverse map = src_kp (N, 2)
    p = np.stack(dst_pts, axis=0)  # control points in dst = dst_kp (N, 2)

    gy = np.arange(0, h, WARP_MAP_STEP, dtype=np.float32)
    gx = np.arange(0, w, WARP_MAP_STEP, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(gx, gy)  # (gh, gw)

    # Inverse-distance squared weights: w_i = 1 / (||v - p_i||^2 + eps)
    dx = grid_x[..., None] - p[None, None, :, 0]  # (gh, gw, N)
    dy = grid_y[..., None] - p[None, None, :, 1]  # (gh, gw, N)
    eps = (0.015 * float(h)) ** 2
    wt = 1.0 / (dx * dx + dy * dy + eps)  # (gh, gw, N)
    w_sum = wt.sum(axis=-1, keepdims=True)  # (gh, gw, 1)

    # Weighted centroids p* and q*
    px_star = (wt * p[None, None, :, 0]).sum(axis=-1, keepdims=True) / w_sum
    py_star = (wt * p[None, None, :, 1]).sum(axis=-1, keepdims=True) / w_sum
    qx_star = (wt * q[None, None, :, 0]).sum(axis=-1, keepdims=True) / w_sum
    qy_star = (wt * q[None, None, :, 1]).sum(axis=-1, keepdims=True) / w_sum

    # Relative coordinates p_hat = p - p*, q_hat = q - q*
    phx = p[None, None, :, 0] - px_star
    phy = p[None, None, :, 1] - py_star
    qhx = q[None, None, :, 0] - qx_star
    qhy = q[None, None, :, 1] - qy_star

    # Similarity normalization mu_s = sum(w_i * ||p_hat_i||^2)
    mu_s = (wt * (phx * phx + phy * phy)).sum(axis=-1) + 1e-6  # (gh, gw)

    # v_rel = v - p*
    vx = grid_x - px_star[..., 0]
    vy = grid_y - py_star[..., 0]

    # Closed-form 2D similarity matrix A_i = [[phx, phy], [phy, -phx]] * [[vx, vy], [vy, -vx]]^T
    # Equivalent complex multiplication: M = sum(w_i * q_hat_i * conj(p_hat_i)) / mu_s
    # where real(M) = a, imag(M) = b => f(v) = q* + [[a, -b], [b, a]] * (v - p*)
    a = (wt * (qhx * phx + qhy * phy)).sum(axis=-1) / mu_s
    b = (wt * (qhy * phx - qhx * phy)).sum(axis=-1) / mu_s

    # Clamp local scale sqrt(a^2 + b^2) for stability in far background
    mag = np.sqrt(a * a + b * b) + 1e-6
    scale_clamped = np.clip(mag, 0.5, 1.8) / mag
    a *= scale_clamped
    b *= scale_clamped

    map_x_small = (qx_star[..., 0] + a * vx - b * vy).astype(np.float32)
    map_y_small = (qy_star[..., 0] + b * vx + a * vy).astype(np.float32)

    map_x = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return map_x, map_y


def _paste_soft_alpha(
    canvas: np.ndarray,
    crop_rgb: np.ndarray,
    crop_alpha: np.ndarray,
    box: np.ndarray,
) -> None:
    """Alpha-composite a warped crop + warped alpha onto the full-res canvas in-place."""
    x0, y0, bw, bh = [float(v) for v in box]
    h_img, w_img = canvas.shape[:2]
    ix0 = max(0, int(math.floor(x0)))
    iy0 = max(0, int(math.floor(y0)))
    ix1 = min(w_img, int(math.ceil(x0 + bw)))
    iy1 = min(h_img, int(math.ceil(y0 + bh)))
    if ix1 <= ix0 or iy1 <= iy0:
        return

    # Map canvas ROI [ix0:ix1, iy0:iy1] back to crop coordinates
    out_w, out_h = ix1 - ix0, iy1 - iy0
    sx = crop_rgb.shape[1] / max(bw, 1e-4)
    sy = crop_rgb.shape[0] / max(bh, 1e-4)
    m = np.array([[sx, 0.0, (ix0 - x0) * sx], [0.0, sy, (iy0 - y0) * sy]], dtype=np.float32)

    warped_rgb = cv2.warpAffine(
        crop_rgb,
        m,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_a = cv2.warpAffine(
        crop_alpha,
        m,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    a3 = warped_a[..., None]
    roi = canvas[iy0:iy1, ix0:ix1].astype(np.float32)
    blended = a3 * warped_rgb.astype(np.float32) + (1.0 - a3) * roi
    canvas[iy0:iy1, ix0:ix1] = np.clip(blended, 0.0, 255.0).astype(np.uint8)


def render_variant(
    variant: str,
    roles: tuple[str, ...],
    unit_inputs: dict[str, dict[str, Any]],
    base_frames: list[np.ndarray],
    supports: Any,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Render all composited frames for a variant on 1 CPU thread and time every phase."""
    n_frames = len(base_frames)
    role_indices = {"near": 0, "far": 1}
    use_warped_mask = variant.endswith("_warped_mask")
    method = variant.replace("_warped_mask", "").replace("_sam", "")

    # Phase 1: One-time t=0 reference alpha extraction (only for warped_mask variants)
    t_matte0 = time.perf_counter()
    ref_alphas: dict[str, np.ndarray] = {}
    if use_warped_mask:
        for role in roles:
            inp = unit_inputs[role]
            ref_box = inp["reference_box"]
            ref_kp = k6b.to_crop(inp["smooth_kp"][0], ref_box)
            ref_alphas[role] = extract_ref_alpha_from_base(
                inp["ref_crop"], base_frames[0], ref_box, ref_kp
            )
    matte_init_s = time.perf_counter() - t_matte0

    # Phase 2: Per-frame warping + compositing
    canvases: list[np.ndarray] = []
    warp_s = 0.0
    comp_s = 0.0

    # Precompute crop-space keypoints (lightweight geometry from decoded PSCP)
    crop_kps: dict[str, list[np.ndarray]] = {}
    ref_kps: dict[str, np.ndarray] = {}
    for role in roles:
        inp = unit_inputs[role]
        ref_kps[role] = k6b.to_crop(inp["smooth_kp"][0], inp["reference_box"])
        crop_kps[role] = [k6b.to_crop(inp["smooth_kp"][t], inp["boxes"][t]) for t in range(n_frames)]

    for t in range(n_frames):
        canvas = base_frames[t].copy()
        boxes_t = supports[t]
        masks_full = None if use_warped_mask else [mask_full_res(boxes_t[i]) for i in range(2)]

        for role in reversed(roles):
            inp = unit_inputs[role]
            ref_crop = inp["ref_crop"]
            h, w = ref_crop.shape[:2]
            ref_kp = ref_kps[role]
            dst_kp = crop_kps[role][t]

            tw0 = time.perf_counter()
            if method == "v1_gauss":
                warped_rgb = _warp_crop_gaussian(ref_crop, ref_kp, dst_kp)
                warped_alpha = None
            else:
                maps = (
                    _compute_lbs_maps(h, w, ref_kp, dst_kp)
                    if method == "v2_lbs"
                    else _compute_mls_similarity_maps(h, w, ref_kp, dst_kp)
                )
                if maps is None:
                    warped_rgb = ref_crop.copy()
                    warped_alpha = ref_alphas[role].copy() if use_warped_mask else None
                else:
                    mx, my = maps
                    warped_rgb = cv2.remap(
                        ref_crop, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                    )
                    warped_alpha = (
                        cv2.remap(
                            ref_alphas[role],
                            mx,
                            my,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0.0,
                        )
                        if use_warped_mask
                        else None
                    )
            warp_s += time.perf_counter() - tw0

            tc0 = time.perf_counter()
            if use_warped_mask:
                assert warped_alpha is not None
                _paste_soft_alpha(canvas, warped_rgb, warped_alpha, inp["boxes"][t])
            else:
                assert masks_full is not None
                k6b.paste(canvas, warped_rgb, inp["boxes"][t], masks_full[role_indices[role]])
            comp_s += time.perf_counter() - tc0

        canvases.append(canvas)

    total_s = matte_init_s + warp_s + comp_s
    timing = {
        "matte_init_seconds": matte_init_s,
        "warp_seconds": warp_s,
        "composite_seconds": comp_s,
        "total_receiver_seconds": total_s,
        "fps_total": float(n_frames) / max(total_s, 1e-6),
        "fps_steady_state": float(n_frames) / max(warp_s + comp_s, 1e-6),
        "speedup_vs_ddim30": DDIM30_48F_SECONDS / max(total_s, 1e-6),
    }
    return canvases, timing


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    stage_name = os.environ["PS_STAGE"]
    stage_dir = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    expected_frames = {"smoke": 16, "full": 48}[stage_name]
    if args.frames != expected_frames:
        raise ValueError(f"frames={args.frames} mismatches stage {stage_name}")

    for path, expected, label in (
        (args.carrier, CARRIER_SHA256, "carrier"),
        (args.pixels, PIXELS_SHA256, "pixels"),
        (args.features, FEATURES_SHA256, "features"),
        (args.witness, WITNESS_SHA256, "witness"),
        (args.vgg, VGG_SHA256, "vgg"),
        (args.alexnet, ALEXNET_SHA256, "alexnet"),
        (args.yolo_pose, YOLO_POSE_SHA256, "yolo_pose"),
    ):
        if sha256(path) != expected:
            raise ValueError(f"{label} sha256 mismatch")

    _, _, decoded = packet.unpack(args.carrier.read_bytes())
    supports, support_hash = get_support(args.features)
    dists_model = quality.load_dists(args.vgg, "cpu")
    lpips_net = quality.load_lpips(args.alexnet, "cpu")

    from ultralytics import YOLO

    yolo_model = YOLO(str(args.yolo_pose))

    with ExitStack() as stack:
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        wit_dir = tmp / "wit"
        with tarfile.open(args.witness) as tf:
            tf.extractall(wit_dir, filter="data")

        roles: tuple[str, ...] = ("near",) if stage_name == "smoke" else ("near", "far")
        base_frames = load_base_frames(wit_dir / "publish/base.yuv", args.frames)
        src_frames = load_source_frames(args.pixels, args.frames)

        unit_inputs: dict[str, dict[str, Any]] = {}
        gt_kps: dict[str, np.ndarray] = {}
        for role in roles:
            run_data = decoded[f"match102_{role}"][0]
            raw_kp = np.asarray(run_data["kp"][: args.frames], dtype=np.float32)
            smooth_kp = smooth_keypoints(raw_kp)
            boxes = np.asarray(run_data["boxes"][: args.frames], dtype=np.float64)
            ref_box = np.asarray(run_data["reference_box"], dtype=np.float64)
            ref_crop = np.asarray(run_data["reference"], dtype=np.uint8)
            unit_inputs[role] = {
                "smooth_kp": smooth_kp,
                "boxes": boxes,
                "reference_box": ref_box,
                "ref_crop": ref_crop,
            }
            gt_kps[role] = smooth_kp

        tlpips_ticks = (
            set(range(1, args.frames, 2))
            if args.frames <= 16
            else (set(range(5, args.frames, 5)) | {16, 20, 32, 40})
        )
        yolo_ticks = set(range(0, args.frames, 4))
        yy, xx = np.indices((544, 960), dtype=np.float32)

        src_960_gray: list[np.ndarray] = []
        src_960_luma: list[np.ndarray] = []
        src_flow: list[np.ndarray | None] = [None]
        src_delta: list[np.ndarray | None] = [None]
        for t in range(args.frames):
            im960 = cv2.resize(src_frames[t], (960, 544), interpolation=cv2.INTER_AREA)
            src_960_gray.append(cv2.cvtColor(im960, cv2.COLOR_RGB2GRAY))
            src_960_luma.append(luma(im960))
            if t > 0:
                flow = dis_flow(src_960_gray[t], src_960_gray[t - 1])
                src_flow.append(flow)
                src_delta.append(src_960_luma[t] - warp(src_960_luma[t - 1], flow))

        variants = (
            "v1_gauss_sam",
            "v2_lbs_sam",
            "v2_mls_sam",
            "v2_lbs_warped_mask",
            "v2_mls_warped_mask",
        )
        role_indices = {"near": 0, "far": 1}
        per_variant: dict[str, Any] = {}

        # Determinism check on smoke
        if stage_name == "smoke":
            c1, _ = render_variant("v2_lbs_warped_mask", roles, unit_inputs, base_frames[:4], supports)
            c2, _ = render_variant("v2_lbs_warped_mask", roles, unit_inputs, base_frames[:4], supports)
            if not all(np.array_equal(a, b) for a, b in zip(c1, c2, strict=True)):
                raise RuntimeError("determinism check failed for v2_lbs_warped_mask")

        t0 = time.monotonic()
        for vname in variants:
            canvases, timing = render_variant(vname, roles, unit_inputs, base_frames, supports)

            dists_rows: list[dict[str, Any]] = []
            flow_rows: list[dict[str, Any]] = []
            tlpips_rows: list[dict[str, Any]] = []
            yolo_rows: list[dict[str, Any]] = []
            prev_960_gray: np.ndarray | None = None
            prev_960_luma: np.ndarray | None = None
            prev_960_rgb: np.ndarray | None = None

            for t in range(args.frames):
                canvas = canvases[t]
                boxes_t = supports[t]
                masks_960 = [mask_960(boxes_t[i]) for i in range(2)]

                for role in roles:
                    idx = role_indices[role]
                    (bx0, by0, bx1, by1), bm = boxes_t[idx]
                    ref = src_frames[t][by0:by1, bx0:bx1]
                    ob = canvas[by0:by1, bx0:bx1]
                    act = torch.from_numpy(bm.copy())[None, None]
                    t_ob = tensor(ob)
                    t_ref = tensor(ref)
                    with torch.inference_mode():
                        dval = float(quality.masked_dists(dists_model, t_ob, t_ref, act))
                        lo_ob, lo_ref = t_ob, t_ref
                        if lo_ob.shape[-2] < 32 or lo_ob.shape[-1] < 32:
                            sz = (max(32, lo_ob.shape[-2]), max(32, lo_ob.shape[-1]))
                            lo_ob = torch.nn.functional.interpolate(
                                lo_ob, size=sz, mode="bilinear", align_corners=False
                            )
                            lo_ref = torch.nn.functional.interpolate(
                                lo_ref, size=sz, mode="bilinear", align_corners=False
                            )
                        lpval = float(lpips_net(lo_ob * 2 - 1, lo_ref * 2 - 1).mean())
                    diff = ob[bm].astype(np.float32) - ref[bm].astype(np.float32)
                    mse = float((diff * diff).mean())
                    psnr = 20.0 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else 99.0
                    dists_rows.append({"tick": t, "role": role, "dists": dval, "lpips": lpval, "psnr": psnr})

                if t in yolo_ticks:
                    for role in roles:
                        idx = role_indices[role]
                        bbox, _ = boxes_t[idx]
                        oks = best_yolo_oks(yolo_model, canvas, gt_kps[role][t], bbox)
                        yolo_rows.append({"tick": t, "role": role, "oks": oks, "recalled": oks >= 0.5})

                cur_960_rgb = cv2.resize(canvas, (960, 544), interpolation=cv2.INTER_AREA)
                cur_960_gray = cv2.cvtColor(cur_960_rgb, cv2.COLOR_RGB2GRAY)
                cur_960_luma = luma(cur_960_rgb)

                if t > 0:
                    assert prev_960_gray is not None and prev_960_luma is not None and prev_960_rgb is not None
                    s_flow = src_flow[t]
                    s_del = src_delta[t]
                    assert s_flow is not None and s_del is not None
                    sx, sy = xx + s_flow[:, :, 0], yy + s_flow[:, :, 1]
                    s_inside = (sx >= 0) & (sy >= 0) & (sx < 959) & (sy < 543)
                    c_src_del = cur_960_luma - warp(prev_960_luma, s_flow)

                    a_flow = dis_flow(cur_960_gray, prev_960_gray)
                    ax, ay = xx + a_flow[:, :, 0], yy + a_flow[:, :, 1]
                    a_inside = (ax >= 0) & (ay >= 0) & (ax < 959) & (ay < 543)
                    a_self_del = cur_960_luma - warp(prev_960_luma, a_flow)

                    prev_masks_960 = [mask_960(supports[t - 1][i]) for i in range(2)]
                    for role in roles:
                        idx = role_indices[role]
                        reg_src = s_inside & (masks_960[idx] > 0) & (warp(prev_masks_960[idx], s_flow) >= 0.5)
                        if not reg_src.any():
                            continue
                        src_err = float(np.abs(c_src_del[reg_src] - s_del[reg_src]).mean())
                        reg_self = a_inside & (masks_960[idx] > 0) & (warp(prev_masks_960[idx], a_flow) >= 0.5)
                        self_err = float(np.abs(a_self_del[reg_self]).mean()) if reg_self.any() else src_err
                        flow_rows.append(
                            {
                                "tick": t,
                                "role": role,
                                "seam": is_seam_tick(t),
                                "source_flow_error": src_err,
                                "self_flow_error": self_err,
                            }
                        )
                        if t in tlpips_ticks:
                            (cx0, cy0, cx1, cy1), _ = boxes_t[idx]
                            (px0, py0, px1, py1), _ = supports[t - 1][idx]
                            ux0 = min(cx0, px0) // 2
                            uy0 = min(cy0, py0) // 2
                            ux1 = min(960, max(cx1, px1) // 2)
                            uy1 = min(544, max(cy1, py1) // 2)
                            if ux1 - ux0 >= 32 and uy1 - uy0 >= 32:
                                prev_warped_rgb = np.stack(
                                    [warp(prev_960_rgb[:, :, c].astype(np.float32), s_flow) for c in range(3)],
                                    axis=2,
                                )
                                cur_crop = cur_960_rgb[uy0:uy1, ux0:ux1].astype(np.float32) / 255.0
                                prv_crop = np.clip(prev_warped_rgb[uy0:uy1, ux0:ux1], 0.0, 255.0) / 255.0
                                tc = torch.from_numpy(cur_crop).permute(2, 0, 1)[None] * 2 - 1
                                tp = torch.from_numpy(prv_crop).permute(2, 0, 1)[None] * 2 - 1
                                with torch.inference_mode():
                                    tlp = float(lpips_net(tc, tp).mean())
                                tlpips_rows.append({"tick": t, "role": role, "tlpips": tlp})

                prev_960_gray = cur_960_gray
                prev_960_luma = cur_960_luma
                prev_960_rgb = cur_960_rgb

            per_variant[vname] = {
                "n_player_frames": len(dists_rows),
                "dists_mean": mean(r["dists"] for r in dists_rows),
                "lpips_mean": mean(r["lpips"] for r in dists_rows),
                "psnr_mean": mean(r["psnr"] for r in dists_rows),
                "source_flow_error_mean": mean(r["source_flow_error"] for r in flow_rows),
                "self_flow_error_mean": mean(r["self_flow_error"] for r in flow_rows),
                "tlpips_mean": mean(r["tlpips"] for r in tlpips_rows) if tlpips_rows else float("nan"),
                "yolo_pose_recall_at_050": mean(1.0 if r["recalled"] else 0.0 for r in yolo_rows),
                "yolo_pose_mean_oks": mean(r["oks"] for r in yolo_rows),
                "yolo_pose_n": len(yolo_rows),
                "timing": timing,
            }

        summary: dict[str, Any] = {
            "stage": stage_name,
            "frames": args.frames,
            "roles": list(roles),
            "support_sha256": support_hash,
            "wall_seconds": time.monotonic() - t0,
            "variants": per_variant,
        }
        text = json.dumps(summary, indent=2) + "\n"
        pub = scratch / "publish"
        pub.mkdir(parents=True, exist_ok=True)
        (pub / "warp_v2_pilot_scores.json").write_text(text)
        (stage_dir / "warp_v2_pilot_scores.json").write_text(text)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "stage": stage_name,
                    "wall_seconds": round(summary["wall_seconds"], 2),
                    "variants": {
                        k: {
                            "dists": round(v["dists_mean"], 4),
                            "lpips": round(v["lpips_mean"], 4),
                            "psnr": round(v["psnr_mean"], 2),
                            "self_flow": round(v["self_flow_error_mean"], 2),
                            "tlpips": round(v["tlpips_mean"], 4),
                            "pose_recall": round(v["yolo_pose_recall_at_050"], 4),
                            "pose_oks": round(v["yolo_pose_mean_oks"], 4),
                            "fps_total": round(v["timing"]["fps_total"], 1),
                            "fps_steady": round(v["timing"]["fps_steady_state"], 1),
                        }
                        for k, v in per_variant.items()
                    },
                }
            )
        )
        return summary


def validate_stage(stage_dir: Path) -> None:
    data = json.loads((stage_dir / "warp_v2_pilot_scores.json").read_text())
    variants = data.get("variants", {})
    expected = {"v1_gauss_sam", "v2_lbs_sam", "v2_mls_sam", "v2_lbs_warped_mask", "v2_mls_warped_mask"}
    checks = {
        "has_all_variants": set(variants) == expected,
        "finite_metrics": all(
            math.isfinite(float(v[m]))
            for v in variants.values()
            for m in ("dists_mean", "lpips_mean", "psnr_mean", "self_flow_error_mean", "yolo_pose_mean_oks")
        ),
        "positive_timing": all(
            float(v["timing"]["total_receiver_seconds"]) > 0.0 for v in variants.values()
        ),
    }
    ok = all(checks.values())
    val_path = os.environ.get("PS_VALIDATION_PATH")
    if val_path:
        Path(val_path).write_text(json.dumps({"passed": ok, "checks": checks}, indent=2) + "\n")
    if not ok:
        raise ValueError(f"warp v2 pilot validation failed: {checks}")
    print(json.dumps({"passed": ok, "checks": checks, "stage": data["stage"]}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track A Stage 1: k6_ref_warp v2 48-frame pilot")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--carrier", type=Path)
    ap.add_argument("--pixels", type=Path)
    ap.add_argument("--features", type=Path)
    ap.add_argument("--witness", type=Path)
    ap.add_argument("--vgg", type=Path)
    ap.add_argument("--alexnet", type=Path)
    ap.add_argument("--yolo-pose", type=Path)
    ap.add_argument("--frames", type=int, default=48)
    args, _ = ap.parse_known_args(argv)
    if args.validate:
        validate_stage(Path(os.environ["PS_STAGE_DIR"]))
        return 0
    run_pilot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

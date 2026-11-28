"""Track A Stage 2: 240-frame end-to-end native CPU receiver benchmark (Shanghai, Sydney, Halle).

Decodes the real transmitted players.pscp wire packet (AV1 IVF base + PSCP poses + PSR reference)
strictly on 1 CPU thread (0 GPU-hours) and evaluates three MLS-Similarity (step=8) receiver arms:
  1. mls_step8_bgsub_alpha: 0 extra wire bytes, 0 SAM, 0 GPU (t=0 matte from |ref_crop - base_0|)
  2. mls_step8_psr_alpha:   paid 1-bit t=0 PNG reference mask in PSR (+billed kbps), 0 receiver SAM, 0 GPU
  3. mls_step8_sam_oracle:  per-frame SAM 3.1 compositing mask (diagnostic upper bound for silhouette drift)
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from fractions import Fraction
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import tempfile
import time
from typing import Any

import cv2
import numpy as np
import torch

from experiments.tennis import common_player_packet as packet
from experiments.tennis import k6b
from experiments.tennis.player_requalification import sha256
from experiments.tennis.tomm_multi_backbone_score import YOLO_POSE_SHA256, best_yolo_oks
from experiments.tennis.tomm_v5_pareto_pilot import smooth_keypoints
from experiments.tennis.tomm_v5_score import ALEXNET_SHA256, VGG_SHA256, mask_full_res
from experiments.tennis.tomm_warp_v2_pilot import (
    COCO_BONES,
    _crop_from_canvas,
    _paste_soft_alpha,
    extract_ref_alpha_from_base,
)
from src.codecs import quality

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/experiments/test-crop-joint-20261122"))
from score_native import get_support, tensor  # noqa: E402

MATCH_RECEIPTS: dict[str, Path] = {
    "match102": ROOT / "docs/experiments/test-crop-joint-20261122/match102-result.json",
    "match127": ROOT / "docs/experiments/test-joint-extension-20261122/match127-crop-result.json",
    "match105": ROOT / "docs/experiments/test-joint-extension-20261122/match105-crop-result.json",
}


def compute_mls_similarity_maps_step(
    h: int,
    w: int,
    src_kp: np.ndarray,
    dst_kp: np.ndarray,
    step: int = 8,
    conf_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Closed-form Moving Least Squares (MLS) Similarity warp on a coarse step grid."""
    valid = (src_kp[:, 2] >= conf_thr) & (dst_kp[:, 2] >= conf_thr)
    src_pts: list[np.ndarray] = [src_kp[i, :2].astype(np.float32) for i in range(len(src_kp)) if valid[i]]
    dst_pts: list[np.ndarray] = [dst_kp[i, :2].astype(np.float32) for i in range(len(dst_kp)) if valid[i]]
    for ja, jb, _ in COCO_BONES:
        if valid[ja] and valid[jb]:
            src_pts.append(0.5 * (src_kp[ja, :2] + src_kp[jb, :2]).astype(np.float32))
            dst_pts.append(0.5 * (dst_kp[ja, :2] + dst_kp[jb, :2]).astype(np.float32))
    if len(src_pts) < 4:
        return None

    q = np.stack(src_pts, axis=0)
    p = np.stack(dst_pts, axis=0)

    gy = np.arange(0, h, step, dtype=np.float32)
    gx = np.arange(0, w, step, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(gx, gy)

    dx = grid_x[..., None] - p[None, None, :, 0]
    dy = grid_y[..., None] - p[None, None, :, 1]
    eps = (0.015 * float(h)) ** 2
    wt = 1.0 / (dx * dx + dy * dy + eps)
    w_sum = wt.sum(axis=-1, keepdims=True)

    px_star = (wt * p[None, None, :, 0]).sum(axis=-1, keepdims=True) / w_sum
    py_star = (wt * p[None, None, :, 1]).sum(axis=-1, keepdims=True) / w_sum
    qx_star = (wt * q[None, None, :, 0]).sum(axis=-1, keepdims=True) / w_sum
    qy_star = (wt * q[None, None, :, 1]).sum(axis=-1, keepdims=True) / w_sum

    phx = p[None, None, :, 0] - px_star
    phy = p[None, None, :, 1] - py_star
    qhx = q[None, None, :, 0] - qx_star
    qhy = q[None, None, :, 1] - qy_star

    mu_s = (wt * (phx * phx + phy * phy)).sum(axis=-1) + 1e-6
    vx = grid_x - px_star[..., 0]
    vy = grid_y - py_star[..., 0]

    a = (wt * (qhx * phx + qhy * phy)).sum(axis=-1) / mu_s
    b = (wt * (qhy * phx - qhx * phy)).sum(axis=-1) / mu_s

    mag = np.sqrt(a * a + b * b) + 1e-6
    scale_clamped = np.clip(mag, 0.5, 1.8) / mag
    a *= scale_clamped
    b *= scale_clamped

    map_x_small = (qx_star[..., 0] + a * vx - b * vy).astype(np.float32)
    map_y_small = (qy_star[..., 0] + b * vx + a * vy).astype(np.float32)

    map_x = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return map_x, map_y


def decode_ivf_to_canvas(
    ivf_bytes: bytes,
    coded_w: int,
    coded_h: int,
    canvas_w: int,
    canvas_h: int,
    n_frames: int,
    scratch: Path,
) -> tuple[list[np.ndarray], float]:
    """Decode native AV1 IVF base from players.pscp and upscale to canvas with cv2.INTER_CUBIC."""
    ivf_path = scratch / "base.ivf"
    ivf_path.write_bytes(ivf_bytes)
    t0 = time.perf_counter()
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        str(ivf_path),
        "-frames:v",
        str(n_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    raw = subprocess.check_output(cmd)
    frame_bytes = coded_w * coded_h * 3
    if len(raw) < n_frames * frame_bytes:
        raise RuntimeError(f"decoded IVF too short: {len(raw)} < {n_frames * frame_bytes}")
    frames: list[np.ndarray] = []
    for i in range(n_frames):
        buf = np.frombuffer(raw, dtype=np.uint8, count=frame_bytes, offset=i * frame_bytes).reshape(
            coded_h, coded_w, 3
        )
        if coded_w != canvas_w or coded_h != canvas_h:
            buf = cv2.resize(buf, (canvas_w, canvas_h), interpolation=cv2.INTER_CUBIC)
        else:
            buf = buf.copy()
        frames.append(buf)
    return frames, time.perf_counter() - t0


def load_source_video(path: Path, n_frames: int, width: int = 1920, height: int = 1080) -> list[np.ndarray]:
    """Decode first n_frames of source .mkv/.mp4 as RGB8 1080p."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        str(path),
        "-frames:v",
        str(n_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    raw = subprocess.check_output(cmd)
    frame_bytes = width * height * 3
    if len(raw) < n_frames * frame_bytes:
        raise RuntimeError(f"decoded source too short: {len(raw)} < {n_frames * frame_bytes}")
    return [
        np.frombuffer(raw, dtype=np.uint8, count=frame_bytes, offset=i * frame_bytes).reshape(height, width, 3).copy()
        for i in range(n_frames)
    ]


def encode_decode_psr_alpha_t0(
    full_mask_t0: np.ndarray,
    ref_box: np.ndarray,
    out_w: int = 512,
    out_h: int = 784,
) -> tuple[np.ndarray, int]:
    """Sender encodes the t=0 1-bit reference silhouette as a compact PNG; receiver decodes + feathers it.

    Returns (float32 alpha matte in [0, 1] of shape (out_h, out_w), exact compressed PNG byte count).
    """
    mask_u8 = (full_mask_t0 > 0).astype(np.uint8) * 255
    crop_mask = _crop_from_canvas(mask_u8[..., None], ref_box, out_w=out_w, out_h=out_h)
    if crop_mask.ndim == 3:
        crop_mask = crop_mask[..., 0]
    # Downsample to 1/4 res (196x128) binary PNG for minimal wire footprint (~200-500 bytes per player)
    small = (cv2.resize(crop_mask, (out_w // 4, out_h // 4), interpolation=cv2.INTER_AREA) >= 128).astype(
        np.uint8
    ) * 255
    ok, encoded = cv2.imencode(".png", small, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError("failed to PNG-encode t=0 reference mask")
    raw_bytes = encoded.tobytes()

    # Receiver decodes PNG and upsamples with 1.2 px edge feather
    dec = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    alpha = cv2.resize(dec.astype(np.float32) / 255.0, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 1.2)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32), len(raw_bytes)


def role_support_indices(receipt_path: Path) -> dict[str, int]:
    """Map 'near' -> taller player index in selected_ids, 'far' -> shorter player index."""
    receipt = json.loads(receipt_path.read_text())
    ids = receipt["selected_ids"]
    heights = receipt["median_height_px"]
    if heights[ids[0]] >= heights[ids[1]]:
        return {"near": 0, "far": 1}
    return {"near": 1, "far": 0}


def render_arm(
    arm: str,
    roles: tuple[str, ...],
    role_idx: dict[str, int],
    unit_inputs: dict[str, dict[str, Any]],
    base_frames: list[np.ndarray],
    supports: Any,
    psr_alphas: dict[str, np.ndarray],
    step: int = 8,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Run one MLS-Similarity (step=8) receiver arm over all frames on 1 CPU thread."""
    n_frames = len(base_frames)
    use_warped_mask = arm != "mls_step8_sam_oracle"

    t_matte0 = time.perf_counter()
    ref_alphas: dict[str, np.ndarray] = {}
    if arm == "mls_step8_bgsub_alpha":
        for role in roles:
            inp = unit_inputs[role]
            ref_kp = k6b.to_crop(inp["smooth_kp"][0], inp["reference_box"])
            ref_alphas[role] = extract_ref_alpha_from_base(
                inp["ref_crop"], base_frames[0], inp["reference_box"], ref_kp
            )
    elif arm == "mls_step8_psr_alpha":
        for role in roles:
            ref_alphas[role] = psr_alphas[role]
    matte_init_s = time.perf_counter() - t_matte0

    ref_kps: dict[str, np.ndarray] = {}
    crop_kps: dict[str, list[np.ndarray]] = {}
    for role in roles:
        inp = unit_inputs[role]
        ref_kps[role] = k6b.to_crop(inp["smooth_kp"][0], inp["reference_box"])
        crop_kps[role] = [k6b.to_crop(inp["smooth_kp"][t], inp["boxes"][t]) for t in range(n_frames)]

    canvases: list[np.ndarray] = []
    warp_s = 0.0
    comp_s = 0.0

    for t in range(n_frames):
        canvas = base_frames[t].copy()
        boxes_t = supports[t]
        masks_full = None if use_warped_mask else [mask_full_res(boxes_t[i]) for i in range(2)]

        for role in reversed(roles):
            inp = unit_inputs[role]
            ref_crop = inp["ref_crop"]
            h, w = ref_crop.shape[:2]

            tw0 = time.perf_counter()
            maps = compute_mls_similarity_maps_step(h, w, ref_kps[role], crop_kps[role][t], step=step)
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
                k6b.paste(canvas, warped_rgb, inp["boxes"][t], masks_full[role_idx[role]])
            comp_s += time.perf_counter() - tc0

        canvases.append(canvas)

    synth_total_s = matte_init_s + warp_s + comp_s
    return canvases, {
        "matte_init_seconds": matte_init_s,
        "warp_seconds": warp_s,
        "composite_seconds": comp_s,
        "synth_and_composite_seconds": synth_total_s,
        "fps_synth_and_composite": float(n_frames) / max(synth_total_s, 1e-6),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    stage_name = os.environ["PS_STAGE"]
    stage_dir = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    expected_frames = {"smoke": 24, "full": 240}[stage_name]
    if args.frames != expected_frames:
        raise ValueError(f"frames={args.frames} mismatches stage {stage_name}")

    for path, expected, label in (
        (args.vgg, VGG_SHA256, "vgg"),
        (args.alexnet, ALEXNET_SHA256, "alexnet"),
        (args.yolo_pose, YOLO_POSE_SHA256, "yolo_pose"),
    ):
        if sha256(path) != expected:
            raise ValueError(f"{label} sha256 mismatch")

    receipt_path = MATCH_RECEIPTS[args.event]
    role_idx = role_support_indices(receipt_path)
    supports, support_hash = get_support(args.features, receipt=receipt_path)

    carrier_bytes = args.carrier.read_bytes()
    info = packet.inspect(carrier_bytes)
    profile = info["profile"]
    sections, accounting, decoded = packet.unpack(carrier_bytes, profile)
    fps = Fraction(*profile["recipe"]["fps"])
    duration_s = float(Fraction(240, 1) / fps)
    warm_bytes = int(accounting["frame"])
    cold_bytes = int(len(carrier_bytes))
    warm_kbps = (warm_bytes * 8.0) / (duration_s * 1000.0)
    cold_kbps = (cold_bytes * 8.0) / (duration_s * 1000.0)

    coded_w, coded_h = int(profile["coded_size"][0]), int(profile["coded_size"][1])
    canvas_w, canvas_h = int(profile["canvas_size"][0]), int(profile["canvas_size"][1])

    with ExitStack() as stack:
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        base_frames, av1_decode_s = decode_ivf_to_canvas(
            sections["base"], coded_w, coded_h, canvas_w, canvas_h, args.frames, tmp
        )
        src_frames = load_source_video(args.source, args.frames, width=canvas_w, height=canvas_h)

        roles: tuple[str, ...] = ("near", "far")
        unit_inputs: dict[str, dict[str, Any]] = {}
        gt_kps: dict[str, np.ndarray] = {}
        psr_alphas: dict[str, np.ndarray] = {}
        psr_alpha_bytes = 0

        for role in roles:
            unit_name = f"{args.event}_{role}"
            run_data = decoded[unit_name][0]
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
            full_m0 = mask_full_res(supports[0][role_idx[role]])
            alpha0, nbytes = encode_decode_psr_alpha_t0(full_m0, ref_box)
            psr_alphas[role] = alpha0
            psr_alpha_bytes += nbytes

        psr_alpha_extra_kbps = (psr_alpha_bytes * 8.0) / (duration_s * 1000.0)

        # Smoke determinism check
        if stage_name == "smoke":
            for check_arm in ("mls_step8_bgsub_alpha", "mls_step8_psr_alpha"):
                c1, _ = render_arm(check_arm, roles, role_idx, unit_inputs, base_frames[:4], supports, psr_alphas)
                c2, _ = render_arm(check_arm, roles, role_idx, unit_inputs, base_frames[:4], supports, psr_alphas)
                if not all(np.array_equal(a, b) for a, b in zip(c1, c2, strict=True)):
                    raise RuntimeError(f"determinism check failed for {check_arm}")

        dists_model = quality.load_dists(args.vgg, "cpu")
        lpips_net = quality.load_lpips(args.alexnet, "cpu")
        from ultralytics import YOLO

        yolo_model = YOLO(str(args.yolo_pose))

        # Evaluate YOLO26x-pose every 10th frame in full (24 frames * 2 players = 48 player-frames) and 4 in smoke
        yolo_ticks = set(range(0, args.frames, 6 if stage_name == "smoke" else 10))
        arms = ("mls_step8_bgsub_alpha", "mls_step8_psr_alpha", "mls_step8_sam_oracle")
        per_arm: dict[str, Any] = {}
        t0 = time.monotonic()

        for arm in arms:
            canvases, timing = render_arm(arm, roles, role_idx, unit_inputs, base_frames, supports, psr_alphas)
            end_to_end_s = av1_decode_s + timing["synth_and_composite_seconds"]
            timing["av1_base_decode_seconds"] = av1_decode_s
            timing["end_to_end_receiver_seconds"] = end_to_end_s
            timing["fps_end_to_end"] = float(args.frames) / max(end_to_end_s, 1e-6)

            dists_rows: list[dict[str, Any]] = []
            yolo_rows: list[dict[str, Any]] = []

            for t in range(args.frames):
                canvas = canvases[t]
                boxes_t = supports[t]
                for role in roles:
                    idx = role_idx[role]
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
                        idx = role_idx[role]
                        bbox, _ = boxes_t[idx]
                        oks = best_yolo_oks(yolo_model, canvas, gt_kps[role][t], bbox)
                        yolo_rows.append({"tick": t, "role": role, "oks": oks, "recalled": oks >= 0.5})

            prefix48 = [r for r in dists_rows if r["tick"] < 48]
            per_arm[arm] = {
                "n_player_frames": len(dists_rows),
                "dists_mean": mean(r["dists"] for r in dists_rows),
                "lpips_mean": mean(r["lpips"] for r in dists_rows),
                "psnr_mean": mean(r["psnr"] for r in dists_rows),
                "near_dists_mean": mean(r["dists"] for r in dists_rows if r["role"] == "near"),
                "far_dists_mean": mean(r["dists"] for r in dists_rows if r["role"] == "far"),
                "prefix48_dists_mean": mean(r["dists"] for r in prefix48) if prefix48 else float("nan"),
                "prefix48_lpips_mean": mean(r["lpips"] for r in prefix48) if prefix48 else float("nan"),
                "yolo_pose_recall_at_050": mean(1.0 if r["recalled"] else 0.0 for r in yolo_rows),
                "yolo_pose_mean_oks": mean(r["oks"] for r in yolo_rows),
                "yolo_pose_n": len(yolo_rows),
                "extra_wire_bytes": psr_alpha_bytes if arm == "mls_step8_psr_alpha" else 0,
                "warm_kbps_total": warm_kbps + (psr_alpha_extra_kbps if arm == "mls_step8_psr_alpha" else 0.0),
                "timing": timing,
            }
            print(
                json.dumps(
                    {
                        "progress": arm,
                        "event": args.event,
                        "dists": round(per_arm[arm]["dists_mean"], 4),
                        "lpips": round(per_arm[arm]["lpips_mean"], 4),
                        "pose_recall": round(per_arm[arm]["yolo_pose_recall_at_050"], 4),
                        "fps_e2e": round(timing["fps_end_to_end"], 1),
                        "fps_synth": round(timing["fps_synth_and_composite"], 1),
                    }
                ),
                flush=True,
            )

        summary: dict[str, Any] = {
            "event": args.event,
            "stage": stage_name,
            "frames": args.frames,
            "fps": str(fps),
            "duration_seconds": duration_s,
            "carrier_sha256": sha256(args.carrier),
            "source_sha256": sha256(args.source),
            "support_sha256": support_hash,
            "warm_bytes": warm_bytes,
            "cold_bytes": cold_bytes,
            "warm_kbps": warm_kbps,
            "cold_kbps": cold_kbps,
            "psr_alpha_bytes": psr_alpha_bytes,
            "psr_alpha_extra_kbps": psr_alpha_extra_kbps,
            "wall_seconds": time.monotonic() - t0,
            "arms": per_arm,
        }
        text = json.dumps(summary, indent=2) + "\n"
        pub = scratch / "publish"
        pub.mkdir(parents=True, exist_ok=True)
        out_name = f"warp_native240_{args.event}.json"
        (pub / out_name).write_text(text)
        (stage_dir / out_name).write_text(text)
        return summary


def validate_stage(args: argparse.Namespace) -> None:
    stage_dir = Path(os.environ["PS_STAGE_DIR"])
    data = json.loads((stage_dir / f"warp_native240_{args.event}.json").read_text())
    arms = data.get("arms", {})
    expected = {"mls_step8_bgsub_alpha", "mls_step8_psr_alpha", "mls_step8_sam_oracle"}
    checks = {
        "has_all_arms": set(arms) == expected,
        "finite_metrics": all(
            math.isfinite(float(v[m]))
            for v in arms.values()
            for m in ("dists_mean", "lpips_mean", "psnr_mean", "yolo_pose_mean_oks")
        ),
        "positive_timing": all(
            float(v["timing"]["end_to_end_receiver_seconds"]) > 0.0 for v in arms.values()
        ),
    }
    ok = all(checks.values())
    val_path = os.environ.get("PS_VALIDATION_PATH")
    if val_path:
        Path(val_path).write_text(json.dumps({"passed": ok, "checks": checks}, indent=2) + "\n")
    if not ok:
        raise ValueError(f"native240 validation failed: {checks}")
    print(json.dumps({"passed": ok, "event": args.event, "stage": data["stage"]}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track A Stage 2: 240-frame native CPU MLS warp benchmark")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--event", choices=("match102", "match127", "match105"), required=True)
    ap.add_argument("--carrier", type=Path)
    ap.add_argument("--source", type=Path)
    ap.add_argument("--features", type=Path)
    ap.add_argument("--vgg", type=Path)
    ap.add_argument("--alexnet", type=Path)
    ap.add_argument("--yolo-pose", type=Path)
    ap.add_argument("--frames", type=int, default=240)
    args, _ = ap.parse_known_args(argv)
    if args.validate:
        validate_stage(args)
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score TOMM multi-backbone Synthesize ablation (DISTS, LPIPS, PSNR, flow, tLPIPS, YOLO26x-pose OKS/recall)."""

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

YOLO_POSE_SHA256 = "08ed9e01d22a6f248b04f2f9992016aca9a32250b9ab57057d886a09d026700d"
DDIM30_48F_SECONDS = 674.4665817180066
COCO_SIGMAS = np.array(
    [
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ],
    dtype=np.float64,
)


def coco_oks(gt_kp: np.ndarray, det_kp: np.ndarray, area: float, conf_thr: float = 0.5) -> float:
    """Compute COCO Object Keypoint Similarity between 17x3 gt_kp and 17x3 det_kp."""
    vis = gt_kp[:, 2] >= conf_thr
    if not vis.any() or area <= 1.0:
        return 0.0
    dx = det_kp[vis, 0] - gt_kp[vis, 0]
    dy = det_kp[vis, 1] - gt_kp[vis, 1]
    vars_k = (2.0 * COCO_SIGMAS[vis]) ** 2
    e = (dx * dx + dy * dy) / (2.0 * area * vars_k)
    return float(np.exp(-e).mean())


def best_yolo_oks(
    yolo_model: Any,
    frame_rgb: np.ndarray,
    gt_kp: np.ndarray,
    mask_box: tuple[int, int, int, int],
) -> float:
    bx0, by0, bx1, by1 = mask_box
    area = float(max(1, (bx1 - bx0) * (by1 - by0)))
    # Run YOLO26x-pose on the full 1080p composited frame (BGR)
    results = yolo_model.predict(frame_rgb[:, :, ::-1], imgsz=1280, conf=0.15, verbose=False)
    if not results or results[0].keypoints is None or len(results[0].keypoints.data) == 0:
        return 0.0
    kps = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)
    return max(coco_oks(gt_kp, kps[i], area) for i in range(len(kps)))


def discover_backbones(gen_dir: Path, roles: tuple[str, ...]) -> tuple[list[str], dict[str, Any]]:
    pub = gen_dir / "publish"
    summary = json.loads((pub / "multi_backbone_summary.json").read_text())
    backbones: list[str] = []
    for sub in sorted(pub.iterdir()):
        if sub.is_dir() and all((sub / f"match102_{role}.npz").is_file() for role in roles):
            backbones.append(sub.name)
    return backbones, summary


def score_run(args: argparse.Namespace) -> dict[str, Any]:
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
        gen_dir = tmp / "gen"
        wit_dir = tmp / "wit"
        gen_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.gen_tar) as tf:
            tf.extractall(gen_dir, filter="data")
        with tarfile.open(args.witness) as tf:
            tf.extractall(wit_dir, filter="data")

        roles: tuple[str, ...] = ("near",) if stage_name == "smoke" else ("near", "far")
        backbones, gen_summary = discover_backbones(gen_dir, roles)
        if not backbones:
            raise ValueError("no complete backbone archives found")

        render_seconds_map: dict[str, float] = {}
        for rec in gen_summary.get("records", []):
            bname = str(rec["backbone"])
            render_seconds_map[bname] = render_seconds_map.get(bname, 0.0) + float(rec["render_seconds"])

        gt_kps: dict[str, np.ndarray] = {}
        for role in roles:
            raw_kp = np.asarray(decoded[f"match102_{role}"][0]["kp"][: args.frames], dtype=np.float32)
            gt_kps[role] = smooth_keypoints(raw_kp)

        gen_pub = gen_dir / "publish"
        base_frames = load_base_frames(wit_dir / "publish/base.yuv", args.frames)
        src_frames = load_source_frames(args.pixels, args.frames)

        tlpips_ticks = (
            set(range(1, args.frames, 2))
            if args.frames <= 16
            else (set(range(5, args.frames, 5)) | {16, 20, 32, 40})
        )
        # Evaluate YOLO26x-pose every 4th frame in full (12 frames * 2 players = 24 player-frames) and 4 frames in smoke
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

        role_indices = {"near": 0, "far": 1}
        per_backbone: dict[str, Any] = {}
        t0 = time.monotonic()

        (x0, y0, x1, y1), m0 = supports[0][0]
        tr0 = tensor(src_frames[0][y0:y1, x0:x1])
        act0 = torch.from_numpy(m0.copy())[None, None]
        with torch.inference_mode():
            self_dists = abs(float(quality.masked_dists(dists_model, tr0, tr0, act0)))
        if self_dists >= 1e-5:
            raise ValueError(f"masked DISTS self-check failed: {self_dists}")

        for bname in backbones:
            unit_data: dict[str, dict[str, np.ndarray]] = {}
            for role in roles:
                npz = np.load(gen_pub / f"{bname}/match102_{role}.npz")
                crops = np.asarray(npz["crops"], dtype=np.uint8)
                boxes = np.asarray(npz["box"], dtype=np.float64)
                unit_data[role] = {"box": boxes, "crops": crops}

            dists_rows: list[dict[str, Any]] = []
            flow_rows: list[dict[str, Any]] = []
            tlpips_rows: list[dict[str, Any]] = []
            yolo_rows: list[dict[str, Any]] = []
            prev_960_gray: np.ndarray | None = None
            prev_960_luma: np.ndarray | None = None
            prev_960_rgb: np.ndarray | None = None

            for t in range(args.frames):
                canvas = base_frames[t].copy()
                boxes_t = supports[t]
                masks_full = [mask_full_res(boxes_t[i]) for i in range(2)]
                masks_960 = [mask_960(boxes_t[i]) for i in range(2)]
                for role in reversed(roles):
                    idx = role_indices[role]
                    k6b.paste(canvas, unit_data[role]["crops"][t], unit_data[role]["box"][t], masks_full[idx])

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
                        # Masked LPIPS on player bbox crop (minimum 32x32 for AlexNet)
                        lo_ob, lo_ref = t_ob, t_ref
                        if lo_ob.shape[-2] < 32 or lo_ob.shape[-1] < 32:
                            sz = (max(32, lo_ob.shape[-2]), max(32, lo_ob.shape[-1]))
                            lo_ob = torch.nn.functional.interpolate(lo_ob, size=sz, mode="bilinear", align_corners=False)
                            lo_ref = torch.nn.functional.interpolate(lo_ref, size=sz, mode="bilinear", align_corners=False)
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
                            c0 = torch.from_numpy(np.ascontiguousarray(prev_960_rgb[uy0:uy1, ux0:ux1])).permute(2, 0, 1)[None].float() / 255.0
                            c1 = torch.from_numpy(np.ascontiguousarray(cur_960_rgb[uy0:uy1, ux0:ux1])).permute(2, 0, 1)[None].float() / 255.0
                            if c0.shape[-2] < 32 or c0.shape[-1] < 32:
                                size = (max(32, c0.shape[-2]), max(32, c0.shape[-1]))
                                c0 = torch.nn.functional.interpolate(c0, size=size, mode="bilinear", align_corners=False)
                                c1 = torch.nn.functional.interpolate(c1, size=size, mode="bilinear", align_corners=False)
                            with torch.inference_mode():
                                lp = float(lpips_net(c0 * 2 - 1, c1 * 2 - 1).mean())
                            tlpips_rows.append({"tick": t, "role": role, "seam": (t % 20 == 0), "tlpips": lp})

                prev_960_gray, prev_960_luma, prev_960_rgb = cur_960_gray, cur_960_luma, cur_960_rgb

            render_s = render_seconds_map.get(bname)
            per_backbone[bname] = {
                "n_dists": len(dists_rows),
                "masked_dists": mean(r["dists"] for r in dists_rows),
                "masked_lpips": mean(r["lpips"] for r in dists_rows),
                "masked_psnr_db": mean(r["psnr"] for r in dists_rows),
                "near_dists": mean(r["dists"] for r in dists_rows if r["role"] == "near"),
                "far_dists": mean(r["dists"] for r in dists_rows if r["role"] == "far") if "far" in roles else None,
                "n_flow": len(flow_rows),
                "source_flow_error": mean(r["source_flow_error"] for r in flow_rows),
                "self_flow_error": mean(r["self_flow_error"] for r in flow_rows),
                "tlpips_n": len(tlpips_rows),
                "tlpips_mean": mean(r["tlpips"] for r in tlpips_rows),
                "yolo26x_pose_n": len(yolo_rows),
                "yolo26x_pose_oks_mean": mean(r["oks"] for r in yolo_rows),
                "yolo26x_pose_recall_at_050": mean(1.0 if r["recalled"] else 0.0 for r in yolo_rows),
                "render_s_total": render_s,
                "sec_per_player_frame": (render_s / (args.frames * len(roles))) if render_s else None,
                "speedup_vs_ddim30": (DDIM30_48F_SECONDS / render_s) if (render_s and stage_name == "full") else None,
            }

    result = {
        "event": "match102",
        "stage": stage_name,
        "frames": args.frames,
        "roles": list(roles),
        "gen_tar_sha256": sha256(args.gen_tar),
        "carrier_sha256": CARRIER_SHA256,
        "source_mask_rle_sha256": support_hash,
        "device": gen_summary.get("device"),
        "timing_contaminated": gen_summary.get("timing_contaminated"),
        "ddim30_48f_seconds_ref": DDIM30_48F_SECONDS,
        "self_dists_check": self_dists,
        "elapsed_s": time.monotonic() - t0,
        "backbones": per_backbone,
    }
    pub = scratch / "publish"
    pub.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2) + "\n"
    (pub / "tomm-multi-backbone-scores.json").write_text(text)
    (stage_dir / "tomm-multi-backbone-scores.json").write_text(text)
    return result


def validate(args: argparse.Namespace) -> None:
    stage_dir = Path(os.environ["PS_STAGE_DIR"])
    data = json.loads((stage_dir / "tomm-multi-backbone-scores.json").read_text())
    bbs = data["backbones"]
    expected_backbones = {b.strip() for b in args.expected_backbones.split(",") if b.strip()}
    checks = {
        "has_all_backbones": set(bbs) == expected_backbones,
        "self_dists_ok": float(data["self_dists_check"]) < 1e-5,
        "finite_scores": all(
            math.isfinite(c["masked_dists"])
            and math.isfinite(c["masked_lpips"])
            and math.isfinite(c["masked_psnr_db"])
            and math.isfinite(c["self_flow_error"])
            and math.isfinite(c["tlpips_mean"])
            and math.isfinite(c["yolo26x_pose_oks_mean"])
            for c in bbs.values()
        ),
    }
    ok = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps({"passed": ok, "checks": checks}, indent=2) + "\n"
    )
    if not ok:
        raise ValueError(f"multi-backbone score validation failed: {checks}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen-tar", type=Path, required="--validate" not in sys.argv)
    for k in ("carrier", "pixels", "features", "witness", "vgg", "alexnet", "yolo-pose"):
        p.add_argument("--" + k, type=Path, required="--validate" not in sys.argv)
    p.add_argument("--frames", type=int, choices=(16, 48), default=48)
    p.add_argument(
        "--expected-backbones",
        default="k6_ref_warp,aa_dpm10_ov8_smooth,cn_openpose_sd15_dpm10",
    )
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    if args.validate:
        validate(args)
    else:
        score_run(args)


if __name__ == "__main__":
    main()

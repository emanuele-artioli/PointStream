"""Score matched latency/pareto (Shanghai) and rendered Q90 + 5-frame pose bridge (Sydney)."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from fractions import Fraction
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

from experiments.tennis import k6b
from experiments.tennis.player_requalification import sha256
from experiments.tennis.tomm_v5_score import is_seam_tick, load_source_frames, mask_960, mask_full_res
from experiments.visor.h3 import dis_flow, warp
from experiments.visor.h3b import luma
from src.codecs import quality
from src.pointstream.tracked_patch import yuv_to_rgb

CARD = Path(__file__).resolve().parents[2] / "docs/experiments/test-crop-joint-20261122"
NATIVE_CARD = Path(__file__).resolve().parents[2] / "docs/experiments/test-joint-extension-20261122/native"
sys.path.insert(0, str(CARD))
from score_native import get_support, tensor  # noqa: E402

VGG_SHA256 = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"

SH_PIXELS_SHA256 = "2fd7f48841adb6ecc37e31856f2d616f50af79c988a37f63244826e5509d5b8a"
SH_FEATURES_SHA256 = "994101c8d3466080777c262d92b9f15953c5636cc4addae06f75ad478d3594ee"
SH_WITNESS_SHA256 = "0c1fc468072ad67f2922e7ea7a81abb4aec1725bb916210c15224c407f0ccbda"

SY_SOURCE_SHA256 = "8ebca8b693202d8bf7afaee515dde4d723a9fb83250115e6eff080038b0bbf50"
SY_FEATURES_SHA256 = "7c7b678cf671e22092efe5284a19cd113bca49ee125af558d844c886fc31c19d"
SY_WITNESS_SHA256 = "4db6e58e761a2b0d118ec2a639dcd9372e4768270b12680b831666c5f5e3972c"


def load_yuv_frames(base_yuv: Path, w: int, h: int, n_frames: int) -> list[np.ndarray]:
    frame_bytes = w * h * 3 // 2
    raw = base_yuv.read_bytes()
    if len(raw) < n_frames * frame_bytes:
        raise ValueError(f"unexpected base.yuv size {len(raw)} for {w}x{h}")
    frames: list[np.ndarray] = []
    for t in range(n_frames):
        chunk = np.frombuffer(
            raw[t * frame_bytes : (t + 1) * frame_bytes], dtype=np.uint8
        ).reshape(h * 3 // 2, w)
        rgb = yuv_to_rgb(chunk, w, h, False)
        frames.append(cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC))
    return frames


def load_video_frames(path: Path, n_frames: int, expected_fps: float | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if expected_fps is not None:
        got_fps = cap.get(cv2.CAP_PROP_FPS)
        if abs(got_fps - expected_fps) >= 0.05:
            cap.release()
            raise ValueError(f"unexpected FPS {got_fps} != {expected_fps}")
    frames: list[np.ndarray] = []
    for t in range(n_frames):
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"failed reading video frame {t}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def score_suite(
    pub_root: Path,
    suite_prefix: str,
    unit_prefix: str,
    arm_names: list[str],
    roles: tuple[str, ...],
    n_frames: int,
    base_frames: list[np.ndarray],
    src_frames: list[np.ndarray],
    supports: list[Any],
    dists_model: Any,
    lpips_net: Any,
    render_times: dict[tuple[str, str, str], float],
) -> dict[str, Any]:
    yy, xx = np.indices((544, 960), dtype=np.float32)
    src_960_gray: list[np.ndarray] = []
    src_960_luma: list[np.ndarray] = []
    src_flow: list[np.ndarray | None] = [None]
    src_delta: list[np.ndarray | None] = [None]
    for t in range(n_frames):
        im960 = cv2.resize(src_frames[t], (960, 544), interpolation=cv2.INTER_AREA)
        src_960_gray.append(cv2.cvtColor(im960, cv2.COLOR_RGB2GRAY))
        src_960_luma.append(luma(im960))
        if t > 0:
            flow = dis_flow(src_960_gray[t], src_960_gray[t - 1])
            src_flow.append(flow)
            src_delta.append(src_960_luma[t] - warp(src_960_luma[t - 1], flow))

    tlpips_ticks = (
        set(range(1, n_frames))
        if n_frames <= 24
        else (set(range(5, n_frames, 5)) | {16, 20, 32, 40})
    )
    role_indices = {"near": 0, "far": 1}
    out: dict[str, Any] = {}

    for arm in arm_names:
        unit_data: dict[str, dict[str, np.ndarray]] = {}
        for role in roles:
            npz = np.load(pub_root / suite_prefix / arm / f"{unit_prefix}_{role}.npz")
            unit_data[role] = {
                "box": np.asarray(npz["box"], dtype=np.float64),
                "crops": np.asarray(npz["crops"], dtype=np.uint8),
            }

        dists_rows: list[dict[str, Any]] = []
        flow_rows: list[dict[str, Any]] = []
        tlpips_rows: list[dict[str, Any]] = []
        prev_960_gray = prev_960_luma = prev_960_rgb = None

        for t in range(n_frames):
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
                with torch.inference_mode():
                    t_ob = tensor(ob)
                    t_ref = tensor(ref)
                    dval = float(quality.masked_dists(dists_model, t_ob, t_ref, act))
                    if t_ob.shape[-2] < 32 or t_ob.shape[-1] < 32:
                        sz = (max(32, t_ob.shape[-2]), max(32, t_ob.shape[-1]))
                        l_ob = torch.nn.functional.interpolate(t_ob, size=sz, mode="bilinear", align_corners=False)
                        l_ref = torch.nn.functional.interpolate(t_ref, size=sz, mode="bilinear", align_corners=False)
                        l_act = torch.nn.functional.interpolate(act.float(), size=sz, mode="nearest")
                    else:
                        l_ob, l_ref, l_act = t_ob, t_ref, act.float()
                    lp_map = lpips_net(l_ob * 2.0 - 1.0, l_ref * 2.0 - 1.0)[:, :1]
                    lpval = float((lp_map * l_act).sum() / l_act.sum().clamp_min(1.0))
                diff = ob[bm].astype(np.float32) - ref[bm].astype(np.float32)
                mse = float((diff * diff).mean())
                psnr = 20.0 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else 99.0
                dists_rows.append({"tick": t, "role": role, "dists": dval, "lpips": lpval, "psnr": psnr})

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
                    if reg_src.any():
                        src_err = float(np.abs(c_src_del[reg_src] - s_del[reg_src]).mean())
                        reg_self = a_inside & (masks_960[idx] > 0) & (warp(prev_masks_960[idx], a_flow) >= 0.5)
                        self_err = float(np.abs(a_self_del[reg_self]).mean()) if reg_self.any() else src_err
                        flow_rows.append({
                            "tick": t,
                            "role": role,
                            "seam": is_seam_tick(t),
                            "source_flow_error": src_err,
                            "self_flow_error": self_err,
                        })
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

        suite_tag = "shanghai_timing" if suite_prefix == "shanghai" else "sydney_ablation"
        near_s = render_times.get((suite_tag, arm, f"{unit_prefix}_near"))
        far_s = render_times.get((suite_tag, arm, f"{unit_prefix}_far"))
        total_s = (near_s or 0.0) + (far_s or 0.0) if (near_s and far_s) else near_s
        gap_rows = [r for r in dists_rows if 9 <= r["tick"] < 14]
        out[arm] = {
            "n_dists": len(dists_rows),
            "masked_dists": mean(r["dists"] for r in dists_rows),
            "masked_lpips": mean(r["lpips"] for r in dists_rows),
            "masked_psnr_db": mean(r["psnr"] for r in dists_rows),
            "near_dists": mean(r["dists"] for r in dists_rows if r["role"] == "near"),
            "far_dists": mean(r["dists"] for r in dists_rows if r["role"] == "far") if "far" in roles else None,
            "gap9_14_dists": mean(r["dists"] for r in gap_rows) if gap_rows else None,
            "gap9_14_near_dists": mean(r["dists"] for r in gap_rows if r["role"] == "near") if gap_rows else None,
            "gap9_14_far_dists": mean(r["dists"] for r in gap_rows if r["role"] == "far") if (gap_rows and "far" in roles) else None,
            "self_flow_error": mean(r["self_flow_error"] for r in flow_rows),
            "source_flow_error": mean(r["source_flow_error"] for r in flow_rows),
            "tlpips_mean": mean(r["tlpips"] for r in tlpips_rows),
            "tlpips_seam_mean": mean(r["tlpips"] for r in tlpips_rows if r["seam"]) if any(r["seam"] for r in tlpips_rows) else None,
            "render_seconds_near": near_s,
            "render_seconds_far": far_s,
            "render_seconds_total": total_s,
        }
    return out


def score_run(args: argparse.Namespace) -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    stage_name = os.environ["PS_STAGE"]
    stage_dir = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    expected_sh = {"smoke": 8, "full": 48}[stage_name]
    if args.frames != expected_sh:
        raise ValueError(f"frames={args.frames} != {expected_sh}")

    for path, expected, label in (
        (args.shanghai_pixels, SH_PIXELS_SHA256, "shanghai_pixels"),
        (args.shanghai_features, SH_FEATURES_SHA256, "shanghai_features"),
        (args.shanghai_witness, SH_WITNESS_SHA256, "shanghai_witness"),
        (args.sydney_source, SY_SOURCE_SHA256, "sydney_source"),
        (args.sydney_features, SY_FEATURES_SHA256, "sydney_features"),
        (args.sydney_witness, SY_WITNESS_SHA256, "sydney_witness"),
        (args.vgg, VGG_SHA256, "vgg"),
        (args.alexnet, ALEXNET_SHA256, "alexnet"),
    ):
        if sha256(path) != expected:
            raise ValueError(f"{label} sha256 mismatch")

    dists_model = quality.load_dists(args.vgg, "cpu")
    lpips_net = quality.load_lpips(args.alexnet, "cpu")
    roles: tuple[str, ...] = ("near",) if stage_name == "smoke" else ("near", "far")

    t0 = time.monotonic()
    with ExitStack() as stack:
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        gpu_dir = tmp / "gpu"
        sh_wit = tmp / "sh_wit"
        sy_wit = tmp / "sy_wit"
        for p, d in ((args.gpu_tar, gpu_dir), (args.shanghai_witness, sh_wit), (args.sydney_witness, sy_wit)):
            with tarfile.open(p) as tf:
                tf.extractall(d, filter="data")

        gpu_pub = gpu_dir / "publish"
        gpu_summary = json.loads((gpu_pub / "matched_summary.json").read_text())
        render_times: dict[tuple[str, str, str], float] = {}
        for rec in gpu_summary["records"]:
            suite_tag = rec["suite"]
            key = rec["config"]["name"] if suite_tag == "shanghai_timing" else rec["arm"]
            render_times[(suite_tag, key, rec["unit"])] = float(rec["render_seconds"])

        sh_supports, _ = get_support(args.shanghai_features)
        sh_base = load_yuv_frames(sh_wit / "publish/base.yuv", 160, 96, expected_sh)
        sh_src = load_source_frames(args.shanghai_pixels, expected_sh)
        sh_arms = [p.name for p in sorted((gpu_pub / "shanghai").iterdir()) if p.is_dir()]
        sh_res = score_suite(
            gpu_pub, "shanghai", "match102", sh_arms, roles, expected_sh,
            sh_base, sh_src, sh_supports, dists_model, lpips_net, render_times,
        )

        sy_frames = 8 if stage_name == "smoke" else 24
        sy_b = json.loads((NATIVE_CARD / "match127-binding.json").read_text())
        sy_receipt = NATIVE_CARD / "match127-crop-receipt.json"
        sy_supports, _ = get_support(
            args.sydney_features,
            receipt=sy_receipt,
            tracks=[sy_b["roles"]["near"], sy_b["roles"]["far"]],
        )
        sy_base = load_yuv_frames(sy_wit / "publish/base.yuv", 480, 272, sy_frames)
        sy_src = load_video_frames(args.sydney_source, sy_frames, float(Fraction(sy_b["fps"])))
        sy_arms = [p.name for p in sorted((gpu_pub / "sydney").iterdir()) if p.is_dir()]
        sy_res = score_suite(
            gpu_pub, "sydney", "match127", sy_arms, roles, sy_frames,
            sy_base, sy_src, sy_supports, dists_model, lpips_net, render_times,
        )

    out = {
        "stage": stage_name,
        "gpu_device": gpu_summary["device"],
        "timing_contaminated": gpu_summary["timing_contaminated"],
        "model_load_seconds": gpu_summary["model_load_seconds"],
        "shanghai_frames": expected_sh,
        "sydney_frames": sy_frames,
        "roles": list(roles),
        "gpu_tar_sha256": sha256(args.gpu_tar),
        "elapsed_s": time.monotonic() - t0,
        "shanghai_timing": sh_res,
        "sydney_ablation": sy_res,
    }
    pub = scratch / "publish"
    pub.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=2) + "\n"
    (pub / "matched_scores.json").write_text(text)
    (stage_dir / "matched_scores.json").write_text(text)


def validate(_args: argparse.Namespace) -> None:
    stage = os.environ["PS_STAGE"]
    stage_dir = Path(os.environ["PS_STAGE_DIR"])
    data = json.loads((stage_dir / "matched_scores.json").read_text())
    sh_ok = len(data["shanghai_timing"]) == 4
    sy_ok = len(data["sydney_ablation"]) >= (2 if stage == "smoke" else 4)
    all_rows = list(data["shanghai_timing"].values()) + list(data["sydney_ablation"].values())
    finite_ok = all(math.isfinite(r["masked_dists"]) and math.isfinite(r["tlpips_mean"]) for r in all_rows)
    ok = sh_ok and sy_ok and finite_ok
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps({"passed": ok, "checks": {"sh_ok": sh_ok, "sy_ok": sy_ok, "finite_ok": finite_ok}}, indent=2) + "\n"
    )
    if not ok:
        raise ValueError("matched checks score validation failed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    req = "--validate" not in sys.argv
    for k in (
        "gpu-tar",
        "shanghai-pixels",
        "shanghai-features",
        "shanghai-witness",
        "sydney-source",
        "sydney-features",
        "sydney-witness",
        "vgg",
        "alexnet",
    ):
        p.add_argument("--" + k, type=Path, required=req)
    p.add_argument("--frames", type=int, choices=(8, 48), default=48)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    if args.validate:
        validate(args)
    else:
        score_run(args)


if __name__ == "__main__":
    main()

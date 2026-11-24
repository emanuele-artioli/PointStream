"""Score V5 Shanghai Pareto & stabilization renders (DISTS, self-flow, source-flow, tLPIPS)."""

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

from experiments.tennis import k6b
from experiments.tennis.player_requalification import sha256
from experiments.visor.h3 import dis_flow, warp
from experiments.visor.h3b import luma
from src.codecs import quality
from src.pointstream.tracked_patch import yuv_to_rgb

CARD = Path(__file__).resolve().parents[2] / "docs/experiments/test-crop-joint-20261122"
sys.path.insert(0, str(CARD))
from score_native import FPS, get_support, tensor  # noqa: E402

PIXELS_SHA256 = "2fd7f48841adb6ecc37e31856f2d616f50af79c988a37f63244826e5509d5b8a"
FEATURES_SHA256 = "994101c8d3466080777c262d92b9f15953c5636cc4addae06f75ad478d3594ee"
WITNESS_SHA256 = "0c1fc468072ad67f2922e7ea7a81abb4aec1725bb916210c15224c407f0ccbda"
VGG_SHA256 = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
BASE_W, BASE_H, BASE_NT = 160, 96, 240
BASE_FRAME_BYTES = BASE_W * BASE_H * 3 // 2


def is_seam_tick(tick: int) -> bool:
    return tick >= 20 and (tick % 20) in (0, 1, 2, 3, 4)


def mask_full_res(crop: tuple[tuple[int, int, int, int], np.ndarray]) -> np.ndarray:
    (x0, y0, x1, y1), m = crop
    mask = np.zeros((1080, 1920), dtype=bool)
    mask[y0:y1, x0:x1] = m
    return mask


def mask_960(crop: tuple[tuple[int, int, int, int], np.ndarray]) -> np.ndarray:
    (x0, y0, x1, y1), m = crop
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    mask[y0:y1, x0:x1] = m.astype(np.uint8)
    return cv2.resize(mask, (960, 544), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def load_base_frames(base_yuv: Path, n_frames: int) -> list[np.ndarray]:
    raw = base_yuv.read_bytes()
    if len(raw) != BASE_NT * BASE_FRAME_BYTES:
        raise ValueError(f"unexpected base.yuv size {len(raw)}")
    frames: list[np.ndarray] = []
    for t in range(n_frames):
        chunk = np.frombuffer(raw[t * BASE_FRAME_BYTES : (t + 1) * BASE_FRAME_BYTES], dtype=np.uint8)
        rgb = yuv_to_rgb(chunk, BASE_W, BASE_H, False)
        frames.append(cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC))
    return frames


def load_source_frames(pixels_path: Path, n_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(pixels_path), cv2.CAP_FFMPEG)
    if abs(cap.get(cv2.CAP_PROP_FPS) - float(FPS)) >= 0.001:
        cap.release()
        raise ValueError("unexpected source FPS")
    frames: list[np.ndarray] = []
    for t in range(n_frames):
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"failed reading source frame {t}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def score_run(args: argparse.Namespace) -> dict[str, Any]:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    stage_name = os.environ["PS_STAGE"]
    stage_dir = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    expected_frames = {"smoke": 8, "full": 48}[stage_name]
    if args.frames != expected_frames:
        raise ValueError(f"frames={args.frames} mismatches stage {stage_name}")

    for path, expected, label in (
        (args.pixels, PIXELS_SHA256, "pixels"),
        (args.features, FEATURES_SHA256, "features"),
        (args.witness, WITNESS_SHA256, "witness"),
        (args.vgg, VGG_SHA256, "vgg"),
        (args.alexnet, ALEXNET_SHA256, "alexnet"),
    ):
        if sha256(path) != expected:
            raise ValueError(f"{label} sha256 mismatch")

    supports, support_hash = get_support(args.features)
    dists_model = quality.load_dists(args.vgg, "cpu")
    lpips_net = quality.load_lpips(args.alexnet, "cpu")

    with ExitStack() as stack:
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        v5_dir = tmp / "v5"
        wit_dir = tmp / "wit"
        with tarfile.open(args.v5_tar) as tf:
            tf.extractall(v5_dir, filter="data")
        with tarfile.open(args.witness) as tf:
            tf.extractall(wit_dir, filter="data")

        v5_pub = v5_dir / "publish"
        v5_summary = json.loads((v5_pub / "v5_summary.json").read_text())
        if int(v5_summary["frames"]) < args.frames:
            raise ValueError("v5 archive has fewer frames than requested")

        base_frames = load_base_frames(wit_dir / "publish/base.yuv", args.frames)
        src_frames = load_source_frames(args.pixels, args.frames)

        configs: list[str] = list(v5_summary["configs"])
        roles: tuple[str, ...] = ("near",) if stage_name == "smoke" else ("near", "far")
        tlpips_ticks = (
            set(range(1, args.frames))
            if args.frames <= 10
            else (set(range(5, args.frames, 5)) | {16, 20, 32, 40})
        )
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
        per_config: dict[str, Any] = {}
        t0 = time.monotonic()

        # Check DISTS identity sanity once on frame 0 near mask.
        (x0, y0, x1, y1), m0 = supports[0][0]
        tr0 = tensor(src_frames[0][y0:y1, x0:x1])
        act0 = torch.from_numpy(m0.copy())[None, None]
        with torch.inference_mode():
            self_dists = abs(float(quality.masked_dists(dists_model, tr0, tr0, act0)))
        if self_dists >= 1e-5:
            raise ValueError(f"masked DISTS self-check failed: {self_dists}")

        for cfg_name in configs:
            unit_data: dict[str, dict[str, np.ndarray]] = {}
            for role in roles:
                npz = np.load(v5_pub / f"{cfg_name}/match102_{role}.npz")
                unit_data[role] = {
                    "box": np.asarray(npz["box"], dtype=np.float64),
                    "crops": np.asarray(npz["crops"], dtype=np.uint8),
                }

            dists_rows: list[dict[str, Any]] = []
            flow_rows: list[dict[str, Any]] = []
            tlpips_rows: list[dict[str, Any]] = []
            prev_960_gray: np.ndarray | None = None
            prev_960_luma: np.ndarray | None = None
            prev_960_rgb: np.ndarray | None = None

            for t in range(args.frames):
                canvas = base_frames[t].copy()
                boxes_t = supports[t]
                masks_full = [mask_full_res(boxes_t[i]) for i in range(2)]
                masks_960 = [mask_960(boxes_t[i]) for i in range(2)]
                # Paste far before near when both exist, matching PointStream layer order.
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
                        dval = float(quality.masked_dists(dists_model, tensor(ob), tensor(ref), act))
                    if not math.isfinite(dval):
                        raise ValueError(f"non-finite DISTS at {cfg_name} t={t} {role}")
                    diff = ob[bm].astype(np.float32) - ref[bm].astype(np.float32)
                    mse = float((diff * diff).mean())
                    psnr = 20.0 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else 99.0
                    dists_rows.append({"tick": t, "role": role, "dists": dval, "psnr": psnr})

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

            seam_rows = [r for r in flow_rows if r["seam"]]
            int_rows = [r for r in flow_rows if not r["seam"]]
            t_seam = [r for r in tlpips_rows if r["seam"]]
            t_int = [r for r in tlpips_rows if not r["seam"]]
            timing_runs = [r for r in v5_summary["runs"] if r["config"] == cfg_name]
            per_config[cfg_name] = {
                "n_dists": len(dists_rows),
                "masked_dists": mean(r["dists"] for r in dists_rows),
                "masked_psnr_db": mean(r["psnr"] for r in dists_rows),
                "near_dists": mean(r["dists"] for r in dists_rows if r["role"] == "near"),
                "far_dists": mean(r["dists"] for r in dists_rows if r["role"] == "far") if "far" in roles else None,
                "n_flow": len(flow_rows),
                "source_flow_error": mean(r["source_flow_error"] for r in flow_rows),
                "self_flow_error": mean(r["self_flow_error"] for r in flow_rows),
                "seam_self_flow_error": mean(r["self_flow_error"] for r in seam_rows) if seam_rows else None,
                "interior_self_flow_error": mean(r["self_flow_error"] for r in int_rows) if int_rows else None,
                "tlpips_n": len(tlpips_rows),
                "tlpips_mean": mean(r["tlpips"] for r in tlpips_rows),
                "tlpips_seam_mean": mean(r["tlpips"] for r in t_seam) if t_seam else None,
                "tlpips_interior_mean": mean(r["tlpips"] for r in t_int) if t_int else None,
                "v5_render_s_total": sum(float(r["render_s"]) for r in timing_runs),
                "v5_render_runs": len(timing_runs),
            }

        base = per_config.get("ddim30_512_ov4")
        if base is not None:
            for c in per_config.values():
                c["delta_dists_vs_base"] = c["masked_dists"] - base["masked_dists"]
                c["delta_self_flow_vs_base"] = c["self_flow_error"] - base["self_flow_error"]
                c["delta_tlpips_vs_base"] = c["tlpips_mean"] - base["tlpips_mean"]
                c["speedup_vs_base"] = (
                    base["v5_render_s_total"] / c["v5_render_s_total"]
                    if c["v5_render_s_total"] > 0
                    else None
                )

    result = {
        "stage": stage_name,
        "frames": args.frames,
        "roles": list(roles),
        "v5_tar_sha256": sha256(args.v5_tar),
        "source_mask_rle_sha256": support_hash,
        "self_dists_check": self_dists,
        "elapsed_s": time.monotonic() - t0,
        "configs": per_config,
    }
    pub = scratch / "publish"
    pub.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2) + "\n"
    (pub / "v5_scores.json").write_text(text)
    (stage_dir / "v5_scores.json").write_text(text)
    return result


def validate(args: argparse.Namespace) -> None:
    stage = os.environ["PS_STAGE"]
    stage_dir = Path(os.environ["PS_STAGE_DIR"])
    data = json.loads((stage_dir / "v5_scores.json").read_text())
    cfgs = data["configs"]
    checks = {
        "has_configs": len(cfgs) >= (4 if stage == "smoke" else 9),
        "has_baseline": "ddim30_512_ov4" in cfgs,
        "self_dists_ok": float(data["self_dists_check"]) < 1e-5,
        "finite_scores": all(
            math.isfinite(c["masked_dists"])
            and math.isfinite(c["self_flow_error"])
            and math.isfinite(c["tlpips_mean"])
            for c in cfgs.values()
        ),
        "forecast_ok": stage != "smoke" or data["elapsed_s"] * 2 * (48 / 8) * (9 / 5) * 1.3 + 60 < 2000,
    }
    ok = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps({"passed": ok, "checks": checks}, indent=2) + "\n"
    )
    if not ok:
        raise ValueError(f"v5 score validation failed: {checks}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("v5-tar", "pixels", "features", "witness", "vgg", "alexnet"):
        p.add_argument("--" + k, type=Path, required="--validate" not in sys.argv)
    p.add_argument("--frames", type=int, default=48)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    if args.validate:
        validate(args)
    else:
        score_run(args)


if __name__ == "__main__":
    main()

"""TOMM multi-backbone Synthesize(reference, poses, seed) ablation on Shanghai match102."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time
from typing import Any

import cv2
import numpy as np

from experiments.tennis import common_player_packet as packet
from experiments.tennis import k6b
from experiments.tennis.tomm_v5_pareto_pilot import (
    CARRIER_SHA256,
    FIT_SHA256,
    PilotConfig,
    build_pipeline as build_aa_pipeline,
    render_config as render_aa_config,
    sha256,
    smooth_keypoints,
)

AA_CFG = PilotConfig("aa_dpm10_ov8_smooth", "dpm", 10, 512, 784, 8, True, 0.0)
DDIM30_48F_SECONDS = 674.4665817180066
WARP_MAP_STEP = 4
WARP_SIGMA_OF_HEIGHT = 0.08
CN_PROMPT = "tennis player on court, photorealistic, broadcast sports video"
CN_NEG_PROMPT = "blurry, distorted, low quality, cartoon, extra limbs"


def _symlink_dir(mapping: dict[str, Path], root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, src in mapping.items():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    return root


def build_controlnet_pipeline(
    cn_files: dict[str, Path],
    sd_files: dict[str, Path],
    scratch: Path,
    device: str,
) -> Any:
    import torch
    from diffusers import (
        ControlNetModel,
        DPMSolverMultistepScheduler,
        StableDiffusionControlNetImg2ImgPipeline,
    )

    cn_dir = _symlink_dir(cn_files, scratch / "cn_openpose")
    sd_dir = _symlink_dir(sd_files, scratch / "sd15")
    controlnet = ControlNetModel.from_pretrained(
        str(cn_dir),
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        str(sd_dir),
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, solver_order=2)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    return pipe


def render_controlnet(
    pipe: Any,
    reference: np.ndarray,
    poses: np.ndarray,
    seed: int,
    steps: int = 10,
    strength: float = 0.65,
    guidance_scale: float = 7.5,
) -> tuple[np.ndarray, float]:
    import torch
    from PIL import Image

    ref_img = Image.fromarray(reference)
    n_frames = len(poses)
    out = np.empty((n_frames, 784, 512, 3), dtype=np.uint8)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for idx in range(n_frames):
            pose_img = Image.fromarray(poses[idx])
            gen = torch.Generator(device="cuda").manual_seed(seed + idx)
            res = pipe(
                prompt=CN_PROMPT,
                negative_prompt=CN_NEG_PROMPT,
                image=ref_img,
                control_image=pose_img,
                num_inference_steps=steps,
                strength=strength,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=1.0,
                generator=gen,
                width=512,
                height=784,
            ).images[0]
            out[idx] = np.asarray(res, dtype=np.uint8)
    torch.cuda.synchronize()
    render_s = time.perf_counter() - t0
    return out, render_s


def _warp_crop_gaussian(
    ref_crop: np.ndarray,
    src_kp: np.ndarray,
    dst_kp: np.ndarray,
    conf_thr: float = 0.5,
) -> np.ndarray:
    """Non-generative K6-style Gaussian-weighted keypoint warp in crop space."""
    h, w = ref_crop.shape[:2]
    valid = (src_kp[:, 2] >= conf_thr) & (dst_kp[:, 2] >= conf_thr)
    if int(valid.sum()) < 4:
        return ref_crop.copy()

    p_src = src_kp[valid, :2].astype(np.float32)
    p_dst = dst_kp[valid, :2].astype(np.float32)
    disp = p_src - p_dst  # inverse map: for each dst pixel, sample from dst + disp

    sigma = max(12.0, WARP_SIGMA_OF_HEIGHT * float(h))
    inv_2sig2 = 1.0 / (2.0 * sigma * sigma)

    gy = np.arange(0, h, WARP_MAP_STEP, dtype=np.float32)
    gx = np.arange(0, w, WARP_MAP_STEP, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(gx, gy)  # (gh, gw)

    dx = grid_x[..., None] - p_dst[None, None, :, 0]
    dy = grid_y[..., None] - p_dst[None, None, :, 1]
    weights = np.exp(-(dx * dx + dy * dy) * inv_2sig2)
    w_sum = weights.sum(axis=-1, keepdims=True) + 1e-6
    flow_x = (weights * disp[None, None, :, 0]).sum(axis=-1) / w_sum[..., 0]
    flow_y = (weights * disp[None, None, :, 1]).sum(axis=-1) / w_sum[..., 0]

    map_x_small = (grid_x + flow_x).astype(np.float32)
    map_y_small = (grid_y + flow_y).astype(np.float32)
    map_x = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.remap(
        ref_crop,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def render_k6_ref_warp(
    ref_crop: np.ndarray,
    smooth_kp: np.ndarray,
    boxes: np.ndarray,
) -> tuple[np.ndarray, float]:
    n_frames = len(smooth_kp)
    crop_kps = [k6b.to_crop(smooth_kp[t], boxes[t]) for t in range(n_frames)]
    ref_kp = crop_kps[0]
    out = np.empty((n_frames, ref_crop.shape[0], ref_crop.shape[1], 3), dtype=np.uint8)
    t0 = time.perf_counter()
    for t in range(n_frames):
        out[t] = _warp_crop_gaussian(ref_crop, ref_kp, crop_kps[t])
    render_s = time.perf_counter() - t0
    return out, render_s


def run_ablation(args: argparse.Namespace) -> None:
    aa = Path(sys.prefix).resolve() / "opt/Moore-AnimateAnyone"
    if str(aa) not in sys.path:
        sys.path.insert(1, str(aa))
    import torch

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    stage = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    stage_name = os.environ["PS_STAGE"]
    expected = {"smoke": 16, "full": 48}[stage_name]
    if args.frames != expected:
        raise ValueError(f"frames={args.frames} mismatches stage {stage_name} ({expected})")
    if sha256(args.carrier) != CARRIER_SHA256:
        raise ValueError("carrier sha256 mismatch")

    fit_root = scratch / "base-fit"
    fit_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.base) as tf:
        tf.extractall(fit_root, filter="data")
    fit_path = fit_root / "publish/tennis-base@fit-100.pt"
    if sha256(fit_path) != FIT_SHA256:
        raise ValueError("fit sha256 mismatch")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    limit_mib = int(os.environ.get("PS_GPU_MEMORY_LIMIT_MIB", "20000"))
    total_mib = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit_mib * 2**20 / total_mib))

    aa_weights = dict(
        zip(
            (
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
                "image_encoder/config.json",
                "image_encoder/pytorch_model.bin",
                "original/denoising_unet.pth",
                "original/reference_unet.pth",
                "original/pose_guider.pth",
                "original/motion_module.pth",
            ),
            map(Path, args.aa_weight),
            strict=True,
        )
    )
    cn_files = dict(
        zip(
            ("config.json", "diffusion_pytorch_model.fp16.safetensors"),
            map(Path, args.cn_weight),
            strict=True,
        )
    )
    sd_files = dict(
        zip(
            (
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.fp16.safetensors",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.fp16.safetensors",
            ),
            map(Path, args.sd_weight),
            strict=True,
        )
    )

    _, _, decoded = packet.unpack(args.carrier.read_bytes())
    requested_units = [u.strip() for u in args.units.split(",") if u.strip()]
    units = [requested_units[0]] if stage_name == "smoke" else requested_units

    unit_inputs: dict[str, dict[str, Any]] = {}
    for unit in units:
        run_data = decoded[unit][0]
        raw_kp = np.asarray(run_data["kp"][: args.frames], dtype=np.float32)
        smooth_kp = smooth_keypoints(raw_kp)
        boxes = np.asarray(run_data["boxes"][: args.frames], dtype=np.float64)
        ref_crop = np.asarray(run_data["reference"], dtype=np.uint8)
        seed = int(run_data["seed"])
        ticks = np.asarray(run_data["t"][: args.frames], dtype=np.int32)
        smooth_poses = np.stack([k6b.pose_map(k6b.to_crop(smooth_kp[t], boxes[t])) for t in range(args.frames)])
        unit_inputs[unit] = {
            "smooth_kp": smooth_kp,
            "boxes": boxes,
            "ref_crop": ref_crop,
            "seed": seed,
            "ticks": ticks,
            "smooth_poses": smooth_poses,
        }

    pub = scratch / "publish"
    pub.mkdir(parents=True, exist_ok=True)
    run_records: list[dict[str, Any]] = []
    determinism: dict[str, int] = {}
    load_times: dict[str, float] = {}

    def save_crops(backbone: str, unit: str, crops: np.ndarray, render_s: float) -> None:
        u = unit_inputs[unit]
        distinct = len({hashlib.sha256(c.tobytes()).hexdigest() for c in crops})
        out_dir = pub / backbone
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / f"{unit}.npz"
        np.savez_compressed(npz_path, t=u["ticks"], box=u["boxes"], crops=crops)
        run_records.append(
            {
                "backbone": backbone,
                "unit": unit,
                "frames": args.frames,
                "seed": u["seed"],
                "render_seconds": render_s,
                "sec_per_frame": render_s / args.frames,
                "distinct_frames": distinct,
                "npz_relpath": f"{backbone}/{unit}.npz",
                "npz_sha256": sha256(npz_path),
            }
        )

    # 1. Non-generative K6 ref pose-warping control
    load_times["k6_ref_warp"] = 0.0
    if stage_name == "smoke":
        u0 = unit_inputs[units[0]]
        a, _ = render_k6_ref_warp(u0["ref_crop"], u0["smooth_kp"][:2], u0["boxes"][:2])
        b, _ = render_k6_ref_warp(u0["ref_crop"], u0["smooth_kp"][:2], u0["boxes"][:2])
        determinism["k6_ref_warp"] = int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))
    for unit in units:
        u = unit_inputs[unit]
        crops, render_s = render_k6_ref_warp(u["ref_crop"], u["smooth_kp"], u["boxes"])
        save_crops("k6_ref_warp", unit, crops, render_s)

    # 2. Moore-AnimateAnyone (dpm10_512_ov8_smooth)
    t0 = time.perf_counter()
    aa_pipe, fit_info = build_aa_pipeline(aa_weights, fit_path, scratch, "cuda")
    load_times["aa_dpm10_ov8_smooth"] = time.perf_counter() - t0
    if stage_name == "smoke":
        u0 = unit_inputs[units[0]]
        a, _ = render_aa_config(aa_pipe, AA_CFG, u0["ref_crop"], u0["smooth_poses"][:2], u0["seed"])
        b, _ = render_aa_config(aa_pipe, AA_CFG, u0["ref_crop"], u0["smooth_poses"][:2], u0["seed"])
        determinism["aa_dpm10_ov8_smooth"] = int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))
    for unit in units:
        u = unit_inputs[unit]
        crops, render_s = render_aa_config(aa_pipe, AA_CFG, u["ref_crop"], u["smooth_poses"], u["seed"])
        save_crops("aa_dpm10_ov8_smooth", unit, crops, render_s)
    del aa_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # 3. ControlNet-v1.1-OpenPose + SD 1.5 (2D frame-by-frame img2img, DPM-10)
    t0 = time.perf_counter()
    cn_pipe = build_controlnet_pipeline(cn_files, sd_files, scratch, "cuda")
    load_times["cn_openpose_sd15_dpm10"] = time.perf_counter() - t0
    if stage_name == "smoke":
        u0 = unit_inputs[units[0]]
        a, _ = render_controlnet(cn_pipe, u0["ref_crop"], u0["smooth_poses"][:2], u0["seed"])
        b, _ = render_controlnet(cn_pipe, u0["ref_crop"], u0["smooth_poses"][:2], u0["seed"])
        determinism["cn_openpose_sd15_dpm10"] = int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))
    for unit in units:
        u = unit_inputs[unit]
        crops, render_s = render_controlnet(cn_pipe, u["ref_crop"], u["smooth_poses"], u["seed"])
        save_crops("cn_openpose_sd15_dpm10", unit, crops, render_s)
    del cn_pipe
    gc.collect()
    torch.cuda.empty_cache()

    dispatch = json.loads((stage / "dispatch.json").read_text()) if (stage / "dispatch.json").exists() else {}
    summary = {
        "event": "match102",
        "stage": stage_name,
        "frames": args.frames,
        "units": units,
        "device": torch.cuda.get_device_name(0),
        "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
        "timing_contaminated": dispatch.get("gpu_sharing", {}).get("mode") == "shared",
        "model_load_seconds": load_times,
        "determinism_max_abs_diff": determinism,
        "fit_info": fit_info,
        "carrier_sha256": CARRIER_SHA256,
        "ddim30_48f_seconds_ref": DDIM30_48F_SECONDS,
        "records": run_records,
    }
    text = json.dumps(summary, indent=2) + "\n"
    (pub / "multi_backbone_summary.json").write_text(text)
    (stage / "multi_backbone_summary.json").write_text(text)

    archive = stage / "published.tar"
    tmp = stage / ".mb-published.tar"
    with tarfile.open(tmp, "x") as tf:
        tf.add(pub, arcname="publish")
    tmp.rename(archive)


def validate(_args: argparse.Namespace) -> None:
    stage = Path(os.environ["PS_STAGE_DIR"])
    stage_name = os.environ["PS_STAGE"]
    expected = {"smoke": 16, "full": 48}[stage_name]
    data = json.loads((stage / "multi_backbone_summary.json").read_text())
    assert data["frames"] == expected
    records = data["records"]
    backbones = {r["backbone"] for r in records}
    assert backbones == {"k6_ref_warp", "aa_dpm10_ov8_smooth", "cn_openpose_sd15_dpm10"}
    for r in records:
        assert r["frames"] == expected
        assert r["distinct_frames"] >= expected - 1
        assert math.isfinite(r["render_seconds"]) and r["render_seconds"] > 0
    if stage_name == "smoke":
        for name, diff in data["determinism_max_abs_diff"].items():
            assert diff <= 1, f"determinism failed for {name}: max_abs_diff={diff}"
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(
            {
                "passed": True,
                "checks": {
                    "frames": expected,
                    "backbones": sorted(backbones),
                    "records": len(records),
                    "cuda_device": data["device"],
                    "gpu_peak_mib": data["gpu_peak_mib"],
                    "determinism_max_abs_diff": data["determinism_max_abs_diff"],
                    "archive_sha256": sha256(stage / "published.tar"),
                },
                "timing_contaminated": data["timing_contaminated"],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--carrier", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--aa-weight", type=Path, action="append", required=True)
    p.add_argument("--cn-weight", type=Path, action="append", required=True)
    p.add_argument("--sd-weight", type=Path, action="append", required=True)
    p.add_argument("--frames", type=int, choices=(16, 48), required=True)
    p.add_argument("--units", default="match102_near,match102_far")
    p.add_argument("--validate", action="store_true")
    if sys.argv[1:] == ["--validate"]:
        stage = Path(os.environ["PS_STAGE_DIR"])
        argv = json.loads((stage / "dispatch.json").read_text())["command"]
        validate(p.parse_args(["--validate", *argv[3:]]))
    else:
        args = p.parse_args()
        if args.validate:
            validate(args)
        else:
            run_ablation(args)


if __name__ == "__main__":
    main()

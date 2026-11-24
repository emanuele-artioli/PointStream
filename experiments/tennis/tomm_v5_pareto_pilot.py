"""V5 48-frame Shanghai GPU pilot: speed-quality Pareto sweep + zero-retraining temporal stabilization."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time
from typing import Any

import numpy as np

from experiments.tennis import common_player_packet as packet
from experiments.tennis import k6b
from experiments.tennis import k6b_generate as gen_mod

CARRIER_SHA256 = "3ebc2e455509db026919f17c17d2bfce3eba899fd29a0888c47440b576a219af"
FIT_SHA256 = "075b001a3ae9dcc9fdda9fb81e4240aaf3a9da228d707ac58733db094d983285"


@dataclass(frozen=True)
class PilotConfig:
    name: str
    scheduler: str
    steps: int
    width: int
    height: int
    overlap: int
    pose_smooth: bool
    corr_noise_alpha: float


CONFIGS: tuple[PilotConfig, ...] = (
    PilotConfig("ddim30_512_ov4", "ddim", 30, 512, 784, 4, False, 0.0),
    PilotConfig("dpm15_512_ov4", "dpm", 15, 512, 784, 4, False, 0.0),
    PilotConfig("dpm10_512_ov4", "dpm", 10, 512, 784, 4, False, 0.0),
    PilotConfig("dpm6_512_ov4", "dpm", 6, 512, 784, 4, False, 0.0),
    PilotConfig("unipc10_512_ov4", "unipc", 10, 512, 784, 4, False, 0.0),
    PilotConfig("dpm10_384_ov4", "dpm", 10, 384, 576, 4, False, 0.0),
    PilotConfig("dpm10_256_ov4", "dpm", 10, 256, 384, 4, False, 0.0),
    PilotConfig("dpm10_512_ov8_smooth", "dpm", 10, 512, 784, 8, True, 0.0),
    PilotConfig("dpm10_512_ov8_smooth_corr05", "dpm", 10, 512, 784, 8, True, 0.5),
)

SMOKE_CONFIGS = {
    "ddim30_512_ov4",
    "dpm10_512_ov4",
    "unipc10_512_ov4",
    "dpm10_256_ov4",
    "dpm10_512_ov8_smooth_corr05",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 2**20):
            digest.update(block)
    return digest.hexdigest()


def smooth_keypoints(kp: np.ndarray, conf_thr: float = 0.5) -> np.ndarray:
    """Zero-phase temporal smoothing on confident (x, y) coordinates before skeleton rendering."""
    out = kp.copy()
    n = len(out)
    if n < 3:
        return out
    savgol5 = np.array([-3.0, 12.0, 17.0, 12.0, -3.0], dtype=np.float64) / 35.0
    tri3 = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    for j in range(17):
        conf = kp[:, j, 2] >= conf_thr
        for t in range(n):
            if not conf[t]:
                continue
            if t >= 2 and t + 2 < n and bool(conf[t - 2 : t + 3].all()):
                out[t, j, :2] = (savgol5[:, None] * kp[t - 2 : t + 3, j, :2]).sum(axis=0)
            elif t >= 1 and t + 1 < n and bool(conf[t - 1 : t + 2].all()):
                out[t, j, :2] = (tri3[:, None] * kp[t - 1 : t + 2, j, :2]).sum(axis=0)
    return out


def make_scheduler(kind: str) -> Any:
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler

    ddim = DDIMScheduler(**gen_mod.SCHEDULER)
    if kind == "ddim":
        return ddim
    if kind == "dpm":
        return DPMSolverMultistepScheduler.from_config(ddim.config, solver_order=2)
    if kind == "unipc":
        return UniPCMultistepScheduler.from_config(ddim.config, solver_order=2)
    raise ValueError(f"unknown scheduler {kind}")


def build_pipeline(weights: dict[str, Path], fit_path: Path, scratch: Path, device: str) -> tuple[Any, dict[str, Any]]:
    import torch
    from animate_anyone.models.pose_guider import PoseGuider
    from animate_anyone.models.unet_2d_condition import UNet2DConditionModel
    from animate_anyone.models.unet_3d import UNet3DConditionModel
    from animate_anyone.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPVisionModelWithProjection
    from experiments.tennis.k6b_fit import apply_fit

    dtype = torch.float16
    vae, _ = AutoencoderKL.from_pretrained(str(gen_mod.folder(weights, "vae", scratch)), output_loading_info=True)
    encoder, _ = CLIPVisionModelWithProjection.from_pretrained(
        str(gen_mod.folder(weights, "image_encoder", scratch)), output_loading_info=True
    )
    reference = UNet2DConditionModel.from_config({**gen_mod.SD15_UNET, "_class_name": "UNet2DConditionModel"})
    gen_mod.load_checked(
        reference,
        torch.load(weights["original/reference_unet.pth"], map_location="cpu", weights_only=True),
        "reference_unet",
        gen_mod.UNUSED_REFERENCE_HEAD,
    )
    config3d = {
        **gen_mod.SD15_UNET,
        "_class_name": "UNet3DConditionModel",
        "down_block_types": ["CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"],
        "up_block_types": ["UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"],
        "mid_block_type": "UNetMidBlock3DCrossAttn",
    }
    denoising = UNet3DConditionModel.from_config(config3d, **gen_mod.UNET_ADDITIONAL)
    state = torch.load(weights["original/denoising_unet.pth"], map_location="cpu", weights_only=True)
    motion = torch.load(weights["original/motion_module.pth"], map_location="cpu", weights_only=True)
    gen_mod.load_checked(denoising, {**state, **motion}, "denoising_unet+motion_module")
    guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256))
    gen_mod.load_checked(guider, torch.load(weights["original/pose_guider.pth"], map_location="cpu", weights_only=True), "pose_guider")
    del state, motion

    fit_info = apply_fit(reference, denoising, guider, fit_path, None, 0)
    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=encoder,
        reference_unet=reference,
        denoising_unet=denoising,
        pose_guider=guider,
        scheduler=DDIMScheduler(**gen_mod.SCHEDULER),
    ).to(device, dtype=dtype)
    return pipe, fit_info


def install_correlated_latents(pipe: Any, corr_alpha: float) -> None:
    """Override prepare_latents so initial noise mixes a shared frame noise with independent noise."""
    from diffusers.utils.torch_utils import randn_tensor

    if not hasattr(pipe, "_orig_prepare_latents"):
        pipe._orig_prepare_latents = pipe.prepare_latents

    if corr_alpha <= 0.0:
        pipe.prepare_latents = pipe._orig_prepare_latents
        return

    def wrapped_prepare_latents(
        batch_size: int,
        num_channels_latents: int,
        width: int,
        height: int,
        video_length: int,
        dtype: Any,
        device: Any,
        generator: Any,
        latents: Any = None,
    ) -> Any:
        if latents is not None:
            return latents.to(device) * pipe.scheduler.init_noise_sigma
        h_lat = height // pipe.vae_scale_factor
        w_lat = width // pipe.vae_scale_factor
        shared_shape = (batch_size, num_channels_latents, 1, h_lat, w_lat)
        full_shape = (batch_size, num_channels_latents, video_length, h_lat, w_lat)
        z_shared = randn_tensor(shared_shape, generator=generator, device=device, dtype=dtype)
        z_indep = randn_tensor(full_shape, generator=generator, device=device, dtype=dtype)
        z = math.sqrt(corr_alpha) * z_shared + math.sqrt(1.0 - corr_alpha) * z_indep
        return z * pipe.scheduler.init_noise_sigma

    pipe.prepare_latents = wrapped_prepare_latents


def render_config(
    pipe: Any,
    cfg: PilotConfig,
    reference: np.ndarray,
    poses: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float]:
    import torch
    from PIL import Image

    pipe.scheduler = make_scheduler(cfg.scheduler)
    install_correlated_latents(pipe, cfg.corr_noise_alpha)

    ref_img = Image.fromarray(reference)
    pose_imgs = [Image.fromarray(p) for p in poses]
    if (cfg.width, cfg.height) != (512, 784):
        ref_img = ref_img.resize((cfg.width, cfg.height), Image.Resampling.BICUBIC)
        pose_imgs = [im.resize((cfg.width, cfg.height), Image.Resampling.BICUBIC) for im in pose_imgs]

    n_frames = len(pose_imgs)
    ctx_frames = min(24, n_frames)
    ctx_overlap = min(cfg.overlap, max(0, ctx_frames // 2))
    generator = torch.Generator(device="cpu").manual_seed(seed)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        video = pipe(
            ref_img,
            pose_imgs,
            cfg.width,
            cfg.height,
            n_frames,
            cfg.steps,
            gen_mod.CFG,
            generator=generator,
            context_frames=ctx_frames,
            context_overlap=ctx_overlap,
        ).videos
    torch.cuda.synchronize()
    render_s = time.perf_counter() - t0

    frames = (video[0].permute(1, 2, 3, 0).float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    if (cfg.width, cfg.height) != (512, 784):
        resized = np.empty((n_frames, 784, 512, 3), dtype=np.uint8)
        for idx in range(n_frames):
            resized[idx] = np.asarray(
                Image.fromarray(frames[idx]).resize((512, 784), Image.Resampling.BICUBIC), dtype=np.uint8
            )
        frames = resized
    return frames, render_s


def run_pilot(args: argparse.Namespace) -> None:
    aa = Path(sys.prefix).resolve() / "opt/Moore-AnimateAnyone"
    if str(aa) not in sys.path:
        sys.path.insert(1, str(aa))
    import torch

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    stage = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    stage_name = os.environ["PS_STAGE"]
    expected = {"smoke": 8, "full": 48}[stage_name]
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
    limit_mib = int(os.environ.get("PS_GPU_MEMORY_LIMIT_MIB", "16000"))
    total_mib = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit_mib * 2**20 / total_mib))

    weights = dict(zip(
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
        map(Path, args.weight),
        strict=True,
    ))

    _, _, decoded = packet.unpack(args.carrier.read_bytes())
    units = ["match102_near"] if stage_name == "smoke" else ["match102_near", "match102_far"]
    selected_cfgs = [c for c in CONFIGS if stage_name != "smoke" or c.name in SMOKE_CONFIGS]
    if args.configs:
        wanted = set(args.configs.split(","))
        selected_cfgs = [c for c in selected_cfgs if c.name in wanted]

    load_t0 = time.perf_counter()
    pipe, fit_info = build_pipeline(weights, fit_path, scratch, "cuda")
    load_s = time.perf_counter() - load_t0

    pub = scratch / "publish"
    pub.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    for unit in units:
        run_data = decoded[unit][0]
        raw_kp = np.asarray(run_data["kp"][: args.frames], dtype=np.float32)
        smooth_kp = smooth_keypoints(raw_kp)
        boxes = np.asarray(run_data["boxes"][: args.frames], dtype=np.float64)
        ref_crop = np.asarray(run_data["reference"], dtype=np.uint8)
        seed = int(run_data["seed"])
        ticks = np.asarray(run_data["t"][: args.frames], dtype=np.int32)

        raw_poses = np.stack([k6b.pose_map(k6b.to_crop(raw_kp[t], boxes[t])) for t in range(args.frames)])
        smooth_poses = np.stack([k6b.pose_map(k6b.to_crop(smooth_kp[t], boxes[t])) for t in range(args.frames)])

        for cfg in selected_cfgs:
            poses_in = smooth_poses if cfg.pose_smooth else raw_poses
            crops, render_s = render_config(pipe, cfg, ref_crop, poses_in, seed)
            distinct = len({hashlib.sha256(c.tobytes()).hexdigest() for c in crops})
            out_dir = pub / cfg.name
            out_dir.mkdir(parents=True, exist_ok=True)
            npz_path = out_dir / f"{unit}.npz"
            np.savez_compressed(npz_path, t=ticks, box=boxes, crops=crops)
            run_records.append(
                {
                    "config": asdict(cfg),
                    "unit": unit,
                    "frames": args.frames,
                    "seed": seed,
                    "render_seconds": render_s,
                    "sec_per_frame": render_s / args.frames,
                    "distinct_frames": distinct,
                    "npz_relpath": f"{cfg.name}/{unit}.npz",
                    "npz_sha256": sha256(npz_path),
                }
            )

    dispatch = json.loads((stage / "dispatch.json").read_text()) if (stage / "dispatch.json").exists() else {}
    summary = {
        "event": "match102",
        "mode": args.mode,
        "stage": stage_name,
        "frames": args.frames,
        "units": units,
        "device": torch.cuda.get_device_name(0),
        "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
        "timing_contaminated": dispatch.get("gpu_sharing", {}).get("mode") == "shared",
        "model_load_seconds": load_s,
        "fit_info": fit_info,
        "carrier_sha256": CARRIER_SHA256,
        "records": run_records,
    }
    (pub / "v5_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (stage / "v5_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    archive = stage / "published.tar"
    tmp = stage / ".v5-published.tar"
    with tarfile.open(tmp, "x") as tf:
        tf.add(pub, arcname="publish")
    tmp.rename(archive)


def validate(_args: argparse.Namespace) -> None:
    stage = Path(os.environ["PS_STAGE_DIR"])
    stage_name = os.environ["PS_STAGE"]
    expected = {"smoke": 8, "full": 48}[stage_name]
    data = json.loads((stage / "v5_summary.json").read_text())
    assert data["frames"] == expected
    records = data["records"]
    assert records
    for r in records:
        assert r["frames"] == expected and r["distinct_frames"] >= expected - 1
        assert math.isfinite(r["render_seconds"]) and r["render_seconds"] > 0
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(
            {
                "passed": True,
                "checks": {
                    "frames": expected,
                    "configs_run": len({r["config"]["name"] for r in records}),
                    "records": len(records),
                    "cuda_device": data["device"],
                    "gpu_peak_mib": data["gpu_peak_mib"],
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
    p.add_argument("--weight", type=Path, action="append", required=True)
    p.add_argument("--frames", type=int, choices=(8, 48), required=True)
    p.add_argument("--configs", default="")
    p.add_argument("--mode", choices=("pareto", "timing"), default="pareto")
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
            run_pilot(args)


if __name__ == "__main__":
    main()

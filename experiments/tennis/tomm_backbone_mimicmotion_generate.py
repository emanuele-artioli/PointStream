"""TOMM Track B (MimicMotion) Synthesize(reference, poses, seed) ablation on Shanghai match102."""

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
    sha256,
    smooth_keypoints,
)

DDIM30_48F_SECONDS = 674.4665817180066
BACKBONE_NAME = "mimicmotion_svdx"
INFER_W = 512
INFER_H = 768  # Exact multiple of 64 for 3-level SVD-XT UNet (nearest to 512x784)
OUT_W = 512
OUT_H = 784


def build_mimicmotion_pipeline(
    mm_tar: Path,
    scratch: Path,
    device: str,
) -> Any:
    import torch
    from diffusers.models import AutoencoderKLTemporalDecoder
    from diffusers.schedulers import EulerDiscreteScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    root = scratch / "mimicmotion"
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(mm_tar) as tf:
        tf.extractall(root, filter="data")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mimicmotion.modules.pose_net import PoseNet
    from mimicmotion.modules.unet import UNetSpatioTemporalConditionModel
    from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
    from mimicmotion.utils.geglu_patch import patch_geglu_inplace

    patch_geglu_inplace()

    svd_dir = root / "stable-video-diffusion-img2vid-xt-1-1"
    ckpt_path = root / "MimicMotion_1-1.pth"

    unet_config = UNetSpatioTemporalConditionModel.load_config(
        str(svd_dir),
        subfolder="unet",
        local_files_only=True,
    )
    unet = UNetSpatioTemporalConditionModel.from_config(unet_config)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        str(svd_dir),
        subfolder="vae",
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        str(svd_dir),
        subfolder="image_encoder",
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    scheduler = EulerDiscreteScheduler.from_pretrained(
        str(svd_dir),
        subfolder="scheduler",
        local_files_only=True,
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        str(svd_dir),
        subfolder="feature_extractor",
        local_files_only=True,
    )
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])

    # Load MimicMotion_1-1.pth (contains fine-tuned unet.* and pose_net.* state dicts)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    unet_sd = {k[len("unet.") :]: v for k, v in checkpoint.items() if k.startswith("unet.")}
    pose_sd = {k[len("pose_net.") :]: v for k, v in checkpoint.items() if k.startswith("pose_net.")}
    if not unet_sd or not pose_sd:
        raise RuntimeError(
            f"Unexpected MimicMotion checkpoint keys: unet={len(unet_sd)}, pose_net={len(pose_sd)}"
        )
    unet_missing, unet_unexpected = unet.load_state_dict(unet_sd, strict=True)
    pose_missing, pose_unexpected = pose_net.load_state_dict(pose_sd, strict=True)
    if unet_missing or unet_unexpected or pose_missing or pose_unexpected:
        raise RuntimeError(
            f"State dict mismatch: unet=({len(unet_missing)},{len(unet_unexpected)}), "
            f"pose=({len(pose_missing)},{len(pose_unexpected)})"
        )
    del checkpoint, unet_sd, pose_sd
    gc.collect()

    unet = unet.to(dtype=torch.float16).eval()
    pose_net = pose_net.to(dtype=torch.float16).eval()
    vae = vae.eval()
    image_encoder = image_encoder.eval()

    pipe = MimicMotionPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        pose_net=pose_net,
    )
    pipe.set_progress_bar_config(disable=True)
    _ = device
    return pipe


def _prepare_pose_tensor(poses_rgb: np.ndarray) -> Any:
    """Convert (N, 784, 512, 3) k6b.pose_map RGB frames to MimicMotion (N+1, 3, 768, 512) fp16 tensor in [-1, 1].

    MimicMotion's dwpose/util.py draws openpose colors onto a BGR canvas and converts BGR->RGB,
    so reversing the RGB channels of k6b.pose_map reproduces MimicMotion's exact PoseNet color convention.
    Frame 0 is prepended as the reference pose and dropped after synthesis (matching MimicMotion inference.py).
    """
    import torch

    n_frames = len(poses_rgb)
    resized = np.empty((n_frames + 1, 3, INFER_H, INFER_W), dtype=np.float32)
    for idx in range(n_frames):
        # Convert to MimicMotion's BGR2RGB channel convention and resize to (512, 768)
        pose_mm = cv2.resize(poses_rgb[idx][:, :, ::-1], (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
        resized[idx + 1] = pose_mm.transpose(2, 0, 1)
    resized[0] = resized[1]
    return torch.from_numpy(resized / 127.5 - 1.0).to(dtype=torch.float16)


def render_mimicmotion(
    pipe: Any,
    reference: np.ndarray,
    poses: np.ndarray,
    seed: int,
    steps: int = 25,
    guidance_scale: float = 2.0,
    noise_aug_strength: float = 0.0,
    fps: int = 7,
    decode_chunk_size: int = 8,
) -> tuple[np.ndarray, float]:
    import torch
    from PIL import Image

    ref_img = Image.fromarray(reference).resize((INFER_W, INFER_H), Image.Resampling.BILINEAR)
    pose_pixels = _prepare_pose_tensor(poses)
    num_frames = int(pose_pixels.shape[0])  # N + 1 (frame 0 = reference anchor)
    tile_size = min(72, num_frames)
    tile_overlap = min(6, max(1, tile_size - 1))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        gen = torch.Generator(device="cuda").manual_seed(seed)
        frames = pipe(
            [ref_img],
            image_pose=pose_pixels,
            num_frames=num_frames,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            height=INFER_H,
            width=INFER_W,
            fps=fps,
            noise_aug_strength=noise_aug_strength,
            num_inference_steps=steps,
            generator=gen,
            min_guidance_scale=guidance_scale,
            max_guidance_scale=guidance_scale,
            decode_chunk_size=decode_chunk_size,
            output_type="pt",
            device="cuda",
        ).frames.cpu()
    torch.cuda.synchronize()
    render_s = time.perf_counter() - t0

    # frames: (1, N+1, 3, INFER_H, INFER_W) in [0, 1]; drop reference frame 0
    vid = (frames[0, 1:].clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    out = np.empty((len(poses), OUT_H, OUT_W, 3), dtype=np.uint8)
    for idx in range(len(poses)):
        out[idx] = cv2.resize(vid[idx], (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    return out, render_s


def run_ablation(args: argparse.Namespace) -> None:
    import torch

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

    stage = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    stage_name = os.environ["PS_STAGE"]
    expected = {"smoke": 16, "full": 48}[stage_name]
    if args.frames != expected:
        raise ValueError(f"frames={args.frames} mismatches stage {stage_name} ({expected})")
    if sha256(args.carrier) != CARRIER_SHA256:
        raise ValueError("carrier sha256 mismatch")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    limit_mib = int(os.environ.get("PS_GPU_MEMORY_LIMIT_MIB", "32000"))
    total_mib = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit_mib * 2**20 / total_mib))

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

    t0 = time.perf_counter()
    mm_pipe = build_mimicmotion_pipeline(args.mm_tar, scratch, "cuda")
    load_s = time.perf_counter() - t0

    if stage_name == "smoke":
        u0 = unit_inputs[units[0]]
        a, _ = render_mimicmotion(
            mm_pipe,
            u0["ref_crop"],
            u0["smooth_poses"][:4],
            u0["seed"],
            steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
        b, _ = render_mimicmotion(
            mm_pipe,
            u0["ref_crop"],
            u0["smooth_poses"][:4],
            u0["seed"],
            steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
        determinism[BACKBONE_NAME] = int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))

    dispatch = json.loads((stage / "dispatch.json").read_text()) if (stage / "dispatch.json").exists() else {}
    timing_contaminated = dispatch.get("gpu_sharing", {}).get("mode") == "shared"

    for unit in units:
        u = unit_inputs[unit]
        crops, render_s = render_mimicmotion(
            mm_pipe,
            u["ref_crop"],
            u["smooth_poses"],
            u["seed"],
            steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
        distinct = len({hashlib.sha256(c.tobytes()).hexdigest() for c in crops})
        out_dir = pub / BACKBONE_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / f"{unit}.npz"
        np.savez_compressed(npz_path, t=u["ticks"], box=u["boxes"], crops=crops)
        run_records.append(
            {
                "backbone": BACKBONE_NAME,
                "unit": unit,
                "frames": args.frames,
                "seed": u["seed"],
                "render_seconds": render_s,
                "sec_per_frame": render_s / args.frames,
                "distinct_frames": distinct,
                "timing_contaminated": timing_contaminated,
                "npz_relpath": f"{BACKBONE_NAME}/{unit}.npz",
                "npz_sha256": sha256(npz_path),
            }
        )

    del mm_pipe
    gc.collect()
    torch.cuda.empty_cache()

    summary = {
        "event": "match102",
        "stage": stage_name,
        "frames": args.frames,
        "units": units,
        "device": torch.cuda.get_device_name(0),
        "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
        "timing_contaminated": timing_contaminated,
        "model_load_seconds": {BACKBONE_NAME: load_s},
        "determinism_max_abs_diff": determinism,
        "inference_config": {
            "infer_width": INFER_W,
            "infer_height": INFER_H,
            "out_width": OUT_W,
            "out_height": OUT_H,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "noise_aug_strength": 0.0,
            "fps": 7,
            "decode_chunk_size": 8,
            "dtype": "float16",
        },
        "carrier_sha256": CARRIER_SHA256,
        "mm_tar_sha256": sha256(args.mm_tar),
        "ddim30_48f_seconds_ref": DDIM30_48F_SECONDS,
        "records": run_records,
    }
    text = json.dumps(summary, indent=2) + "\n"
    (pub / "multi_backbone_summary.json").write_text(text)
    (stage / "multi_backbone_summary.json").write_text(text)

    archive = stage / "published.tar"
    tmp = stage / ".mb-mm-published.tar"
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
    assert backbones == {BACKBONE_NAME}
    expected_units = 1 if stage_name == "smoke" else 2
    assert len(records) == expected_units
    for r in records:
        assert r["frames"] == expected
        assert r["distinct_frames"] >= expected - 1
        assert math.isfinite(r["render_seconds"]) and r["render_seconds"] > 0
        assert r["timing_contaminated"] is False
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
    p.add_argument("--mm-tar", type=Path, required=True)
    p.add_argument("--frames", type=int, choices=(16, 48), required=True)
    p.add_argument("--units", default="match102_near,match102_far")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance-scale", type=float, default=2.0)
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

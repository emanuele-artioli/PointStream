"""TOMM multi-backbone Track C: Wan2.2-Animate-14B generator on Shanghai match102."""

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
import types
from typing import Any

import numpy as np

from experiments.tennis import common_player_packet as packet
from experiments.tennis import k6b
from experiments.tennis.tomm_v5_pareto_pilot import (
    CARRIER_SHA256,
    sha256,
    smooth_keypoints,
)

BACKBONE_NAME = "wan_animate_14b"
DDIM30_48F_SECONDS = 674.4665817180066
WAN_DEFAULT_PROMPT = "视频中的人在做动作"
WAN_DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def _install_shims_and_sdpa_patch(wan_code_root: Path) -> dict[str, Any]:
    """Install lightweight shims (decord, easydict) and patch Wan attention to Flash SDPA."""
    import importlib.machinery
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if "decord" not in sys.modules:
        decord_mod = types.ModuleType("decord")
        decord_mod.__spec__ = importlib.machinery.ModuleSpec("decord", None)

        class _DummyVideoReader:
            def __init__(self, *_a: Any, **_kw: Any) -> None:
                raise RuntimeError("decord.VideoReader should not be called in memory-only PointStream path")

        decord_mod.VideoReader = _DummyVideoReader  # type: ignore[attr-defined]
        sys.modules["decord"] = decord_mod

    if "easydict" not in sys.modules:
        ed_mod = types.ModuleType("easydict")
        ed_mod.__spec__ = importlib.machinery.ModuleSpec("easydict", None)

        class EasyDict(dict[str, Any]):
            def __init__(self, d: dict[str, Any] | None = None, **kwargs: Any) -> None:
                super().__init__()
                if d is not None:
                    for k, v in d.items():
                        setattr(self, k, v)
                for k, v in kwargs.items():
                    setattr(self, k, v)

            def __setattr__(self, name: str, value: Any) -> None:
                if isinstance(value, dict) and not isinstance(value, EasyDict):
                    value = EasyDict(value)
                super().__setitem__(name, value)
                super().__setattr__(name, value)

            def __setitem__(self, name: str, value: Any) -> None:
                self.__setattr__(name, value)

            def update(self, e: Any = None, **f: Any) -> None:  # type: ignore[override]
                if e is not None:
                    for k, v in (e.items() if isinstance(e, dict) else e):
                        setattr(self, k, v)
                for k, v in f.items():
                    setattr(self, k, v)

        ed_mod.EasyDict = EasyDict  # type: ignore[attr-defined]
        sys.modules["easydict"] = ed_mod

    if str(wan_code_root) not in sys.path:
        sys.path.insert(0, str(wan_code_root))

    # Avoid executing wan/__init__.py which eagerly imports speech2video (librosa)
    if "wan" not in sys.modules:
        wan_pkg = types.ModuleType("wan")
        wan_pkg.__path__ = [str(wan_code_root / "wan")]
        wan_pkg.__spec__ = importlib.machinery.ModuleSpec("wan", None, is_package=True)
        sys.modules["wan"] = wan_pkg

    import wan.modules.animate.face_blocks as face_blocks_mod
    import wan.modules.animate.model_animate as model_animate_mod
    import wan.modules.attention as attn_mod
    import wan.modules.model as model_mod

    kernel_stats = {"flash_sdpa_calls": 0, "face_sdpa_calls": 0}

    def sdpa_flash_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: torch.Tensor | None = None,
        k_lens: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        q_scale: float | None = None,
        causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        deterministic: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        version: int | None = None,
    ) -> torch.Tensor:
        del window_size, deterministic, version
        out_dtype = q.dtype
        half_dtypes = (torch.float16, torch.bfloat16)
        q_h = q if q.dtype in half_dtypes else q.to(dtype)
        k_h = k if k.dtype in half_dtypes else k.to(dtype)
        v_h = v if v.dtype in half_dtypes else v.to(dtype)
        q_h = q_h.to(v_h.dtype)
        k_h = k_h.to(v_h.dtype)
        if q_scale is not None:
            q_h = q_h * q_scale

        # Check whether k_lens actually trims any tokens
        b, _lq, _nq, _d = q_h.shape
        lk = k_h.shape[1]
        has_padding = (q_lens is not None) or (
            k_lens is not None and not bool(torch.all(k_lens == lk).item())
        )
        qt = q_h.transpose(1, 2)
        kt = k_h.transpose(1, 2)
        vt = v_h.transpose(1, 2)
        if not has_padding:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(
                    qt,
                    kt,
                    vt,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=causal,
                    scale=softmax_scale,
                )
            kernel_stats["flash_sdpa_calls"] += 1
        else:
            # Build exact boolean mask if padded
            attn_mask = None
            if k_lens is not None:
                pos = torch.arange(lk, device=k_h.device)[None, :]
                attn_mask = (pos < k_lens.to(k_h.device)[:, None])[:, None, None, :]
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = F.scaled_dot_product_attention(
                    qt,
                    kt,
                    vt,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=causal,
                    scale=softmax_scale,
                )
        return out.transpose(1, 2).contiguous().to(out_dtype)

    def sdpa_face_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mode: str = "torch",
        drop_rate: float = 0.0,
        attn_mask: torch.Tensor | None = None,
        causal: bool = False,
        max_seqlen_q: int | None = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        del mode, max_seqlen_q, batch_size
        qt = q.transpose(1, 2)
        kt = k.transpose(1, 2)
        vt = v.transpose(1, 2)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(qt.dtype)
        backends = (
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            if attn_mask is None
            else [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        )
        with sdpa_kernel(backends):
            x = F.scaled_dot_product_attention(
                qt, kt, vt, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
            )
        kernel_stats["face_sdpa_calls"] += 1
        return x.transpose(1, 2).contiguous().reshape(q.shape[0], q.shape[1], -1)

    attn_mod.flash_attention = sdpa_flash_attention
    model_mod.flash_attention = sdpa_flash_attention
    model_animate_mod.flash_attention = sdpa_flash_attention
    face_blocks_mod.attention = sdpa_face_attention
    return kernel_stats


def _assemble_ckpt_dir(args: argparse.Namespace, scratch: Path) -> tuple[Path, Path]:
    """Assemble local checkpoint directory via symlinks to staged weights + extracted configs/code."""
    root = scratch / "wan_bundle"
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.code_configs_tar) as tf:
        tf.extractall(root, filter="data")

    code_root = root / "wan22-code"
    ckpt_dir = root / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    weight_map = {
        "Wan2.1_VAE.pth": args.vae,
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": args.clip,
        "models_t5_umt5-xxl-enc-bf16.pth": args.t5,
        "diffusion_pytorch_model-00001-of-00004.safetensors": args.dit1,
        "diffusion_pytorch_model-00002-of-00004.safetensors": args.dit2,
        "diffusion_pytorch_model-00003-of-00004.safetensors": args.dit3,
        "diffusion_pytorch_model-00004-of-00004.safetensors": args.dit4,
    }
    for name, src in weight_map.items():
        dst = ckpt_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(Path(src).resolve())
    return code_root, ckpt_dir


def _pad_to_4n_plus_1(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad sequence length N to smallest 4k+1 >= N by repeating the last frame."""
    n = arr.shape[0]
    target = ((n - 1 + 3) // 4) * 4 + 1
    if target == n:
        return arr, target
    pad = np.repeat(arr[-1:], target - n, axis=0)
    return np.concatenate([arr, pad], axis=0), target


def _get_i2v_mask(lat_t: int, lat_h: int, lat_w: int, mask_len: int, device: Any) -> Any:
    import torch

    msk = torch.zeros(1, (lat_t - 1) * 4 + 1, lat_h, lat_w, device=device)
    msk[:, :mask_len] = 1
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    return msk.transpose(1, 2)[0]


def _render_wan_animate_unit(
    noise_model: Any,
    vae: Any,
    clip: Any,
    context: list[Any],
    ref_crop: np.ndarray,
    poses: np.ndarray,
    seed: int,
    steps: int,
    shift: float,
    device: Any,
) -> tuple[np.ndarray, float, dict[str, float]]:
    import torch
    from einops import rearrange
    from wan.modules.animate.animate_utils import TensorList
    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )

    real_len = poses.shape[0]
    poses_padded, clip_len = _pad_to_4n_plus_1(poses)
    h, w = ref_crop.shape[:2]
    assert h % 16 == 0 and w % 16 == 0, f"Expected H,W divisible by 16, got {h}x{w}"

    # Zero-leakage unconditional face input (-1.0 in normalized [-1, 1] space)
    face_pixel_values = torch.full(
        (1, 3, clip_len, 512, 512),
        -1.0,
        dtype=torch.bfloat16,
        device=device,
    )
    cond_pixel_values = rearrange(
        torch.from_numpy(poses_padded.astype(np.float32) / 127.5 - 1.0),
        "t h w c -> 1 c t h w",
    ).to(device=device, dtype=torch.bfloat16)
    ref_pixel_values = rearrange(
        torch.from_numpy(ref_crop.astype(np.float32) / 127.5 - 1.0),
        "h w c -> 1 c 1 h w",
    ).to(device=device, dtype=torch.bfloat16)

    lat_h = h // 8
    lat_w = w // 8
    lat_t = clip_len // 4 + 1
    target_shape = [lat_t + 1, lat_h, lat_w]
    max_seq_len = int(math.ceil(int(np.prod(target_shape)) // 4))

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        # 1. CLIP visual features (stage CLIP on GPU, then offload to CPU)
        t0 = time.perf_counter()
        clip.model.to(device)
        img = ref_pixel_values[0, :, 0]
        clip_context = clip.visual([img[:, None, :, :]]).to(dtype=torch.bfloat16, device=device)
        clip.model.to("cpu")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        clip_s = time.perf_counter() - t0

        # 2. VAE encode pose, reference, and empty temporal context
        t0 = time.perf_counter()
        vae.model.to(device)
        pose_latents = torch.stack(vae.encode(cond_pixel_values))
        ref_latents = torch.stack(vae.encode(ref_pixel_values))
        mask_ref = _get_i2v_mask(1, lat_h, lat_w, 1, device=device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=device)
        y_reft = vae.encode([torch.zeros(3, clip_len, h, w, device=device, dtype=torch.bfloat16)])[0]
        msk_reft = _get_i2v_mask(lat_t, lat_h, lat_w, 0, device=device)
        y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=device)
        y = torch.concat([y_ref, y_reft], dim=1)
        vae.model.to("cpu")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        vae_enc_s = time.perf_counter() - t0

        # 3. DiT denoising (stage noise_model on GPU in bf16 with CLIP/VAE on CPU)
        t0 = time.perf_counter()
        noise_model.to(device=device, dtype=torch.bfloat16)
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        noise = [
            torch.randn(
                16,
                target_shape[0],
                target_shape[1],
                target_shape[2],
                dtype=torch.float32,
                device=device,
                generator=seed_g,
            )
        ]

        sample_scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )
        sampling_sigmas = get_sampling_sigmas(steps, shift)
        timesteps, _ = retrieve_timesteps(
            sample_scheduler,
            device=device,
            sigmas=sampling_sigmas,
        )

        arg_c = {
            "context": [c.to(device=device, dtype=torch.bfloat16) for c in context],
            "seq_len": max_seq_len,
            "clip_fea": clip_context,
            "y": [y],
            "pose_latents": pose_latents,
            "face_pixel_values": face_pixel_values,
        }

        latents = noise
        for t_step in timesteps:
            timestep = torch.stack([t_step])
            noise_pred = TensorList(noise_model(TensorList(latents), t=timestep, **arg_c))
            temp_x0 = sample_scheduler.step(
                noise_pred[0].unsqueeze(0),
                t_step,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents[0] = temp_x0.squeeze(0)

        x0 = [latents[0][:, 1:].to(dtype=torch.float32)]
        torch.cuda.synchronize()
        dit_s = time.perf_counter() - t0

        # 4. VAE decode (offload DiT to CPU so VAE decode has full VRAM headroom)
        del pose_latents, ref_latents, y_ref, y_reft, y, clip_context, arg_c, cond_pixel_values, face_pixel_values
        noise_model.to("cpu")
        torch.cuda.empty_cache()

        t0 = time.perf_counter()
        vae.model.to(device)
        out_frames = vae.decode(x0)[0]  # (3, clip_len, H, W) in [-1, 1]
        vae.model.to("cpu")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        vae_dec_s = time.perf_counter() - t0

    render_s = time.perf_counter() - t_start
    # Convert (3, clip_len, H, W) [-1, 1] -> (real_len, H, W, 3) uint8
    vid = out_frames[:, :real_len].float().clamp(-1.0, 1.0)
    vid = ((vid + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    breakdown = {
        "clip_s": clip_s,
        "vae_enc_s": vae_enc_s,
        "dit_s": dit_s,
        "vae_dec_s": vae_dec_s,
    }
    return np.ascontiguousarray(vid), render_s, breakdown


def run_ablation(args: argparse.Namespace) -> None:
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(0)

    code_root, ckpt_dir = _assemble_ckpt_dir(args, scratch)
    kernel_stats = _install_shims_and_sdpa_patch(code_root)

    from wan.modules.animate import CLIPModel, WanAnimateModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_1 import Wan2_1_VAE

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
            "boxes": boxes,
            "ref_crop": ref_crop,
            "seed": seed,
            "ticks": ticks,
            "smooth_poses": smooth_poses,
        }

    # 1. Encode fixed prompt on CPU with UMT5-XXL, then free T5 RAM
    t_load0 = time.perf_counter()
    t5 = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=str(ckpt_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(ckpt_dir / "google/umt5-xxl"),
    )
    t_t5_0 = time.perf_counter()
    with torch.inference_mode():
        context = t5([WAN_DEFAULT_PROMPT], torch.device("cpu"))
    t5_encode_s = time.perf_counter() - t_t5_0
    del t5
    gc.collect()

    # 2. Initialize CLIP and VAE on CPU (moved to GPU briefly during encode/decode)
    clip = CLIPModel(
        dtype=torch.float16,
        device=torch.device("cpu"),
        checkpoint_path=str(ckpt_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        tokenizer_path=str(ckpt_dir / "xlm-roberta-large"),
    )
    vae = Wan2_1_VAE(
        vae_pth=str(ckpt_dir / "Wan2.1_VAE.pth"),
        device=torch.device("cpu"),
    )
    # Keep VAE's internal device pointer synced when moved to GPU
    vae.device = device
    vae.mean = vae.mean.to(device)
    vae.std = vae.std.to(device)
    vae.scale = [vae.mean, 1.0 / vae.std]
    clip.device = device

    # 3. Load 14B WanAnimateModel in bf16 on CPU (staged onto GPU during DiT denoising)
    noise_model = WanAnimateModel.from_pretrained(
        str(ckpt_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    noise_model.eval().requires_grad_(False)
    noise_model.to(dtype=torch.bfloat16)
    load_s = time.perf_counter() - t_load0

    pub = scratch / "publish"
    pub.mkdir(parents=True, exist_ok=True)
    run_records: list[dict[str, Any]] = []
    determinism: dict[str, int] = {}

    if stage_name == "smoke":
        u0 = unit_inputs[units[0]]
        a, _, _ = _render_wan_animate_unit(
            noise_model, vae, clip, context, u0["ref_crop"], u0["smooth_poses"][:5],
            u0["seed"], steps=4, shift=args.shift, device=device,
        )
        b, _, _ = _render_wan_animate_unit(
            noise_model, vae, clip, context, u0["ref_crop"], u0["smooth_poses"][:5],
            u0["seed"], steps=4, shift=args.shift, device=device,
        )
        determinism[BACKBONE_NAME] = int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))

    for unit in units:
        u = unit_inputs[unit]
        crops, render_s, breakdown = _render_wan_animate_unit(
            noise_model,
            vae,
            clip,
            context,
            u["ref_crop"],
            u["smooth_poses"],
            u["seed"],
            steps=args.steps,
            shift=args.shift,
            device=device,
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
                "timing_breakdown": breakdown,
                "npz_relpath": f"{BACKBONE_NAME}/{unit}.npz",
                "npz_sha256": sha256(npz_path),
            }
        )

    peak_alloc_mib = torch.cuda.max_memory_allocated(device) / 2**20
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 2**20
    dispatch = json.loads((stage / "dispatch.json").read_text()) if (stage / "dispatch.json").exists() else {}
    summary = {
        "event": "match102",
        "stage": stage_name,
        "frames": args.frames,
        "units": units,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_peak_mib": peak_alloc_mib,
        "gpu_peak_reserved_mib": peak_reserved_mib,
        "timing_contaminated": dispatch.get("gpu_sharing", {}).get("mode") == "shared",
        "model_load_seconds": {BACKBONE_NAME: load_s, "t5_cpu_encode_s": t5_encode_s},
        "kernel_stats": kernel_stats,
        "determinism_max_abs_diff": determinism,
        "wan_config": {
            "steps": args.steps,
            "shift": args.shift,
            "guide_scale": 1.0,
            "solver": "dpm++",
            "dtype": "bfloat16",
            "face_conditioning": "unconditional_black",
        },
        "carrier_sha256": CARRIER_SHA256,
        "ddim30_48f_seconds_ref": DDIM30_48F_SECONDS,
        "records": run_records,
    }
    text = json.dumps(summary, indent=2) + "\n"
    (pub / "multi_backbone_summary.json").write_text(text)
    (stage / "multi_backbone_summary.json").write_text(text)

    archive = stage / "published.tar"
    tmp = stage / ".wan-published.tar"
    with tarfile.open(tmp, "x") as tf:
        tf.add(pub, arcname="publish")
    tmp.rename(archive)


def validate(_args: argparse.Namespace) -> None:
    stage = Path(os.environ["PS_STAGE_DIR"])
    stage_name = os.environ["PS_STAGE"]
    expected = {"smoke": 16, "full": 48}[stage_name]
    data = json.loads((stage / "multi_backbone_summary.json").read_text())
    assert data["frames"] == expected
    assert data["kernel_stats"]["flash_sdpa_calls"] > 0, "FlashAttention SDPA kernel was not exercised"
    assert data["gpu_peak_mib"] < 44 * 1024, f"Peak VRAM {data['gpu_peak_mib']:.1f} MiB exceeded 44 GiB"
    records = data["records"]
    backbones = {r["backbone"] for r in records}
    assert backbones == {BACKBONE_NAME}
    for r in records:
        assert r["frames"] == expected
        assert r["distinct_frames"] >= expected - 1
        assert math.isfinite(r["render_seconds"]) and r["render_seconds"] > 0
    if stage_name == "smoke":
        diff = data["determinism_max_abs_diff"][BACKBONE_NAME]
        assert diff <= 1, f"determinism failed for {BACKBONE_NAME}: max_abs_diff={diff}"
        # Forecast 48-frame 2-player runtime (conservative linear * 1.35 for longer sequence)
        est_48f_2p = records[0]["render_seconds"] * 3.0 * 2.0 * 1.35
        assert est_48f_2p <= 2700.0, f"Forecast 48f 2-player runtime {est_48f_2p:.1f}s > 2700s"
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(
            {
                "passed": True,
                "checks": {
                    "frames": expected,
                    "backbones": sorted(backbones),
                    "records": len(records),
                    "cuda_device": data["device"],
                    "compute_capability": data["compute_capability"],
                    "gpu_peak_mib": data["gpu_peak_mib"],
                    "gpu_peak_reserved_mib": data["gpu_peak_reserved_mib"],
                    "kernel_stats": data["kernel_stats"],
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
    p.add_argument("--carrier", type=Path, required="--validate" not in sys.argv)
    p.add_argument("--code-configs-tar", type=Path, required="--validate" not in sys.argv)
    p.add_argument("--vae", type=Path, required="--validate" not in sys.argv)
    p.add_argument("--clip", type=Path, required="--validate" not in sys.argv)
    p.add_argument("--t5", type=Path, required="--validate" not in sys.argv)
    p.add_argument("--dit1", type=Path, required="--validate" not in sys.argv)
    p.add_argument("--dit2", type=Path, required="--validate" not in sys.argv)
    p.add_argument("--dit3", type=Path, required="--validate" not in sys.argv)
    p.add_argument("--dit4", type=Path, required="--validate" not in sys.argv)
    p.add_argument("--frames", type=int, choices=(16, 48), required="--validate" not in sys.argv)
    p.add_argument("--units", default="match102_near,match102_far")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    if args.validate:
        validate(args)
    else:
        run_ablation(args)


if __name__ == "__main__":
    main()

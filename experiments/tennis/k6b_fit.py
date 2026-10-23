"""K6b (players) 3's fit: AnimateAnyone (the `original` profile) fitted on a window's own frames, the overfit oracle.
Run in ``pointstream-jcv`` (as `k6b_generate`), never imported by PointStream:

    python -m experiments.tennis.k6b_fit fit --fit-data TAR --window ID --weight NAME PATH ... --minutes M
        [--max-t T] [--fractions 0.25,0.5,1] [--max-steps N]
    python -m experiments.tennis.k6b_fit validate

``--fit-data`` is `experiments.tennis.k6b prepare-fit`'s published.tar: per unit (window, player) the true crops and the
DWPose maps of the uncoded keypoints. The objective is the fork's ``finetune_stage1.py``: a target frame's latent noised
and predicted (v-prediction, zero-SNR DDIM schedule, SNR-gamma 5 weighting, noise offset 0.05) by the denoising UNet
(no motion module) conditioned on its pose map and on a reference frame of the same player through the ReferenceNet
and its CLIP embedding, which are dropped 10% of the time; the ReferenceNet's last up block is frozen, as there. What is
fitted is a LoRA on both UNets' attention projections (attn1/attn2 to_q, to_k, to_v, to_out.0) and the whole pose
guider: a full fit of both UNets with AdamW needs about 30-40 GB, more than a shared 48 GB card leaves (the rule's
LoRA branch). Pairs come from frames with ``t < --max-t`` (the pilot's held-out split). A checkpoint (the LoRA and the
pose guider) is written at each fraction of ``--minutes`` of fitting; `k6b_generate --fit` renders with it merged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis.k6b_generate import (GEN_H, GEN_W, SD15_UNET, UNET_ADDITIONAL, folder, load_checked, loading_record,
                                             sha256, stage_dir, write_json, UNUSED_REFERENCE_HEAD)

LORA_RANK = 64
LORA_ALPHA = 64
#: Spatial attention projections; the motion module's are named ``attention_blocks`` and are not matched.
DENOISING_TARGETS = r".*(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)$"
REFERENCE_TARGETS = r"^(?!up_blocks\.3\.).*(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)$"
LORA_LR = 1e-4
GUIDER_LR = 1e-5
WARMUP = 50
BATCH = 1
ACCUMULATE = 4
UNCOND_RATIO = 0.1
NOISE_OFFSET = 0.05
SNR_GAMMA = 5.0
MAX_GRAD_NORM = 1.0
#: finetune_stage1.yaml's noise schedule with enable_zero_snr (rescaled, v-prediction, trailing).
TRAIN_SCHEDULER: dict[str, Any] = {"num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012,
                                   "beta_schedule": "scaled_linear", "steps_offset": 1, "clip_sample": False,
                                   "rescale_betas_zero_snr": True, "timestep_spacing": "trailing", "prediction_type": "v_prediction"}
PROBE_T = 500  # the smoke's x0 check: one timestep, fixed noise, the same frames before and after


QUANT_MIN_VALUES = 1024  # g5c's: smaller tensors go as fp16


def lora_config(targets: str, rank: int = LORA_RANK) -> Any:
    from peft import LoraConfig

    return LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=targets)  # alpha = rank (step 3)


def inject(reference: Any, denoising: Any, rank: int = LORA_RANK) -> None:
    from peft import inject_adapter_in_model

    inject_adapter_in_model(lora_config(REFERENCE_TARGETS, rank), reference)
    inject_adapter_in_model(lora_config(DENOISING_TARGETS, rank), denoising)


def quantize_channels(t: Any, bits: int) -> tuple[Any, Any, Any]:
    """`experiments.background.g5c.quantize_channels`, copied (g5c's module needs the main environment): symmetric
    uniform quantization per output channel (dim 0), int8 levels, fp16 scales, and the dequantized tensor."""
    import torch

    levels = 2 ** (bits - 1) - 1
    flat = t.reshape(t.shape[0], -1).float()
    scale = (flat.abs().amax(1) / levels).half().float()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(flat / safe[:, None]), -levels, levels)
    q[scale == 0] = 0
    return q.to(torch.int8), scale.half(), (q * scale[:, None]).reshape(t.shape)


def quantize_tensor(t: Any, bits: int, payloads: list[bytes]) -> Any:
    """One billed tensor: per output channel at ``bits`` when it has QUANT_MIN_VALUES or more values, else fp16; its
    payloads appended; the decoder's rebuilt tensor returned (fp32)."""
    if t.numel() < QUANT_MIN_VALUES or t.dim() == 0:
        sent = t.half()
        payloads.append(sent.numpy().tobytes())
        return sent.float()
    q, scale, rebuilt = quantize_channels(t.float(), bits)
    payloads += [q.numpy().tobytes(), scale.numpy().tobytes()]
    return rebuilt


def quantize_adapters(state: dict[str, Any], bits: int, guider_base: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """K6b (players) 4's billed form of a checkpoint: every adapter tensor of QUANT_MIN_VALUES or more values
    quantized per output channel to ``bits`` (smaller ones fp16), each payload LZMA-coded on its own (g5c's
    ``whole`` variant). Returns the state the decoder rebuilds (fp16) and the billed size. A trained pose guider
    (step 5) is billed the same way as its change from the profile's guider (``guider_base``), which the decoder
    holds: rebuilt = base + dequantized change. Frozen, it is absent and costs nothing."""
    import lzma

    out: dict[str, Any] = {k: v for k, v in state.items() if k not in ("reference_unet", "denoising_unet", "pose_guider")}
    payloads: list[bytes] = []
    values = 0
    for part in ("reference_unet", "denoising_unet"):
        out[part] = {}
        for name, t in state[part].items():
            values += t.numel()
            out[part][name] = quantize_tensor(t.float(), bits, payloads).half()
    adapter_payloads = len(payloads)
    guider_values = 0
    if "pose_guider" in state:
        if guider_base is None:
            raise SystemExit("a trained pose guider is billed as its change: give the profile's guider weights")
        out["pose_guider"] = {}
        for name, t in state["pose_guider"].items():
            base = guider_base[name].float()
            if not t.is_floating_point():  # integer buffers (none in PoseGuider today) go unchanged and unbilled
                out["pose_guider"][name] = t
                continue
            guider_values += t.numel()
            out["pose_guider"][name] = base + quantize_tensor(t.float() - base, bits, payloads)
    sizes = [len(lzma.compress(b, preset=6)) for b in payloads]
    return out, {"bits": bits, "bytes": sum(sizes), "adapter_values": values, "raw_fp16_bytes": 2 * values,
                 "pose_guider_billed": "pose_guider" in state, "pose_guider_values": guider_values,
                 "pose_guider_bytes": sum(sizes[adapter_payloads:])}


def merge_base(reference: Any, denoising: Any, base: dict[str, Any]) -> dict[str, Any]:
    """Step 5's tennis base: its LoRA merged into both UNets' weights and the LoRA layers removed, so a match's
    LoRA can be injected (fitting) or merged (rendering) on top. Not billed: the base is part of the decoder."""
    from peft import set_peft_model_state_dict
    from peft.tuners.lora import LoraLayer

    rank = int(base.get("rank", LORA_RANK))
    inject(reference, denoising, rank)
    record: dict[str, Any] = {"rank": rank}
    for name, module in (("reference_unet", reference), ("denoising_unet", denoising)):
        result = set_peft_model_state_dict(module, base[name])
        missing = [k for k in getattr(result, "missing_keys", []) if "lora_" in k]
        if missing:
            raise SystemExit(f"base {name}: {len(missing)} LoRA keys missing, e.g. {missing[:3]}")
        layers = [(n, m) for n, m in module.named_modules() if isinstance(m, LoraLayer)]
        for n, layer in layers:
            layer.merge()
            parent, _, child = n.rpartition(".")
            setattr(module.get_submodule(parent) if parent else module, child, layer.get_base_layer())
        if any(isinstance(m, LoraLayer) for m in module.modules()):
            raise SystemExit(f"base {name}: LoRA layers left after merging")
        record[name] = {"merged_layers": len(layers)}
        if hasattr(module, "peft_config"):  # the base's adapter config goes with its layers; the match's is injected fresh
            del module.peft_config
    return record


def apply_fit(reference: Any, denoising: Any, guider: Any, checkpoint: Path, base: Path | None = None,
              requantize: int = 0) -> dict[str, Any]:
    """Load a fitted checkpoint into rendering models and merge the LoRA into their weights (k6b_generate --fit).
    Step 5: ``base``, a tennis base merged first; ``requantize``, the checkpoint (an fp16 one) quantized to these bits
    as billed before merging, the billed size recorded."""
    import torch
    from peft import set_peft_model_state_dict
    from peft.tuners.lora import LoraLayer

    record: dict[str, Any] = {}
    if base is not None:
        record["base"] = {"sha256": sha256(base), **merge_base(reference, denoising, torch.load(base, map_location="cpu", weights_only=True))}
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if requantize:
        guider_base = {k: v.detach().cpu() for k, v in guider.state_dict().items()}
        state, record["billing"] = quantize_adapters({**state, **{k: {n: v.float() for n, v in state[k].items()}
                                                                 for k in ("reference_unet", "denoising_unet")}}, requantize, guider_base)
    rank = int(state.get("rank", LORA_RANK))
    inject(reference, denoising, rank)
    record["rank"] = rank
    for name, module in (("reference_unet", reference), ("denoising_unet", denoising)):
        result = set_peft_model_state_dict(module, state[name])
        missing = [k for k in getattr(result, "missing_keys", []) if "lora_" in k]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} LoRA keys missing from the checkpoint, e.g. {missing[:3]}")
        layers = [m for m in module.modules() if isinstance(m, LoraLayer)]
        for layer in layers:
            layer.merge()
        record[name] = {"lora_layers": len(layers), "lora_tensors": len(state[name])}
    if "pose_guider" in state:
        record["pose_guider"] = load_checked(guider, state["pose_guider"], "pose_guider")
    else:
        record["pose_guider"] = "frozen (the profile's)"
    return record


class Trainer:
    def __init__(self, weights: dict[str, Path], device: str, scratch: Path, rank: int = LORA_RANK, freeze_guider: bool = False,
                 base: dict[str, Any] | None = None):
        import torch
        from animate_anyone.models.mutual_self_attention import ReferenceAttentionControl
        from animate_anyone.models.pose_guider import PoseGuider
        from animate_anyone.models.unet_2d_condition import UNet2DConditionModel
        from animate_anyone.models.unet_3d import UNet3DConditionModel
        from diffusers import AutoencoderKL, DDIMScheduler
        from transformers import CLIPVisionModelWithProjection

        self.device = device
        dtype = torch.bfloat16
        record: dict[str, Any] = {"dtype": "bfloat16 base weights and autocast; LoRA and pose guider float32"}
        vae, info = AutoencoderKL.from_pretrained(str(folder(weights, "vae", scratch)), output_loading_info=True)
        record["vae"] = loading_record(info, "vae")
        encoder, info = CLIPVisionModelWithProjection.from_pretrained(str(folder(weights, "image_encoder", scratch)), output_loading_info=True)
        record["image_encoder"] = loading_record(info, "image_encoder")
        reference = UNet2DConditionModel.from_config({**SD15_UNET, "_class_name": "UNet2DConditionModel"})
        record["reference_unet"] = load_checked(reference, torch.load(weights["original/reference_unet.pth"], map_location="cpu", weights_only=True),
                                                "reference_unet", UNUSED_REFERENCE_HEAD)
        config3d = {**SD15_UNET, "_class_name": "UNet3DConditionModel",
                    "down_block_types": ["CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"],
                    "up_block_types": ["UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"],
                    "mid_block_type": "UNetMidBlock3DCrossAttn"}
        stage1 = {**UNET_ADDITIONAL, "use_motion_module": False, "unet_use_temporal_attention": False}  # finetune_stage1.py's
        denoising = UNet3DConditionModel.from_config(config3d, **stage1)
        record["denoising_unet"] = load_checked(denoising, torch.load(weights["original/denoising_unet.pth"], map_location="cpu", weights_only=True),
                                                "denoising_unet")
        guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256))
        record["pose_guider"] = load_checked(guider, torch.load(weights["original/pose_guider.pth"], map_location="cpu", weights_only=True), "pose_guider")
        self.guider_base = {k: v.detach().clone() for k, v in guider.state_dict().items()}  # the decoder's: a trained guider is billed as its change
        if base is not None:  # step 5's tennis base, merged before the match's LoRA is injected
            record["base"] = merge_base(reference, denoising, base)
        for m in (vae, encoder, reference, denoising):
            m.requires_grad_(False)
        vae.to(device, dtype=torch.float32)  # encoding only, once per frame
        encoder.to(device, dtype=dtype)
        reference.to(device, dtype=dtype)
        denoising.to(device, dtype=dtype)
        inject(reference, denoising, rank)
        self.rank, self.freeze_guider = rank, freeze_guider
        lora = [p for m in (reference, denoising) for n, p in m.named_parameters() if "lora_" in n]
        for p in lora:
            p.data = p.data.float()
            p.requires_grad_(True)
        guider.to(device, dtype=torch.float32).requires_grad_(not freeze_guider)
        record["trainable"] = {"lora_tensors": len(lora), "lora_params": int(sum(p.numel() for p in lora)),
                               "pose_guider_params": 0 if freeze_guider else int(sum(p.numel() for p in guider.parameters())),
                               "rank": rank, "pose_guider_frozen": freeze_guider}
        groups = [{"params": lora, "lr": LORA_LR}] + ([] if freeze_guider else [{"params": list(guider.parameters()), "lr": GUIDER_LR}])
        self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
        self.trainable = lora + ([] if freeze_guider else list(guider.parameters()))
        self.writer = ReferenceAttentionControl(reference, do_classifier_free_guidance=False, mode="write", fusion_blocks="full")
        self.reader = ReferenceAttentionControl(denoising, do_classifier_free_guidance=False, mode="read", fusion_blocks="full")
        self.scheduler = DDIMScheduler(**TRAIN_SCHEDULER)
        self.vae, self.encoder, self.reference, self.denoising, self.guider = vae, encoder, reference, denoising, guider
        self.lora = lora
        self.dtype = dtype
        self.record = record

    def encode(self, images: np.ndarray) -> tuple[Any, Any, Any]:
        """Latent means and stds (VAE, float32) and CLIP image embeddings of uint8 crops."""
        import torch
        from transformers import CLIPImageProcessor

        processor = CLIPImageProcessor()
        means, stds, embeds = [], [], []
        with torch.inference_mode():
            for first in range(0, len(images), 8):
                x = torch.from_numpy(images[first:first + 8]).to(self.device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
                dist = self.vae.encode(x).latent_dist
                means.append(dist.mean)
                stds.append(dist.std)
                clip = processor(images=list(images[first:first + 8]), return_tensors="pt").pixel_values
                embeds.append(self.encoder(clip.to(self.device, dtype=self.dtype)).image_embeds.float())
        return torch.cat(means), torch.cat(stds), torch.cat(embeds)

    def snr(self, timesteps: Any) -> Any:
        """finetune_stage1.py's compute_snr."""
        import torch

        alphas = self.scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
        return (alphas ** 0.5 / torch.clamp((1 - alphas) ** 0.5, min=1e-8)) ** 2

    def forward(self, target_latents: Any, ref_latents: Any, clip: Any, pose: Any, timesteps: Any, noise: Any, uncond: bool) -> tuple[Any, Any]:
        """(prediction, target) for a batch; the reference banks are cleared after."""
        import torch

        noisy = self.scheduler.add_noise(target_latents, noise, timesteps)
        target = self.scheduler.get_velocity(target_latents, noise, timesteps)
        with torch.autocast("cuda", dtype=self.dtype):
            pose_fea = self.guider(pose)
            embeds = (torch.zeros_like(clip) if uncond else clip).unsqueeze(1).to(self.dtype)
            if not uncond:
                self.reference(ref_latents.to(self.dtype), torch.zeros_like(timesteps), encoder_hidden_states=embeds, return_dict=False)
                self.reader.update(self.writer)
            pred = self.denoising(noisy.to(self.dtype), timesteps, pose_cond_fea=pose_fea, encoder_hidden_states=embeds).sample
        self.reader.clear()
        self.writer.clear()
        return pred.float(), target

    def loss(self, pred: Any, target: Any, timesteps: Any) -> Any:
        import torch
        import torch.nn.functional as F

        snr = self.snr(timesteps) + 1  # v-prediction, as finetune_stage1.py
        weights = torch.stack([snr, SNR_GAMMA * torch.ones_like(timesteps)], 1).min(1)[0] / snr
        per = F.mse_loss(pred, target, reduction="none").mean(dim=list(range(1, pred.ndim)))
        return (per * weights).mean()

    def x0(self, pred: Any, noisy: Any, timesteps: Any) -> Any:
        """x0 from a v-prediction: sqrt(a) x_t - sqrt(1 - a) v."""
        a = self.scheduler.alphas_cumprod.to(noisy.device)[timesteps].view(-1, 1, 1, 1, 1)
        return a ** 0.5 * noisy - (1 - a) ** 0.5 * pred

    def state(self) -> dict[str, Any]:
        from peft import get_peft_model_state_dict

        out: dict[str, Any] = {"rank": self.rank,
                               "reference_unet": {k: v.detach().cpu().float() for k, v in get_peft_model_state_dict(self.reference).items()},
                               "denoising_unet": {k: v.detach().cpu().float() for k, v in get_peft_model_state_dict(self.denoising).items()}}
        if not self.freeze_guider:
            out["pose_guider"] = {k: v.detach().cpu() for k, v in self.guider.state_dict().items()}
        return out


def command_fit(args: argparse.Namespace) -> int:
    import torch

    from experiments.audit.env_smoke import device_record, profile_cuda
    from experiments.jobs.monitor import publish_progress

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    began = time.time()
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    weights = {name: Path(path) for name, path in args.weight}
    # Step 5: several prepare tars (step 4's capped windows and the uncapped rest); a unit's npz is found in its own.
    unit_path: dict[str, Path] = {}
    windows_used: list[str] = []
    for n, tar in enumerate(args.fit_data):
        root = scratch / "fit-data" / str(n)
        if not root.is_dir():
            root.mkdir(parents=True)
            with tarfile.open(tar) as archive:
                archive.extractall(root, filter="data")
        manifest = json.loads((root / "publish" / "prepare-fit.json").read_text())
        for w in manifest["windows"]:
            if w["window"] in args.exclude_window:
                continue
            if args.window:
                take = w["window"] == args.window
            else:  # step 4: one match; step 5: a player's matches, or every match (--all, the tennis base)
                take = args.all or w["window"].split("/")[0] in args.match
            if not take:
                continue
            windows_used.append(w["window"])
            for u in w["units"]:
                if u["unit"] in unit_path:
                    raise SystemExit(f"{u['unit']} is in two fit-data archives")
                unit_path[u["unit"]] = root / "publish" / "fit" / f"{u['unit']}.npz"
    units = sorted(unit_path)
    if args.max_units:  # the smoke: a few units, spread over the archives' order
        units = units[:: max(1, len(units) // args.max_units)][:args.max_units]
    if not units:
        raise SystemExit(f"no fit data for {args.window_name}")
    base_state = None
    base_path = Path()
    if args.base:
        base_root = scratch / "base-archive"
        if not base_root.is_dir():
            base_root.mkdir(parents=True)
            with tarfile.open(args.base_archive) as archive:
                archive.extractall(base_root, filter="data")
        base_path = base_root / "publish" / args.base
        base_state = torch.load(base_path, map_location="cpu", weights_only=True)
    device = "cuda"
    limit = int(os.environ.get("PS_GPU_MEMORY_LIMIT_MIB", "24000"))
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    seed = int.from_bytes(args.window_name.encode(), "little") % 2**31
    random.seed(seed)
    torch.manual_seed(seed)
    trainer = Trainer(weights, device, scratch, args.rank, args.freeze_guider, base_state)
    # Every training frame encoded once: latent mean/std, CLIP embedding, pose map (as the dataset's ToTensor).
    # Pose maps stay uint8 in host memory (step 5's base holds about 9,000 crops) and the true images are read back
    # only for the probe's four frames.
    data = []
    for name in units:
        a = np.load(unit_path[name])
        keep = a["t"] < args.max_t if args.max_t else np.ones(len(a["t"]), bool)
        if args.frame_stride > 1:  # step 5's base: every n-th frame (neighbours are near copies), for the 45-minute stage
            keep &= np.arange(len(keep)) % args.frame_stride == 0
        if keep.sum() < 2:
            continue
        mean, std, clip = trainer.encode(a["image"][keep])
        pose = torch.from_numpy(np.ascontiguousarray(a["pose"][keep])).permute(0, 3, 1, 2)
        data.append({"unit": name, "t": a["t"][keep], "rows": np.nonzero(keep)[0], "mean": mean, "std": std, "clip": clip, "pose": pose})
    frames = sum(len(d["t"]) for d in data)

    def batch(gen: random.Random) -> tuple[Any, ...]:
        d = gen.choice(data)
        i, r = gen.randrange(len(d["t"])), gen.randrange(len(d["t"]))
        lat = (d["mean"][i:i + 1] + d["std"][i:i + 1] * torch.randn_like(d["mean"][i:i + 1])) * 0.18215
        ref = (d["mean"][r:r + 1] + d["std"][r:r + 1] * torch.randn_like(d["mean"][r:r + 1])) * 0.18215
        return lat.unsqueeze(2), ref, d["clip"][r:r + 1], d["pose"][i:i + 1].to(device).float().div(255).unsqueeze(2), d, i

    # The smoke's check, before and after: x0 at PROBE_T with fixed noise on 4 fixed training frames, against the truth.
    probe_gen = random.Random(1)
    probe = [batch(probe_gen) for _ in range(4)]
    probe_noise = [torch.randn_like(p[0]) for p in probe]
    truths = {(p[4]["unit"], p[5]): np.load(unit_path[p[4]["unit"]])["image"][p[4]["rows"][p[5]]] for p in probe}

    def probe_error() -> float:
        errs = []
        with torch.inference_mode():
            for (lat, ref, clip, pose, d, i), noise in zip(probe, probe_noise):
                t = torch.full((1,), PROBE_T, device=device, dtype=torch.long)
                pred, _ = trainer.forward(lat, ref, clip, pose, t, noise, False)
                noisy = trainer.scheduler.add_noise(lat, noise, t)
                x0 = trainer.x0(pred, noisy, t)[:, :, 0] / 0.18215
                image = trainer.vae.decode(x0.float()).sample.clamp(-1, 1)
                truth = torch.from_numpy(truths[(d["unit"], i)]).to(device).permute(2, 0, 1)[None].float() / 127.5 - 1.0
                errs.append(float(((image - truth) ** 2).mean()))
        return float(np.mean(errs))

    def save(state: dict[str, Any], path: Path) -> dict[str, Any]:
        """The checkpoint rendering reads: quantized as billed (``--quantize``), else fp16; with ``--also-unquantized`` the
        fp16 adapters beside it (`-fp16`), the quantization check."""
        rec: dict[str, Any] = {}
        if args.also_unquantized:
            raw = path.with_name(path.stem + "-fp16.pt")
            torch.save({**state, **{k: {n: v.half() for n, v in state[k].items()} for k in ("reference_unet", "denoising_unet")}}, raw)
            rec["unquantized"] = {"path": raw.name, "sha256": sha256(raw)}
        if args.quantize:
            state, rec["billing"] = quantize_adapters(state, args.quantize, {k: v.cpu() for k, v in trainer.guider_base.items()})
        else:
            state = {**state, **{k: {n: v.half() for n, v in state[k].items()} for k in ("reference_unet", "denoising_unet")}}
        torch.save(state, path)
        return {"path": path.name, "sha256": sha256(path), **rec}

    error_before = probe_error()
    p0 = probe[0]
    _, kernels = profile_cuda(lambda: trainer.forward(p0[0], p0[1], p0[2], p0[3], torch.full((1,), PROBE_T, device=device, dtype=torch.long),
                                                      probe_noise[0], False))
    fractions = sorted(float(f) for f in args.fractions.split(","))
    budget = args.minutes * 60
    gen = random.Random(seed)
    losses: list[float] = []
    saved: list[dict[str, Any]] = []
    step = micro = 0
    fit_began = time.time()
    trainer.optimizer.zero_grad()
    while True:
        elapsed = time.time() - fit_began
        due = [f for f in fractions if f * budget <= elapsed and not any(s["fraction"] == f for s in saved)]
        for f in due:
            path = publish / f"{args.window_name}@fit-{int(round(f * 100))}.pt"
            saved.append({"fraction": f, "minutes": round(elapsed / 60, 2), "optimizer_steps": step, **save(trainer.state(), path)})
            print(f"phase=fit checkpoint={path.name} steps={step} minutes={elapsed / 60:.1f}", flush=True)
            publish_progress(os.environ.get("PS_STAGE", "local"), len(saved))
        if len(saved) == len(fractions) or (args.max_steps and step >= args.max_steps):
            break
        lat, ref, clip, pose, _, _ = batch(gen)
        noise = torch.randn_like(lat) + NOISE_OFFSET * torch.randn((lat.shape[0], lat.shape[1], 1, 1, 1), device=device)
        t = torch.randint(0, 1000, (lat.shape[0],), device=device).long()
        pred, target = trainer.forward(lat, ref, clip, pose, t, noise, random.random() < UNCOND_RATIO)
        loss = trainer.loss(pred, target, t) / ACCUMULATE
        loss.backward()
        losses.append(float(loss) * ACCUMULATE)
        micro += 1
        if micro % ACCUMULATE == 0:
            torch.nn.utils.clip_grad_norm_(trainer.trainable, MAX_GRAD_NORM)
            for group, base in zip(trainer.optimizer.param_groups, (LORA_LR, GUIDER_LR)):
                group["lr"] = base * min(1.0, (step + 1) / WARMUP)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"phase=fit step={step} loss={np.mean(losses[-50 * ACCUMULATE:]):.4f} minutes={(time.time() - fit_began) / 60:.1f}", flush=True)
    error_after = probe_error()
    if args.max_steps:  # the smoke: a final checkpoint whatever the time, for the render smoke
        path = publish / f"{args.window_name}@fit-smoke.pt"
        saved.append({"fraction": None, "optimizer_steps": step, **save(trainer.state(), path)})
    window = max(1, len(losses) // 10)
    write_json(stage_dir() / "k6b_fit.json", {
        "command": "fit", "stage": os.environ.get("PS_STAGE"), "window": args.window, "match": ",".join(args.match),
        "name": args.window_name, "all": args.all, "excluded": args.exclude_window, "windows": windows_used,
        "base": {"member": args.base, "sha256": sha256(base_path)} if args.base else None, "units": [d["unit"] for d in data],
        "frames": frames, "frame_stride": args.frame_stride, "max_t": args.max_t, "minutes": args.minutes, "max_steps": args.max_steps, "optimizer_steps": step,
        "micro_steps": micro, "checkpoints": saved, "loss_first_tenth": float(np.mean(losses[:window])) if losses else None,
        "loss_last_tenth": float(np.mean(losses[-window:])) if losses else None, "loss_curve": [float(np.mean(losses[i:i + 200])) for i in range(0, len(losses), 200)],
        "probe_mse_before": error_before, "probe_mse_after": error_after, "model": trainer.record, "kernels": kernels,
        "device": device_record(args.gpu_class), "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "gpu_memory_limit_mib": limit, "settings": {"lora_rank": args.rank, "lora_alpha": args.rank, "quantize": args.quantize,
        "pose_guider_frozen": args.freeze_guider, "lora_lr": LORA_LR,
        "guider_lr": GUIDER_LR, "batch": BATCH, "accumulate": ACCUMULATE, "uncond": UNCOND_RATIO, "noise_offset": NOISE_OFFSET,
        "snr_gamma": SNR_GAMMA, "scheduler": TRAIN_SCHEDULER, "size": [GEN_W, GEN_H]},
        "inputs": {"fit_data": [sha256(Path(t)) for t in args.fit_data], "weights": {k: sha256(v) for k, v in sorted(weights.items())}},
        "seconds": round(time.time() - began, 2)})
    with tarfile.open(stage_dir() / "published.tar", "w") as tar:
        tar.add(publish, arcname="publish")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    stage = Path(args.stage) if args.stage else stage_dir()
    r = json.loads((stage / "k6b_fit.json").read_text())
    checks = {
        "device_class": bool(r["device"]["class_matches"]),
        "peak_within_declaration": r["gpu_peak_gib"] * 1024 <= r["gpu_memory_limit_mib"],
        "no_missing_keys": all(r["model"][k]["missing"] == 0 for k in ("vae", "image_encoder", "reference_unet", "denoising_unet", "pose_guider")),
        "lora_injected": r["model"]["trainable"]["lora_tensors"] > 100,
        "kernels_on_gpu": r["kernels"]["kernel_launches"] > 0,
        "loss_falls": r["loss_last_tenth"] is not None and r["loss_last_tenth"] < r["loss_first_tenth"],
        "sample_moves_to_target": r["probe_mse_after"] < r["probe_mse_before"],
        "checkpoints": bool(r["checkpoints"]) and all(c["sha256"] for c in r["checkpoints"]),
        "finite": all(math.isfinite(x) for x in r["loss_curve"]),
    }
    if r.get("base"):  # step 5: the tennis base merged into both UNets before the match's LoRA
        checks["base_merged"] = all(r["model"]["base"][k]["merged_layers"] > 50 for k in ("reference_unet", "denoising_unet"))
    if not r["settings"]["pose_guider_frozen"] and r["settings"]["quantize"]:  # step 5: the trained guider billed
        checks["guider_billed"] = all(c["billing"]["pose_guider_billed"] and c["billing"]["pose_guider_bytes"] > 0
                                      for c in r["checkpoints"] if "billing" in c)
    report = {"passed": all(checks.values()), "checks": checks}
    if target := os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--fit-data", action="append", required=True, help="prepare-fit/prepare-match published.tar (repeatable)")
    fit.add_argument("--window", default="", help="step 3: fit on this window's frames")
    fit.add_argument("--match", action="append", default=[], help="step 4: fit on every training window of this match; "
                     "step 5: repeat for a player's matches")
    fit.add_argument("--all", action="store_true", help="step 5's tennis base: every window of every archive")
    fit.add_argument("--exclude-window", action="append", default=[], help="never fitted on (step 5: K6's windows)")
    fit.add_argument("--name", default="", help="the checkpoints' stem (default: from --window or --match)")
    fit.add_argument("--base-archive", default="", help="step 5: the tennis base fit's published.tar")
    fit.add_argument("--base", default="", help="step 5: the base checkpoint (a member of --base-archive), merged first")
    fit.add_argument("--rank", type=int, default=LORA_RANK)
    fit.add_argument("--freeze-guider", action="store_true", help="step 4: the pose guider stays the profile's (not sent)")
    fit.add_argument("--quantize", type=int, default=0, help="step 4: adapter bits as billed (8); 0 keeps fp16")
    fit.add_argument("--also-unquantized", action="store_true", help="also save the fp16 adapters (the quantization check)")
    fit.add_argument("--weight", action="append", nargs=2, required=True, metavar=("NAME", "PATH"))
    fit.add_argument("--minutes", type=float, required=True)
    fit.add_argument("--fractions", default="0.25,0.5,1")
    fit.add_argument("--max-t", type=int, default=0, help="train on frames t < this (0: all)")
    fit.add_argument("--max-units", type=int, default=0, help="the smoke: fit on this many units only (0: all)")
    fit.add_argument("--frame-stride", type=int, default=1, help="step 5's base: train on every n-th frame of each unit")
    fit.add_argument("--max-steps", type=int, default=0, help="the smoke: stop after this many optimizer steps")
    fit.add_argument("--gpu-class", default="RTX")
    val = sub.add_parser("validate")
    val.add_argument("--stage", default="")
    args = ap.parse_args(argv)
    if args.command == "fit":
        if sum(map(bool, (args.window, args.match, args.all))) != 1:
            raise SystemExit("give exactly one of --window, --match and --all")
        if bool(args.base) != bool(args.base_archive):
            raise SystemExit("--base and --base-archive go together")
        if args.name:
            args.window_name = args.name
        else:
            args.window_name = (args.window or "+".join(args.match) or "tennis").replace("/", "__").replace("@", "_at")
            if not args.window:  # step 4's pilot fits several ranks of one match: their checkpoints must not share names
                args.window_name += f"-r{args.rank}"
        return command_fit(args)
    return command_validate(args)


if __name__ == "__main__":
    raise SystemExit(main())

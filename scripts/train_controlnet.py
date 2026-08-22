import os
import argparse
import glob
import json
import logging
import random
import time
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from accelerate import Accelerator
from diffusers import (
    ControlNetModel,
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Derived directories and sidecars under a scene dir -- never training items.
DERIVED_SUFFIXES = ("_skeleton", "_canny", "_caption", "_pose_racket", "_pose_body")

def pad_to_square(img, fill=0):
    w, h = img.size
    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top = (max_dim - h) // 2
    pad_right = max_dim - w - pad_left
    pad_bottom = max_dim - h - pad_top
    import torchvision.transforms.functional as TF
    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill, padding_mode='constant')

class ControlNetDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        condition_type: str,
        target_size: int = 512,
        tokenizer=None,
        include_reference: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.condition_type = condition_type
        self.target_size = target_size
        self.tokenizer = tokenizer
        self.include_reference = include_reference
        self.items = []
        self.track_to_colors: dict[str, list[str]] = {}
        
        search_pattern = os.path.join(str(self.root_dir), "*", "segmentations", "scene_*", "track_*")
        all_tracks = glob.glob(search_pattern)
        
        for track_dir_str in all_tracks:
            # Skip derived directories and sidecar files -- only primary track
            # dirs are training items. Missing a suffix here makes the loader
            # treat e.g. `track_0036_pose_body` as a track and hunt for
            # `track_0036_pose_body_pose_body`.
            if track_dir_str.endswith(DERIVED_SUFFIXES) or not os.path.isdir(track_dir_str):
                continue

            track_dir = Path(track_dir_str)
            
            # Read caption if available, otherwise fallback
            caption_path = track_dir.parent / f"{track_dir.name}_caption.json"
            prompt = "photorealistic tennis player, broadcast sports shot"
            if caption_path.exists():
                with open(caption_path, "r") as f:
                    cdata = json.load(f)
                    prompt = cdata.get("caption", prompt)
                    
            color_frames = sorted(list(track_dir.glob("frame_*.png")))
            parts = track_dir.parts
            unique_track_id = f"{parts[-4]}_{parts[-2]}_{parts[-1]}"
            if self.include_reference and len(color_frames) < 2:
                continue

            # Identify the condition directory.
            # `pose` selects the body-only variant, which the decoder reproduces
            # for free and bit-identically; `pose-racket` matches the legacy
            # checkpoints but needs racket geometry the wire format does not
            # carry. The legacy `_skeleton` tree is NOT selectable here -- its
            # filenames are positional and pairing it wrecked this dataset.
            if self.condition_type == "pose":
                cond_dir = track_dir.with_name(f"{track_dir.name}_pose_body")
            elif self.condition_type == "pose-racket":
                cond_dir = track_dir.with_name(f"{track_dir.name}_pose_racket")
            elif self.condition_type == "canny":
                cond_dir = track_dir.with_name(f"{track_dir.name}_canny")
            elif self.condition_type in ["seg", "ip-adapter"]:
                cond_dir = None
            else:
                raise ValueError(f"Unknown condition type: {self.condition_type}")

            # Pair POSITIONALLY, matching src/shared/tennis_dataset.py.
            # Pairing by filename silently produced garbage: `_skeleton` frames
            # were named by position while colour frames carry the absolute
            # source frame id, so across the training view 32.7% of items were
            # paired with the WRONG pose and 22.9% were dropped without a word.
            # A count mismatch now raises instead of quietly shrinking the
            # dataset -- a training set that silently loses a quarter of its
            # data is indistinguishable from one that trained fine.
            if cond_dir is None:
                for color_path in color_frames:
                    self.items.append({
                        "image_path": str(color_path),
                        "cond_path": None,
                        "prompt": prompt,
                        "track_id": unique_track_id,
                    })
                    self.track_to_colors.setdefault(unique_track_id, []).append(str(color_path))
                continue

            if not cond_dir.exists():
                raise FileNotFoundError(
                    f"Condition directory missing for {track_dir}: {cond_dir}. "
                    f"Run scripts/process_dataset.py's '{self.condition_type}' stage first."
                )

            cond_frames = sorted(cond_dir.glob("frame_*.png"))
            if len(cond_frames) != len(color_frames):
                raise ValueError(
                    f"Frame-count mismatch for {track_dir.name}: {len(color_frames)} colour frames "
                    f"vs {len(cond_frames)} '{self.condition_type}' frames in {cond_dir.name}. "
                    "Colour and condition sequences must correspond one-to-one; regenerate the "
                    "condition stage rather than training on a partial pairing."
                )

            for color_path, cond_path in zip(color_frames, cond_frames):
                self.items.append({
                    "image_path": str(color_path),
                    "cond_path": str(cond_path),
                    "prompt": prompt,
                    "track_id": unique_track_id,
                })
                self.track_to_colors.setdefault(unique_track_id, []).append(str(color_path))

        self.transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.cond_transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        img = Image.open(item["image_path"])
        
        seg_mask = None
        if self.condition_type == "seg":
            if img.mode == 'RGBA':
                alpha = img.split()[-1]
                seg_mask = Image.merge("RGB", (alpha, alpha, alpha))
            else:
                seg_mask = Image.new("RGB", img.size, (255, 255, 255))
        
        if img.mode == 'RGBA':
            background = Image.new('RGBA', img.size, (0, 0, 0, 255))
            img = Image.alpha_composite(background, img).convert("RGB")
        else:
            img = img.convert("RGB")
            
        img = pad_to_square(img, fill=0)
        image_tensor = self.transform(img)
        
        if self.condition_type in ("pose", "pose-racket", "canny"):
            cond_img = Image.open(item["cond_path"]).convert("RGB")
            cond_img = pad_to_square(cond_img, fill=0)
            cond_tensor = self.cond_transform(cond_img)
        elif self.condition_type == "seg":
            cond_img = pad_to_square(seg_mask, fill=0)
            cond_tensor = self.cond_transform(cond_img)
        elif self.condition_type == "ip-adapter":
            cond_tensor = self.cond_transform(img)
        else:
            raise ValueError(f"Unknown condition type: {self.condition_type}")
            
        tokens = self.tokenizer(
            item["prompt"], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.squeeze(0)
        
        sample = {
            "pixel_values": image_tensor,
            "conditioning_pixel_values": cond_tensor,
            "input_ids": tokens
        }
        if self.include_reference:
            track_id = item["track_id"]
            colors = self.track_to_colors[track_id]
            candidates = [path for path in colors if path != item["image_path"]]
            ref_path = random.choice(candidates or colors)
            ref_img = Image.open(ref_path)
            if ref_img.mode == "RGBA":
                background = Image.new("RGBA", ref_img.size, (0, 0, 0, 255))
                ref_img = Image.alpha_composite(background, ref_img).convert("RGB")
            else:
                ref_img = ref_img.convert("RGB")
            ref_img = pad_to_square(ref_img, fill=0)
            sample["reference_pixel_values"] = self.cond_transform(ref_img)
        return sample


def compose_pose_on_appearance_tensor(
    pose: torch.Tensor, appearance: torch.Tensor, *, threshold: float = 8.0 / 255.0
) -> torch.Tensor:
    """Tensor twin of ``compose_pose_on_appearance``. CHW, values in [0, 1]."""
    mask = pose.amax(dim=1, keepdim=True) > threshold
    return torch.where(mask, pose, appearance)


def assert_reference_enters_controlnet(
    controlnet, batch, *, weight_dtype, timesteps, noisy_latents, encoder_hidden_states
) -> float:
    """Drive two forwards that differ only in the reference; residuals must move."""
    pose = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
    reference = batch["reference_pixel_values"].to(dtype=weight_dtype)
    composed = compose_pose_on_appearance_tensor(pose, reference)
    blank_ref = torch.zeros_like(reference)
    composed_blank = compose_pose_on_appearance_tensor(pose, blank_ref)
    if torch.equal(composed, composed_blank):
        raise RuntimeError(
            "composed control equals pose-on-black; the reference never reached the canvas."
        )
    with torch.no_grad():
        _, mid_ref = controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=composed,
            return_dict=False,
        )
        _, mid_blank = controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=composed_blank,
            return_dict=False,
        )
    delta = float((mid_ref.float() - mid_blank.float()).abs().mean().cpu())
    logging.info("reference-enters-controlnet residual_delta=%.6g", delta)
    if delta <= 1e-6:
        raise RuntimeError(
            f"ControlNet residuals did not change when the reference was zeroed "
            f"(delta={delta}). Appearance is not an input."
        )
    return delta


def main():
    parser = argparse.ArgumentParser(description="Train ControlNet")
    parser.add_argument("--data-root", type=str, default="assets/dataset", help="Dataset root")
    parser.add_argument("--condition-type", type=str,
                        choices=["pose", "pose-racket", "canny", "seg", "ip-adapter"], required=True,
                        help="pose = body-only skeleton (decoder-reproducible for free); "
                             "pose-racket = body + racket (needs racket geometry in ActorPacket)")
    parser.add_argument("--model-id", type=str, default="assets/weights/stable-diffusion-v1-5")
    parser.add_argument("--controlnet-model-id", type=str, default=None, help="Path to pre-trained ControlNet to fine-tune")
    parser.add_argument("--from-scratch", action="store_true", help="Initialize ControlNet from scratch (no fine-tuning)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=str, default="assets/weights/custom-controlnet")
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Sample a same-track reference frame and paint it under the pose (BP8).",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps.")
    parser.add_argument(
        "--checkpoint-every-minutes",
        type=int,
        default=60,
        help="Wall-clock checkpoint interval. Long jobs must checkpoint at least hourly.",
    )
    parser.add_argument(
        "--progress-every-minutes",
        type=int,
        default=10,
        help="Append a progress line at least this often so a hang is visible.",
    )
    parser.add_argument(
        "--smoke-check-reference",
        action="store_true",
        help="Before training, drive two ControlNet forwards that differ only in the reference.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if args.include_reference and args.controlnet_model_id is None and not args.from_scratch:
        args.controlnet_model_id = "assets/weights/pose-controlnet/checkpoint-epoch-10"
        logging.info(
            "include-reference defaults to fine-tuning the tennis pose ControlNet at %s",
            args.controlnet_model_id,
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
    )

    logging.info(f"Loading models from {args.model_id}...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    if args.from_scratch:
        logging.info("Initializing ControlNet from scratch using UNet config.")
        controlnet = ControlNetModel.from_unet(unet)
    elif args.controlnet_model_id:
        logging.info(f"Loading pre-trained ControlNet from {args.controlnet_model_id} for fine-tuning.")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_id)
    else:
        defaults = {
            "pose": "assets/weights/control_v11p_sd15_openpose",
            "pose-racket": "assets/weights/control_v11p_sd15_openpose",
            "canny": "lllyasviel/control_v11p_sd15_canny",
            "seg": "lllyasviel/control_v11p_sd15_seg",
            "ip-adapter": "assets/weights/control_v11p_sd15_openpose"
        }
        cnet_id = defaults[args.condition_type]
        logging.info(f"Loading default pre-trained ControlNet {cnet_id} for fine-tuning.")
        controlnet = ControlNetModel.from_pretrained(cnet_id)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

    dataset = ControlNetDataset(
        args.data_root,
        args.condition_type,
        tokenizer=tokenizer,
        include_reference=args.include_reference,
    )
    if args.include_reference and len(dataset) == 0:
        raise RuntimeError("include-reference yielded an empty dataset; tracks need at least 2 colour frames.")
    logging.info("Dataset size %s include_reference=%s", len(dataset), args.include_reference)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    controlnet, optimizer, dataloader = accelerator.prepare(
        controlnet, optimizer, dataloader
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    if args.include_reference and args.controlnet_model_id is None and not args.from_scratch:
        logging.info(
            "include-reference fine-tunes the pose ControlNet with appearance under the skeleton. "
            "Pass --controlnet-model-id to choose weights; default OpenPose is used otherwise."
        )

    global_step = 0
    last_ckpt_wall = time.monotonic()
    last_progress_wall = time.monotonic()
    started_wall = time.monotonic()

    def _save(tag: str) -> None:
        if not accelerator.is_main_process:
            return
        dest = os.path.join(args.output_dir, tag)
        logging.info("Saving ControlNet checkpoint to %s", dest)
        accelerator.unwrap_model(controlnet).save_pretrained(dest)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(tqdm(dataloader, disable=not accelerator.is_local_main_process)):
            with accelerator.accumulate(controlnet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                if args.include_reference:
                    controlnet_image = compose_pose_on_appearance_tensor(
                        controlnet_image,
                        batch["reference_pixel_values"].to(dtype=weight_dtype),
                    )

                if args.smoke_check_reference and global_step == 0:
                    if not args.include_reference:
                        raise ValueError("--smoke-check-reference requires --include-reference")
                    delta = assert_reference_enters_controlnet(
                        controlnet,
                        batch,
                        weight_dtype=weight_dtype,
                        timesteps=timesteps,
                        noisy_latents=noisy_latents,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                    logging.info("smoke-check-reference passed residual_delta=%.6g", delta)
                    if accelerator.is_main_process:
                        dest = Path(args.output_dir)
                        dest.mkdir(parents=True, exist_ok=True)
                        pose_cpu = batch["conditioning_pixel_values"].detach().float().cpu()
                        ref_cpu = batch["reference_pixel_values"].detach().float().cpu()
                        composed_cpu = compose_pose_on_appearance_tensor(pose_cpu, ref_cpu)

                        def _chw_to_png(tensor: torch.Tensor, path: Path) -> None:
                            array = (
                                tensor[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
                            )
                            Image.fromarray(array).save(path)

                        _chw_to_png(pose_cpu, dest / "smoke-pose.png")
                        _chw_to_png(ref_cpu, dest / "smoke-reference.png")
                        _chw_to_png(composed_cpu, dest / "smoke-composed.png")
                        (dest / "smoke-check.json").write_text(
                            json.dumps({"residual_delta": delta, "passed": True}, indent=2)
                            + "\n"
                        )
                    if args.max_steps is None:
                        accelerator.wait_for_everyone()
                        return

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            now = time.monotonic()
            if accelerator.is_local_main_process and now - last_progress_wall >= args.progress_every_minutes * 60:
                logging.info(
                    "progress step=%s epoch=%s/%s loss=%.5f elapsed_min=%.1f",
                    global_step,
                    epoch + 1,
                    args.epochs,
                    float(loss.detach().float().cpu()),
                    (now - started_wall) / 60.0,
                )
                last_progress_wall = now
            if accelerator.is_main_process and now - last_ckpt_wall >= args.checkpoint_every_minutes * 60:
                accelerator.wait_for_everyone()
                _save(f"checkpoint-step-{global_step}")
                last_ckpt_wall = now
            if args.max_steps is not None and global_step >= args.max_steps:
                logging.info("Reached --max-steps %s", args.max_steps)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _save(f"checkpoint-step-{global_step}")
                return

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _save(f"checkpoint-epoch-{epoch+1}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logging.info(f"Saving ControlNet to {args.output_dir}")
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

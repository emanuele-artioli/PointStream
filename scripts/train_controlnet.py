import os
import argparse
import glob
import json
import logging
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
    def __init__(self, root_dir: str, condition_type: str, target_size: int = 512, tokenizer=None):
        self.root_dir = Path(root_dir)
        self.condition_type = condition_type
        self.target_size = target_size
        self.tokenizer = tokenizer
        
        self.items = []
        
        search_pattern = os.path.join(str(self.root_dir), "*", "segmentations", "scene_*", "track_*")
        all_tracks = glob.glob(search_pattern)
        
        for track_dir_str in all_tracks:
            # Skip non-primary track folders
            if track_dir_str.endswith("_skeleton") or track_dir_str.endswith("_canny") or track_dir_str.endswith("_caption"):
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

            # Identify the condition directory
            if self.condition_type == "pose":
                cond_dir = track_dir.with_name(f"{track_dir.name}_skeleton")
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
                    self.items.append({"image_path": str(color_path), "cond_path": None, "prompt": prompt})
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
                    "prompt": prompt
                })

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
        
        if self.condition_type == "pose" or self.condition_type == "canny":
            cond_img = Image.open(item["cond_path"]).convert("RGB")
            cond_img = pad_to_square(cond_img, fill=0)
            cond_tensor = self.cond_transform(cond_img)
        elif self.condition_type == "seg":
            cond_img = pad_to_square(seg_mask, fill=0)
            cond_tensor = self.cond_transform(cond_img)
        elif self.condition_type == "ip-adapter":
            cond_tensor = self.cond_transform(img)
            
        tokens = self.tokenizer(
            item["prompt"], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.squeeze(0)
        
        return {
            "pixel_values": image_tensor,
            "conditioning_pixel_values": cond_tensor,
            "input_ids": tokens
        }

def main():
    parser = argparse.ArgumentParser(description="Train ControlNet")
    parser.add_argument("--data-root", type=str, default="assets/dataset", help="Dataset root")
    parser.add_argument("--condition-type", type=str, choices=["pose", "canny", "seg", "ip-adapter"], required=True)
    parser.add_argument("--model-id", type=str, default="assets/weights/stable-diffusion-v1-5")
    parser.add_argument("--controlnet-model-id", type=str, default=None, help="Path to pre-trained ControlNet to fine-tune")
    parser.add_argument("--from-scratch", action="store_true", help="Initialize ControlNet from scratch (no fine-tuning)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=str, default="assets/weights/custom-controlnet")
    args = parser.parse_args()

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

    dataset = ControlNetDataset(args.data_root, args.condition_type, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

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

    global_step = 0
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

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            epoch_output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            logging.info(f"Saving ControlNet checkpoint to {epoch_output_dir}")
            controlnet_unwrapped = accelerator.unwrap_model(controlnet)
            controlnet_unwrapped.save_pretrained(epoch_output_dir)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logging.info(f"Saving ControlNet to {args.output_dir}")
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()

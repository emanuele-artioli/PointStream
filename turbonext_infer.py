import argparse
from pathlib import Path
from typing import List

import torch
from diffusers import LCMScheduler
from PIL import Image
from torchvision import transforms

from turbonext_model import TurboNextConfig, TurboNextModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Turbo-NeXt realtime inference")
    parser.add_argument("--pose_dir", type=str, required=True)
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/home/itec/emanuele/pointstream/experiments/turbonext_outputs")
    parser.add_argument("--pose_guider_ckpt", type=str, default="/home/itec/emanuele/pointstream/experiments/turbonext/pose_guider_epoch_0.pt", help="Path to trained pose guider checkpoint")
    parser.add_argument("--lcm_lora", type=str, default="latent-consistency/lcm-lora-sdv1-5")
    parser.add_argument("--tiny_vae", type=str, default="madebyollin/taesd")
    parser.add_argument("--buffer_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", type=str, default="")
    return parser.parse_args()


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    config = TurboNextConfig()
    model = TurboNextModel(config)
    
    # Load trained pose guider checkpoint if provided
    if args.pose_guider_ckpt:
        checkpoint = torch.load(args.pose_guider_ckpt, map_location="cpu")
        model.pose_guider.load_state_dict(checkpoint)
        print(f"Loaded pose guider from {args.pose_guider_ckpt}")
    
    model.to(device)
    model.eval()

    scheduler = LCMScheduler.from_config(model.unet.config)
    scheduler.set_timesteps(args.steps, device=device)

    # Try to load TinyVAE for faster decoding
    try:
        from diffusers import AutoencoderTiny
        model.vae = AutoencoderTiny.from_pretrained(args.tiny_vae).to(device)
        print("Using TinyVAE decoder")
    except Exception as e:
        print(f"Could not load TinyVAE: {e}. Using standard VAE.")

    # Try to load LCM-LoRA for faster inference
    if args.lcm_lora:
        try:
            model.unet.load_attn_procs(args.lcm_lora)
            print("LCM-LoRA loaded successfully")
        except Exception as e:
            print(f"Could not load LCM-LoRA: {e}. Proceeding without it.")

    pose_paths = sorted(Path(args.pose_dir).glob("*.jpg")) + sorted(Path(args.pose_dir).glob("*.png"))
    if not pose_paths:
        raise ValueError(f"No pose images found in {args.pose_dir}")

    ref_image = load_image(Path(args.ref_image))

    image_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    pose_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )

    ref_tensor = image_transform(ref_image).unsqueeze(0).to(device)
    prompt_embed = model.encode_prompt([args.prompt], device)
    model.cache_reference(ref_tensor, scheduler.timesteps[:1], [args.prompt])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process frames individually (no FIFO buffer for standard UNet)
    # The FIFO/StreamDiffusion approach requires AnimateDiff motion modules
    with torch.no_grad():
        for idx, pose_path in enumerate(pose_paths):
            pose_image = pose_transform(load_image(pose_path)).unsqueeze(0).to(device)

            # Start from noise, add pose conditioning
            pose_cond = model.pose_guider(pose_image)
            latent = torch.randn(1, 4, 64, 64, device=device)
            latent = latent + pose_cond

            # Denoise step by step
            for t in scheduler.timesteps:
                model.reference_store.reset_index()
                model_output = model.unet(
                    latent,
                    t,
                    encoder_hidden_states=prompt_embed,
                    return_dict=False,
                )[0]
                latent = scheduler.step(model_output, t, latent, return_dict=False)[0]

            # Decode latent to image
            # TinyVAE returns output without .sample attribute
            vae_output = model.vae.decode(latent / 0.18215)
            if hasattr(vae_output, 'sample'):
                decoded = vae_output.sample
            else:
                decoded = vae_output
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            output_image = transforms.ToPILImage()(decoded[0].cpu())
            output_image.save(output_dir / f"frame_{idx:05d}.png")
            
            if idx % 10 == 0:
                print(f"Processed frame {idx}/{len(pose_paths)}")


if __name__ == "__main__":
    main()

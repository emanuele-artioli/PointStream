import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from turbonext_dataset import VideoPoseDataset
from turbonext_model import TurboNextConfig, TurboNextModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Turbo-NeXt pose transfer model")
    parser.add_argument("--dataset_root", type=str, default="/home/itec/emanuele/pointstream/experiments/dataset")
    parser.add_argument("--output_dir", type=str, default="/home/itec/emanuele/pointstream/experiments/turbonext")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=1000, help="Max number of training samples to use (None for all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    device = accelerator.device

    dataset = VideoPoseDataset(args.dataset_root)
    if len(dataset) == 0:
        raise ValueError(f"No training samples found in {args.dataset_root}")
    
    if args.max_train_samples is not None:
        dataset.samples = dataset.samples[:args.max_train_samples]
    
    # Limit dataset size if requested
    if args.max_train_samples is not None and args.max_train_samples < len(dataset):
        dataset.samples = dataset.samples[:args.max_train_samples]

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    config = TurboNextConfig(dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32)
    model = TurboNextModel(config)
    model.to(device)

    trainable_params = list(model.pose_guider.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    model.train()
    for epoch in range(args.num_epochs):
        progress = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch in progress:
            pixel_values = batch["pixel_values"].to(device)
            pose_images = batch["pose_images"].to(device)
            ref_image = batch["ref_image"].to(device)
            
            latents = model.encode_image(pixel_values)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=device).long()

            with accelerator.accumulate(model):
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_pred = model(
                    pixel_values=pixel_values,
                    pose_images=pose_images,
                    ref_image=ref_image,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                )

                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            progress.set_description(f"epoch {epoch} loss {loss.item():.4f}")

        if accelerator.is_local_main_process:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.pose_guider.state_dict(), output_dir / f"pose_guider_epoch_{epoch}.pt")


if __name__ == "__main__":
    main()

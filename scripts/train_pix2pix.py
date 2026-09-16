import sqlite3  # noqa: F401
import argparse
import logging
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
import torchvision.utils as vutils
from tqdm import tqdm

from src.shared.tennis_dataset import TennisSkeletonDataset

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=9):
        super().__init__()
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def save_checkpoint_atomic(state: dict[str, Any] | Any, target_path: str | Path) -> None:
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


class ReconstructibleEpochSampler(Sampler):
    """Deterministic epoch-seeded sampler supporting partial-epoch cursor resumption."""

    def __init__(
        self,
        data_source: Any,
        batch_size: int,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = max(1, batch_size)
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.epoch = 0
        self.start_step = 0

    def set_epoch(self, epoch: int, start_step: int = 0) -> None:
        self.epoch = epoch
        self.start_step = start_step

    def __iter__(self):
        n = len(self.data_source)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        if self.num_replicas > 1:
            total_size = (
                (n // self.num_replicas) * self.num_replicas
                if self.drop_last
                else math.ceil(n / self.num_replicas) * self.num_replicas
            )
            indices = indices[:total_size]
            indices = indices[self.rank : total_size : self.num_replicas]

        n_batches = len(indices) // self.batch_size
        if self.drop_last:
            indices = indices[: n_batches * self.batch_size]

        if self.start_step > 0:
            start_idx = self.start_step * self.batch_size
            indices = indices[start_idx:]

        return iter(indices)

    def __len__(self) -> int:
        n = len(self.data_source)
        if self.num_replicas > 1:
            n = (
                n // self.num_replicas
                if self.drop_last
                else math.ceil(n / self.num_replicas)
            )
        n_batches = n // self.batch_size
        if self.start_step > 0:
            return max(0, (n_batches - self.start_step) * self.batch_size)
        return n_batches * self.batch_size if self.drop_last else n


def build_checkpoint_state(
    epoch: int,
    step: int,
    total_steps: int,
    generator: nn.Module,
    discriminator: nn.Module | None = None,
    optimizer_G: optim.Optimizer | None = None,
    optimizer_D: optim.Optimizer | None = None,
    ngpus: int = 1,
    base_seed: int = 42,
) -> dict[str, Any]:
    state_dict_G = (
        generator.module.state_dict() if ngpus > 1 else generator.state_dict()
    ) if generator is not None else None
    state_dict_D = (
        discriminator.module.state_dict() if ngpus > 1 else discriminator.state_dict()
    ) if discriminator is not None else None
    rng_cuda = torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None
    rng_torch = torch.get_rng_state().cpu()
    return {
        "epoch": epoch,
        "step": step,
        "total_steps_in_epoch": total_steps,
        "base_seed": base_seed,
        "G": state_dict_G,
        "D": state_dict_D,
        "opt_G": optimizer_G.state_dict() if optimizer_G is not None else None,
        "opt_D": optimizer_D.state_dict() if optimizer_D is not None else None,
        "rng_torch": rng_torch,
        "rng_cuda": rng_cuda,
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
        "timestamp_unix": time.time(),
    }


def main_worker(gpu, ngpus_per_node, args):
    is_main_process = (gpu == 0)
    
    if ngpus_per_node > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:23456',
            world_size=ngpus_per_node,
            rank=gpu
        )
    
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    base_seed = getattr(args, "seed", 42)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)
    np.random.seed(base_seed)
    random.seed(base_seed)

    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    start_epoch = 0
    start_step = 0
    checkpoint = None
    if args.resume and os.path.exists(args.checkpoint_path):
        if is_main_process:
            logging.info(f"Resuming from checkpoint {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint["G"])
        discriminator.load_state_dict(checkpoint["D"])
        saved_step = checkpoint.get("step", -1)
        total_steps = checkpoint.get("total_steps_in_epoch", -1)
        if "base_seed" in checkpoint:
            base_seed = int(checkpoint["base_seed"])
        if saved_step >= 0 and total_steps > 0 and saved_step < total_steps - 1:
            start_epoch = checkpoint["epoch"]
            start_step = saved_step + 1
        else:
            start_epoch = checkpoint["epoch"] + 1
            start_step = 0

        # Selected-device resume requires CPU ByteTensors for torch and cuda RNG states
        if "rng_torch" in checkpoint and checkpoint["rng_torch"] is not None:
            t_rng = checkpoint["rng_torch"]
            if isinstance(t_rng, torch.Tensor):
                t_rng = t_rng.cpu()
            torch.set_rng_state(t_rng)
        if "rng_cuda" in checkpoint and checkpoint["rng_cuda"] is not None and torch.cuda.is_available():
            c_rng = checkpoint["rng_cuda"]
            if isinstance(c_rng, torch.Tensor):
                c_rng = c_rng.cpu()
            torch.cuda.set_rng_state(c_rng)
        if "rng_numpy" in checkpoint and checkpoint["rng_numpy"] is not None:
            np.random.set_state(checkpoint["rng_numpy"])
        if "rng_python" in checkpoint and checkpoint["rng_python"] is not None:
            random.setstate(checkpoint["rng_python"])
    else:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Wrap models with DDP
    if ngpus_per_node > 1:
        generator = DDP(generator, device_ids=[gpu])
        discriminator = DDP(discriminator, device_ids=[gpu])

    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    if checkpoint is not None:
        if "opt_G" in checkpoint:
            optimizer_G.load_state_dict(checkpoint["opt_G"])
        if "opt_D" in checkpoint:
            optimizer_D.load_state_dict(checkpoint["opt_D"])

    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    criterion_pixelwise = nn.L1Loss().to(device)

    ref_mode = getattr(args, "reference_mode", "deterministic")
    dataset = TennisSkeletonDataset(
        args.data_root,
        target_size=args.img_size,
        include_reference=True,
        condition=args.condition,
        reference_mode=ref_mode,
    )

    if args.resume and getattr(args, "num_workers", 0) > 0:
        logging.warning(
            "Worker mode notice: num_workers=%d > 0 requested with resume. "
            "Multi-process async worker DataLoader prefetching cannot guarantee per-worker RNG restoration across process restarts. "
            "Constrain to num_workers=0 for verified bit-identical fresh-process continuation.",
            args.num_workers,
        )

    # Resolve batch size, defaulting to 64 if 'auto' was passed to better utilize 48GB VRAM
    batch_size = 64 if str(args.batch_size).lower() == "auto" else int(args.batch_size)

    sampler = ReconstructibleEpochSampler(
        dataset,
        batch_size=batch_size,
        seed=base_seed,
        shuffle=True,
        drop_last=True,
        num_replicas=ngpus_per_node,
        rank=gpu,
    )

    dl_gen = torch.Generator()
    dl_gen.manual_seed(base_seed + start_epoch)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=(torch.cuda.is_available() and device.type == "cuda"),
        drop_last=False,
        sampler=sampler,
        generator=dl_gen,
        persistent_workers=(args.num_workers > 0)
    )

    total_batches = len(dataset) // batch_size
    if ngpus_per_node > 1:
        total_batches = (len(dataset) // ngpus_per_node) // batch_size

    if is_main_process:
        logging.info(f"Starting training on {len(dataset)} images for {args.epochs} epochs with batch size {batch_size} (per GPU)")

    if total_batches == 0:
        if is_main_process:
            logging.error(f"Dataloader is empty! Dataset size ({len(dataset)}) is too small for batch size {batch_size} across {ngpus_per_node} GPUs with drop_last=True.")
        return

    last_checkpoint_time = time.time()
    last_progress_time = time.time()
    ckpt_interval = getattr(args, "checkpoint_interval_sec", 3600.0)

    for epoch in range(start_epoch, args.epochs):
        current_start_step = start_step if epoch == start_epoch else 0
        sampler.set_epoch(epoch, start_step=current_start_step)

        if is_main_process:
            pbar = tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                initial=current_start_step,
                desc=f"Epoch {epoch}/{args.epochs}",
            )
            iterator = pbar
        else:
            iterator = enumerate(dataloader)

        max_steps = getattr(args, "max_steps_per_epoch", None)
        last_step_executed = None
        for batch_offset, (real_A, ref_img, real_B) in iterator:
            i = current_start_step + batch_offset
            if max_steps is not None and i >= max_steps:
                break
            last_step_executed = i

            real_A = real_A.to(device)
            ref_img = ref_img.to(device)
            real_B = real_B.to(device)

            valid = torch.ones((real_A.size(0), 1, real_A.size(2) // 16, real_A.size(3) // 16), device=device, requires_grad=False)
            fake = torch.zeros((real_A.size(0), 1, real_A.size(2) // 16, real_A.size(3) // 16), device=device, requires_grad=False)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Generate fake image conditioned on skeleton and reference image
            gen_input = torch.cat((real_A, ref_img), 1)
            fake_B = generator(gen_input)
            
            # In DDP, calling a module multiple times before backward() can cause hook/graph collisions.
            # Combine real and fake into a single forward pass:
            combined_A = torch.cat((gen_input, gen_input), 0)
            combined_B = torch.cat((real_B, fake_B.detach()), 0)
            pred_combined = discriminator(combined_A, combined_B)
            
            pred_real, pred_fake_detached = torch.chunk(pred_combined, 2, dim=0)
            
            # Real loss
            loss_real = criterion_GAN(pred_real, valid)
            
            # Fake loss
            loss_fake = criterion_GAN(pred_fake_detached, fake)
            
            # Total D loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # G needs to fool D, so we evaluate D(fake_B) with the UPDATED discriminator weights
            pred_fake = discriminator(gen_input, fake_B)
            loss_GAN = criterion_GAN(pred_fake, valid)
            
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            
            # Total G loss
            loss_G = loss_GAN + args.lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            last_real_A = real_A
            last_ref_img = ref_img
            last_real_B = real_B
            last_fake_B = fake_B

            if is_main_process:
                pbar.set_postfix({"D_loss": f"{loss_D.item():.4f}", "G_loss": f"{loss_G.item():.4f}"})
                now = time.time()
                if now - last_progress_time >= 600.0:
                    logging.info(f"Progress heartbeat: epoch {epoch}/{args.epochs}, step {i}/{total_batches}")
                    last_progress_time = now

                # Intra-epoch hourly/periodic wall-clock checkpoint deadline check
                if now - last_checkpoint_time >= ckpt_interval:
                    state_dict_G = generator.module.state_dict() if ngpus_per_node > 1 else generator.state_dict()
                    save_checkpoint_atomic(state_dict_G, args.out_weights)
                    ckpt_state = build_checkpoint_state(
                        epoch=epoch,
                        step=i,
                        total_steps=total_batches,
                        generator=generator,
                        discriminator=discriminator,
                        optimizer_G=optimizer_G,
                        optimizer_D=optimizer_D,
                        ngpus=ngpus_per_node,
                        base_seed=base_seed,
                    )
                    save_checkpoint_atomic(ckpt_state, args.checkpoint_path)
                    last_checkpoint_time = now
                    logging.info(f"Intra-epoch checkpoint saved at epoch {epoch}, step {i}/{total_batches}")

        start_step = 0

        if is_main_process and "last_fake_B" in locals():
            sample_img = torch.cat(
                (
                    last_real_A[:4].detach().cpu(),
                    last_ref_img[:4].detach().cpu(),
                    last_real_B[:4].detach().cpu(),
                    last_fake_B[:4].detach().cpu(),
                ),
                -1,
            )
            vutils.save_image(sample_img, f"{args.sample_dir}/epoch_{epoch:03d}.png", nrow=4, normalize=True)

            now = time.time()
            epoch_completed = (last_step_executed is not None and last_step_executed == total_batches - 1)
            step_to_save = (total_batches - 1) if epoch_completed else (last_step_executed if last_step_executed is not None else 0)
            # Wall-clock checkpointing (or every 10 epochs or final epoch)
            if (epoch + 1) % 10 == 0 or (now - last_checkpoint_time >= ckpt_interval) or (epoch + 1 == args.epochs):
                state_dict_G = generator.module.state_dict() if ngpus_per_node > 1 else generator.state_dict()
                save_checkpoint_atomic(state_dict_G, args.out_weights)
                ckpt_state = build_checkpoint_state(
                    epoch=epoch,
                    step=step_to_save,
                    total_steps=total_batches,
                    generator=generator,
                    discriminator=discriminator,
                    optimizer_G=optimizer_G,
                    optimizer_D=optimizer_D,
                    ngpus=ngpus_per_node,
                    base_seed=base_seed,
                )
                save_checkpoint_atomic(ckpt_state, args.checkpoint_path)
                last_checkpoint_time = now

    if is_main_process:
        state_dict_G = generator.module.state_dict() if ngpus_per_node > 1 else generator.state_dict()
        save_checkpoint_atomic(state_dict_G, args.out_weights)
        logging.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train Pix2Pix for Pointstream GenAI Backend")
    parser.add_argument("--data-root", type=str, default="assets/dataset", help="Path to dataset root")
    parser.add_argument("--condition", type=str, default="pose_body",
                        choices=["pose_body", "pose_racket", "skeleton"],
                        help="Pose-condition variant. pose_body is reproducible by the decoder for free "
                             "and bit-identically; pose_racket matches the legacy checkpoints but needs "
                             "racket geometry the wire format does not carry; skeleton is the legacy tree "
                             "(positional filenames) kept only to reproduce old runs.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=str, default="auto", help="Batch size per GPU ('auto' maps to 64)")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 4), help="Number of CPU workers for data loading")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--lambda-pixel", type=float, default=100)
    parser.add_argument("--out-weights", type=str, default="assets/weights/pix2pix_generator.pt")
    parser.add_argument("--checkpoint-path", type=str, default="assets/weights/pix2pix_checkpoint.pt")
    parser.add_argument("--checkpoint-interval-sec", type=float, default=3600.0,
                        help="Periodic checkpoint wall-clock interval in seconds (default: 3600.0)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--sample-dir", type=str, default="assets/samples")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--reference-mode", type=str, default="deterministic",
                        choices=["deterministic", "first", "random"],
                        help="Reference selection mode for TennisSkeletonDataset (default: deterministic)")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None,
                        help="Optional cap on steps per epoch for fast integration tests")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_weights), exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        logging.info(f"Using {ngpus_per_node} GPUs with DistributedDataParallel (DDP)!")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(0, 1, args)

if __name__ == "__main__":
    main()

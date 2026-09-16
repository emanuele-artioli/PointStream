"""Spade4Tennis: Reference-SPADE Player+Racket Synthesis – Training Script.

Architecture
============
* Generator : SPADE-conditioned ResNet-9 (lite) or UNet+LocalEnhancer (full)
* Discriminator : Multi-scale PatchGAN with spectral normalisation
* Losses : Hinge GAN + VGG-19 perceptual + Feature matching + L1 (λ=10)

Usage
-----
    conda run -n pointstream python scripts/train_spade4tennis.py --model-size lite --epochs 200
    conda run -n pointstream python scripts/train_spade4tennis.py --model-size full --epochs 200 \\
        --pretrained-g assets/weights/spade4tennis_lite_generator.pt
"""
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
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
import torchvision.utils as vutils
from tqdm import tqdm

from src.components.generation.spade4tennis_arch import SPADEResNet9Generator
from src.shared.tennis_dataset import TennisSkeletonDataset

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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


# ---------------------------------------------------------------------------
# Multi-Scale Discriminator
# ---------------------------------------------------------------------------

class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral normalisation.

    Returns a list of intermediate features (for feature matching loss)
    and the final prediction map.
    """

    def __init__(self, input_nc: int = 6, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        kw = 4
        padw = int(math.ceil((kw - 1.0) / 2))

        layers: list[nn.Module] = [
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, inplace=False),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=padw)),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=False),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=1, padding=padw)),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=False),
        ]

        layers += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kw, stride=1, padding=padw)),
        ]

        # Split into sub-networks so we can extract intermediate features
        self.blocks = nn.ModuleList()
        block: list[nn.Module] = []
        for layer in layers:
            block.append(layer)
            if isinstance(layer, nn.LeakyReLU):
                self.blocks.append(nn.Sequential(*block))
                block = []
        if block:
            self.blocks.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Returns (intermediate_features, final_prediction)."""
        features: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features[:-1], features[-1]


class MultiscaleDiscriminator(nn.Module):
    """Multi-scale discriminator: runs ``num_D`` PatchGANs at different scales."""

    def __init__(self, input_nc: int = 6, ndf: int = 64, n_layers: int = 3, num_D: int = 2):
        super().__init__()
        self.num_D = num_D

        self.discriminators = nn.ModuleList()
        for _ in range(num_D):
            self.discriminators.append(NLayerDiscriminator(input_nc, ndf, n_layers))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, real_or_fake: torch.Tensor, condition: torch.Tensor):
        """Returns list of (features, prediction) per scale."""
        results: list[tuple[list[torch.Tensor], torch.Tensor]] = []
        x = torch.cat([condition, real_or_fake], dim=1)  # Shape: [B, input_nc, H, W]

        for i, D in enumerate(self.discriminators):
            features, pred = D(x)
            results.append((features, pred))
            if i < self.num_D - 1:
                x = self.downsample(x)  # Downsample for next scale

        return results


# ---------------------------------------------------------------------------
# VGG-19 Perceptual Loss
# ---------------------------------------------------------------------------

class VGG19PerceptualLoss(nn.Module):
    """Extracts features from VGG-19 BN at multiple layers and computes L1
    distance between real and generated feature maps."""

    _LAYER_INDICES = [3, 8, 17, 26]  # relu1_1, relu2_1, relu3_1, relu4_1

    def __init__(self, weights_path: str | None = None):
        super().__init__()
        import torchvision.models as models

        if weights_path and os.path.exists(weights_path):
            vgg = models.vgg19_bn(weights=None)
            vgg.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)

        features = vgg.features
        self.slices = nn.ModuleList()
        prev = 0
        for idx in self._LAYER_INDICES:
            self.slices.append(nn.Sequential(*[features[i] for i in range(prev, idx + 1)]))
            prev = idx + 1

        # Freeze all VGG weights
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalisation constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Shift from [-1,1] to ImageNet normalisation."""
        x = (x + 1.0) / 2.0  # Shape: [B, 3, H, W] in [0,1]
        return (x - self.mean) / self.std  # Shape: [B, 3, H, W]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred   Shape: [B, 3, H, W]
        # target Shape: [B, 3, H, W]
        pred = self._normalise(pred)
        target = self._normalise(target)

        loss = torch.tensor(0.0, device=pred.device)
        x_pred = pred
        x_target = target
        for s in self.slices:
            x_pred = s(x_pred)
            x_target = s(x_target)
            loss = loss + F.l1_loss(x_pred, x_target)

        return loss


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def hinge_loss_d(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator."""
    return 0.5 * (
        torch.mean(F.relu(1.0 - real_pred)) +
        torch.mean(F.relu(1.0 + fake_pred))
    )


def hinge_loss_g(fake_pred: torch.Tensor) -> torch.Tensor:
    """Hinge loss for generator."""
    return -torch.mean(fake_pred)


def feature_matching_loss(real_features: list[list[torch.Tensor]],
                          fake_features: list[list[torch.Tensor]]) -> torch.Tensor:
    """L1 loss between discriminator intermediate features for real vs fake."""
    loss = torch.tensor(0.0, device=real_features[0][0].device)
    for r_feats, f_feats in zip(real_features, fake_features):
        for rf, ff in zip(r_feats, f_feats):
            loss = loss + F.l1_loss(ff, rf.detach())
    return loss


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def weights_init_normal(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("InstanceNorm") != -1 and hasattr(m, "weight") and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ---------------------------------------------------------------------------
# Training worker
# ---------------------------------------------------------------------------

def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace) -> None:
    is_main = (gpu == 0)

    if ngpus_per_node > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:23457",
            world_size=ngpus_per_node,
            rank=gpu,
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

    # --- Build models ---
    if args.model_size == "lite":
        generator = SPADEResNet9Generator(in_nc=3, out_nc=3, ngf=64, n_blocks=9).to(device)
        num_D = 2
    else:  # full – currently same ResNet9 with 3-scale discriminator (local enhancer is future work)
        generator = SPADEResNet9Generator(in_nc=3, out_nc=3, ngf=64, n_blocks=9).to(device)
        num_D = 3

    discriminator = MultiscaleDiscriminator(input_nc=6, ndf=64, n_layers=3, num_D=num_D).to(device)

    # Base weight initialization first so pretrained weights are never overwritten
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    start_epoch = 0
    start_step = 0
    ckpt = None

    # Optionally load pretrained generator (for progressive training)
    if args.pretrained_g and os.path.exists(args.pretrained_g):
        if is_main:
            logging.info(f"Loading pretrained generator from {args.pretrained_g}")
        state = torch.load(args.pretrained_g, map_location=device)
        generator.load_state_dict(state, strict=False)

    if args.resume and os.path.exists(args.checkpoint_path):
        if is_main:
            logging.info(f"Resuming from checkpoint {args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        generator.load_state_dict(ckpt["G"])
        discriminator.load_state_dict(ckpt["D"])
        saved_step = ckpt.get("step", -1)
        total_steps = ckpt.get("total_steps_in_epoch", -1)
        if "base_seed" in ckpt:
            base_seed = int(ckpt["base_seed"])
        if saved_step >= 0 and total_steps > 0 and saved_step < total_steps - 1:
            start_epoch = ckpt["epoch"]
            start_step = saved_step + 1
        else:
            start_epoch = ckpt["epoch"] + 1
            start_step = 0

        # Selected-device resume requires CPU ByteTensors for torch and cuda RNG states
        if "rng_torch" in ckpt and ckpt["rng_torch"] is not None:
            t_rng = ckpt["rng_torch"]
            if isinstance(t_rng, torch.Tensor):
                t_rng = t_rng.cpu()
            torch.set_rng_state(t_rng)
        if "rng_cuda" in ckpt and ckpt["rng_cuda"] is not None and torch.cuda.is_available():
            c_rng = ckpt["rng_cuda"]
            if isinstance(c_rng, torch.Tensor):
                c_rng = c_rng.cpu()
            torch.cuda.set_rng_state(c_rng)
        if "rng_numpy" in ckpt and ckpt["rng_numpy"] is not None:
            np.random.set_state(ckpt["rng_numpy"])
        if "rng_python" in ckpt and ckpt["rng_python"] is not None:
            random.setstate(ckpt["rng_python"])

    # DDP wrapping
    if ngpus_per_node > 1:
        generator = DDP(generator, device_ids=[gpu])  # type: ignore[assignment]
        discriminator = DDP(discriminator, device_ids=[gpu])  # type: ignore[assignment]

    # --- Optimisers ---
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    if ckpt is not None:
        if "opt_G" in ckpt:
            optimizer_G.load_state_dict(ckpt["opt_G"])
        if "opt_D" in ckpt:
            optimizer_D.load_state_dict(ckpt["opt_D"])

    # --- Losses ---
    vgg_weights: str | None = "assets/weights/vgg19-bn.pth"
    if vgg_weights and not os.path.exists(vgg_weights):
        vgg_weights = None
    vgg_loss = VGG19PerceptualLoss(weights_path=vgg_weights).to(device)

    # --- Data ---
    ref_mode = getattr(args, "reference_mode", "deterministic")
    dataset = TennisSkeletonDataset(
        root_dir=args.data_root,
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

    batch_size = 32 if str(args.batch_size).lower() == "auto" else int(args.batch_size)

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
        persistent_workers=(args.num_workers > 0),
    )

    total_batches = len(dataset) // batch_size
    if ngpus_per_node > 1:
        total_batches = (len(dataset) // ngpus_per_node) // batch_size

    if is_main:
        n_params_g = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        n_params_d = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        logging.info(
            f"Spade4Tennis ({args.model_size}) | G: {n_params_g/1e6:.1f}M params | "
            f"D: {n_params_d/1e6:.1f}M params | Dataset: {len(dataset)} images | "
            f"Batch: {batch_size}/GPU | Epochs: {args.epochs}"
        )

    if total_batches == 0:
        if is_main:
            logging.error(
                f"Dataloader empty! Dataset size ({len(dataset)}) too small for "
                f"batch {batch_size} × {ngpus_per_node} GPUs with drop_last."
            )
        return

    last_checkpoint_time = time.time()
    last_progress_time = time.time()
    ckpt_interval = getattr(args, "checkpoint_interval_sec", 3600.0)

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        current_start_step = start_step if epoch == start_epoch else 0
        sampler.set_epoch(epoch, start_step=current_start_step)

        if is_main:
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
        for batch_offset, (skeleton, ref_img, real_img) in iterator:
            i = current_start_step + batch_offset
            if max_steps is not None and i >= max_steps:
                break
            last_step_executed = i

            skeleton = skeleton.to(device)  # Shape: [B, 3, H, W]
            ref_img = ref_img.to(device)    # Shape: [B, 3, H, W]
            real_img = real_img.to(device)  # Shape: [B, 3, H, W]

            # =====================
            #  Train Discriminator
            # =====================
            optimizer_D.zero_grad()

            with torch.no_grad():
                fake_img = generator(skeleton, ref_img)  # Shape: [B, 3, H, W]

            # Combined forward pass for DDP compatibility
            combined_imgs = torch.cat([real_img, fake_img], dim=0)          # Shape: [2B, 3, H, W]
            combined_cond = torch.cat([skeleton, skeleton], dim=0)          # Shape: [2B, 3, H, W]
            combined_results = discriminator(combined_imgs, combined_cond)  # list of (feats, pred) per scale

            loss_D = torch.tensor(0.0, device=device)
            for feats, pred in combined_results:
                real_pred, fake_pred = torch.chunk(pred, 2, dim=0)
                loss_D = loss_D + hinge_loss_d(real_pred, fake_pred)

            loss_D.backward()
            optimizer_D.step()

            # =====================
            #  Train Generator
            # =====================
            optimizer_G.zero_grad()

            fake_img = generator(skeleton, ref_img)  # Shape: [B, 3, H, W]

            # Discriminator forward for G and also get real features for FM loss
            combined_imgs_g = torch.cat([real_img, fake_img], dim=0)
            combined_cond_g = torch.cat([skeleton, skeleton], dim=0)
            combined_results_g = discriminator(combined_imgs_g, combined_cond_g)

            loss_G_gan = torch.tensor(0.0, device=device)
            real_feat_list: list[list[torch.Tensor]] = []
            fake_feat_list: list[list[torch.Tensor]] = []

            for feats, pred in combined_results_g:
                real_pred, fake_pred = torch.chunk(pred, 2, dim=0)
                loss_G_gan = loss_G_gan + hinge_loss_g(fake_pred)

                r_feats = [f[:f.shape[0] // 2] for f in feats]
                f_feats = [f[f.shape[0] // 2:] for f in feats]
                real_feat_list.append(r_feats)
                fake_feat_list.append(f_feats)

            # Feature matching loss
            loss_fm = feature_matching_loss(real_feat_list, fake_feat_list)

            # VGG perceptual loss
            loss_vgg = vgg_loss(fake_img, real_img)

            # L1 pixel loss
            loss_l1 = F.l1_loss(fake_img, real_img)

            # Total generator loss
            loss_G = (
                loss_G_gan +
                args.lambda_fm * loss_fm +
                args.lambda_vgg * loss_vgg +
                args.lambda_pixel * loss_l1
            )

            loss_G.backward()
            optimizer_G.step()

            last_skeleton = skeleton
            last_ref_img = ref_img
            last_real_img = real_img

            if is_main:
                pbar.set_postfix({
                    "D": f"{loss_D.item():.3f}",
                    "G": f"{loss_G.item():.3f}",
                    "vgg": f"{loss_vgg.item():.3f}",
                    "fm": f"{loss_fm.item():.3f}",
                    "l1": f"{loss_l1.item():.3f}",
                })
                now = time.time()
                if now - last_progress_time >= 600.0:
                    logging.info(f"Progress heartbeat: epoch {epoch}/{args.epochs}, step {i}/{total_batches}")
                    last_progress_time = now

                # Intra-epoch hourly/periodic wall-clock checkpoint deadline check
                if now - last_checkpoint_time >= ckpt_interval:
                    g_mod = generator.module if ngpus_per_node > 1 else generator
                    d_mod = discriminator.module if ngpus_per_node > 1 else discriminator
                    save_checkpoint_atomic(g_mod.state_dict(), args.out_weights)
                    rng_cuda = torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None
                    save_checkpoint_atomic({
                        "epoch": epoch,
                        "step": i,
                        "total_steps_in_epoch": total_batches,
                        "base_seed": base_seed,
                        "model_size": args.model_size,
                        "G": g_mod.state_dict(),
                        "D": d_mod.state_dict(),
                        "opt_G": optimizer_G.state_dict(),
                        "opt_D": optimizer_D.state_dict(),
                        "rng_torch": torch.get_rng_state().cpu(),
                        "rng_cuda": rng_cuda,
                        "rng_numpy": np.random.get_state(),
                        "rng_python": random.getstate(),
                        "timestamp_unix": now,
                    }, args.checkpoint_path)
                    last_checkpoint_time = now
                    logging.info(f"Intra-epoch checkpoint saved at epoch {epoch}, step {i}/{total_batches}")

        start_step = 0

        # --- End of epoch ---
        if is_main and "last_skeleton" in locals():
            # Save sample images
            with torch.no_grad():
                sample_fake = generator(last_skeleton[:4], last_ref_img[:4])  # Shape: [4, 3, H, W]
            sample = torch.cat(
                (
                    last_skeleton[:4].detach().cpu(),
                    last_ref_img[:4].detach().cpu(),
                    last_real_img[:4].detach().cpu(),
                    sample_fake.detach().cpu(),
                ),
                -1,
            )
            vutils.save_image(sample, f"{args.sample_dir}/s4t_epoch_{epoch:03d}.png", nrow=4, normalize=True)

            now = time.time()
            epoch_completed = (last_step_executed is not None and last_step_executed == total_batches - 1)
            step_to_save = (total_batches - 1) if epoch_completed else (last_step_executed if last_step_executed is not None else 0)
            # Wall-clock checkpointing (or every 10 epochs or final epoch)
            if (epoch + 1) % 10 == 0 or (now - last_checkpoint_time >= ckpt_interval) or (epoch + 1 == args.epochs):
                g_mod = generator.module if ngpus_per_node > 1 else generator
                d_mod = discriminator.module if ngpus_per_node > 1 else discriminator

                save_checkpoint_atomic(g_mod.state_dict(), args.out_weights)
                rng_cuda = torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None
                save_checkpoint_atomic({
                    "epoch": epoch,
                    "step": step_to_save,
                    "total_steps_in_epoch": total_batches,
                    "base_seed": base_seed,
                    "model_size": args.model_size,
                    "G": g_mod.state_dict(),
                    "D": d_mod.state_dict(),
                    "opt_G": optimizer_G.state_dict(),
                    "opt_D": optimizer_D.state_dict(),
                    "rng_torch": torch.get_rng_state().cpu(),
                    "rng_cuda": rng_cuda,
                    "rng_numpy": np.random.get_state(),
                    "rng_python": random.getstate(),
                    "timestamp_unix": now,
                }, args.checkpoint_path)
                last_checkpoint_time = now

    # Final save
    if is_main:
        g_mod = generator.module if ngpus_per_node > 1 else generator
        save_checkpoint_atomic(g_mod.state_dict(), args.out_weights)
        logging.info("Spade4Tennis training complete.")

    if ngpus_per_node > 1:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spade4Tennis for Pointstream GenAI Backend")
    parser.add_argument("--model-size", type=str, default="lite", choices=["lite", "full"],
                        help="Model tier: lite (ResNet-9, 2 discriminators) or full (ResNet-9, 3 discriminators; local enhancer is future work)")
    parser.add_argument("--data-root", type=str, default="assets/dataset", help="Dataset root path")
    parser.add_argument("--condition", type=str, default="pose_body",
                        choices=["pose_body", "pose_racket", "skeleton"],
                        help="Pose-condition variant. pose_body is reproducible by the decoder for free "
                             "and bit-identically; pose_racket matches the legacy checkpoints but needs "
                             "racket geometry the wire format does not carry; skeleton is the legacy tree "
                             "(positional filenames) kept only to reproduce old runs.")
    parser.add_argument("--subset", type=str, default="all",
                        help="Subset name or 'all'")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=str, default="auto",
                        help="Batch size per GPU ('auto' → 32)")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--b1", type=float, default=0.0,
                        help="Adam beta1 (0.0 for hinge GAN stability)")
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--lambda-pixel", type=float, default=10.0,
                        help="L1 pixel loss weight (reduced from 100 to 10)")
    parser.add_argument("--lambda-vgg", type=float, default=10.0,
                        help="VGG perceptual loss weight")
    parser.add_argument("--lambda-fm", type=float, default=10.0,
                        help="Feature matching loss weight")
    parser.add_argument("--out-weights", type=str, default=None,
                        help="Output generator weights path (auto-named per model-size)")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Checkpoint path for resume (auto-named per model-size)")
    parser.add_argument("--checkpoint-interval-sec", type=float, default=3600.0,
                        help="Periodic checkpoint wall-clock interval in seconds (default: 3600.0)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--sample-dir", type=str, default="assets/samples")
    parser.add_argument("--pretrained-g", type=str, default=None,
                        help="Path to pretrained generator (for progressive training)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    parser.add_argument("--reference-mode", type=str, default="deterministic",
                        choices=["deterministic", "first", "random"],
                        help="Reference selection mode for TennisSkeletonDataset (default: deterministic)")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None,
                        help="Optional cap on steps per epoch for fast integration tests")
    args = parser.parse_args()

    # Auto-name outputs by model size
    if args.out_weights is None:
        args.out_weights = f"assets/weights/spade4tennis_{args.model_size}_generator.pt"
    if args.checkpoint_path is None:
        args.checkpoint_path = f"assets/weights/spade4tennis_{args.model_size}_checkpoint.pt"

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

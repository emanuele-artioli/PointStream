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
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import argparse
import logging
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from src.shared.tennis_dataset import TennisSkeletonDataset
from src.shared.spade4tennis_arch import SPADE, SPADEResBlock, ReferenceEncoder, SPADEResNet9Generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SPADE Normalization
# ---------------------------------------------------------------------------

class SPADE(nn.Module):
    """Spatially-Adaptive Denormalization.

    Replaces standard BatchNorm: instead of using learned affine parameters,
    SPADE computes spatially-varying gamma/beta from a conditioning input
    (the reference image features) via a small conv network.
    """

    def __init__(self, norm_nc: int, cond_nc: int, hidden_nc: int = 128):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)  # Instance norm (no learnt affine)
        self.shared = nn.Sequential(
            nn.Conv2d(cond_nc, hidden_nc, 3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.gamma = nn.Conv2d(hidden_nc, norm_nc, 3, padding=1)  # Learned scale
        self.beta = nn.Conv2d(hidden_nc, norm_nc, 3, padding=1)   # Learned bias

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x     Shape: [B, norm_nc, H, W]
        # cond  Shape: [B, cond_nc, H_c, W_c]
        normalised = self.norm(x)  # Shape: [B, norm_nc, H, W]

        # Resize condition to match spatial dims of x
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode="bilinear", align_corners=False)
            # Shape: [B, cond_nc, H, W]

        shared = self.shared(cond)        # Shape: [B, hidden_nc, H, W]
        gamma = self.gamma(shared)        # Shape: [B, norm_nc, H, W]
        beta = self.beta(shared)          # Shape: [B, norm_nc, H, W]

        return normalised * (1 + gamma) + beta  # Shape: [B, norm_nc, H, W]


class SPADEResBlock(nn.Module):
    """Residual block with SPADE normalization at each conv layer."""

    def __init__(self, fin: int, fout: int, cond_nc: int):
        super().__init__()
        fmid = min(fin, fout)
        self.learned_skip = (fin != fout)

        self.norm_0 = SPADE(fin, cond_nc)
        self.conv_0 = nn.Conv2d(fin, fmid, 3, padding=1)
        self.norm_1 = SPADE(fmid, cond_nc)
        self.conv_1 = nn.Conv2d(fmid, fout, 3, padding=1)

        if self.learned_skip:
            self.norm_s = SPADE(fin, cond_nc)
            self.conv_s = nn.Conv2d(fin, fout, 1, bias=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x    Shape: [B, fin, H, W]
        # cond Shape: [B, cond_nc, H_c, W_c]
        dx = self.conv_0(F.leaky_relu(self.norm_0(x, cond), 0.2))  # Shape: [B, fmid, H, W]
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, cond), 0.2)) # Shape: [B, fout, H, W]

        if self.learned_skip:
            skip = self.conv_s(F.leaky_relu(self.norm_s(x, cond), 0.2))  # Shape: [B, fout, H, W]
        else:
            skip = x  # Shape: [B, fout, H, W]

        return dx + skip  # Shape: [B, fout, H, W]


# ---------------------------------------------------------------------------
# Reference Encoder (lightweight)
# ---------------------------------------------------------------------------

class ReferenceEncoder(nn.Module):
    """Encodes the reference image into a multi-scale feature representation
    that will be injected into the generator via SPADE layers."""

    def __init__(self, in_nc: int = 3, nf: int = 64):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, stride=2, padding=1),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=False),
        )  # Output: [B, 64, H/2, W/2]
        self.layer2 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=False),
        )  # Output: [B, 128, H/4, W/4]
        self.layer3 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=False),
        )  # Output: [B, 256, H/8, W/8]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x Shape: [B, 3, H, W]
        x = self.layer1(x)  # Shape: [B, 64, H/2, W/2]
        x = self.layer2(x)  # Shape: [B, 128, H/4, W/4]
        x = self.layer3(x)  # Shape: [B, 256, H/8, W/8]
        return x  # Shape: [B, 256, H/8, W/8]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

class SPADEResNet9Generator(nn.Module):
    """Lite generator: ResNet-9 backbone with SPADE reference injection.

    The skeleton is encoded via a standard conv encoder. The reference image
    is encoded separately by a ``ReferenceEncoder`` and injected at every
    SPADE residual block in the decoder.
    """

    def __init__(self, in_nc: int = 3, out_nc: int = 3, ngf: int = 64, n_blocks: int = 9):
        super().__init__()
        self.ref_encoder = ReferenceEncoder(in_nc=3, nf=ngf)
        cond_nc = ngf * 4  # 256 – output channels of reference encoder

        # --- Skeleton encoder (no SPADE, just standard convs) ---
        self.enc_head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_nc, ngf, 7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=False),
        )  # Output: [B, 64, H, W]

        self.enc_down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=False),
        )  # Output: [B, 128, H/2, W/2]

        self.enc_down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=False),
        )  # Output: [B, 256, H/4, W/4]

        # --- SPADE ResNet blocks in bottleneck ---
        self.spade_blocks = nn.ModuleList([
            SPADEResBlock(ngf * 4, ngf * 4, cond_nc) for _ in range(n_blocks)
        ])

        # --- Decoder (upsampling) ---
        self.dec_up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=False),
        )  # Output: [B, 128, H/2, W/2]

        self.dec_up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=False),
        )  # Output: [B, 64, H, W]

        self.dec_head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_nc, 7, padding=0),
            nn.Tanh(),
        )  # Output: [B, 3, H, W]

    def forward(self, skeleton: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # skeleton  Shape: [B, 3, H, W]
        # reference Shape: [B, 3, H, W]

        # Encode reference
        ref_feat = self.ref_encoder(reference)  # Shape: [B, 256, H/8, W/8]

        # Encode skeleton
        x = self.enc_head(skeleton)  # Shape: [B, 64, H, W]
        x = self.enc_down1(x)       # Shape: [B, 128, H/2, W/2]
        x = self.enc_down2(x)       # Shape: [B, 256, H/4, W/4]

        # SPADE bottleneck: reference features injected here
        for block in self.spade_blocks:
            x = block(x, ref_feat)   # Shape: [B, 256, H/4, W/4]

        # Decode
        x = self.dec_up1(x)          # Shape: [B, 128, H/2, W/2]
        x = self.dec_up2(x)          # Shape: [B, 64, H, W]
        x = self.dec_head(x)         # Shape: [B, 3, H, W]

        return x


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

    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")

    # --- Build models ---
    if args.model_size == "lite":
        generator = SPADEResNet9Generator(in_nc=3, out_nc=3, ngf=64, n_blocks=9).to(device)
        num_D = 2
    else:  # full – will use same ResNet9 + local enhancer in future
        generator = SPADEResNet9Generator(in_nc=3, out_nc=3, ngf=64, n_blocks=9).to(device)
        num_D = 3

    discriminator = MultiscaleDiscriminator(input_nc=6, ndf=64, n_layers=3, num_D=num_D).to(device)

    # Optionally load pretrained generator (for progressive training)
    start_epoch = 0
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
        start_epoch = ckpt["epoch"] + 1
    else:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # DDP wrapping
    if ngpus_per_node > 1:
        generator = DDP(generator, device_ids=[gpu])
        discriminator = DDP(discriminator, device_ids=[gpu])

    # --- Optimisers ---
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    if args.resume and os.path.exists(args.checkpoint_path):
        optimizer_G.load_state_dict(ckpt["opt_G"])
        optimizer_D.load_state_dict(ckpt["opt_D"])

    # --- Losses ---
    vgg_weights = "assets/weights/vgg19-bn.pth"
    if not os.path.exists(vgg_weights):
        vgg_weights = None
    vgg_loss = VGG19PerceptualLoss(weights_path=vgg_weights).to(device)

    # --- Data ---
    dataset = TennisSkeletonDataset(
        root_dir=args.data_root, 
        target_size=args.img_size, 
        include_reference=True
    )

    if ngpus_per_node > 1:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    batch_size = 32 if str(args.batch_size).lower() == "auto" else int(args.batch_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        persistent_workers=(args.num_workers > 0),
    )

    if is_main:
        n_params_g = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        n_params_d = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        logging.info(
            f"Spade4Tennis ({args.model_size}) | G: {n_params_g/1e6:.1f}M params | "
            f"D: {n_params_d/1e6:.1f}M params | Dataset: {len(dataset)} images | "
            f"Batch: {batch_size}/GPU | Epochs: {args.epochs}"
        )

    if len(dataloader) == 0:
        if is_main:
            logging.error(
                f"Dataloader empty! Dataset size ({len(dataset)}) too small for "
                f"batch {batch_size} × {ngpus_per_node} GPUs with drop_last."
            )
        return

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if is_main:
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{args.epochs}")
            iterator = pbar
        else:
            iterator = enumerate(dataloader)

        for i, (skeleton, ref_img, real_img) in iterator:
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

            if is_main:
                pbar.set_postfix({
                    "D": f"{loss_D.item():.3f}",
                    "G": f"{loss_G.item():.3f}",
                    "vgg": f"{loss_vgg.item():.3f}",
                    "fm": f"{loss_fm.item():.3f}",
                    "l1": f"{loss_l1.item():.3f}",
                })

        # --- End of epoch ---
        if is_main:
            # Save sample images
            with torch.no_grad():
                sample_fake = generator(skeleton[:4], ref_img[:4])  # Shape: [4, 3, H, W]
            sample = torch.cat((skeleton[:4], ref_img[:4], real_img[:4], sample_fake), -1)
            vutils.save_image(sample, f"{args.sample_dir}/s4t_epoch_{epoch:03d}.png", nrow=4, normalize=True)

            # Save checkpoints every 10 epochs
            if (epoch + 1) % 10 == 0:
                g_mod = generator.module if ngpus_per_node > 1 else generator
                d_mod = discriminator.module if ngpus_per_node > 1 else discriminator

                torch.save(g_mod.state_dict(), args.out_weights)
                torch.save({
                    "epoch": epoch,
                    "model_size": args.model_size,
                    "G": g_mod.state_dict(),
                    "D": d_mod.state_dict(),
                    "opt_G": optimizer_G.state_dict(),
                    "opt_D": optimizer_D.state_dict(),
                }, args.checkpoint_path)

    # Final save
    if is_main:
        g_mod = generator.module if ngpus_per_node > 1 else generator
        torch.save(g_mod.state_dict(), args.out_weights)
        logging.info("Spade4Tennis training complete.")

    if ngpus_per_node > 1:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spade4Tennis for Pointstream GenAI Backend")
    parser.add_argument("--model-size", type=str, default="lite", choices=["lite", "full"],
                        help="Model tier: lite (ResNet-9, fast) or full (with local enhancer)")
    parser.add_argument("--data-root", type=str, default="assets/dataset", help="Dataset root path")
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
    parser.add_argument("--sample-dir", type=str, default="assets/samples")
    parser.add_argument("--pretrained-g", type=str, default=None,
                        help="Path to pretrained generator (for progressive training)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
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

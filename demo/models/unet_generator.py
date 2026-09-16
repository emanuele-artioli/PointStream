"""Lightweight conditional UNet generator for hand synthesis from pose keypoints and appearance anchor."""

from __future__ import annotations

import sqlite3  # noqa: F401

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat((x, skip_input), dim=1)


class HandPix2PixUNet(nn.Module):
    """Conditional UNet: 6 channels (3 appearance + 3 skeleton) -> 3 channels (RGB hand crop)."""

    def __init__(self, in_channels: int = 6, out_channels: int = 3) -> None:
        super().__init__()
        # Downsampling: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.0)
        self.down5 = UNetDown(512, 512, dropout=0.0)
        self.down6 = UNetDown(512, 512, dropout=0.0)

        # Upsampling: 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.up1 = UNetUp(512, 512, dropout=0.0)
        self.up2 = UNetUp(1024, 512, dropout=0.0)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, appearance_and_pose: torch.Tensor) -> torch.Tensor:
        # appearance_and_pose: [B, 6, H, W] in range [-1, 1]
        d1 = self.down1(appearance_and_pose)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)


class SPADE(nn.Module):
    """Spatially-Adaptive Denormalization layer."""

    def __init__(self, norm_nc: int, cond_nc: int = 3, hidden_nc: int = 64) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.shared = nn.Sequential(
            nn.Conv2d(cond_nc, hidden_nc, 3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.gamma = nn.Conv2d(hidden_nc, norm_nc, 3, padding=1)
        self.beta = nn.Conv2d(hidden_nc, norm_nc, 3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, norm_nc, H, W]
        # cond: [B, cond_nc, H_c, W_c]
        normalised = self.norm(x)
        if cond.shape[2:] != x.shape[2:]:
            cond = nn.functional.interpolate(cond, size=x.shape[2:], mode="bilinear", align_corners=False)
        shared = self.shared(cond)
        return normalised * (1.0 + self.gamma(shared)) + self.beta(shared)


class SPADEResBlock(nn.Module):
    """Residual block with SPADE normalization for pose-conditioned feature modulation."""

    def __init__(self, fin: int, fout: int, cond_nc: int = 3) -> None:
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
        dx = self.conv_0(nn.functional.leaky_relu(self.norm_0(x, cond), 0.2))
        dx = self.conv_1(nn.functional.leaky_relu(self.norm_1(dx, cond), 0.2))
        skip = self.conv_s(nn.functional.leaky_relu(self.norm_s(x, cond), 0.2)) if self.learned_skip else x
        return dx + skip


class HandSPADEUNet(nn.Module):
    """Upgraded conditional generator: encodes appearance anchor and decodes with SPADE pose modulation.

    Accepts either:
    - A single 6-channel tensor [B, 6, H, W] (first 3 channels appearance, last 3 skeleton)
    - Two 3-channel tensors (appearance [B, 3, H, W], skeleton [B, 3, H, W])
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 3, ngf: int = 64) -> None:
        super().__init__()
        # 1. Appearance Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, ngf, 4, stride=2, padding=1),  # 256 -> 128
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, stride=2, padding=1, bias=False),  # 128 -> 64
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, stride=2, padding=1, bias=False),  # 64 -> 32
            nn.InstanceNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, 4, stride=2, padding=1, bias=False),  # 32 -> 16
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1, bias=False),  # 16 -> 8
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2. Bottleneck SPADE Residual Blocks
        self.res1 = SPADEResBlock(ngf * 8, ngf * 8, cond_nc=3)
        self.res2 = SPADEResBlock(ngf * 8, ngf * 8, cond_nc=3)

        # 3. SPADE Decoder with Skip Connections
        self.up_conv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, stride=2, padding=1, bias=False)  # 8 -> 16
        self.dec_block1 = SPADEResBlock(ngf * 16, ngf * 4, cond_nc=3)

        self.up_conv2 = nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, stride=2, padding=1, bias=False)  # 16 -> 32
        self.dec_block2 = SPADEResBlock(ngf * 8, ngf * 2, cond_nc=3)

        self.up_conv3 = nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, stride=2, padding=1, bias=False)  # 32 -> 64
        self.dec_block3 = SPADEResBlock(ngf * 4, ngf, cond_nc=3)

        self.up_conv4 = nn.ConvTranspose2d(ngf, ngf, 4, stride=2, padding=1, bias=False)  # 64 -> 128
        self.dec_block4 = SPADEResBlock(ngf * 2, ngf, cond_nc=3)

        self.up_conv5 = nn.ConvTranspose2d(ngf, ngf // 2, 4, stride=2, padding=1, bias=False)  # 128 -> 256
        self.dec_block5 = SPADEResBlock(ngf // 2, ngf // 2, cond_nc=3)

        self.final = nn.Sequential(
            nn.Conv2d(ngf // 2, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, appearance_and_pose: torch.Tensor, skeleton_cond: torch.Tensor | None = None) -> torch.Tensor:
        if skeleton_cond is not None:
            app = appearance_and_pose
            skel = skeleton_cond
        else:
            if appearance_and_pose.shape[1] == 6:
                app = appearance_and_pose[:, :3]
                skel = appearance_and_pose[:, 3:]
            else:
                raise ValueError(f"Expected 6 channels or separate skeleton_cond, got shape {appearance_and_pose.shape}")

        # Encoder pass on appearance anchor
        e1 = self.enc1(app)   # [B, 64, 128, 128]
        e2 = self.enc2(e1)    # [B, 128, 64, 64]
        e3 = self.enc3(e2)    # [B, 256, 32, 32]
        e4 = self.enc4(e3)    # [B, 512, 16, 16]
        e5 = self.enc5(e4)    # [B, 512, 8, 8]

        # Bottleneck modulation with skeleton pose
        b = self.res1(e5, skel)
        b = self.res2(b, skel)

        # Decoder pass with skip connections and SPADE modulation
        u1 = self.up_conv1(b)
        d1 = self.dec_block1(torch.cat([u1, e4], dim=1), skel)

        u2 = self.up_conv2(d1)
        d2 = self.dec_block2(torch.cat([u2, e3], dim=1), skel)

        u3 = self.up_conv3(d2)
        d3 = self.dec_block3(torch.cat([u3, e2], dim=1), skel)

        u4 = self.up_conv4(d3)
        d4 = self.dec_block4(torch.cat([u4, e1], dim=1), skel)

        u5 = self.up_conv5(d4)
        d5 = self.dec_block5(u5, skel)

        return self.final(d5)



import torch
import torch.nn as nn
import torch.nn.functional as F

class SPADE(nn.Module):
    """Spatially-Adaptive Denormalization."""

    def __init__(self, norm_nc: int, cond_nc: int, hidden_nc: int = 128):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.shared = nn.Sequential(
            nn.Conv2d(cond_nc, hidden_nc, 3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.gamma = nn.Conv2d(hidden_nc, norm_nc, 3, padding=1)
        self.beta = nn.Conv2d(hidden_nc, norm_nc, 3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x     Shape: [B, norm_nc, H, W]
        # cond  Shape: [B, cond_nc, H_c, W_c]
        normalised = self.norm(x)  # Shape: [B, norm_nc, H, W]
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode="bilinear", align_corners=False)
        shared = self.shared(cond)  # Shape: [B, hidden_nc, H, W]
        return normalised * (1 + self.gamma(shared)) + self.beta(shared)


class SPADEResBlock(nn.Module):
    """Residual block with SPADE normalization."""

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
        dx = self.conv_0(F.leaky_relu(self.norm_0(x, cond), 0.2))
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, cond), 0.2))
        skip = self.conv_s(F.leaky_relu(self.norm_s(x, cond), 0.2)) if self.learned_skip else x
        return dx + skip


class ReferenceEncoder(nn.Module):
    """Lightweight encoder for the reference image."""

    def __init__(self, in_nc: int = 3, nf: int = 64):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, stride=2, padding=1),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x Shape: [B, 3, H, W]
        x = self.layer1(x)  # Shape: [B, 64, H/2, W/2]
        x = self.layer2(x)  # Shape: [B, 128, H/4, W/4]
        x = self.layer3(x)  # Shape: [B, 256, H/8, W/8]
        return x


class SPADEResNet9Generator(nn.Module):
    """SPADE-conditioned ResNet-9 generator."""

    def __init__(self, in_nc: int = 3, out_nc: int = 3, ngf: int = 64, n_blocks: int = 9):
        super().__init__()
        self.ref_encoder = ReferenceEncoder(in_nc=3, nf=ngf)
        cond_nc = ngf * 4  # 256

        self.enc_head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_nc, ngf, 7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=False),
        )
        self.enc_down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=False),
        )
        self.enc_down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=False),
        )
        self.spade_blocks = nn.ModuleList([
            SPADEResBlock(ngf * 4, ngf * 4, cond_nc) for _ in range(n_blocks)
        ])
        self.dec_up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=False),
        )
        self.dec_up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=False),
        )
        self.dec_head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_nc, 7, padding=0),
            nn.Tanh(),
        )

    def forward(self, skeleton: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # skeleton  Shape: [B, 3, H, W]
        # reference Shape: [B, 3, H, W]
        ref_feat = self.ref_encoder(reference)  # Shape: [B, 256, H/8, W/8]
        x = self.enc_head(skeleton)   # Shape: [B, 64, H, W]
        x = self.enc_down1(x)         # Shape: [B, 128, H/2, W/2]
        x = self.enc_down2(x)         # Shape: [B, 256, H/4, W/4]
        for block in self.spade_blocks:
            x = block(x, ref_feat)    # Shape: [B, 256, H/4, W/4]
        x = self.dec_up1(x)           # Shape: [B, 128, H/2, W/2]
        x = self.dec_up2(x)           # Shape: [B, 64, H, W]
        x = self.dec_head(x)          # Shape: [B, 3, H, W]
        return x

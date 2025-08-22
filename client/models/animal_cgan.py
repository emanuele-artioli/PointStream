#!/usr/bin/env python3
"""
Animal cGAN Model

Conditional GAN for generating animal figures from keypoints and reference images.
Uses animal pose keypoint format (20 keypoints) for pose conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AnimalResidualBlock(nn.Module):
    """Residual block for animal generator."""
    
    def __init__(self, channels: int):
        super(AnimalResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AnimalGenerator(nn.Module):
    """Generator network for animal figures."""
    
    def __init__(self, input_channels: int = 4, output_channels: int = 3, ngf: int = 64):
        """
        Initialize animal generator.
        
        Args:
            input_channels: Number of input channels (3 for RGB + 1 for pose)
            output_channels: Number of output channels (3 for RGB)
            ngf: Number of generator filters
        """
        super(AnimalGenerator, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder5 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Bottleneck with residual blocks (more blocks for complex animal poses)
        self.bottleneck = nn.Sequential(
            AnimalResidualBlock(ngf * 8),
            AnimalResidualBlock(ngf * 8),
            AnimalResidualBlock(ngf * 8),
            AnimalResidualBlock(ngf * 8)
        )
        
        # Decoder
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Bottleneck
        bottleneck = self.bottleneck(e5)
        
        # Decoder with skip connections
        d5 = self.decoder5(bottleneck)
        d5 = torch.cat([d5, e4], 1)
        
        d4 = self.decoder4(d5)
        d4 = torch.cat([d4, e3], 1)
        
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], 1)
        
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e1], 1)
        
        output = self.decoder1(d2)
        
        return output


class AnimalDiscriminator(nn.Module):
    """Discriminator network for animal figures."""
    
    def __init__(self, input_channels: int = 6, ndf: int = 64):
        """
        Initialize animal discriminator.
        
        Args:
            input_channels: Number of input channels (real/fake + condition)
            ndf: Number of discriminator filters
        """
        super(AnimalDiscriminator, self).__init__()
        
        # Add spectral normalization for training stability
        self.main = nn.Sequential(
            # Input: [B, 6, 256, 256]
            nn.utils.spectral_norm(nn.Conv2d(input_channels, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, 64, 128, 128]
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, 128, 64, 64]
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, 256, 32, 32]
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, 512, 16, 16]
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, 512, 8, 8]
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
            # [B, 1, 5, 5]
        )
        
    def forward(self, input_img, condition_img):
        x = torch.cat([input_img, condition_img], 1)
        return self.main(x)


class AnimalCGAN(nn.Module):
    """Complete Animal cGAN model."""
    
    def __init__(self, input_size: int = 256, keypoint_channels: int = 20):
        """
        Initialize Animal cGAN.
        
        Args:
            input_size: Input image size
            keypoint_channels: Number of keypoint channels (20 for animals)
        """
        super(AnimalCGAN, self).__init__()
        
        self.input_size = input_size
        self.keypoint_channels = keypoint_channels
        
        # Generator takes 4 channels: 3 (RGB reference) + 1 (pose map)
        self.generator = AnimalGenerator(input_channels=4, output_channels=3)
        
        # Discriminator takes 6 channels: 3 (real/fake) + 3 (reference condition)
        self.discriminator = AnimalDiscriminator(input_channels=6)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        """Forward pass through generator only (for inference)."""
        return self.generator(x)
    
    def generate(self, reference_img: torch.Tensor, pose_map: torch.Tensor) -> torch.Tensor:
        """
        Generate animal figure from reference and pose.
        
        Args:
            reference_img: Reference image [B, 3, H, W]
            pose_map: Pose map [B, 1, H, W]
            
        Returns:
            Generated image [B, 3, H, W]
        """
        input_tensor = torch.cat([reference_img, pose_map], dim=1)
        return self.generator(input_tensor)
    
    def discriminate(self, img: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real vs fake images.
        
        Args:
            img: Image to discriminate [B, 3, H, W]
            condition: Conditioning image [B, 3, H, W]
            
        Returns:
            Discriminator output [B, 1, H_out, W_out]
        """
        return self.discriminator(img, condition)

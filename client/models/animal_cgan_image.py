#!/usr/bin/env python3
"""
Animal cGAN Model (Image-based)

Conditional GAN for generating animal figures from appearance vectors and pose skeleton images.
Takes pose as image input instead of vector for better spatial understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
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
    """Generator network for animal figures from appearance vector and pose image."""
    
    def __init__(self, appearance_vector_size: int = 2048, pose_channels: int = 1,
                 output_channels: int = 3, ngf: int = 64):
        """Initialize animal generator."""
        super(AnimalGenerator, self).__init__()
        
        self.appearance_vector_size = appearance_vector_size
        self.pose_channels = pose_channels
        self.ngf = ngf
        
        # Store model metadata
        self.model_metadata = {
            'appearance_vector_size': appearance_vector_size,
            'pose_channels': pose_channels,
            'model_version': '2.0',
            'input_type': 'image',
            'category': 'animal'
        }
        
        # Similar architecture to human model but adapted for animal poses
        self.pose_encoder = nn.Sequential(
            nn.Conv2d(pose_channels, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.appearance_projection = nn.Sequential(
            nn.Linear(appearance_vector_size, ngf * 8 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(ngf * 16, ngf * 8, 3, 1, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(ngf * 8),
            ResidualBlock(ngf * 8),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, output_channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def get_model_metadata(self):
        return self.model_metadata.copy()
        
    def forward(self, appearance_vec: torch.Tensor, pose_img: torch.Tensor) -> torch.Tensor:
        pose_features = self.pose_encoder(pose_img)
        appearance_projected = self.appearance_projection(appearance_vec)
        appearance_features = appearance_projected.view(-1, self.ngf * 8, 8, 8)
        combined_features = torch.cat([pose_features, appearance_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        output = self.decoder(fused_features)
        return output


class AnimalDiscriminator(nn.Module):
    """Discriminator network for animal figures."""
    
    def __init__(self, input_channels: int = 3, pose_channels: int = 1, ndf: int = 64):
        super(AnimalDiscriminator, self).__init__()
        
        self.model_metadata = {
            'input_channels': input_channels,
            'pose_channels': pose_channels,
            'model_version': '2.0',
            'input_type': 'image',
            'category': 'animal'
        }
        
        self.main = nn.Sequential(
            nn.Conv2d(input_channels + pose_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 2, 1),
        )
        
    def get_model_metadata(self):
        return self.model_metadata.copy()
        
    def forward(self, input_img: torch.Tensor, pose_img: torch.Tensor) -> torch.Tensor:
        x = torch.cat([input_img, pose_img], dim=1)
        return self.main(x)


class AnimalCGAN(nn.Module):
    """Complete Animal cGAN model with pose image input."""
    
    def __init__(self, input_size: int = 256, pose_channels: int = 1, device: torch.device = None):
        super(AnimalCGAN, self).__init__()
        
        self.input_size = input_size
        self.pose_channels = pose_channels
        self.device = device
        
        self.generator = AnimalGenerator(
            appearance_vector_size=2048,
            pose_channels=pose_channels,
            output_channels=3
        )
        
        self.discriminator = AnimalDiscriminator(
            input_channels=3,
            pose_channels=pose_channels
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def get_model_metadata(self):
        return {
            'input_size': self.input_size,
            'pose_channels': self.pose_channels,
            'model_version': '2.0',
            'input_type': 'image',
            'category': 'animal',
            'generator_metadata': self.generator.get_model_metadata(),
            'discriminator_metadata': self.discriminator.get_model_metadata()
        }
    
    def generate(self, appearance_vec: torch.Tensor, pose_img: torch.Tensor) -> torch.Tensor:
        return self.generator(appearance_vec, pose_img)
    
    def discriminate(self, img: torch.Tensor, pose_img: torch.Tensor) -> torch.Tensor:
        return self.discriminator(img, pose_img)
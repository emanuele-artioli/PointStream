#!/usr/bin/env python3
"""
Human cGAN Model

Conditional GAN for generating human figures from keypoints and reference images.
Uses COCO keypoint format (17 keypoints) for pose conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


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


class HumanGenerator(nn.Module):
    """Generator network for human figures from a concatenated appearance and pose vector."""
    
    def __init__(self, vector_input_size: int = 2099, output_channels: int = 3, ngf: int = 64, latent_dim: int = 512):
        """
        Initialize human generator.
        
        Args:
            vector_input_size: Dimension of the concatenated input vector [v_appearance, p_t].
            output_channels: Number of output channels (3 for RGB).
            ngf: Number of generator filters.
            latent_dim: Dimension of the latent space to project the input vector to.
        """
        super(HumanGenerator, self).__init__()
        
        # Mapping network to project the input vector into a latent space
        self.mapping_network = nn.Sequential(
            nn.Linear(vector_input_size, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Project the latent vector to the starting size of the decoder
        # We'll start at 4x4 resolution
        self.initial_projection = nn.Sequential(
            nn.Linear(latent_dim, ngf * 8 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.ngf = ngf

        # Decoder (upsampling from 4x4 to 256x256)
        self.decoder = nn.Sequential(
            # Start from 4x4
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False), # -> 8x8
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), # -> 16x16
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # -> 32x32
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),     # -> 64x64
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),         # -> 128x128
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, output_channels, 4, 2, 1, bias=False), # -> 256x256
            nn.Tanh()
        )
        
    def forward(self, vec):
        # Map the input vector to the latent space
        latent_vec = self.mapping_network(vec)
        
        # Project and reshape to start the convolutional decoder
        initial_h = self.initial_projection(latent_vec)
        initial_h = initial_h.view(-1, self.ngf * 8, 4, 4)
        
        # Generate the image
        output = self.decoder(initial_h)
        
        return output


class HumanDiscriminator(nn.Module):
    """Discriminator network for human figures, conditioned on a pose vector."""
    
    def __init__(self, input_channels: int = 3, pose_vector_size: int = 51, ndf: int = 64):
        """
        Initialize human discriminator.
        
        Args:
            input_channels: Number of input channels (3 for real/fake image).
            pose_vector_size: Dimension of the pose vector p_t.
            ndf: Number of discriminator filters.
        """
        super(HumanDiscriminator, self).__init__()
        
        self.pose_projection = nn.Sequential(
            nn.Linear(pose_vector_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256 * 256) # Project to image size
        )

        self.main = nn.Sequential(
            # Input: [B, 4, 256, 256] (3 for image + 1 for pose map)
            nn.Conv2d(input_channels + 1, ndf, 4, 2, 1),
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
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
        )
        
    def forward(self, input_img, pose_vector):
        # Project pose vector to a spatial map
        pose_map = self.pose_projection(pose_vector).view(-1, 1, 256, 256)

        # Concatenate image with the projected pose map
        x = torch.cat([input_img, pose_map], 1)
        return self.main(x)


class HumanCGAN(nn.Module):
    """Complete Human cGAN model."""
    
    def __init__(self, input_size: int = 256, vector_input_size: int = 2099):
        """
        Initialize Human cGAN.
        
        Args:
            input_size: Output image size.
            vector_input_size: Dimension of the concatenated input vector [v_appearance, p_t].
        """
        super(HumanCGAN, self).__init__()
        
        self.input_size = input_size
        self.vector_input_size = vector_input_size
        
        self.generator = HumanGenerator(vector_input_size=vector_input_size, output_channels=3)
        
        # Discriminator is now conditioned on the pose vector
        human_pose_size = 17 * 3 # 17 keypoints, 3 values (x, y, conf)
        self.discriminator = HumanDiscriminator(input_channels=3, pose_vector_size=human_pose_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator only (for inference)."""
        return self.generator(vec)
    
    def generate(self, v_appearance: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
        """
        Generate human figure from appearance and pose vectors.
        
        Args:
            v_appearance: Appearance vector [B, 2048]
            p_t: Pose vector [B, 51]
            
        Returns:
            Generated image [B, 3, H, W]
        """
        # Concatenate appearance and pose vectors
        input_vec = torch.cat([v_appearance, p_t], dim=1)
        return self.generator(input_vec)
    
    def discriminate(self, img: torch.Tensor, pose_vector: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real vs fake images.
        
        Args:
            img: Image to discriminate [B, 3, H, W]
            pose_vector: Pose vector p_t [B, 51]
            
        Returns:
            Discriminator output
        """
        return self.discriminator(img, pose_vector)

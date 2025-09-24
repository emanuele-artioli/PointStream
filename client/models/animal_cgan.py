#!/usr/bin/env python3
"""
Animal cGAN Model

Conditional GAN for generating animal figures from keypoints and reference images.
Uses animal pose keypoint format (12 keypoints) for pose conditioning.
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
    """Generator network for animal figures from a concatenated appearance and pose vector with temporal context."""
    
    def __init__(self, vector_input_size: int = 2156, output_channels: int = 3, ngf: int = 64, 
                 latent_dim: int = 512, temporal_frames: int = 0):
        """
        Initialize animal generator.
        
        Args:
            vector_input_size: Dimension of the concatenated input vector [v_appearance, p_t, temporal_context].
            output_channels: Number of output channels (3 for RGB).
            ngf: Number of generator filters.
            latent_dim: Dimension of the latent space to project the input vector to.
            temporal_frames: Number of temporal frames included in input (0 = no temporal context).
        """
        super(AnimalGenerator, self).__init__()
        
        # Store model metadata for compatibility checking
        self.model_metadata = {
            'vector_input_size': vector_input_size,
            'temporal_frames': temporal_frames,
            'keypoint_channels': 12,  # Animal pose
            'include_confidence': True,  # Will be set based on actual input
            'model_version': '2.0'  # Version with temporal support
        }
        
        self.mapping_network = nn.Sequential(
            nn.Linear(vector_input_size, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.initial_projection = nn.Sequential(
            nn.Linear(latent_dim, ngf * 8 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.ngf = ngf

        self.decoder = nn.Sequential(
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
        
    def get_model_metadata(self):
        """Get generator metadata."""
        return {
            'type': 'AnimalGenerator',
            'vector_input_size': self.vector_input_size,
            'image_size': self.image_size,
            'ngf': self.ngf
        }
        
    def forward(self, vec):
        latent_vec = self.mapping_network(vec)
        initial_h = self.initial_projection(latent_vec)
        initial_h = initial_h.view(-1, self.ngf * 8, 4, 4)
        output = self.decoder(initial_h)
        return output


class AnimalDiscriminator(nn.Module):
    """Discriminator network for animal figures, conditioned on a pose vector."""
    
    def __init__(self, input_channels: int = 3, pose_vector_size: int = 24, ndf: int = 64):
        """
        Initialize animal discriminator.
        
        Args:
            input_channels: Number of input channels (3 for real/fake image).
            pose_vector_size: Dimension of the pose vector p_t (12 keypoints × 3 coordinates = 36).
            ndf: Number of discriminator filters.
        """
        super(AnimalDiscriminator, self).__init__()
        
        self.pose_projection = nn.Sequential(
            nn.Linear(pose_vector_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256 * 256)
        )

        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(input_channels + 1, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
        )
        
    def get_model_metadata(self):
        """Get discriminator metadata."""
        return {
            'type': 'AnimalDiscriminator',
            'pose_vector_size': self.pose_vector_size,
            'image_size': self.image_size,
            'ndf': self.ndf
        }
        
    def forward(self, input_img, pose_vector):
        pose_map = self.pose_projection(pose_vector).view(-1, 1, 256, 256)
        x = torch.cat([input_img, pose_map], 1)
        return self.main(x)


class AnimalCGAN(nn.Module):
    """Complete Animal cGAN model."""
    
    def __init__(self, input_size: int = 256, vector_input_size: int = 2156, 
                 temporal_frames: int = 0, include_confidence: bool = True):
        """
        Initialize Animal cGAN.
        
        Args:
            input_size: Output image size.
            vector_input_size: Dimension of the concatenated input vector [v_appearance, p_t].
            temporal_frames: Number of temporal frames included in input (for metadata).
            include_confidence: Whether confidence values are included in vectors (for metadata).
        """
        super(AnimalCGAN, self).__init__()
        
        self.input_size = input_size
        self.vector_input_size = vector_input_size
        self.temporal_frames = temporal_frames
        self.include_confidence = include_confidence
        
        self.generator = AnimalGenerator(vector_input_size=vector_input_size, 
                                       output_channels=3, temporal_frames=temporal_frames)
        
        # Calculate pose size from vector_input_size: total - appearance (2048)
        animal_pose_size = vector_input_size - 2048
        self.discriminator = AnimalDiscriminator(input_channels=3, pose_vector_size=animal_pose_size)
        
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
    
    def get_model_metadata(self):
        """Get comprehensive model metadata for compatibility checking."""
        return {
            'input_size': self.input_size,
            'vector_input_size': self.vector_input_size,
            'temporal_frames': self.temporal_frames,
            'include_confidence': self.include_confidence,
            'keypoint_channels': 12,  # Animal pose
            'model_version': '2.0',  # Version with temporal support
            'generator_metadata': self.generator.get_model_metadata(),
            'discriminator_metadata': self.discriminator.get_model_metadata()
        }
    
    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator only (for inference)."""
        return self.generator(vec)
    
    def generate(self, v_appearance: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
        """
        Generate animal figure from appearance and pose vectors.
        
        Args:
            v_appearance: Appearance vector [B, 2048]
            p_t: Pose vector [B, 36] (12 keypoints × 3 coordinates)
            
        Returns:
            Generated image [B, 3, H, W]
        """
        input_vec = torch.cat([v_appearance, p_t], dim=1)
        return self.generator(input_vec)
    
    def discriminate(self, img: torch.Tensor, pose_vector: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real vs fake images.
        
        Args:
            img: Image to discriminate [B, 3, H, W]
            pose_vector: Pose vector p_t
            
        Returns:
            Discriminator output
        """
        return self.discriminator(img, pose_vector)

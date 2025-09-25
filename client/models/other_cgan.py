#!/usr/bin/env python3
"""
Other Objects cGAN Model

Conditional GAN for generating other objects from edge/corner features and reference images.
Uses edge detection and corner features for conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class OtherResidualBlock(nn.Module):
    """Residual block for other objects generator."""
    
    def __init__(self, channels: int):
        super(OtherResidualBlock, self).__init__()
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


class AttentionBlock(nn.Module):
    """Self-attention block for better feature understanding."""
    
    def __init__(self, in_channels: int):
        super(AttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        # Attention
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=2)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Apply gamma weighting and residual connection
        out = self.gamma * out + x
        return out


class OtherGenerator(nn.Module):
    """Generator network for other objects from a concatenated appearance and pose vector with temporal context."""
    
    def __init__(self, vector_input_size: int = 2264, output_channels: int = 3, ngf: int = 64, 
                 latent_dim: int = 512, temporal_frames: int = 0):
        """
        Initialize other objects generator.
        
        Args:
            vector_input_size: Dimension of the concatenated input vector [v_appearance, p_t, temporal_context].
            output_channels: Number of output channels (3 for RGB).
            ngf: Number of generator filters.
            latent_dim: Dimension of the latent space to project the input vector to.
            temporal_frames: Number of temporal frames included in input (0 = no temporal context).
        """
        super(OtherGenerator, self).__init__()
        
        # Store vector input size as instance attribute
        self.vector_input_size = vector_input_size
        self.temporal_frames = temporal_frames
        self.image_size = 256  # Output image size (256x256)
        self.ngf = ngf
        
        # Store model metadata for compatibility checking
        self.model_metadata = {
            'vector_input_size': vector_input_size,
            'temporal_frames': temporal_frames,
            'keypoint_channels': 24,  # Enhanced other keypoints
            'include_confidence': True,  # Will be set based on actual input
            'model_version': '2.0'  # Version with temporal support and enhanced keypoints
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

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False), # -> 8x8
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            AttentionBlock(ngf * 8), # Add attention for better structure
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
            'type': 'OtherGenerator',
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


class OtherDiscriminator(nn.Module):
    """Discriminator network for other objects, conditioned on a pose vector."""
    
    def __init__(self, input_channels: int = 3, pose_vector_size: int = 216, ndf: int = 64):
        """
        Initialize other objects discriminator.
        
        Args:
            input_channels: Number of input channels (3 for real/fake image).
            pose_vector_size: Dimension of the pose vector p_t (enhanced keypoints with temporal context).
            ndf: Number of discriminator filters.
        """
        super(OtherDiscriminator, self).__init__()
        
        # Store attributes
        self.pose_vector_size = pose_vector_size
        self.image_size = 256
        self.ndf = ndf
        
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
        
        # Fine-detail discriminator remains unconditioned for simplicity
        self.fine_detail = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(input_channels, ndf // 2, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf // 2, ndf, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 3, 1, 1),
        )
        
    def get_model_metadata(self):
        """Get discriminator metadata."""
        return {
            'type': 'OtherDiscriminator',
            'pose_vector_size': self.pose_vector_size,
            'image_size': self.image_size,
            'ndf': self.ndf
        }
        
    def forward(self, input_img, pose_vector):
        # Condition the main branch on the pose vector
        pose_map = self.pose_projection(pose_vector).view(-1, 1, 256, 256)
        x_main = torch.cat([input_img, pose_map], 1)
        main_out = self.main(x_main)
        
        # The fine_detail branch works on the image only
        fine_out = self.fine_detail(input_img)
        
        return main_out, fine_out


class OtherCGAN(nn.Module):
    """Complete Other objects cGAN model."""
    
    def __init__(self, input_size: int = 256, vector_input_size: int = 2264, 
                 temporal_frames: int = 0, include_confidence: bool = True):
        """
        Initialize Other objects cGAN.
        
        Args:
            input_size: Output image size.
            vector_input_size: Dimension of the concatenated input vector [v_appearance, p_t].
            temporal_frames: Number of temporal frames included in input (for metadata).
            include_confidence: Whether confidence values are included in vectors (for metadata).
        """
        super(OtherCGAN, self).__init__()
        
        self.input_size = input_size
        self.vector_input_size = vector_input_size
        self.temporal_frames = temporal_frames
        self.include_confidence = include_confidence
        
        self.generator = OtherGenerator(vector_input_size=vector_input_size, 
                                      output_channels=3, temporal_frames=temporal_frames)
        
        # Calculate pose vector size based on configuration
        # For 'other' category: 24 keypoints * 3 coords * (1 + temporal_frames) = 24*3*3 = 216 with 2 temporal frames
        other_pose_size = vector_input_size - 2048  # Subtract appearance vector size to get pose vector size
        self.discriminator = OtherDiscriminator(input_channels=3, pose_vector_size=other_pose_size)
        
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
            'keypoint_channels': 24,  # Enhanced other objects pose
            'model_version': '2.0',  # Version with temporal support
            'generator_metadata': self.generator.get_model_metadata(),
            'discriminator_metadata': self.discriminator.get_model_metadata()
        }
    
    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator only (for inference)."""
        return self.generator(vec)
    
    def generate(self, v_appearance: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
        """
        Generate object from appearance and pose vectors.
        
        Args:
            v_appearance: Appearance vector [B, 2048]
            p_t: Pose vector (normalized bbox) [B, 4]
            
        Returns:
            Generated image [B, 3, H, W]
        """
        input_vec = torch.cat([v_appearance, p_t], dim=1)
        return self.generator(input_vec)
    
    def discriminate(self, img: torch.Tensor, pose_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discriminate real vs fake images with multi-scale output.
        
        Args:
            img: Image to discriminate [B, 3, H, W]
            pose_vector: Pose vector p_t (bbox) [B, 4]
            
        Returns:
            Tuple of (main_output, fine_detail_output)
        """
        return self.discriminator(img, pose_vector)

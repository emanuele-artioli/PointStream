#!/usr/bin/env python3
"""
Human cGAN Model (Image-based)

Conditional GAN for generating human figures from appearance vectors and pose skeleton images.
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


class HumanGenerator(nn.Module):
    """Generator network for human figures from appearance vector and pose image."""
    
    def __init__(self, appearance_vector_size: int = 2048, pose_channels: int = 1,
                 output_channels: int = 3, ngf: int = 64):
        """
        Initialize human generator.
        
        Args:
            appearance_vector_size: Dimension of the appearance vector (e.g., 2048 from ResNet).
            pose_channels: Number of channels in pose image (1 for grayscale, 3 for RGB).
            output_channels: Number of output channels (3 for RGB).
            ngf: Number of generator filters.
        """
        super(HumanGenerator, self).__init__()
        
        self.appearance_vector_size = appearance_vector_size
        self.pose_channels = pose_channels
        self.ngf = ngf  # Store ngf as instance variable
        
        # Store model metadata
        self.model_metadata = {
            'appearance_vector_size': appearance_vector_size,
            'pose_channels': pose_channels,
            'model_version': '2.0',
            'input_type': 'image',
            'category': 'human'
        }
        
        # Pose encoder - encode pose image to feature maps
        self.pose_encoder = nn.Sequential(
            # Input: [B, pose_channels, 256, 256] -> [B, ngf, 128, 128]
            nn.Conv2d(pose_channels, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, ngf, 128, 128] -> [B, ngf*2, 64, 64]
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, ngf*2, 64, 64] -> [B, ngf*4, 32, 32]
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, ngf*4, 32, 32] -> [B, ngf*8, 16, 16]
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, ngf*8, 16, 16] -> [B, ngf*8, 8, 8]
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Appearance projection - project appearance vector to spatial features
        # Project to same spatial size as pose features for combination
        self.appearance_projection = nn.Sequential(
            nn.Linear(appearance_vector_size, ngf * 8 * 8 * 8),  # 8x8 feature map
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Feature fusion - combine pose and appearance features
        # Input: [B, ngf*8 + ngf*8, 8, 8] = [B, ngf*16, 8, 8]
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(ngf * 16, ngf * 8, 3, 1, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(ngf * 8),
            ResidualBlock(ngf * 8),
        )
        
        # Decoder - upsample fused features to final image
        self.decoder = nn.Sequential(
            # [B, ngf*8, 8, 8] -> [B, ngf*8, 16, 16]
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # [B, ngf*8, 16, 16] -> [B, ngf*4, 32, 32]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # [B, ngf*4, 32, 32] -> [B, ngf*2, 64, 64]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # [B, ngf*2, 64, 64] -> [B, ngf, 128, 128]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # [B, ngf, 128, 128] -> [B, output_channels, 256, 256]
            nn.ConvTranspose2d(ngf, output_channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def get_model_metadata(self):
        """Get model metadata for compatibility checking."""
        return self.model_metadata.copy()
        
    def forward(self, appearance_vec: torch.Tensor, pose_img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            appearance_vec: Appearance vector [B, 2048]
            pose_img: Pose skeleton image [B, pose_channels, 256, 256]
            
        Returns:
            Generated image [B, 3, 256, 256]
        """
        # Encode pose image to spatial features
        pose_features = self.pose_encoder(pose_img)  # [B, ngf*8, 8, 8]
        
        # Project appearance vector to spatial features
        appearance_projected = self.appearance_projection(appearance_vec)  # [B, ngf*8*8*8]
        appearance_features = appearance_projected.view(-1, self.ngf * 8, 8, 8)  # [B, ngf*8, 8, 8]
        
        # Combine pose and appearance features
        combined_features = torch.cat([pose_features, appearance_features], dim=1)  # [B, ngf*16, 8, 8]
        
        # Fuse features
        fused_features = self.feature_fusion(combined_features)  # [B, ngf*8, 8, 8]
        
        # Decode to final image
        output = self.decoder(fused_features)  # [B, 3, 256, 256]
        
        return output


class HumanDiscriminator(nn.Module):
    """Discriminator network for human figures, conditioned on pose image."""
    
    def __init__(self, input_channels: int = 3, pose_channels: int = 1, ndf: int = 64):
        """
        Initialize human discriminator.
        
        Args:
            input_channels: Number of channels in real/fake image (3 for RGB).
            pose_channels: Number of channels in pose image.
            ndf: Number of discriminator filters.
        """
        super(HumanDiscriminator, self).__init__()
        
        self.model_metadata = {
            'input_channels': input_channels,
            'pose_channels': pose_channels,
            'model_version': '2.0',
            'input_type': 'image',
            'category': 'human'
        }
        
        # Main discriminator network
        # Input: [B, input_channels + pose_channels, 256, 256]
        self.main = nn.Sequential(
            # [B, input_channels + pose_channels, 256, 256] -> [B, ndf, 128, 128]
            nn.Conv2d(input_channels + pose_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, ndf, 128, 128] -> [B, ndf*2, 64, 64]
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, ndf*2, 64, 64] -> [B, ndf*4, 32, 32]
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, ndf*4, 32, 32] -> [B, ndf*8, 16, 16]
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, ndf*8, 16, 16] -> [B, ndf*8, 8, 8]
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [B, ndf*8, 8, 8] -> [B, 1, 4, 4]
            nn.Conv2d(ndf * 8, 1, 4, 2, 1),
        )
        
    def get_model_metadata(self):
        """Get model metadata for compatibility checking."""
        return self.model_metadata.copy()
        
    def forward(self, input_img: torch.Tensor, pose_img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_img: Real or fake image [B, 3, 256, 256]
            pose_img: Pose skeleton image [B, pose_channels, 256, 256]
            
        Returns:
            Discriminator output [B, 1, 4, 4]
        """
        # Concatenate image with pose image
        x = torch.cat([input_img, pose_img], dim=1)
        return self.main(x)


class HumanCGAN(nn.Module):
    """Complete Human cGAN model with pose image input."""
    
    def __init__(self, input_size: int = 256, pose_channels: int = 1, device: torch.device = None):
        """
        Initialize Human cGAN.
        
        Args:
            input_size: Output image size.
            pose_channels: Number of channels in pose image (1 for grayscale, 3 for RGB).
            device: PyTorch device.
        """
        super(HumanCGAN, self).__init__()
        
        self.input_size = input_size
        self.pose_channels = pose_channels
        self.device = device
        
        self.generator = HumanGenerator(
            appearance_vector_size=2048,
            pose_channels=pose_channels,
            output_channels=3
        )
        
        self.discriminator = HumanDiscriminator(
            input_channels=3,
            pose_channels=pose_channels
        )
        
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
            'pose_channels': self.pose_channels,
            'model_version': '2.0',
            'input_type': 'image',
            'category': 'human',
            'generator_metadata': self.generator.get_model_metadata(),
            'discriminator_metadata': self.discriminator.get_model_metadata()
        }
    
    def forward(self, appearance_vec: torch.Tensor, pose_img: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator only (for inference)."""
        return self.generator(appearance_vec, pose_img)
    
    def generate(self, appearance_vec: torch.Tensor, pose_img: torch.Tensor) -> torch.Tensor:
        """
        Generate human figure from appearance vector and pose image.
        
        Args:
            appearance_vec: Appearance vector [B, 2048]
            pose_img: Pose skeleton image [B, pose_channels, 256, 256]
            
        Returns:
            Generated image [B, 3, 256, 256]
        """
        return self.generator(appearance_vec, pose_img)
    
    def discriminate(self, img: torch.Tensor, pose_img: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real vs fake images.
        
        Args:
            img: Image to discriminate [B, 3, 256, 256]
            pose_img: Pose skeleton image [B, pose_channels, 256, 256]
            
        Returns:
            Discriminator output
        """
        return self.discriminator(img, pose_img)
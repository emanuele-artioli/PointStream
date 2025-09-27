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
    """Generator network for human figures from appearance vector and pose image."""
    
    def __init__(self, appearance_vector_size: int = 2048, pose_image_channels: int = 3, 
                 output_channels: int = 3, ngf: int = 64):
        """
        Initialize human generator that takes appearance vector + pose image.
        
        Args:
            appearance_vector_size: Dimension of the appearance vector (default: 2048 from ResNet).
            pose_image_channels: Number of channels in pose image (3 for RGB skeleton).
            output_channels: Number of output channels (3 for RGB).
            ngf: Number of generator filters.
        """
        super(HumanGenerator, self).__init__()
        
        self.appearance_vector_size = appearance_vector_size
        self.pose_image_channels = pose_image_channels
        
        # Store model metadata for compatibility checking
        self.model_metadata = {
            'appearance_vector_size': appearance_vector_size,
            'pose_image_channels': pose_image_channels,
            'input_type': 'vector_plus_image',  # New input type
            'model_version': '3.0'  # Version with pose images
        }
        
        # Pose image encoder - extract features from pose skeleton
        self.pose_encoder = nn.Sequential(
            # Input: pose image [B, 3, 256, 256]
            nn.Conv2d(pose_image_channels, ngf, 4, 2, 1),  # -> [B, 64, 128, 128]
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),  # -> [B, 128, 64, 64]
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),  # -> [B, 256, 32, 32]
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1),  # -> [B, 512, 16, 16]
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),  # -> [B, 512, 8, 8]
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Appearance vector processor
        self.appearance_processor = nn.Sequential(
            nn.Linear(appearance_vector_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Fusion layer - combine pose features with appearance
        # Pose features: [B, 512, 8, 8] = 512 * 64 = 32768 when flattened
        # Appearance features: [B, 512]
        self.fusion_layer = nn.Sequential(
            nn.Linear(32768 + 512, 512),  # Fuse pose spatial features + appearance
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, ngf * 8 * 4 * 4),  # Project to decoder input
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
        
    def get_model_metadata(self):
        """Get model metadata for compatibility checking."""
        return self.model_metadata.copy()
        
    def forward(self, appearance_vec, pose_img):
        """
        Forward pass with appearance vector and pose image.
        
        Args:
            appearance_vec: Appearance vector [B, 2048]
            pose_img: Pose skeleton image [B, 3, 256, 256]
        """
        # Extract features from pose image
        pose_features = self.pose_encoder(pose_img)  # [B, 512, 8, 8]
        pose_features_flat = pose_features.view(pose_features.size(0), -1)  # [B, 32768]
        
        # Process appearance vector
        appearance_features = self.appearance_processor(appearance_vec)  # [B, 512]
        
        # Fuse features
        fused_features = torch.cat([pose_features_flat, appearance_features], dim=1)  # [B, 33280]
        fused_output = self.fusion_layer(fused_features)  # [B, ngf*8*4*4]
        
        # Reshape for decoder
        decoder_input = fused_output.view(-1, self.ngf * 8, 4, 4)
        
        # Generate the image
        output = self.decoder(decoder_input)
        
        return output


class HumanDiscriminator(nn.Module):
    """Discriminator network for human figures, conditioned on a pose image."""
    
    def __init__(self, input_channels: int = 3, pose_image_channels: int = 3, ndf: int = 64):
        """
        Initialize human discriminator.
        
        Args:
            input_channels: Number of input channels for real/fake image (3 for RGB).
            pose_image_channels: Number of channels in pose image (3 for RGB skeleton).
            ndf: Number of discriminator filters.
        """
        super(HumanDiscriminator, self).__init__()
        
        # Store model metadata for compatibility checking
        self.model_metadata = {
            'input_channels': input_channels,
            'pose_image_channels': pose_image_channels,
            'input_type': 'image_plus_image',  # New input type
            'model_version': '3.0'  # Version with pose images
        }
        
        # Discriminator takes concatenated real/fake image + pose image
        self.main = nn.Sequential(
            # Input: [B, 6, 256, 256] (3 for image + 3 for pose image)
            nn.Conv2d(input_channels + pose_image_channels, ndf, 4, 2, 1),
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
        
    def get_model_metadata(self):
        """Get model metadata for compatibility checking."""
        return self.model_metadata.copy()
        
    def forward(self, input_img, pose_img):
        """
        Forward pass with input image and pose image.
        
        Args:
            input_img: Real or fake image [B, 3, 256, 256]
            pose_img: Pose skeleton image [B, 3, 256, 256]
        """
        # Concatenate image with pose image
        x = torch.cat([input_img, pose_img], 1)  # [B, 6, 256, 256]
        return self.main(x)


class HumanCGAN(nn.Module):
    """Complete Human cGAN model with pose image input."""
    
    def __init__(self, input_size: int = 256, appearance_vector_size: int = 2048, 
                 pose_image_channels: int = 3):
        """
        Initialize Human cGAN.
        
        Args:
            input_size: Output image size.
            appearance_vector_size: Dimension of the appearance vector (default: 2048 from ResNet).
            pose_image_channels: Number of channels in pose image (3 for RGB).
        """
        super(HumanCGAN, self).__init__()
        
        self.input_size = input_size
        self.appearance_vector_size = appearance_vector_size
        self.pose_image_channels = pose_image_channels
        
        self.generator = HumanGenerator(
            appearance_vector_size=appearance_vector_size,
            pose_image_channels=pose_image_channels,
            output_channels=3
        )
        
        self.discriminator = HumanDiscriminator(
            input_channels=3, 
            pose_image_channels=pose_image_channels
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
            'appearance_vector_size': self.appearance_vector_size,
            'pose_image_channels': self.pose_image_channels,
            'input_type': 'vector_plus_image',  # New input type
            'model_version': '3.0',  # Version with pose images
            'generator_metadata': self.generator.get_model_metadata(),
            'discriminator_metadata': self.discriminator.get_model_metadata()
        }
    
    def forward(self, appearance_vec: torch.Tensor, pose_img: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator only (for inference)."""
        return self.generator(appearance_vec, pose_img)
    
    def generate(self, v_appearance: torch.Tensor, pose_img: torch.Tensor) -> torch.Tensor:
        """
        Generate human figure from appearance vector and pose image.
        
        Args:
            v_appearance: Appearance vector [B, 2048]
            pose_img: Pose skeleton image [B, 3, 256, 256]
            
        Returns:
            Generated image [B, 3, H, W]
        """
        return self.generator(v_appearance, pose_img)
    
    def discriminate(self, img: torch.Tensor, pose_img: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real vs fake images.
        
        Args:
            img: Image to discriminate [B, 3, H, W]
            pose_img: Pose skeleton image [B, 3, H, W]
            
        Returns:
            Discriminator output
        """
        return self.discriminator(img, pose_img)

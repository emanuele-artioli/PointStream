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
    """Generator network for other objects."""
    
    def __init__(self, input_channels: int = 4, output_channels: int = 3, ngf: int = 64):
        """
        Initialize other objects generator.
        
        Args:
            input_channels: Number of input channels (3 for RGB + 1 for features)
            output_channels: Number of output channels (3 for RGB)
            ngf: Number of generator filters
        """
        super(OtherGenerator, self).__init__()
        
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
        
        # Bottleneck with residual blocks and attention
        self.bottleneck = nn.Sequential(
            OtherResidualBlock(ngf * 8),
            AttentionBlock(ngf * 8),
            OtherResidualBlock(ngf * 8),
            OtherResidualBlock(ngf * 8)
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


class OtherDiscriminator(nn.Module):
    """Discriminator network for other objects."""
    
    def __init__(self, input_channels: int = 6, ndf: int = 64):
        """
        Initialize other objects discriminator.
        
        Args:
            input_channels: Number of input channels (real/fake + condition)
            ndf: Number of discriminator filters
        """
        super(OtherDiscriminator, self).__init__()
        
        # Multi-scale discriminator for better detail capture
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
        
        # Additional fine-detail discriminator
        self.fine_detail = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(input_channels, ndf // 2, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf // 2, ndf, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 3, 1, 1),
        )
        
    def forward(self, input_img, condition_img):
        x = torch.cat([input_img, condition_img], 1)
        
        # Main discriminator output
        main_out = self.main(x)
        
        # Fine detail discriminator output
        fine_out = self.fine_detail(x)
        
        return main_out, fine_out


class OtherCGAN(nn.Module):
    """Complete Other objects cGAN model."""
    
    def __init__(self, input_size: int = 256, feature_channels: int = 50):
        """
        Initialize Other objects cGAN.
        
        Args:
            input_size: Input image size
            feature_channels: Number of feature channels (edges, corners, etc.)
        """
        super(OtherCGAN, self).__init__()
        
        self.input_size = input_size
        self.feature_channels = feature_channels
        
        # Generator takes 4 channels: 3 (RGB reference) + 1 (feature map)
        self.generator = OtherGenerator(input_channels=4, output_channels=3)
        
        # Discriminator takes 6 channels: 3 (real/fake) + 3 (reference condition)
        self.discriminator = OtherDiscriminator(input_channels=6)
        
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
    
    def generate(self, reference_img: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Generate object from reference and features.
        
        Args:
            reference_img: Reference image [B, 3, H, W]
            feature_map: Feature map [B, 1, H, W]
            
        Returns:
            Generated image [B, 3, H, W]
        """
        input_tensor = torch.cat([reference_img, feature_map], dim=1)
        return self.generator(input_tensor)
    
    def discriminate(self, img: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discriminate real vs fake images with multi-scale output.
        
        Args:
            img: Image to discriminate [B, 3, H, W]
            condition: Conditioning image [B, 3, H, W]
            
        Returns:
            Tuple of (main_output, fine_detail_output)
        """
        return self.discriminator(img, condition)

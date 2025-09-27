#!/usr/bin/env python3
"""
Other Objects cGAN Model (Image-based)

Conditional GAN for generating other objects from appearance vectors and pose/bbox images.
Takes pose as image input instead of vector for better spatial understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from client.models.animal_cgan_image import AnimalGenerator, AnimalDiscriminator


class OtherGenerator(AnimalGenerator):
    """Generator for other objects - reuses animal architecture."""
    
    def __init__(self, appearance_vector_size: int = 2048, pose_channels: int = 1,
                 output_channels: int = 3, ngf: int = 64):
        super(OtherGenerator, self).__init__(appearance_vector_size, pose_channels, output_channels, ngf)
        self.model_metadata['category'] = 'other'


class OtherDiscriminator(AnimalDiscriminator):
    """Discriminator for other objects - reuses animal architecture."""
    
    def __init__(self, input_channels: int = 3, pose_channels: int = 1, ndf: int = 64):
        super(OtherDiscriminator, self).__init__(input_channels, pose_channels, ndf)
        self.model_metadata['category'] = 'other'


class OtherCGAN(nn.Module):
    """Complete Other objects cGAN model with pose image input."""
    
    def __init__(self, input_size: int = 256, pose_channels: int = 1, device: torch.device = None):
        super(OtherCGAN, self).__init__()
        
        self.input_size = input_size
        self.pose_channels = pose_channels
        self.device = device
        
        self.generator = OtherGenerator(
            appearance_vector_size=2048,
            pose_channels=pose_channels,
            output_channels=3
        )
        
        self.discriminator = OtherDiscriminator(
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
            'category': 'other',
            'generator_metadata': self.generator.get_model_metadata(),
            'discriminator_metadata': self.discriminator.get_model_metadata()
        }
    
    def generate(self, appearance_vec: torch.Tensor, pose_img: torch.Tensor) -> torch.Tensor:
        return self.generator(appearance_vec, pose_img)
    
    def discriminate(self, img: torch.Tensor, pose_img: torch.Tensor) -> torch.Tensor:
        return self.discriminator(img, pose_img)
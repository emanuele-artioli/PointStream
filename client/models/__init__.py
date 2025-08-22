#!/usr/bin/env python3
"""
PointStream Client Models

Generative models for client-side video reconstruction.
"""

from .human_cgan import HumanCGAN, HumanGenerator, HumanDiscriminator
from .animal_cgan import AnimalCGAN, AnimalGenerator, AnimalDiscriminator
from .other_cgan import OtherCGAN, OtherGenerator, OtherDiscriminator

__all__ = [
    # Human models
    'HumanCGAN',
    'HumanGenerator', 
    'HumanDiscriminator',
    
    # Animal models
    'AnimalCGAN',
    'AnimalGenerator',
    'AnimalDiscriminator',
    
    # Other objects models
    'OtherCGAN',
    'OtherGenerator',
    'OtherDiscriminator'
]

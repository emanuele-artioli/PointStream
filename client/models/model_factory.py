#!/usr/bin/env python3
"""
Model Factory

Factory functions to create the appropriate model architectures based on configuration.
Supports both vector-based and image-based pose inputs for all categories.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from utils import config

# Import both vector and image-based models
from client.models.human_cgan import HumanCGAN as VectorHumanCGAN
from client.models.human_cgan_image import HumanCGAN as ImageHumanCGAN
from client.models.animal_cgan import AnimalCGAN as VectorAnimalCGAN  
from client.models.animal_cgan_image import AnimalCGAN as ImageAnimalCGAN
from client.models.other_cgan import OtherCGAN as VectorOtherCGAN
from client.models.other_cgan_image import OtherCGAN as ImageOtherCGAN


def create_human_model(device: torch.device, **kwargs) -> nn.Module:
    """
    Create human model based on configuration.
    
    Args:
        device: PyTorch device
        **kwargs: Additional model parameters
        
    Returns:
        Configured human model
    """
    pose_input_type = config.get_str('models', 'pose_input_type', 'vector')
    
    if pose_input_type == 'image':
        # Image-based model
        grayscale = config.get_bool('models', 'pose_image_grayscale', False)
        pose_channels = 1 if grayscale else 3
        
        return ImageHumanCGAN(
            input_size=config.get_int('models', 'human_input_size', 256),
            pose_channels=pose_channels,
            device=device,
            **kwargs
        )
    else:
        # Vector-based model (original)
        keypoint_channels = config.get_int('keypoints', 'human_num_keypoints', 17)
        include_confidence = config.get_bool('keypoints', 'include_confidence_in_vectors', True)
        temporal_frames = config.get_int('keypoints', 'temporal_frames', 0)
        
        # Calculate vector size
        if include_confidence:
            pose_size = keypoint_channels * 3  # x, y, confidence
        else:
            pose_size = keypoint_channels * 2  # x, y only
        
        pose_size *= (1 + temporal_frames)  # Include temporal context
        vector_input_size = 2048 + pose_size  # appearance + pose
        
        return VectorHumanCGAN(
            input_size=config.get_int('models', 'human_input_size', 256),
            vector_input_size=vector_input_size,
            temporal_frames=temporal_frames,
            include_confidence=include_confidence,
            **kwargs
        )


def create_animal_model(device: torch.device, **kwargs) -> nn.Module:
    """Create animal model based on configuration."""
    pose_input_type = config.get_str('models', 'pose_input_type', 'vector')
    
    if pose_input_type == 'image':
        # Image-based model
        grayscale = config.get_bool('models', 'pose_image_grayscale', False)
        pose_channels = 1 if grayscale else 3
        
        return ImageAnimalCGAN(
            input_size=config.get_int('models', 'animal_input_size', 256),
            pose_channels=pose_channels,
            device=device,
            **kwargs
        )
    else:
        # Vector-based model (original)
        keypoint_channels = config.get_int('keypoints', 'animal_num_keypoints', 12)
        include_confidence = config.get_bool('keypoints', 'include_confidence_in_vectors', True)
        temporal_frames = config.get_int('keypoints', 'temporal_frames', 0)
        
        # Calculate vector size
        if include_confidence:
            pose_size = keypoint_channels * 3  # x, y, confidence
        else:
            pose_size = keypoint_channels * 2  # x, y only
        
        pose_size *= (1 + temporal_frames)  # Include temporal context
        vector_input_size = 2048 + pose_size  # appearance + pose
        
        return VectorAnimalCGAN(
            input_size=config.get_int('models', 'animal_input_size', 256),
            vector_input_size=vector_input_size,
            temporal_frames=temporal_frames,
            include_confidence=include_confidence,
            **kwargs
        )


def create_other_model(device: torch.device, **kwargs) -> nn.Module:
    """Create other objects model based on configuration."""
    pose_input_type = config.get_str('models', 'pose_input_type', 'vector')
    
    if pose_input_type == 'image':
        # Image-based model
        grayscale = config.get_bool('models', 'pose_image_grayscale', False)
        pose_channels = 1 if grayscale else 3
        
        return ImageOtherCGAN(
            input_size=config.get_int('models', 'other_input_size', 256),
            pose_channels=pose_channels,
            device=device,
            **kwargs
        )
    else:
        # Vector-based model (original)
        keypoint_channels = config.get_int('keypoints', 'other_num_keypoints', 24)
        include_confidence = config.get_bool('keypoints', 'include_confidence_in_vectors', True)
        temporal_frames = config.get_int('keypoints', 'temporal_frames', 0)
        
        # Calculate vector size
        if include_confidence:
            pose_size = keypoint_channels * 3  # x, y, confidence
        else:
            pose_size = keypoint_channels * 2  # x, y only
        
        pose_size *= (1 + temporal_frames)  # Include temporal context
        vector_input_size = 2048 + pose_size  # appearance + pose
        
        return VectorOtherCGAN(
            input_size=config.get_int('models', 'other_input_size', 256),
            vector_input_size=vector_input_size,
            temporal_frames=temporal_frames,
            include_confidence=include_confidence,
            **kwargs
        )


def get_model_input_requirements(category: str) -> Dict[str, Any]:
    """
    Get input requirements for a specific category model.
    
    Args:
        category: Model category (human, animal, other)
        
    Returns:
        Dictionary with input requirements
    """
    pose_input_type = config.get_str('models', 'pose_input_type', 'vector')
    
    requirements = {
        'pose_input_type': pose_input_type,
        'appearance_vector_size': 2048,  # ResNet-50 features
    }
    
    if pose_input_type == 'image':
        # Image-based requirements
        grayscale = config.get_bool('models', 'pose_image_grayscale', False)
        low_resolution = config.get_bool('models', 'pose_image_low_resolution', False)
        
        if category == 'human':
            image_size = config.get_int('models', 'human_input_size', 256)
        elif category == 'animal':
            image_size = config.get_int('models', 'animal_input_size', 256)
        else:
            image_size = config.get_int('models', 'other_input_size', 256)
        
        requirements.update({
            'pose_image_size': (image_size, image_size),
            'pose_image_channels': 1 if grayscale else 3,
            'pose_image_grayscale': grayscale,
            'pose_image_low_resolution': low_resolution,
        })
    else:
        # Vector-based requirements
        if category == 'human':
            keypoint_channels = config.get_int('keypoints', 'human_num_keypoints', 17)
        elif category == 'animal':
            keypoint_channels = config.get_int('keypoints', 'animal_num_keypoints', 12)
        else:
            keypoint_channels = config.get_int('keypoints', 'other_num_keypoints', 24)
        
        include_confidence = config.get_bool('keypoints', 'include_confidence_in_vectors', True)
        temporal_frames = config.get_int('keypoints', 'temporal_frames', 0)
        
        if include_confidence:
            pose_size = keypoint_channels * 3
        else:
            pose_size = keypoint_channels * 2
            
        pose_size *= (1 + temporal_frames)
        
        requirements.update({
            'pose_vector_size': pose_size,
            'keypoint_channels': keypoint_channels,
            'include_confidence': include_confidence,
            'temporal_frames': temporal_frames,
            'total_vector_size': 2048 + pose_size,
        })
    
    return requirements


def is_pose_image_mode() -> bool:
    """Check if we're using pose images instead of vectors."""
    return config.get_str('models', 'pose_input_type', 'vector') == 'image'


def get_pose_image_config() -> Dict[str, Any]:
    """Get pose image configuration."""
    return {
        'grayscale': config.get_bool('models', 'pose_image_grayscale', False),
        'low_resolution': config.get_bool('models', 'pose_image_low_resolution', False),
    }
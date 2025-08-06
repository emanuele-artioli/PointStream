"""
Model handler for MMPose using the high-level MMPoseInferencer API.
"""
from typing import List, Dict, Any
import numpy as np
import torch
from mmpose.apis import MMPoseInferencer
from .. import config

class MMPoseHandler:
    """A wrapper for the MMPoseInferencer to handle different model types."""

    def __init__(self, model_alias: str = 'human'):
        """Initializes the MMPose inferencers for different categories."""
        print(f" -> Initializing MMPose inferencer for {model_alias}...")
        self.model_alias = model_alias
        
        # Determine device with CUDA fallback to CPU
        device = self._get_safe_device()
        print(f" -> Using device: {device}")
        
        try:
            if model_alias == 'human':
                self.inferencer = MMPoseInferencer(
                    pose2d=config.MMPOSE_HUMAN_MODEL_ALIAS, device=device)
            elif model_alias == 'animal':
                self.inferencer = MMPoseInferencer(
                    pose2d=config.MMPOSE_ANIMAL_MODEL_ALIAS, device=device)
            else:
                # Default to human model
                self.inferencer = MMPoseInferencer(
                    pose2d=config.MMPOSE_HUMAN_MODEL_ALIAS, device=device)
            
            print(f" -> MMPose inferencer for {model_alias} initialized successfully.")
        except Exception as e:
            print(f" -> Warning: Failed to initialize MMPose inferencer: {e}")
            print(" -> Falling back to mock implementation")
            self.inferencer = None
    
    def _get_safe_device(self):
        """Get a safe device for MMPose, with fallback to CPU on CUDA issues."""
        if not torch.cuda.is_available():
            return 'cpu'
        
        try:
            # Test CUDA availability with a simple operation
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
            return config.DEVICE
        except Exception as e:
            print(f" -> CUDA test failed: {e}")
            print(" -> Falling back to CPU for MMPose")
            return 'cpu'

    def extract_poses(self, frames: List[np.ndarray], category: str) -> List[np.ndarray]:
        """Extracts poses from a series of frames for a given category."""
        all_keypoints = []
        
        # If inferencer failed to initialize, return empty keypoints
        if self.inferencer is None:
            print(f"     -> MMPose not available, returning empty keypoints")
            for frame in frames:
                all_keypoints.append(np.array([]))
            return all_keypoints
        
        # Skip animal pose estimation for now due to potential CUDA compatibility issues
        if category == 'animal':
            print(f"     -> Skipping animal pose estimation due to CUDA compatibility issues")
            # Return empty keypoints for animals
            for frame in frames:
                all_keypoints.append(np.array([]))
            return all_keypoints
        
        for frame in frames:
            try:
                result_generator = self.inferencer(frame, return_datasamples=True)
                result = next(result_generator)
                
                if 'predictions' in result and result['predictions'] and result['predictions'][0].pred_instances:
                    keypoints = result['predictions'][0].pred_instances.keypoints[0]
                    all_keypoints.append(keypoints)
                else:
                    # Return empty array if no pose detected
                    all_keypoints.append(np.array([]))
            except Exception as e:
                print(f"     -> Error extracting pose: {e}")
                print(f"     -> This is likely a CUDA compatibility issue, continuing with empty keypoints")
                all_keypoints.append(np.array([]))
        
        return all_keypoints
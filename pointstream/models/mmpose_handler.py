"""
Model handler for MMPose using the high-level MMPoseInferencer API.
"""
from typing import List, Dict, Any
import numpy as np
import torch

# Import warning suppression first
from ..utils.warning_suppression import with_suppressed_warnings, WarningFilter

from mmpose.apis import MMPoseInferencer
from .. import config

class MMPoseHandler:
    """A wrapper for the MMPoseInferencer to handle different model types."""

    @with_suppressed_warnings
    def __init__(self, model_alias: str = 'human'):
        """Initializes the MMPose inferencers for different categories."""
        print(f" -> Initializing MMPose inferencer for {model_alias}...")
        self.model_alias = model_alias
        self.cpu_fallback_triggered = False  # Track if we've fallen back to CPU
        
        # Determine device with CUDA fallback to CPU
        device = self._get_safe_device()
        print(f" -> Using device: {device}")
        self.current_device = device
        
        try:
            with WarningFilter():  # Suppress warnings during initialization
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
            # Test CUDA availability with a more comprehensive check
            test_tensor = torch.tensor([1.0]).cuda()
            test_result = test_tensor * 2  # Simple operation
            del test_tensor, test_result
            
            # Additional check for CUDA kernel compatibility
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print(" -> nvidia-smi not available, forcing CPU")
                return 'cpu'
            
            return config.DEVICE
        except Exception as e:
            print(f" -> CUDA compatibility test failed: {e}")
            print(" -> Forcing CPU fallback for MMPose due to CUDA issues")
            return 'cpu'

    def extract_poses(self, frames: List[np.ndarray], category: str) -> List[np.ndarray]:
        """Extracts poses from a series of frames for a given category."""
        all_keypoints = []
        
        # If inferencer failed to initialize, return empty keypoints
        if self.inferencer is None:
            print(f"     -> MMPose not available, returning empty keypoints for {len(frames)} frames")
            return [np.array([]) for _ in frames]
        
        # Process frames with better error handling
        success_count = 0
        for i, frame in enumerate(frames):
            try:
                with WarningFilter():  # Suppress warnings during inference
                    result_generator = self.inferencer(frame, return_datasamples=True)
                    result = next(result_generator)
                
                if 'predictions' in result and result['predictions'] and result['predictions'][0].pred_instances:
                    keypoints = result['predictions'][0].pred_instances.keypoints[0]
                    all_keypoints.append(keypoints)
                    success_count += 1
                else:
                    # Return empty array if no pose detected
                    all_keypoints.append(np.array([]))
                    
            except RuntimeError as e:
                if "CUDA" in str(e) and not self.cpu_fallback_triggered:
                    print(f"     -> CUDA error detected: {str(e)[:100]}...")
                    print(f"     -> Permanently switching to CPU for this session")
                    self.cpu_fallback_triggered = True
                    # Reinitialize inferencer on CPU
                    try:
                        with WarningFilter():
                            if self.model_alias == 'human':
                                self.inferencer = MMPoseInferencer(
                                    pose2d=config.MMPOSE_HUMAN_MODEL_ALIAS, device='cpu')
                            elif self.model_alias == 'animal':
                                self.inferencer = MMPoseInferencer(
                                    pose2d=config.MMPOSE_ANIMAL_MODEL_ALIAS, device='cpu')
                            else:
                                self.inferencer = MMPoseInferencer(
                                    pose2d=config.MMPOSE_HUMAN_MODEL_ALIAS, device='cpu')
                        self.current_device = 'cpu'
                        
                        # Retry the current frame on CPU
                        result_generator = self.inferencer(frame, return_datasamples=True)
                        result = next(result_generator)
                        if 'predictions' in result and result['predictions'] and result['predictions'][0].pred_instances:
                            keypoints = result['predictions'][0].pred_instances.keypoints[0]
                            all_keypoints.append(keypoints)
                            success_count += 1
                        else:
                            all_keypoints.append(np.array([]))
                    except Exception:
                        all_keypoints.append(np.array([]))
                        # Mark inferencer as failed
                        self.inferencer = None
                        print("     -> CPU fallback also failed, disabling MMPose for this session")
                        break
                else:
                    all_keypoints.append(np.array([]))
            except Exception as e:
                all_keypoints.append(np.array([]))
        
        # If we hit an error and disabled the inferencer, fill remaining frames with empty arrays
        while len(all_keypoints) < len(frames):
            all_keypoints.append(np.array([]))
        
        if success_count > 0:
            print(f"     -> Successfully extracted keypoints from {success_count}/{len(frames)} frames")
        else:
            print(f"     -> No valid keypoints extracted from {len(frames)} frames")
        
        return all_keypoints
"""
Model handler for MMPose using the high-level MMPoseInferencer API.
"""
from typing import List, Dict, Any
import numpy as np
from mmpose.apis import MMPoseInferencer
from .. import config

class MMPoseHandler:
    """A wrapper for the MMPoseInferencer to handle different model types."""

    def __init__(self):
        """Initializes the MMPose inferencers for different categories."""
        print(" -> Initializing MMPose inferencers...")
        self.human_inferencer = MMPoseInferencer(
            pose2d=config.MMPOSE_HUMAN_MODEL_ALIAS, device=config.DEVICE)
        self.animal_inferencer = MMPoseInferencer(
            pose2d=config.MMPOSE_ANIMAL_MODEL_ALIAS, device=config.DEVICE)
        print(" -> MMPose inferencers initialized.")

    def extract_poses(self, frames: List[np.ndarray], category: str) -> List[np.ndarray]:
        """Extracts poses from a series of frames for a given category."""
        if category == 'person':
            inferencer = self.human_inferencer
        elif category == 'animal':
            inferencer = self.animal_inferencer
        else:
            return []

        all_keypoints = []
        for frame in frames:
            result_generator = inferencer(frame, return_datasamples=True)
            result = next(result_generator)
            
            if 'predictions' in result and result['predictions'] and result['predictions'][0].pred_instances:
                keypoints = result['predictions'][0].pred_instances.keypoints[0]
                all_keypoints.append(keypoints)
            else:
                all_keypoints.append(np.array([]))
        
        return all_keypoints
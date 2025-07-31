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
        """
        Initializes the MMPose inferencers.
        Models and weights are downloaded automatically on first use.
        """
        print(" -> Initializing MMPose inferencers...")
        self.human_inferencer = MMPoseInferencer(
            pose2d=config.MMPOSE_MODEL_ALIAS,
            device=config.DEVICE
        )
        print(" -> MMPose inferencers initialized.")

    def extract_poses(self, frames: List[np.ndarray], bboxes: List[List[int]], category: str) -> List[np.ndarray]:
        """
        Extracts poses from a series of frames given bounding boxes and a category.
        """
        if category == 'person':
            inferencer = self.human_inferencer
        else:
            return []

        all_keypoints = []
        # Call the inferencer for each frame; this is the most robust method.
        for frame, bbox in zip(frames, bboxes):
            # Pass bbox as a standard Python list to avoid internal library errors.
            result_generator = inferencer(frame, bboxes=[bbox], return_datasamples=True)
            result = next(result_generator)
            
            if 'predictions' in result and result['predictions'] and result['predictions'][0].pred_instances:
                keypoints = result['predictions'][0].pred_instances.keypoints[0]
                all_keypoints.append(keypoints)
            else:
                print("     -> Warning: No pose detected in one of the frames.")
        
        return all_keypoints
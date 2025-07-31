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
        Initializes the MMPose inferencers for different categories.
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
        # FIX: Call the inferencer explicitly for each frame in a loop. This is the
        # most robust way to interact with the API and avoids internal bugs.
        for frame, bbox in zip(frames, bboxes):
            # The input is the frame, and bboxes are passed as a keyword argument.
            # Crucially, pass a standard Python list `[bbox]` to avoid the ValueError.
            result_generator = inferencer(frame, bboxes=[bbox], return_datasamples=True)
            result = next(result_generator)
            
            # Add a defensive check to ensure a pose was detected.
            if 'predictions' in result and result['predictions'] and result['predictions'][0].pred_instances:
                keypoints = result['predictions'][0].pred_instances.keypoints[0]
                all_keypoints.append(keypoints)
            else:
                print("     -> Warning: No pose detected in one of the frames.")
        
        return all_keypoints
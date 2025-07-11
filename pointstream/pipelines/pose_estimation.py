from .base import PoseEstimator
from typing import List, Any
import logging

class MMPoseEstimator(PoseEstimator):
    def estimate(self, frame: Any, objects: List[Any]) -> List[Any]:
        logging.info("Estimating poses with MMPose...")
        # Placeholder: Return dummy poses
        return [{"keypoints": [[20, 20, 0.9], [30, 30, 0.8]]}] * len(objects)

class MediaPipePoseEstimator(PoseEstimator):
     def estimate(self, frame: Any, objects: List[Any]) -> List[Any]:
        raise NotImplementedError("MediaPipe has not been implemented.")
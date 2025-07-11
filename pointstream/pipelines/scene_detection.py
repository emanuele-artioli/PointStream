from .base import SceneDetector
from typing import List, Any
import logging

class SSIMSceneDetector(SceneDetector):
    def detect(self, frame_buffer: List[Any]) -> List[int]:
        logging.info("Detecting scene cuts using SSIM...")
        # Placeholder: Pretend a cut happens every 100 frames
        if len(frame_buffer) > 100:
            return [100]
        return []

class PySceneDetectSceneDetector(SceneDetector):
    def detect(self, frame_buffer: List[Any]) -> List[int]:
        raise NotImplementedError("PySceneDetect has not been implemented.")
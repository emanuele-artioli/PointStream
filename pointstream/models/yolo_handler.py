"""
Model handler for YOLO object detection and tracking.
"""
from typing import List, Dict, Any
import numpy as np
from ultralytics import YOLO
from .. import config

class YOLOHandler:
    """A wrapper for the ultralytics YOLO model."""

    def __init__(self, model_path: str = config.STUDENT_MODEL_PATH):
        """Initializes the YOLO model."""
        print(f" -> Initializing YOLO model from: {model_path}")
        self.model = YOLO(model_path)

    def track_objects(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Tracks objects through a list of frames for a single scene.

        Args:
            frames: A list of video frames.

        Returns:
            A list of lists, where each inner list contains detection
            dictionaries for a single frame.
        """
        print(f"  -> Tracking objects across {len(frames)} frames...")
        
        # Use stream=True for memory-efficient processing of frame lists
        results_generator = self.model.track(
            source=frames, 
            persist=True, 
            stream=True, 
            verbose=False,
            conf=config.DETECTION_CONFIDENCE_THRESHOLD
        )

        all_frames_detections = []
        for frame_results in results_generator:
            current_frame_detections = []
            if frame_results.boxes.id is not None:
                for i in range(len(frame_results.boxes.id)):
                    detection = {
                        "track_id": int(frame_results.boxes.id[i]),
                        "class_id": int(frame_results.boxes.cls[i]),
                        "class_name": self.model.names[int(frame_results.boxes.cls[i])],
                        "confidence": float(frame_results.boxes.conf[i]),
                        "bbox_normalized": frame_results.boxes.xyxyn[i].cpu().numpy().tolist()
                    }
                    current_frame_detections.append(detection)
            
            all_frames_detections.append(current_frame_detections)
            
        return all_frames_detections

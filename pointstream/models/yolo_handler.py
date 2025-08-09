"""
Model handler for YOLO object detection and tracking.
"""
from typing import List, Dict, Any
import numpy as np
from ultralytics import YOLO
from .. import config

class YOLOHandler:
    """A wrapper for the ultralytics YOLO model."""

    # FIX: The __init__ method now accepts a model_path
    def __init__(self, model_path: str):
        """Initializes the YOLO model from a given path."""
        print(f" -> Initializing YOLO model from: {model_path}")
        self.model = YOLO(model_path)

    def track_objects(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Tracks objects through a list of frames for a single scene."""
        print(f"  -> Tracking objects across {len(frames)} frames...")
        
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
        
        # Clear CUDA cache after processing scene to prevent memory accumulation
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return all_frames_detections
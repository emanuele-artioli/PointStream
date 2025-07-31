"""
Model handler for YOLOv8 object detection and tracking.
"""
from typing import List, Dict, Any
import numpy as np
from ultralytics import YOLO
from .. import config

class YOLOHandler:
    """A wrapper for the ultralytics YOLO model."""

    def __init__(self, model_path: str = config.STUDENT_MODEL_PATH):
        """
        Initializes the YOLO model.

        Args:
            model_path: Path to the YOLO model weights file (.pt).
                        The model will be downloaded automatically by ultralytics
                        if the path doesn't exist.
        """
        print(f" -> Initializing YOLO model from: {model_path}")
        self.model = YOLO(model_path)

    def track_objects(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Tracks objects through a list of frames.

        Args:
            frames: A list of video frames for a single scene.

        Returns:
            A list of lists, where each inner list contains detection
            dictionaries for a single frame.
        """
        print(f"  -> Tracking objects across {len(frames)} frames...")
        
        # The 'stream=True' argument is more memory-efficient for lists of frames
        results_generator = self.model.track(source=frames, persist=True, stream=True, verbose=False)

        all_frames_detections = []
        for frame_results in results_generator:
            current_frame_detections = []
            # Check if there are any detections with tracking IDs
            if frame_results.boxes.id is not None:
                for i in range(len(frame_results.boxes.id)):
                    track_id = int(frame_results.boxes.id[i])
                    class_id = int(frame_results.boxes.cls[i])
                    class_name = self.model.names[class_id]
                    confidence = float(frame_results.boxes.conf[i])
                    bbox = frame_results.boxes.xyxyn[i].cpu().numpy().tolist() # Normalized [x1, y1, x2, y2]

                    detection = {
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox_normalized": bbox
                    }
                    current_frame_detections.append(detection)
            
            all_frames_detections.append(current_frame_detections)
            
        return all_frames_detections
"""
Stage 2: Foreground Object Detection and Tracking.
"""
from typing import List, Dict, Any, Generator
from ..models.yolo_handler import AdaptiveTracker

Scene = Dict[str, Any]

# FIX: The function now accepts a model_path argument and content_type for adaptive tracking
def run_detection_pipeline(scene_generator: Generator[Scene, None, None], 
                         model_path: str, 
                         content_type: str = "general") -> Generator[Scene, None, None]:
    """
    Orchestrates the object detection stage in a streaming fashion.
    """
    print("\n--- Starting Stage 2: Object Detection & Tracking (Streaming) ---")
    # FIX: Initialize the adaptive tracker with the specific model for this run
    yolo_handler = AdaptiveTracker(model_path=model_path)
    
    for scene in scene_generator:
        print(f"  -> Stage 2 processing Scene {scene['scene_index']}...")
        
        if scene["motion_type"] in ["STATIC", "SIMPLE"]:
            # Use adaptive tracking with content type for threshold learning
            detections = yolo_handler.track_objects(scene["frames"], content_type)
            scene["detections"] = detections
        else:
            print("     -> Skipping detection for COMPLEX scene.")
            scene["detections"] = []
        
        yield scene
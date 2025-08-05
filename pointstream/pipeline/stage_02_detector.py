"""
Stage 2: Foreground Object Detection and Tracking.
"""
from typing import List, Dict, Any, Generator
from ..models.yolo_handler import YOLOHandler

Scene = Dict[str, Any]

# FIX: The function now accepts a model_path argument
def run_detection_pipeline(scene_generator: Generator[Scene, None, None], model_path: str) -> Generator[Scene, None, None]:
    """
    Orchestrates the object detection stage in a streaming fashion.
    """
    print("\n--- Starting Stage 2: Object Detection & Tracking (Streaming) ---")
    # FIX: Initialize the handler with the specific model for this run
    yolo_handler = YOLOHandler(model_path=model_path)
    
    for scene in scene_generator:
        print(f"  -> Stage 2 processing Scene {scene['scene_index']}...")
        
        if scene["motion_type"] in ["STATIC", "SIMPLE"]:
            detections = yolo_handler.track_objects(scene["frames"])
            scene["detections"] = detections
        else:
            print("     -> Skipping detection for COMPLEX scene.")
            scene["detections"] = []
        
        yield scene
"""
Stage 2: Foreground Object Detection and Tracking.

This module consumes a generator of scenes, and for each 'STATIC' or 'SIMPLE'
scene, it runs a YOLO model to detect and track foreground objects.
"""
from typing import Dict, Any, Generator
from ..models.yolo_handler import YOLOHandler

# A type alias for clarity
Scene = Dict[str, Any]

def run_detection_pipeline(scene_generator: Generator[Scene, None, None]) -> Generator[Scene, None, None]:
    """
    Orchestrates the object detection stage in a streaming fashion.
    This version KEEPS the frames in the scene dictionary for the next stage.
    """
    print("\n--- Starting Stage 2: Object Detection & Tracking (Streaming) ---")
    yolo_handler = YOLOHandler()
    
    for scene in scene_generator:
        print(f"  -> Stage 2 processing Scene {scene['scene_index']}...")
        
        frames = scene["frames"] # Use the frames, but don't remove them
        
        if scene["motion_type"] in ["STATIC", "SIMPLE"]:
            detections = yolo_handler.track_objects(frames)
            scene["detections"] = detections
        else:
            print("     -> Skipping detection for COMPLEX scene.")
            scene["detections"] = []
        
        yield scene
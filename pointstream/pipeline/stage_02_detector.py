"""
Stage 2: Foreground Object Detection and Tracking.
"""
from typing import List, Dict, Any, Generator
from ..models.yolo_handler import YOLOHandler

Scene = Dict[str, Any] # Type alias for clarity

def run_detection_pipeline(scene_generator: Generator[Scene, None, None]) -> Generator[Scene, None, None]:
    """
    Orchestrates the object detection stage in a streaming fashion.

    Args:
        scene_generator: A generator that yields scene dictionaries from Stage 1.

    Yields:
        The updated scene dictionaries, now with detection data. The 'frames'
        key is passed through for use in subsequent stages.
    """
    print("\n--- Starting Stage 2: Object Detection & Tracking (Streaming) ---")
    yolo_handler = YOLOHandler()
    
    for scene in scene_generator:
        print(f"  -> Stage 2 processing Scene {scene['scene_index']}...")
        
        if scene["motion_type"] in ["STATIC", "SIMPLE"]:
            # This is a scene we want to process
            detections = yolo_handler.track_objects(scene["frames"])
            scene["detections"] = detections
        else:
            # This is a 'COMPLEX' scene, so we skip detection
            print("     -> Skipping detection for COMPLEX scene.")
            scene["detections"] = []
        
        yield scene
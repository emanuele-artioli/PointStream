"""
Stage 3: Background Modeling.

This module consumes scenes from the detection stage. For each scene, it
generates a static background image and calculates camera motion.
"""
from typing import Dict, Any, Generator, List
import numpy as np
import cv2
from pathlib import Path
from .. import config

# A type alias for clarity
Scene = Dict[str, Any]

def _create_static_background(frames: List[np.ndarray]) -> np.ndarray:
    """
    Creates a clean background from a list of frames by taking the median.
    This is effective at removing moving objects in static camera scenes.
    """
    print("  -> Creating background from median of frames...")
    # Stack frames along a new axis (depth)
    frame_stack = np.stack(frames, axis=0)
    # Calculate the median along the depth axis
    median_frame = np.median(frame_stack, axis=0).astype(np.uint8)
    return median_frame

def _calculate_camera_motion(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Calculates the frame-to-frame affine transformation for a list of frames.
    
    Returns:
        A list of 2x3 affine transformation matrices.
    """
    print("  -> Calculating camera motion...")
    motion_matrices = []
    
    scale = config.MOTION_DOWNSAMPLE_FACTOR
    prev_frame_gray = cv2.cvtColor(cv2.resize(frames[0], (0, 0), fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)

    identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    motion_matrices.append(identity_matrix)

    for i in range(1, len(frames)):
        frame_gray = cv2.cvtColor(cv2.resize(frames[i], (0, 0), fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        
        try:
            # FIX: Wrap the estimation in a try-except block to handle cases
            # where OpenCV cannot find enough matching points.
            transform_matrix, _ = cv2.estimateAffine2D(prev_frame_gray, frame_gray)
            
            if transform_matrix is None:
                motion_matrices.append(identity_matrix)
            else:
                motion_matrices.append(transform_matrix)
        except cv2.error:
            # If an internal OpenCV error occurs, default to no motion.
            motion_matrices.append(identity_matrix)
        
        prev_frame_gray = frame_gray
        
    return motion_matrices


def run_background_modeling_pipeline(scene_generator: Generator[Scene, None, None], video_stem: str) -> Generator[Scene, None, None]:
    """
    Orchestrates the background modeling stage in a streaming fashion.
    """
    print("\n--- Starting Stage 3: Background Modeling (Streaming) ---")
    
    for scene in scene_generator:
        print(f"  -> Stage 3 processing Scene {scene['scene_index']}...")
        
        # FIX: Use the frames but DO NOT pop them from the scene dictionary.
        frames = scene["frames"]
        background_image = None
        camera_motion = []

        if scene['motion_type'] == 'STATIC':
            background_image = _create_static_background(frames)
            identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            camera_motion = [identity_matrix] * len(frames)
        
        elif scene['motion_type'] == 'SIMPLE':
            camera_motion = _calculate_camera_motion(frames)
            background_image = frames[0]
        
        else: # COMPLEX scene
             print("     -> Skipping background modeling for COMPLEX scene.")

        if background_image is not None:
            bg_filename = f"{video_stem}_scene_{scene['scene_index']}_background.png"
            bg_path = config.OUTPUT_DIR / bg_filename
            cv2.imwrite(str(bg_path), background_image)
            scene['background_image_path'] = str(bg_path)
        else:
            scene['background_image_path'] = None
            
        scene['camera_motion'] = camera_motion
        
        yield scene
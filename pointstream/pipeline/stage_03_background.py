"""
Stage 3: Background Modeling.
"""
from typing import Dict, Any, Generator, List
import numpy as np
import cv2
from pathlib import Path
from .. import config

Scene = Dict[str, Any] # Type alias for clarity

def _create_static_background(frames: List[np.ndarray]) -> np.ndarray:
    """Creates a clean background from a list of frames by taking the median."""
    print("  -> Creating background from median of frames...")
    frame_stack = np.stack(frames, axis=0)
    median_frame = np.median(frame_stack, axis=0).astype(np.uint8)
    return median_frame

def _calculate_camera_motion(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Calculates the frame-to-frame affine transformation for a list of frames."""
    print("  -> Calculating camera motion...")
    motion_matrices = []
    scale = config.MOTION_DOWNSAMPLE_FACTOR
    prev_gray = cv2.cvtColor(cv2.resize(frames[0], (0,0), fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    
    identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    motion_matrices.append(identity_matrix)

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(cv2.resize(frames[i], (0,0), fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        try:
            transform_matrix, _ = cv2.estimateAffine2D(prev_gray, curr_gray)
            if transform_matrix is None:
                motion_matrices.append(identity_matrix)
            else:
                motion_matrices.append(transform_matrix)
        except cv2.error:
            motion_matrices.append(identity_matrix)
        prev_gray = curr_gray
        
    return motion_matrices

def run_background_modeling_pipeline(scene_generator: Generator[Scene, None, None], video_stem: str) -> Generator[Scene, None, None]:
    """Orchestrates the background modeling stage in a streaming fashion."""
    print("\n--- Starting Stage 3: Background Modeling (Streaming) ---")
    
    for scene in scene_generator:
        print(f"  -> Stage 3 processing Scene {scene['scene_index']}...")
        
        frames = scene["frames"]
        background_image = None
        camera_motion = []

        if scene['motion_type'] == 'STATIC':
            background_image = _create_static_background(frames)
            identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            camera_motion = [identity_matrix] * len(frames)
        
        elif scene['motion_type'] == 'SIMPLE':
            camera_motion = _calculate_camera_motion(frames)
            # As a placeholder, we use the first frame. A full implementation
            # would use the motion matrices to stitch a panorama.
            background_image = frames[0]
        
        else: # COMPLEX scene
             print("     -> Skipping background modeling for COMPLEX scene.")

        # Save the background image and add its path to the scene dict
        if background_image is not None:
            bg_filename = f"{video_stem}_scene_{scene['scene_index']}_background.png"
            bg_path = config.OUTPUT_DIR / bg_filename
            cv2.imwrite(str(bg_path), background_image)
            scene['background_image_path'] = str(bg_path)
        else:
            scene['background_image_path'] = None
            
        scene['camera_motion'] = camera_motion
        
        # This is the last stage to use the raw frames, so we remove them.
        del scene['frames']
        
        yield scene
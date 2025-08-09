"""
Stage 3: Background Modeling.
"""
from typing import Dict, Any, Generator, List, Tuple, Optional
import numpy as np
import cv2
from pathlib import Path
from .. import config
from ..models.segmentation_handler import get_segmentation_handler

Scene = Dict[str, Any] # Type alias for clarity

def _create_object_masks_from_detections(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Create precise segmentation masks for all detected objects in a frame."""
    segmentation_handler = get_segmentation_handler()
    return segmentation_handler.create_precise_masks(frame, detections)

def _inpaint_background(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint masked regions using OpenCV's inpainting algorithm."""
    if np.sum(mask) == 0:  # No objects to inpaint
        return frame
    
    # Use Navier-Stokes based inpainting for better results
    inpainted = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    return inpainted

def _create_static_background_with_inpainting(frames: List[np.ndarray], 
                                            frame_detections: List[List[Dict[str, Any]]]) -> np.ndarray:
    """Creates a clean background by taking the first frame and inpainting foreground objects."""
    print("  -> Creating background from first frame with foreground inpainting...")
    
    # Use the first frame as base
    base_frame = frames[0].copy()
    
    # Create mask for all detected objects in the first frame
    if frame_detections and len(frame_detections) > 0:
        object_mask = _create_object_masks_from_detections(base_frame, frame_detections[0])
        
        # Inpaint the masked regions
        background = _inpaint_background(base_frame, object_mask)
        print(f"     -> Inpainted {np.sum(object_mask > 0)} pixels")
    else:
        background = base_frame
        print("     -> No objects detected, using first frame as-is")
    
    return background

def _calculate_camera_motion(frames: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
    """Calculates the frame-to-frame affine transformation and cumulative motion."""
    print("  -> Calculating camera motion...")
    motion_matrices = []
    cumulative_motion_vectors = []
    scale = config.MOTION_DOWNSAMPLE_FACTOR
    
    # Convert frames to grayscale for motion estimation
    gray_frames = []
    for frame in frames:
        gray = cv2.cvtColor(cv2.resize(frame, (0,0), fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)
    
    identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    motion_matrices.append(identity_matrix)
    cumulative_motion_vectors.append([0.0, 0.0])
    
    # Track cumulative motion from first frame
    cumulative_dx, cumulative_dy = 0.0, 0.0
    
    for i in range(1, len(frames)):
        try:
            # Estimate affine transformation between consecutive frames
            transform_matrix = cv2.estimateAffinePartial2D(
                cv2.goodFeaturesToTrack(gray_frames[i-1], maxCorners=500, qualityLevel=0.01, minDistance=10),
                cv2.goodFeaturesToTrack(gray_frames[i], maxCorners=500, qualityLevel=0.01, minDistance=10)
            )[0]
            
            if transform_matrix is None:
                motion_matrices.append(identity_matrix)
                cumulative_motion_vectors.append([cumulative_dx, cumulative_dy])
            else:
                # Extract translation components
                dx = transform_matrix[0, 2] / scale  # Scale back to original resolution
                dy = transform_matrix[1, 2] / scale
                
                # Accumulate motion
                cumulative_dx += dx
                cumulative_dy += dy
                
                # Create cumulative transformation matrix
                cumulative_matrix = np.array([
                    [1.0, 0.0, cumulative_dx],
                    [0.0, 1.0, cumulative_dy]
                ])
                
                motion_matrices.append(cumulative_matrix)
                cumulative_motion_vectors.append([cumulative_dx, cumulative_dy])
                
        except (cv2.error, TypeError):
            motion_matrices.append(identity_matrix)
            cumulative_motion_vectors.append([cumulative_dx, cumulative_dy])
    
    # Calculate average motion vector for metadata
    if cumulative_motion_vectors:
        avg_motion = np.mean(cumulative_motion_vectors, axis=0)
    else:
        avg_motion = np.array([0.0, 0.0])
        
    print(f"     -> Total motion: dx={cumulative_dx:.2f}, dy={cumulative_dy:.2f}")
    print(f"     -> Average motion vector: dx={avg_motion[0]:.2f}, dy={avg_motion[1]:.2f}")
    
    return motion_matrices, avg_motion

def _create_panorama_background(frames: List[np.ndarray], 
                              motion_matrices: List[np.ndarray],
                              frame_detections: List[List[Dict[str, Any]]]) -> np.ndarray:
    """Creates a panoramic background by stitching frames with proper cumulative motion."""
    print("  -> Creating panoramic background by stitching frames...")
    
    if len(frames) < 2:
        return frames[0]
    
    h, w = frames[0].shape[:2]
    
    # Calculate the panorama dimensions based on motion matrices
    min_x, max_x = 0, w
    min_y, max_y = 0, h
    
    for matrix in motion_matrices:
        if matrix is not None:
            dx, dy = matrix[0, 2], matrix[1, 2]
            min_x = min(min_x, dx)
            max_x = max(max_x, w + dx)
            min_y = min(min_y, dy)
            max_y = max(max_y, h + dy)
    
    # Create panorama canvas
    panorama_width = int(max_x - min_x) + 100  # Add some padding
    panorama_height = int(max_y - min_y) + 100
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    weight_map = np.zeros((panorama_height, panorama_width), dtype=np.float32)
    
    # Calculate offset to ensure all frames fit in panorama
    offset_x = int(-min_x) + 50
    offset_y = int(-min_y) + 50
    
    print(f"     -> Panorama size: {panorama_width}x{panorama_height}")
    
    # Stitch all frames into panorama
    for i, (frame, matrix) in enumerate(zip(frames, motion_matrices)):
        if matrix is None:
            continue
            
        # Remove foreground objects from frame
        clean_frame = frame.copy()
        if i < len(frame_detections) and frame_detections[i]:
            object_mask = _create_object_masks_from_detections(clean_frame, frame_detections[i])
            clean_frame = _inpaint_background(clean_frame, object_mask)
        
        # Calculate position for this frame
        dx, dy = matrix[0, 2], matrix[1, 2]
        frame_x = int(offset_x + dx)
        frame_y = int(offset_y + dy)
        
        # Check bounds
        end_x = min(frame_x + w, panorama_width)
        end_y = min(frame_y + h, panorama_height)
        start_x = max(frame_x, 0)
        start_y = max(frame_y, 0)
        
        if start_x >= end_x or start_y >= end_y:
            continue
            
        # Calculate corresponding region in the frame
        frame_start_x = max(0, -frame_x + start_x)
        frame_start_y = max(0, -frame_y + start_y)
        frame_end_x = frame_start_x + (end_x - start_x)
        frame_end_y = frame_start_y + (end_y - start_y)
        
        # Weight for blending (higher weight for center frames)
        frame_weight = 1.0 if i == 0 else 0.7
        
        # Extract regions
        panorama_region = panorama[start_y:end_y, start_x:end_x]
        weight_region = weight_map[start_y:end_y, start_x:end_x]
        frame_region = clean_frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x]
        
        # Blend frames where they overlap
        mask = weight_region > 0
        if np.any(mask):
            # Weighted blending for overlapping regions
            total_weight = weight_region + frame_weight
            panorama_region[mask] = (
                (panorama_region[mask].astype(np.float32) * weight_region[mask, None] + 
                 frame_region[mask].astype(np.float32) * frame_weight) / 
                total_weight[mask, None]
            ).astype(np.uint8)
        else:
            # First frame in this region
            panorama_region[:] = frame_region
        
        # Update weight map
        weight_region += frame_weight
        
        print(f"     -> Stitched frame {i} at position ({frame_x}, {frame_y})")
    
    return panorama

def run_background_modeling_pipeline(scene_generator: Generator[Scene, None, None], video_stem: str) -> Generator[Scene, None, None]:
    """Orchestrates the background modeling stage in a streaming fashion."""
    print("\n--- Starting Stage 3: Background Modeling (Streaming) ---")
    
    for scene in scene_generator:
        print(f"  -> Stage 3 processing Scene {scene['scene_index']}...")
        
        frames = scene["frames"]
        detections = scene.get("detections", [])  # Frame-by-frame detections from Stage 2
        background_image = None
        camera_motion = []
        avg_motion_vector = np.array([0.0, 0.0])

        if scene['motion_type'] == 'STATIC':
            # For static scenes: take first frame, remove foreground objects, inpaint
            background_image = _create_static_background_with_inpainting(frames, detections)
            identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            camera_motion = [identity_matrix] * len(frames)
        
        elif scene['motion_type'] == 'SIMPLE':
            # For simple motion: calculate motion and create panorama
            motion_matrices, avg_motion_vector = _calculate_camera_motion(frames)
            background_image = _create_panorama_background(frames, motion_matrices, detections)
            camera_motion = motion_matrices
        
        else: # COMPLEX scene
            print("     -> Keeping frames for AV1 encoding of COMPLEX scene.")
            # For complex scenes, we'll encode with AV1 so keep the frames
            # Don't delete frames here, they'll be handled in run_server.py

        # Save the background image and add its path to the scene dict
        if background_image is not None:
            bg_filename = f"{video_stem}_scene_{scene['scene_index']}_background.png"
            bg_path = config.OUTPUT_DIR / bg_filename
            cv2.imwrite(str(bg_path), background_image)
            scene['background_image_path'] = str(bg_path)
            print(f"     -> Background saved to: {bg_filename}")
        else:
            scene['background_image_path'] = None
            
        scene['camera_motion'] = camera_motion
        scene['avg_motion_vector'] = avg_motion_vector.tolist()  # Convert to list for JSON serialization
        
        # Remove frames to save memory (except for complex scenes which need them for AV1 encoding)
        if scene['motion_type'] != 'COMPLEX':
            del scene['frames']
        
        yield scene
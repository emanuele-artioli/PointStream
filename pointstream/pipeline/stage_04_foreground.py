"""
Stage 4: Foreground Representation.
"""
from typing import Dict, Any, Generator, List
import numpy as np
import cv2
from pathlib import Path
from .. import config
try:
    from ..models.mmpose_handler import MMPoseHandler
except ImportError:
    print("[WARNING] MMPose not available, using mock implementation")
    from ..models.mock_mmpose_handler import MockMMPoseHandler as MMPoseHandler

# Import our new rigid keypoint extractor
from ..models.rigid_keypoint_extractor import extract_rigid_object_keypoints

Scene = Dict[str, Any]

def _extract_feature_keypoints(frames: List[np.ndarray], bboxes: List[List[int]]) -> List[np.ndarray]:
    """
    Extract meaningful keypoints from rigid objects using computer vision techniques.
    
    This replaces the simple bbox corner approach with proper edge detection,
    corner detection, and feature extraction.
    """
    print("  -> Extracting RIGID OBJECT keypoints using CV methods (edge detection, corners, features)...")
    
    keypoints_list = []
    
    for frame, bbox in zip(frames, bboxes):
        try:
            # Convert bbox format from [x1, y1, x2, y2] to [x, y, w, h]
            x1, y1, x2, y2 = bbox
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
            
            # Extract keypoints using our new method
            keypoints = extract_rigid_object_keypoints(
                image=frame,
                bbox=bbox_xywh,
                max_keypoints=15  # Reasonable number for rigid objects
            )
            
            # Convert to numpy array format expected by the pipeline
            if keypoints:
                keypoints_array = np.array(keypoints, dtype=np.float32)
            else:
                # Fallback to bbox corners if extraction fails
                keypoints_array = np.array([
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ], dtype=np.float32)
            
            keypoints_list.append(keypoints_array)
            
        except Exception as e:
            print(f"     -> Warning: Error extracting rigid keypoints: {e}")
            # Fallback to bbox corners
            x1, y1, x2, y2 = bbox
            keypoints_array = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=np.float32)
            keypoints_list.append(keypoints_array)
    
    return keypoints_list


def run_foreground_pipeline(scene_generator: Generator[Scene, None, None], video_path: str) -> Generator[Scene, None, None]:
    """Orchestrates the foreground representation stage."""
    print("\n--- Starting Stage 4: Foreground Representation (Streaming) ---")
    mmpose_handler = MMPoseHandler('human')  # Use default human model
    video_stem = Path(video_path).stem

    for scene in scene_generator:
        print(f"  -> Stage 4 processing Scene {scene['scene_index']}...")
        
        if scene['motion_type'] == 'COMPLEX' or not scene.get('detections'):
            print("     -> Skipping foreground analysis for COMPLEX or empty scene.")
            scene['foreground_objects'] = []
        else:
            frames = scene['frames']
            height, width, _ = frames[0].shape
            
            objects_by_track_id = {}
            # --- ADD LOGGING HERE ---
            print("     -> Raw Detections from Stage 2:")
            for frame_idx, frame_detections in enumerate(scene['detections']):
                if frame_detections:
                    print(f"        - Frame {scene['start_frame'] + frame_idx}: {frame_detections}")
                for detection in frame_detections:
                    track_id = detection['track_id']
                    if track_id not in objects_by_track_id:
                        objects_by_track_id[track_id] = {'class_name': detection['class_name'], 'frames': [], 'bboxes_abs': []}
                    
                    bbox_norm = detection['bbox_normalized']
                    bbox_abs = [int(bbox_norm[0] * width), int(bbox_norm[1] * height), int(bbox_norm[2] * width), int(bbox_norm[3] * height)]
                    objects_by_track_id[track_id]['frames'].append(frames[frame_idx])
                    objects_by_track_id[track_id]['bboxes_abs'].append(bbox_abs)

            scene['foreground_objects'] = []
            for track_id, data in objects_by_track_id.items():
                class_name = data['class_name']
                obj_frames, bboxes = data['frames'], data['bboxes_abs']
                keypoints, keypoint_type = [], "none"

                # --- UPDATED LOGIC ---
                if class_name == 'person':
                    print(f"     -> Extracting HUMAN keypoints for track ID {track_id}...")
                    keypoints = mmpose_handler.extract_poses(obj_frames, 'person')
                    keypoint_type = "mmpose_human"
                elif class_name in ['bear', 'horse', 'dog', 'cat']: # Expand as needed
                    print(f"     -> Extracting ANIMAL keypoints for track ID {track_id}...")
                    keypoints = mmpose_handler.extract_poses(obj_frames, 'animal')
                    keypoint_type = "mmpose_animal"
                else:
                    keypoints = _extract_feature_keypoints(obj_frames, bboxes)
                    keypoint_type = "rigid_cv_features"

                valid_frames_data = [(frame, bbox, kp) for frame, bbox, kp in zip(obj_frames, bboxes, keypoints) if kp.size > 0]
                if not valid_frames_data:
                    print(f"     -> No valid keypoints found for track ID {track_id}. Skipping.")
                    continue

                valid_frames, valid_bboxes, valid_keypoints = zip(*valid_frames_data)

                x1, y1, x2, y2 = valid_bboxes[0]
                appearance_patch = valid_frames[0][y1:y2, x1:x2]
                
                app_filename = f"{video_stem}_scene_{scene['scene_index']}_track_{track_id}_appearance.png"
                app_path = config.OUTPUT_DIR / app_filename
                cv2.imwrite(str(app_path), appearance_patch)
                
                scene['foreground_objects'].append({
                    "track_id": track_id, "class_name": class_name,
                    "appearance_path": str(app_path), "keypoint_type": keypoint_type,
                    "keypoints": valid_keypoints
                })

        # Final cleanup
        if 'frames' in scene: del scene['frames']
        if 'detections' in scene: del scene['detections']

        yield scene
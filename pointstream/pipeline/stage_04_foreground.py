"""
Stage 4: Foreground Representation.

This module processes scenes to extract appearance and motion (keypoints)
for each tracked foreground object.
"""
from typing import Dict, Any, Generator, List
import numpy as np
import cv2
from pathlib import Path
from .. import config
from ..models.mmpose_handler import MMPoseHandler

Scene = Dict[str, Any]
TrackedObjectData = Dict[str, Any]

# ... (_collect_tracked_objects and _extract_feature_keypoints are unchanged)
def _collect_tracked_objects(scene: Scene) -> Dict[int, TrackedObjectData]:
    tracked_objects: Dict[int, TrackedObjectData] = {}
    frames = scene['frames']
    height, width, _ = frames[0].shape
    for frame_idx, frame_detections in enumerate(scene['detections']):
        for detection in frame_detections:
            track_id = detection['track_id']
            if track_id not in tracked_objects:
                tracked_objects[track_id] = { "class_name": detection['class_name'], "frames": [], "bboxes_abs": [] }
            bbox_norm = detection['bbox_normalized']
            bbox_abs = [ int(bbox_norm[0] * width), int(bbox_norm[1] * height), int(bbox_norm[2] * width), int(bbox_norm[3] * height) ]
            tracked_objects[track_id]['frames'].append(frames[frame_idx])
            tracked_objects[track_id]['bboxes_abs'].append(bbox_abs)
    return tracked_objects

def _extract_feature_keypoints(frames: List[np.ndarray], bboxes: List[List[int]]) -> List[np.ndarray]:
    print("  -> NOTE: Using placeholder for rigid object keypoint extraction.")
    return [np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[2]], [bbox[0], bbox[2]]]) for bbox in bboxes]


def run_foreground_pipeline(scene_generator: Generator[Scene, None, None], video_stem: str) -> Generator[Scene, None, None]:
    """Orchestrates the foreground representation stage."""
    print("\n--- Starting Stage 4: Foreground Representation (Streaming) ---")
    mmpose_handler = MMPoseHandler()

    for scene in scene_generator:
        print(f"  -> Stage 4 processing Scene {scene['scene_index']}...")
        
        if scene['motion_type'] == 'COMPLEX' or not scene.get('detections'):
            print("     -> Skipping foreground analysis for COMPLEX or empty scene.")
            scene['foreground_objects'] = []
        else:
            tracked_objects = _collect_tracked_objects(scene)
            scene['foreground_objects'] = []

            for track_id, data in tracked_objects.items():
                class_name = data['class_name']
                frames = data['frames']
                bboxes = data['bboxes_abs']
                keypoints, keypoint_type = [], "none"

                if class_name == 'person':
                    print(f"     -> Extracting MMPose keypoints for track ID {track_id} ({class_name})...")
                    keypoints = mmpose_handler.extract_poses(frames, bboxes, category='person')
                    keypoint_type = "mmpose_human"
                else:
                    keypoints = _extract_feature_keypoints(frames, bboxes)
                    keypoint_type = "features_bbox_corners"

                if not keypoints:
                    print(f"     -> Could not extract keypoints for track ID {track_id}. Skipping.")
                    continue

                x1, y1, x2, y2 = bboxes[0]
                appearance_patch = frames[0][y1:y2, x1:x2]
                
                appearance_filename = f"{video_stem}_scene_{scene['scene_index']}_track_{track_id}_appearance.png"
                appearance_path = config.OUTPUT_DIR / appearance_filename
                cv2.imwrite(str(appearance_path), appearance_patch)
                
                scene['foreground_objects'].append({
                    "track_id": track_id,
                    "class_name": class_name,
                    "appearance_path": str(appearance_path),
                    "keypoint_type": keypoint_type,
                    "keypoints": keypoints
                })

        # FIX: Final cleanup before yielding to ensure a clean output.
        if 'frames' in scene: del scene['frames']
        if 'detections' in scene: del scene['detections']

        yield scene
"""
Client-side logic for reconstructing a video from the structured data
produced by the Pointstream server pipeline.
"""
from typing import Dict, Any, List, Generator
from pathlib import Path
import json
import cv2
import numpy as np
from ..utils.video_utils import save_frames_as_video

Scene = Dict[str, Any]

class Reconstructor:
    """
    Reconstructs a video from a Pointstream JSON data file.
    """

    def __init__(self, data_path: str):
        """
        Initializes the Reconstructor.

        Args:
            data_path: Path to the _final_results.json file.
        """
        print(f" -> Initializing Reconstructor with data from: {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Read metadata and scenes from the JSON file
        self.metadata = data.get("metadata", {})
        self.scenes = data.get("scenes", [])
        
        self.fps = self.metadata.get("fps", 30.0)
        self.resolution = self.metadata.get("resolution", [1280, 720]) # Default fallback

    def _reconstruct_scene(self, scene: Scene) -> Generator[np.ndarray, None, None]:
        """A generator that reconstructs and yields frames for a single scene."""
        if not scene.get('background_image_path'):
            print(f"  -> Skipping Scene {scene['scene_index']} (COMPLEX or no data).")
            return

        print(f"  -> Reconstructing Scene {scene['scene_index']}...")
        bg_image = cv2.imread(scene['background_image_path'])
        
        objects = {}
        for obj_data in scene['foreground_objects']:
            track_id = obj_data['track_id']
            appearance = cv2.imread(obj_data['appearance_path'], cv2.IMREAD_UNCHANGED)
            objects[track_id] = {
                'appearance': appearance,
                'keypoints': obj_data['keypoints'],  # Keep as list, don't convert to numpy array
            }
        
        num_frames = scene['end_frame'] - scene['start_frame'] + 1
        for i in range(num_frames):
            current_frame = bg_image.copy()

            for track_id, data in objects.items():
                if i >= len(data['keypoints']) or data['appearance'] is None:
                    continue
                
                appearance, keypoints = data['appearance'], data['keypoints'][i]
                
                # Convert keypoints list to numpy array
                keypoints = np.array(keypoints)
                
                # Handle both 2D and 3D keypoint formats (x,y) or (x,y,confidence)
                if len(keypoints) == 0:
                    continue  # Skip if no keypoints
                
                if keypoints.shape[-1] == 3:
                    # Extract only x, y coordinates, ignore confidence
                    keypoints_2d = keypoints[:, :2]
                else:
                    keypoints_2d = keypoints
                
                min_x, min_y = np.min(keypoints_2d, axis=0)
                max_x, max_y = np.max(keypoints_2d, axis=0)
                
                dst_pts = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y]], dtype=np.float32)
                h, w, _ = appearance.shape
                src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1]], dtype=np.float32)

                warp_matrix = cv2.getAffineTransform(src_pts, dst_pts)
                warped_appearance = cv2.warpAffine(appearance, warp_matrix, (self.resolution[0], self.resolution[1]))

                mask = np.all(warped_appearance > [0, 0, 0], axis=-1).astype(np.uint8) * 255
                
                locs = np.where(mask != 0)
                current_frame[locs[0], locs[1]] = warped_appearance[locs[0], locs[1]]
            
            yield current_frame

    def run(self, output_dir):
        """
        Runs the reconstruction for all scenes and saves each as a separate video file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n -> Starting reconstruction. Output will be saved in: {output_dir}")

        for scene in self.scenes:
            scene_frames = list(self._reconstruct_scene(scene))
            
            if scene_frames:
                scene_idx = scene['scene_index']
                output_path = output_dir / f"scene_{scene_idx}_reconstructed.mp4"
                print(f"  -> Saving Scene {scene_idx} to {output_path}...")
                save_frames_as_video(str(output_path), scene_frames, self.fps)
        
        print("\n -> Reconstruction complete.")
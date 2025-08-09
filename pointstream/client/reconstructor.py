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
        
        # Get motion information for proper background handling
        motion_type = scene.get('motion_type', 'STATIC')
        camera_motion = scene.get('camera_motion', [])
        
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
            # Get the correct background for this frame
            if motion_type == 'SIMPLE' and len(camera_motion) > i:
                # Use camera motion matrices to extract correct panorama portion
                current_frame = self._extract_frame_with_motion_matrix(bg_image, camera_motion[i])
            else:
                # For static scenes or single-frame backgrounds, use as-is
                current_frame = cv2.resize(bg_image, (self.resolution[0], self.resolution[1]))

            # Render foreground objects using improved method
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
                
                # Render object using improved generative method
                current_frame = self._render_object_generative(
                    current_frame, appearance, keypoints_2d, track_id
                )
            
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

    def _extract_frame_from_panorama(self, panorama: np.ndarray, frame_index: int, avg_motion_vector: List[float]) -> np.ndarray:
        """
        Extracts the correct portion of a panoramic background for a given frame.
        
        Args:
            panorama: The full panoramic background image
            frame_index: The current frame index (0-based within the scene)
            avg_motion_vector: The average motion vector [dx, dy] per frame
            
        Returns:
            The extracted background portion resized to match the original video resolution
        """
        target_width, target_height = self.resolution
        pano_height, pano_width = panorama.shape[:2]
        
        # Calculate the cumulative motion for this frame
        cumulative_dx = frame_index * avg_motion_vector[0]
        cumulative_dy = frame_index * avg_motion_vector[1]
        
        # Calculate the starting position in the panorama
        # We need to account for the fact that the panorama was built with an offset
        # to accommodate negative motions
        start_x = int(cumulative_dx)
        start_y = int(cumulative_dy)
        
        # Ensure we don't go out of bounds
        start_x = max(0, min(start_x, pano_width - target_width))
        start_y = max(0, min(start_y, pano_height - target_height))
        
        # Extract the portion
        end_x = min(start_x + target_width, pano_width)
        end_y = min(start_y + target_height, pano_height)
        
        extracted = panorama[start_y:end_y, start_x:end_x]
        
        # If the extracted portion is smaller than target resolution, pad it
        if extracted.shape[0] < target_height or extracted.shape[1] < target_width:
            padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            padded[:extracted.shape[0], :extracted.shape[1]] = extracted
            return padded
        
        # If it's exactly the right size or larger, crop/resize as needed
        if extracted.shape[0] != target_height or extracted.shape[1] != target_width:
            return cv2.resize(extracted, (target_width, target_height))
        
        return extracted

    def _extract_frame_with_motion_matrix(self, panorama: np.ndarray, motion_matrix: np.ndarray) -> np.ndarray:
        """
        Extracts the correct portion of panoramic background using camera motion matrix.
        
        Args:
            panorama: The full panoramic background image
            motion_matrix: The 2x3 affine transformation matrix for this frame
            
        Returns:
            The extracted background portion at target resolution
        """
        target_width, target_height = self.resolution
        pano_height, pano_width = panorama.shape[:2]
        
        # Get translation from motion matrix
        dx = motion_matrix[0, 2] if motion_matrix is not None else 0
        dy = motion_matrix[1, 2] if motion_matrix is not None else 0
        
        # Calculate starting position (account for panorama padding)
        start_x = int(50 + dx)  # 50 is the padding added during panorama creation
        start_y = int(50 + dy)
        
        # Ensure we don't go out of bounds
        start_x = max(0, min(start_x, pano_width - target_width))
        start_y = max(0, min(start_y, pano_height - target_height))
        
        # Extract portion
        end_x = min(start_x + target_width, pano_width)
        end_y = min(start_y + target_height, pano_height)
        
        extracted = panorama[start_y:end_y, start_x:end_x]
        
        # Resize to target resolution if needed
        if extracted.shape[:2] != (target_height, target_width):
            extracted = cv2.resize(extracted, (target_width, target_height))
        
        return extracted
    
    def _render_object_generative(self, frame: np.ndarray, appearance: np.ndarray, 
                                keypoints: np.ndarray, track_id: int) -> np.ndarray:
        """
        Render an object using improved generative reconstruction with keypoint guidance.
        
        Args:
            frame: Current frame to render object onto
            appearance: Object appearance template
            keypoints: Object keypoints for this frame
            track_id: Unique identifier for the object
            
        Returns:
            Frame with object rendered
        """
        if len(keypoints) == 0 or appearance is None:
            return frame
        
        try:
            # Calculate bounding box from keypoints
            valid_keypoints = keypoints[keypoints[:, 0] > 0]  # Filter out invalid keypoints
            if len(valid_keypoints) < 3:
                return frame  # Need at least 3 points for affine transform
            
            min_x, min_y = np.min(valid_keypoints, axis=0)
            max_x, max_y = np.max(valid_keypoints, axis=0)
            
            # Add some padding to the bounding box
            padding = 10
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(frame.shape[1], max_x + padding)
            max_y = min(frame.shape[0], max_y + padding)
            
            # Create more sophisticated transformation using multiple keypoints
            h, w = appearance.shape[:2]
            
            # Use 4 corners of appearance as source points
            src_pts = np.array([
                [0, 0],
                [w, 0], 
                [w, h],
                [0, h]
            ], dtype=np.float32)
            
            # Map to keypoint-derived destination
            dst_pts = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ], dtype=np.float32)
            
            # Use perspective transform for more realistic warping
            transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_appearance = cv2.warpPerspective(
                appearance, transform_matrix, 
                (frame.shape[1], frame.shape[0])
            )
            
            # Create alpha mask from non-zero pixels
            if appearance.shape[2] == 4:  # Has alpha channel
                alpha_mask = appearance[:, :, 3]
                alpha_mask = cv2.warpPerspective(alpha_mask, transform_matrix, (frame.shape[1], frame.shape[0]))
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
            else:
                # Create mask from non-black pixels
                gray_appearance = cv2.cvtColor(appearance, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_appearance, 10, 255, cv2.THRESH_BINARY)
                alpha_mask = cv2.warpPerspective(mask, transform_matrix, (frame.shape[1], frame.shape[0]))
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
            
            # Apply alpha blending for smooth integration
            for c in range(3):  # RGB channels
                frame[:, :, c] = (
                    frame[:, :, c] * (1 - alpha_mask) + 
                    warped_appearance[:, :, c] * alpha_mask
                ).astype(np.uint8)
            
        except Exception as e:
            print(f"    -> Warning: Failed to render object {track_id}: {e}")
            # Fallback to simple method
            return self._render_object_simple(frame, appearance, keypoints)
        
        return frame
    
    def _render_object_simple(self, frame: np.ndarray, appearance: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Simple fallback rendering method."""
        if len(keypoints) == 0:
            return frame
            
        min_x, min_y = np.min(keypoints, axis=0)
        max_x, max_y = np.max(keypoints, axis=0)
        
        dst_pts = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y]], dtype=np.float32)
        h, w = appearance.shape[:2]
        src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1]], dtype=np.float32)

        warp_matrix = cv2.getAffineTransform(src_pts, dst_pts)
        warped_appearance = cv2.warpAffine(appearance, warp_matrix, (frame.shape[1], frame.shape[0]))

        mask = np.all(warped_appearance > [0, 0, 0], axis=-1).astype(np.uint8) * 255
        locs = np.where(mask != 0)
        frame[locs[0], locs[1]] = warped_appearance[locs[0], locs[1]]
        
        return frame
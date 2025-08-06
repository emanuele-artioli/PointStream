"""
Mock implementation of MMPose handler for demonstration purposes.
This allows the pipeline to run without the full MMPose installation.
"""
import numpy as np
from typing import List, Dict, Any

class MockMMPoseHandler:
    """Mock MMPose handler that simulates pose estimation."""
    
    def __init__(self, model_alias: str):
        self.model_alias = model_alias
        print(f"[MOCK] Initialized MMPose handler for {model_alias}")
    
    def estimate_poses(self, image: np.ndarray, bboxes: List[Dict]) -> List[Dict[str, Any]]:
        """
        Mock pose estimation that returns fake keypoints.
        
        Args:
            image: Input image as numpy array
            bboxes: List of bounding boxes from object detection
            
        Returns:
            List of pose estimations with mock keypoints
        """
        poses = []
        
        for i, bbox in enumerate(bboxes):
            # Create mock keypoints based on bounding box
            x, y, w, h = bbox.get('bbox', [0, 0, 100, 100])
            
            # Generate mock keypoints (simplified human pose - 17 keypoints)
            mock_keypoints = []
            for j in range(17):  # COCO-style 17 keypoints
                # Distribute keypoints within the bounding box
                kx = x + (j % 4) * w / 4 + np.random.uniform(-10, 10)
                ky = y + (j // 4) * h / 4 + np.random.uniform(-10, 10)
                confidence = np.random.uniform(0.5, 0.9)  # Mock confidence
                mock_keypoints.extend([kx, ky, confidence])
            
            pose_data = {
                'keypoints': mock_keypoints,
                'bbox': bbox['bbox'],
                'score': bbox.get('confidence', 0.8),
                'category': bbox.get('class', 'person')
            }
            poses.append(pose_data)
            
        print(f"[MOCK] Generated {len(poses)} mock pose estimations")
        return poses
    
    def extract_poses(self, frames: List[np.ndarray], model_type: str) -> List[np.ndarray]:
        """
        Mock pose extraction for multiple frames.
        
        Args:
            frames: List of input frames
            model_type: Type of model ('person' or 'animal')
            
        Returns:
            List of mock keypoint arrays
        """
        print(f"[MOCK] Extracting {model_type} keypoints for {len(frames)} frames")
        
        # Generate mock keypoints for each frame
        keypoints_list = []
        num_keypoints = 17 if model_type == 'person' else 20  # Different for animals
        
        for frame in frames:
            # Create mock keypoints array (num_keypoints x 3: x, y, confidence)
            mock_keypoints = np.random.uniform(0, 1, (num_keypoints, 3))
            mock_keypoints[:, 2] = np.random.uniform(0.5, 0.9, num_keypoints)  # confidence scores
            keypoints_list.append(mock_keypoints)
            
        print(f"[MOCK] Generated keypoints for {len(keypoints_list)} frames")
        return keypoints_list
    
    def cleanup(self):
        """Mock cleanup method."""
        print(f"[MOCK] Cleaned up MMPose handler for {self.model_alias}")

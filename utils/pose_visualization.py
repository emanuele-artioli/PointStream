#!/usr/bin/env python3
"""
Pose Visualization Utilities

Converts pose keypoints to visual skeleton images for improved model training.
Creates skeleton images with keypoints and connections on black backgrounds.
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Any, Optional, Tuple


class PoseVisualizer:
    """Converts keypoint data to skeleton images for model training."""
    
    # Human pose connections (COCO format - 17 keypoints)
    HUMAN_CONNECTIONS = [
        # Head
        (0, 1),   # nose -> left_eye
        (0, 2),   # nose -> right_eye
        (1, 3),   # left_eye -> left_ear
        (2, 4),   # right_eye -> right_ear
        
        # Torso
        (5, 6),   # left_shoulder -> right_shoulder
        (5, 11),  # left_shoulder -> left_hip
        (6, 12),  # right_shoulder -> right_hip
        (11, 12), # left_hip -> right_hip
        
        # Left arm
        (5, 7),   # left_shoulder -> left_elbow
        (7, 9),   # left_elbow -> left_wrist
        
        # Right arm
        (6, 8),   # right_shoulder -> right_elbow
        (8, 10),  # right_elbow -> right_wrist
        
        # Left leg
        (11, 13), # left_hip -> left_knee
        (13, 15), # left_knee -> left_ankle
        
        # Right leg
        (12, 14), # right_hip -> right_knee
        (14, 16), # right_knee -> right_ankle
    ]
    
    # Animal pose connections (simplified - 12 keypoints)
    ANIMAL_CONNECTIONS = [
        # Head to neck
        (0, 1),   # nose -> neck
        
        # Spine
        (1, 2),   # neck -> spine_mid
        (2, 3),   # spine_mid -> spine_end
        
        # Front legs
        (1, 4),   # neck -> front_left_shoulder
        (4, 5),   # front_left_shoulder -> front_left_foot
        (1, 6),   # neck -> front_right_shoulder
        (6, 7),   # front_right_shoulder -> front_right_foot
        
        # Back legs
        (3, 8),   # spine_end -> back_left_hip
        (8, 9),   # back_left_hip -> back_left_foot
        (3, 10),  # spine_end -> back_right_hip
        (10, 11), # back_right_hip -> back_right_foot
    ]
    
    def __init__(self):
        """Initialize pose visualizer."""
        self.keypoint_radius = 3
        self.connection_thickness = 2
        self.keypoint_color = (255, 255, 255)  # White keypoints
        self.connection_color = (128, 128, 128)  # Gray connections
        self.confidence_threshold = 0.3  # Min confidence to draw keypoint
        
    def keypoints_to_image(self, keypoints_data: List, category: str, 
                          image_size: int = 256, normalize_coords: bool = True,
                          grayscale: bool = False, low_resolution: bool = False) -> np.ndarray:
        """
        Convert keypoints to skeleton image.
        
        Args:
            keypoints_data: List of keypoints [[x, y, conf], ...] or [[x, y], ...]
            category: Object category (human, animal, other)
            image_size: Output image size (square)
            normalize_coords: Whether coordinates are normalized [0,1] or absolute
            grayscale: If True, create single-channel grayscale image
            low_resolution: If True, create at 64x64 then upscale
            
        Returns:
            Skeleton image as numpy array (H, W, 3) or (H, W, 1)
        """
        # Determine working resolution
        if low_resolution:
            work_size = 64
            # Adjust drawing parameters for smaller image
            keypoint_radius = max(1, self.keypoint_radius // 4)
            connection_thickness = max(1, self.connection_thickness // 4)
        else:
            work_size = image_size
            keypoint_radius = self.keypoint_radius
            connection_thickness = self.connection_thickness
        
        # Create black background with appropriate channels
        if grayscale:
            skeleton_img = np.zeros((work_size, work_size), dtype=np.uint8)
            keypoint_color = 255  # White on black for grayscale
            connection_color = 128  # Gray for connections
        else:
            skeleton_img = np.zeros((work_size, work_size, 3), dtype=np.uint8)
            keypoint_color = self.keypoint_color
            connection_color = self.connection_color
        
        if not keypoints_data:
            return skeleton_img
        
        # Parse keypoints with confidence
        keypoints = []
        for i, kp in enumerate(keypoints_data):
            if isinstance(kp, (list, tuple)):
                if len(kp) >= 3:  # [x, y, conf]
                    x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
                elif len(kp) >= 2:  # [x, y]
                    x, y, conf = float(kp[0]), float(kp[1]), 1.0
                else:
                    continue
            else:
                # Flat list format: [x1, y1, c1, x2, y2, c2, ...]
                if i * 3 + 2 < len(keypoints_data):
                    x = float(keypoints_data[i * 3])
                    y = float(keypoints_data[i * 3 + 1]) 
                    conf = float(keypoints_data[i * 3 + 2])
                else:
                    continue
                    
            keypoints.append((x, y, conf))
        
        # Convert coordinates to image space
        image_keypoints = []
        for x, y, conf in keypoints:
            if normalize_coords:
                # Coordinates are in [0, 1] range
                img_x = int(x * work_size)
                img_y = int(y * work_size)
            else:
                # Coordinates are already in image space - scale to work size
                img_x = int(x * work_size / image_size)
                img_y = int(y * work_size / image_size)
                
            # Clamp to image bounds
            img_x = max(0, min(img_x, work_size - 1))
            img_y = max(0, min(img_y, work_size - 1))
            
            image_keypoints.append((img_x, img_y, conf))
        
        # Draw connections first (so they appear behind keypoints)
        connections = self._get_connections_for_category(category)
        for conn in connections:
            start_idx, end_idx = conn
            if (start_idx < len(image_keypoints) and end_idx < len(image_keypoints)):
                start_kp = image_keypoints[start_idx]
                end_kp = image_keypoints[end_idx]
                
                # Only draw if both keypoints have sufficient confidence
                if (start_kp[2] >= self.confidence_threshold and 
                    end_kp[2] >= self.confidence_threshold):
                    
                    cv2.line(skeleton_img, 
                            (start_kp[0], start_kp[1]), 
                            (end_kp[0], end_kp[1]),
                            connection_color, 
                            connection_thickness)
        
        # Draw keypoints on top
        for img_x, img_y, conf in image_keypoints:
            if conf >= self.confidence_threshold:
                cv2.circle(skeleton_img, (img_x, img_y), 
                          keypoint_radius, keypoint_color, -1)
        
        # Upscale if we worked at low resolution
        if low_resolution and work_size != image_size:
            skeleton_img = cv2.resize(skeleton_img, (image_size, image_size), 
                                    interpolation=cv2.INTER_NEAREST)
        
        # Add channel dimension for grayscale if needed
        if grayscale and len(skeleton_img.shape) == 2:
            skeleton_img = skeleton_img[:, :, np.newaxis]
        
        return skeleton_img
    
    def bbox_to_image(self, bbox_data: List, image_size: int = 256, 
                     normalize_coords: bool = True, grayscale: bool = False,
                     low_resolution: bool = False) -> np.ndarray:
        """
        Convert bounding box to simple rectangle image for 'other' objects.
        
        Args:
            bbox_data: Bounding box as [x1, y1, x2, y2] or [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            image_size: Output image size
            normalize_coords: Whether coordinates are normalized
            grayscale: If True, create single-channel grayscale image
            low_resolution: If True, create at lower resolution then upscale
            
        Returns:
            Rectangle image as numpy array
        """
        # Determine working resolution
        if low_resolution:
            work_size = 64
            thickness = max(1, self.connection_thickness // 4)
            radius = max(1, self.keypoint_radius // 4)
        else:
            work_size = image_size
            thickness = self.connection_thickness
            radius = self.keypoint_radius
        
        # Create black background with appropriate channels
        if grayscale:
            rect_img = np.zeros((work_size, work_size), dtype=np.uint8)
            color = 255  # White on black
        else:
            rect_img = np.zeros((work_size, work_size, 3), dtype=np.uint8)
            color = self.keypoint_color
        
        if not bbox_data:
            return rect_img
        
        # Parse different bbox formats
        if len(bbox_data) == 4 and all(isinstance(x, (int, float)) for x in bbox_data):
            # Format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox_data
        elif len(bbox_data) >= 4 and all(isinstance(pt, (list, tuple)) for pt in bbox_data):
            # Format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] (4 corners)
            x_coords = [pt[0] for pt in bbox_data[:4]]
            y_coords = [pt[1] for pt in bbox_data[:4]]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
        else:
            logging.warning(f"Unknown bbox format: {bbox_data}")
            return rect_img
        
        # Convert to image coordinates
        if normalize_coords:
            x1 = int(x1 * work_size)
            y1 = int(y1 * work_size) 
            x2 = int(x2 * work_size)
            y2 = int(y2 * work_size)
        else:
            # Scale from original image size to work size
            scale = work_size / image_size
            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
        
        # Clamp to image bounds
        x1 = max(0, min(x1, work_size - 1))
        y1 = max(0, min(y1, work_size - 1))
        x2 = max(0, min(x2, work_size - 1))
        y2 = max(0, min(y2, work_size - 1))
        
        # Draw rectangle outline
        cv2.rectangle(rect_img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw corner points
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for corner in corners:
            cv2.circle(rect_img, corner, radius, color, -1)
        
        # Draw center point
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(rect_img, (center_x, center_y), radius + 1, color, -1)
        
        # Upscale if we worked at low resolution
        if low_resolution and work_size != image_size:
            rect_img = cv2.resize(rect_img, (image_size, image_size), 
                                interpolation=cv2.INTER_NEAREST)
        
        # Add channel dimension for grayscale if needed
        if grayscale and len(rect_img.shape) == 2:
            rect_img = rect_img[:, :, np.newaxis]
        
        return rect_img
    
    def _get_connections_for_category(self, category: str) -> List[Tuple[int, int]]:
        """Get skeletal connections for a specific category."""
        if category == 'human':
            return self.HUMAN_CONNECTIONS
        elif category == 'animal':
            return self.ANIMAL_CONNECTIONS
        else:
            # For 'other' objects, no predefined connections
            return []
    
    def pose_data_to_image(self, obj_data: Dict[str, Any], image_size: int = 256,
                          grayscale: bool = False, low_resolution: bool = False) -> np.ndarray:
        """
        Convert object pose data to skeleton image.
        Handles different keypoint formats automatically.
        
        Args:
            obj_data: Object data containing keypoint information
            image_size: Output image size
            grayscale: If True, create single-channel grayscale image
            low_resolution: If True, create at lower resolution then upscale
            
        Returns:
            Skeleton image
        """
        category = obj_data.get('category', obj_data.get('semantic_category', 'other'))
        
        # Try to extract keypoints from various possible locations
        keypoints_data = None
        keypoints_method = 'unknown'
        
        if 'keypoints' in obj_data and isinstance(obj_data['keypoints'], dict):
            keypoints_info = obj_data['keypoints']
            keypoints_data = keypoints_info.get('points', [])
            keypoints_method = keypoints_info.get('method', 'keypoints')
        elif 'p_pose' in obj_data and isinstance(obj_data['p_pose'], dict):
            keypoints_data = obj_data['p_pose'].get('points', [])
            keypoints_method = 'p_pose'
        elif 'p_pose_data' in obj_data:
            keypoints_data = obj_data['p_pose_data']
            keypoints_method = 'p_pose_data'
        
        # Handle different types of pose data
        if keypoints_data and keypoints_method.startswith('bbox'):
            # Bounding box data - convert to rectangle
            return self.bbox_to_image(keypoints_data, image_size, normalize_coords=True,
                                    grayscale=grayscale, low_resolution=low_resolution)
        elif keypoints_data:
            # Regular keypoints - convert to skeleton
            return self.keypoints_to_image(keypoints_data, category, image_size, 
                                         normalize_coords=True, grayscale=grayscale, 
                                         low_resolution=low_resolution)
        else:
            # No pose data - return black image with appropriate channels
            logging.warning(f"No pose data found for object {obj_data.get('object_id', 'unknown')}")
            if grayscale:
                return np.zeros((image_size, image_size, 1), dtype=np.uint8)
            else:
                return np.zeros((image_size, image_size, 3), dtype=np.uint8)


# Global instance for easy access
pose_visualizer = PoseVisualizer()


def create_pose_image(obj_data: Dict[str, Any], image_size: int = 256,
                     grayscale: bool = False, low_resolution: bool = False) -> np.ndarray:
    """
    Convenience function to create pose image from object data.
    
    Args:
        obj_data: Object data dictionary
        image_size: Output image size
        grayscale: If True, create single-channel grayscale image
        low_resolution: If True, create at lower resolution then upscale
        
    Returns:
        Skeleton image as numpy array
    """
    return pose_visualizer.pose_data_to_image(obj_data, image_size, grayscale, low_resolution)


def save_pose_image(obj_data: Dict[str, Any], output_path: str, image_size: int = 256,
                   grayscale: bool = False, low_resolution: bool = False) -> bool:
    """
    Create and save pose image to file.
    
    Args:
        obj_data: Object data dictionary
        output_path: Path to save image
        image_size: Output image size
        grayscale: If True, create single-channel grayscale image
        low_resolution: If True, create at lower resolution then upscale
        
    Returns:
        True if successful, False otherwise
    """
    try:
        pose_img = create_pose_image(obj_data, image_size, grayscale, low_resolution)
        success = cv2.imwrite(output_path, pose_img)
        if success:
            logging.debug(f"Saved pose image: {output_path}")
        else:
            logging.warning(f"Failed to save pose image: {output_path}")
        return success
    except Exception as e:
        logging.error(f"Error saving pose image {output_path}: {e}")
        return False
#!/usr/bin/env python3
"""
Frame Composer

This module composes final frames by overlaying generated objects on reconstructed backgrounds.
Handles blending, occlusion, and temporal consistency for smooth video output.
"""

import logging
import time
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

try:
    from utils.decorators import track_performance
    from utils import config
except ImportError as e:
    logging.error(f"Failed to import PointStream utilities: {e}")
    raise


class FrameComposer:
    """
    Composes final frames by overlaying objects on backgrounds.
    
    This component takes reconstructed backgrounds and generated objects
    to create the final frame composition with proper blending and occlusion handling.
    """
    
    def __init__(self):
        """Initialize the frame composer."""
        self.enable_temporal_smoothing = config.get_bool('video_reconstruction', 'temporal_smoothing', True)
        self.smoothing_window = config.get_int('video_reconstruction', 'smoothing_window', 3)
        self.motion_compensation = config.get_bool('video_reconstruction', 'motion_compensation', True)
        
        # Blending parameters
        self.alpha_blending = True
        self.edge_feathering = 5  # pixels
        self.shadow_simulation = True
        
        logging.info("🎨 Frame Composer initialized")
        logging.info(f"   Temporal smoothing: {self.enable_temporal_smoothing}")
        logging.info(f"   Motion compensation: {self.motion_compensation}")
        logging.info(f"   Edge feathering: {self.edge_feathering}px")
    
    @track_performance
    def compose_frames(self, backgrounds: List[np.ndarray], 
                      generated_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compose final frames from backgrounds and objects.
        
        Args:
            backgrounds: List of background frames
            generated_objects: List of generated objects with frame indices
            
        Returns:
            Dictionary containing composed frames
        """
        start_time = time.time()
        
        if not backgrounds:
            raise ValueError("No background frames provided")
        
        logging.info(f"🎬 Composing {len(backgrounds)} frames with {len(generated_objects)} objects")
        
        # Group objects by frame
        objects_by_frame = self._group_objects_by_frame(generated_objects)
        
        # Compose each frame
        composed_frames = []
        
        for frame_idx, background in enumerate(backgrounds):
            frame_objects = objects_by_frame.get(frame_idx, [])
            
            composed_frame = self._compose_single_frame(
                background, 
                frame_objects, 
                frame_idx
            )
            
            composed_frames.append(composed_frame)
        
        # Apply temporal smoothing if enabled
        if self.enable_temporal_smoothing and len(composed_frames) > 1:
            composed_frames = self._apply_temporal_smoothing(composed_frames)
        
        processing_time = time.time() - start_time
        
        result = {
            'composed_frames': composed_frames,
            'frame_count': len(composed_frames),
            'objects_placed': len(generated_objects),
            'processing_time': processing_time
        }
        
        logging.info(f"✅ Frame composition completed in {processing_time:.2f}s")
        return result
    
    def _group_objects_by_frame(self, generated_objects: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Group objects by their frame index."""
        objects_by_frame = {}
        
        for obj in generated_objects:
            frame_idx = obj.get('frame_index', 0)
            
            if frame_idx not in objects_by_frame:
                objects_by_frame[frame_idx] = []
            
            objects_by_frame[frame_idx].append(obj)
        
        return objects_by_frame
    
    def _compose_single_frame(self, background: np.ndarray, 
                            frame_objects: List[Dict[str, Any]], 
                            frame_idx: int) -> np.ndarray:
        """
        Compose a single frame with objects overlaid on background.
        
        Args:
            background: Background frame
            frame_objects: Objects to place in this frame
            frame_idx: Frame index for logging
            
        Returns:
            Composed frame
        """
        if not frame_objects:
            return background.copy()
        
        composed_frame = background.copy()
        
        # Sort objects by depth (z-order) for proper occlusion
        # Objects with larger bboxes (closer to camera) are rendered last
        sorted_objects = sorted(frame_objects, 
                              key=lambda obj: self._calculate_object_depth(obj))
        
        for obj in sorted_objects:
            try:
                composed_frame = self._place_object(composed_frame, obj)
            except Exception as e:
                logging.warning(f"Failed to place object {obj.get('object_id')} in frame {frame_idx}: {e}")
                continue
        
        return composed_frame
    
    def _calculate_object_depth(self, obj: Dict[str, Any]) -> float:
        """Calculate object depth based on bounding box size (larger = closer)."""
        bbox = obj.get('bbox', [])
        if len(bbox) >= 4:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return width * height  # Area as depth proxy
        return 0.0
    
    def _place_object(self, frame: np.ndarray, obj: Dict[str, Any]) -> np.ndarray:
        """
        Place a single object on the frame.
        
        Args:
            frame: Current frame
            obj: Object to place
            
        Returns:
            Frame with object placed
        """
        generated_image = obj.get('generated_image')
        bbox = obj.get('bbox', [])
        
        if generated_image is None or len(bbox) < 4:
            return frame
        
        # Get placement coordinates
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Ensure coordinates are within frame bounds
        frame_h, frame_w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return frame
        
        # Resize generated object to fit bounding box
        target_w, target_h = x2 - x1, y2 - y1
        resized_object = cv2.resize(generated_image, (target_w, target_h))
        
        # Create alpha mask for blending
        alpha_mask = self._create_alpha_mask(resized_object, obj)
        
        # Blend object onto frame
        frame_region = frame[y1:y2, x1:x2]
        blended_region = self._blend_with_alpha(frame_region, resized_object, alpha_mask)
        
        frame[y1:y2, x1:x2] = blended_region
        
        return frame
    
    def _create_alpha_mask(self, object_image: np.ndarray, obj: Dict[str, Any]) -> np.ndarray:
        """
        Create alpha mask for object blending.
        
        Args:
            object_image: Generated object image
            obj: Object metadata
            
        Returns:
            Alpha mask [0-1]
        """
        h, w = object_image.shape[:2]
        
        # Create basic mask from non-black pixels
        gray = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)
        basic_mask = (gray > 10).astype(np.float32)
        
        # Apply edge feathering for smooth blending
        if self.edge_feathering > 0:
            # Create distance transform
            kernel_size = max(3, self.edge_feathering)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Erode mask slightly to create soft edges
            eroded_mask = cv2.erode(basic_mask, kernel, iterations=1)
            
            # Create distance-based soft edges
            distance = cv2.distanceTransform((eroded_mask * 255).astype(np.uint8), 
                                           cv2.DIST_L2, 5)
            
            # Normalize and apply feathering
            feather_radius = self.edge_feathering
            alpha_mask = np.clip(distance / feather_radius, 0, 1)
            
            # Combine with original mask
            alpha_mask = np.minimum(alpha_mask, basic_mask)
        else:
            alpha_mask = basic_mask
        
        # Adjust based on object confidence
        confidence = obj.get('confidence', 1.0)
        alpha_mask *= confidence
        
        return alpha_mask
    
    def _blend_with_alpha(self, background: np.ndarray, 
                         foreground: np.ndarray, 
                         alpha: np.ndarray) -> np.ndarray:
        """
        Blend foreground onto background using alpha mask.
        
        Args:
            background: Background region
            foreground: Foreground object
            alpha: Alpha mask [0-1]
            
        Returns:
            Blended result
        """
        # Ensure alpha has 3 channels for RGB blending
        if len(alpha.shape) == 2:
            alpha = np.stack([alpha] * 3, axis=2)
        
        # Blend using alpha compositing
        blended = (alpha * foreground.astype(np.float32) + 
                  (1 - alpha) * background.astype(np.float32))
        
        return blended.astype(np.uint8)
    
    def _apply_temporal_smoothing(self, composed_frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply temporal smoothing to reduce flickering.
        
        Args:
            composed_frames: List of composed frames
            
        Returns:
            Temporally smoothed frames
        """
        if len(composed_frames) < 3:
            return composed_frames
        
        smoothed_frames = []
        window_size = min(self.smoothing_window, len(composed_frames))
        half_window = window_size // 2
        
        for i, frame in enumerate(composed_frames):
            if i < half_window or i >= len(composed_frames) - half_window:
                # Keep boundary frames unchanged
                smoothed_frames.append(frame)
            else:
                # Apply temporal filtering
                frame_window = composed_frames[i - half_window:i + half_window + 1]
                
                # Weighted average with emphasis on current frame
                weights = self._create_temporal_weights(window_size)
                
                smoothed_frame = np.zeros_like(frame, dtype=np.float32)
                for j, window_frame in enumerate(frame_window):
                    smoothed_frame += weights[j] * window_frame.astype(np.float32)
                
                smoothed_frames.append(smoothed_frame.astype(np.uint8))
        
        logging.info(f"🎯 Applied temporal smoothing to {len(composed_frames)} frames")
        return smoothed_frames
    
    def _create_temporal_weights(self, window_size: int) -> List[float]:
        """Create gaussian-like weights for temporal filtering."""
        center = window_size // 2
        weights = []
        
        for i in range(window_size):
            # Gaussian-like weighting centered on current frame
            distance = abs(i - center)
            weight = np.exp(-distance**2 / (2 * (center/2)**2))
            weights.append(weight)
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        return weights
    
    def preview_composition(self, background: np.ndarray, 
                          frame_objects: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate preview of frame composition.
        
        Args:
            background: Background frame
            frame_objects: Objects to place
            
        Returns:
            Preview composed frame
        """
        return self._compose_single_frame(background, frame_objects, 0)
    
    def create_composition_debug_view(self, background: np.ndarray, 
                                    frame_objects: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Create debug view showing composition layers.
        
        Args:
            background: Background frame
            frame_objects: Objects to place
            
        Returns:
            Dictionary with debug views
        """
        debug_views = {
            'background': background.copy(),
            'objects_only': np.zeros_like(background),
            'composition': self._compose_single_frame(background, frame_objects, 0)
        }
        
        # Create objects-only view
        objects_frame = np.zeros_like(background)
        for obj in frame_objects:
            objects_frame = self._place_object(objects_frame, obj)
        debug_views['objects_only'] = objects_frame
        
        return debug_views

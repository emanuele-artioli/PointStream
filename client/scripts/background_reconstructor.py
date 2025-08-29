#!/usr/bin/env python3
"""
Background Reconstructor

This module reconstructs background frames from panoramas and homographies.
It uses the panorama image and homography matrices to generate the background
for each frame position in the scene.
"""

import logging
import time
import numpy as np
import cv2
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from utils.decorators import track_performance
    from utils import config
except ImportError as e:
    logging.error(f"Failed to import PointStream utilities: {e}")
    raise


class BackgroundReconstructor:
    """
    Reconstructs frame backgrounds from panorama and homography data.
    
    This component takes a scene panorama and the homography matrices
    for each frame to generate the background portion of each frame.
    """
    
    def __init__(self):
        """Initialize the background reconstructor."""
        self.blending_mode = config.get_str('reconstruction', 'background_blending', 'gaussian')
        self.blur_kernel = config.get_int('reconstruction', 'background_blur_kernel', 5)
        
        # Map interpolation methods correctly for OpenCV
        interp_method = config.get_str('reconstruction', 'background_interpolation', 'bilinear')
        interp_mapping = {
            'bilinear': cv2.INTER_LINEAR,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'nearest': cv2.INTER_NEAREST,
            'area': cv2.INTER_AREA,
            'lanczos4': cv2.INTER_LANCZOS4
        }
        self.interpolation = interp_mapping.get(interp_method.lower(), cv2.INTER_LINEAR)
        
        self.enable_smoothing = config.get_bool('reconstruction', 'homography_smoothing', True)
        self.temporal_consistency = config.get_bool('reconstruction', 'temporal_consistency', True)
        
        logging.info("🖼️  Background Reconstructor initialized")
        logging.info(f"   Blending mode: {self.blending_mode}")
        logging.info(f"   Interpolation: {interp_method}")
        logging.info(f"   Temporal consistency: {self.temporal_consistency}")
    
    @track_performance
    def reconstruct_backgrounds(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct backgrounds for all frames in a scene.
        
        Args:
            scene_data: Scene metadata including panorama path and homographies
            
        Returns:
            Dictionary containing reconstructed background frames
        """
        start_time = time.time()
        
        # Load panorama
        panorama_path = scene_data.get('panorama_path')
        if not panorama_path or not Path(panorama_path).exists():
            raise FileNotFoundError(f"Panorama not found: {panorama_path}")
        
        panorama = cv2.imread(panorama_path)
        if panorama is None:
            raise ValueError(f"Failed to load panorama: {panorama_path}")
        
        logging.info(f"📷 Loaded panorama: {panorama.shape}")
        
        # Get homographies
        homographies = scene_data.get('homographies', [])
        if not homographies:
            raise ValueError("No homographies found in scene data")
        
        # Determine frame size from first homography or use default
        frame_height = scene_data.get('frame_height', 1080)
        frame_width = scene_data.get('frame_width', 1920)
        
        logging.info(f"🎬 Reconstructing {len(homographies)} background frames ({frame_width}x{frame_height})")
        
        # Smooth homographies if enabled
        if self.enable_smoothing and len(homographies) > 1:
            homographies = self._smooth_homographies(homographies)
        
        # Reconstruct each frame
        background_frames = []
        for i, homography in enumerate(homographies):
            background_frame = self._reconstruct_single_background(
                panorama, 
                homography, 
                (frame_width, frame_height),
                frame_index=i
            )
            background_frames.append(background_frame)
        
        # Apply temporal consistency if enabled
        if self.temporal_consistency and len(background_frames) > 1:
            background_frames = self._apply_temporal_consistency(background_frames)
        
        processing_time = time.time() - start_time
        
        result = {
            'backgrounds': background_frames,
            'frame_count': len(background_frames),
            'frame_size': (frame_width, frame_height),
            'panorama_size': panorama.shape[:2],
            'processing_time': processing_time
        }
        
        logging.info(f"✅ Background reconstruction completed in {processing_time:.2f}s")
        return result
    
    def _reconstruct_single_background(self, panorama: np.ndarray, homography: np.ndarray, 
                                     frame_size: tuple, frame_index: int = 0) -> np.ndarray:
        """
        Reconstruct a single background frame.
        
        Args:
            panorama: Source panorama image
            homography: Homography matrix for this frame
            frame_size: Target frame size (width, height)
            frame_index: Frame index for logging
            
        Returns:
            Reconstructed background frame
        """
        try:
            # Convert homography to numpy array if needed
            if isinstance(homography, list):
                homography = np.array(homography, dtype=np.float32)
            
            # Ensure homography is 3x3
            if homography.shape != (3, 3):
                raise ValueError(f"Invalid homography shape: {homography.shape}")
            
            # The stored homography transforms from panorama → frame
            # For reconstruction, we can use it directly
            background_frame = cv2.warpPerspective(
                panorama, 
                homography, 
                frame_size,
                flags=self.interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
            # Flip horizontally to correct the direction
            # background_frame = cv2.flip(background_frame, 1)
            
            # If the result is shifted, try to center it
            # Check if there's a significant black border on one side
            gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Check for black borders
            left_border = np.mean(gray[:, :10])  # Left 10 pixels
            right_border = np.mean(gray[:, -10:])  # Right 10 pixels
            top_border = np.mean(gray[:10, :])  # Top 10 pixels
            bottom_border = np.mean(gray[-10:, :])  # Bottom 10 pixels
            
            # If there's a significant black border on the right, try to shift the content
            if right_border < 10 and left_border > right_border * 2:
                logging.debug(f"Detected right black border in frame {frame_index}, attempting to center content")
                # Find the actual content bounds
                non_black = np.where(gray > 10)
                if len(non_black[0]) > 0 and len(non_black[1]) > 0:
                    min_y, max_y = np.min(non_black[0]), np.max(non_black[0])
                    min_x, max_x = np.min(non_black[1]), np.max(non_black[1])
                    
                    # Center the content in the frame
                    content_h, content_w = max_y - min_y, max_x - min_x
                    center_y, center_x = h // 2, w // 2
                    target_min_y = center_y - content_h // 2
                    target_min_x = center_x - content_w // 2
                    
                    # Create a new frame and place the content in the center
                    centered_frame = np.zeros_like(background_frame)
                    src_roi = background_frame[min_y:max_y, min_x:max_x]
                    dest_roi = centered_frame[target_min_y:target_min_y+content_h, 
                                            target_min_x:target_min_x+content_w]
                    dest_roi[:] = src_roi
                    
                    return centered_frame
            
            return background_frame
            
        except Exception as e:
            logging.warning(f"Failed to reconstruct background for frame {frame_index}: {e}")
            # Return black frame as fallback
            return np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    
    def _smooth_homographies(self, homographies: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply temporal smoothing to homography matrices.
        
        Args:
            homographies: List of homography matrices
            
        Returns:
            Smoothed homography matrices
        """
        if len(homographies) < 3:
            return homographies
        
        smoothed_homographies = []
        
        # First frame - no smoothing
        smoothed_homographies.append(homographies[0])
        
        # Middle frames - apply moving average
        for i in range(1, len(homographies) - 1):
            prev_h = homographies[i - 1]
            curr_h = homographies[i]
            next_h = homographies[i + 1]
            
            # Simple moving average of homography elements
            smoothed_h = (prev_h + 2 * curr_h + next_h) / 4.0
            smoothed_homographies.append(smoothed_h)
        
        # Last frame - no smoothing
        smoothed_homographies.append(homographies[-1])
        
        logging.info(f"📐 Applied temporal smoothing to {len(homographies)} homographies")
        return smoothed_homographies
    
    def _apply_temporal_consistency(self, background_frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply temporal consistency filtering to reduce flickering.
        
        Args:
            background_frames: List of background frames
            
        Returns:
            Temporally consistent background frames
        """
        if len(background_frames) < 3:
            return background_frames
        
        consistent_frames = []
        window_size = 3  # Use 3-frame window for temporal filtering
        
        # Process each frame with temporal filtering
        for i, frame in enumerate(background_frames):
            if i == 0 or i == len(background_frames) - 1:
                # Keep first and last frames unchanged
                consistent_frames.append(frame)
            else:
                # Apply temporal filtering
                prev_frame = background_frames[i - 1]
                next_frame = background_frames[i + 1]
                
                # Weighted average with stronger weight on current frame
                weights = [0.25, 0.5, 0.25]
                filtered_frame = (
                    weights[0] * prev_frame.astype(np.float32) +
                    weights[1] * frame.astype(np.float32) +
                    weights[2] * next_frame.astype(np.float32)
                ).astype(np.uint8)
                
                consistent_frames.append(filtered_frame)
        
        logging.info(f"🎯 Applied temporal consistency to {len(background_frames)} frames")
        return consistent_frames
    
    def preview_background_reconstruction(self, scene_data: Dict[str, Any], 
                                        frame_indices: List[int] = None) -> List[np.ndarray]:
        """
        Generate preview of background reconstruction for specific frames.
        
        Args:
            scene_data: Scene metadata
            frame_indices: Specific frame indices to preview (or None for all)
            
        Returns:
            List of preview background frames
        """
        if frame_indices is None:
            return self.reconstruct_backgrounds(scene_data)['backgrounds']
        
        # Load panorama
        panorama_path = scene_data.get('panorama_path')
        panorama = cv2.imread(panorama_path)
        
        # Get homographies
        homographies = scene_data.get('homographies', [])
        frame_height = scene_data.get('frame_height', 1080)
        frame_width = scene_data.get('frame_width', 1920)
        
        preview_frames = []
        for idx in frame_indices:
            if 0 <= idx < len(homographies):
                background_frame = self._reconstruct_single_background(
                    panorama, 
                    homographies[idx], 
                    (frame_width, frame_height),
                    frame_index=idx
                )
                preview_frames.append(background_frame)
        
        return preview_frames

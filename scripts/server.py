#!/usr/bin/env python3
"""
PointStream Server Pipeline - Simplified Sequential Processing

This module implements a simplified processing pipeline for the PointStream system.
Removes redundant classes and functions while maintaining all functionality.
"""

import os
import sys
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import json
import shutil
import hashlib

# Suppress warnings before importing other modules
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Using a slow image processor')

# Configure PySceneDetect logging to reduce verbosity
pyscene_logger = logging.getLogger('pyscenedetect')
pyscene_logger.setLevel(logging.WARNING)

# Now import other modules
import torch
import cv2
import numpy as np
from PIL import Image

# Import all PointStream components
try:
    from .decorators import track_performance
    from .stitcher import Stitcher
    from .segmenter import Segmenter 
    from .keypointer import Keypointer
    from .saver import Saver
    from .splitter import VideoSceneSplitter
    from . import config
except ImportError as e:
    logging.error(f"Failed to import PointStream components: {e}")
    print("Error: Cannot import required PointStream components")
    print("Make sure all component files are in the same directory")
    sys.exit(1)

# Check for optional dependencies
try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logging.warning("imagehash not available, disabling frame similarity caching")


class PointStreamPipeline:
    """
    Simplified PointStream pipeline that combines processing and orchestration.
    
    This class eliminates the redundant PointStreamProcessor wrapper and handles
    all processing, statistics, and I/O in a single coherent class.
    """
    
    def __init__(self, config_file: str = None):
        """Initialize the pipeline with configuration."""
        # Load configuration
        if config_file:
            config.load_config(config_file)
        
        # Initialize all processing components directly
        logging.info("🚀 Initializing PointStream pipeline...")
        self._initialize_components()
        
        # Statistics tracking
        self.processed_scenes = 0
        self.complex_scenes = 0
        self.processing_times = []
        
        logging.info("✅ PointStream Pipeline initialized")
        logging.info("🎯 Workflow: Segmentation → Masking → Stitching → Keypoints")
    
    def _initialize_components(self):
        """Initialize all processing components."""
        try:
            logging.info("🔧 Loading components...")
            
            # Initialize components in order of dependency
            logging.info("   📐 Initializing Stitcher...")
            self.stitcher = Stitcher()
            
            logging.info("   🎯 Initializing Segmenter...")
            self.segmenter = Segmenter()
            
            logging.info("   🫴 Initializing Keypointer...")
            self.keypointer = Keypointer()
            
            logging.info("   💾 Initializing Saver...")
            self.saver = Saver()
            
            logging.info("✅ All components loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize components: {e}")
            raise
    
    def _create_masked_frame_for_objects(self, frame: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create a frame where all object backgrounds are masked to black.
        This ensures that when objects are cropped, they have black backgrounds instead of scene background.
        
        Args:
            frame: Original frame
            objects: List of objects with their masks
            
        Returns:
            Frame with object backgrounds masked to black
        """
        masked_frame = frame.copy()
        
        for obj in objects:
            if 'mask' in obj:
                obj_mask = obj['mask']
                
                # Create inverted mask (background areas)
                background_mask = (obj_mask == 0)
                
                # Get object bounding box to limit the masking area
                bbox = obj.get('bbox', [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    # Ensure coordinates are within frame bounds
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Apply black masking only within the bounding box region
                    bbox_background_mask = background_mask[y1:y2, x1:x2]
                    masked_frame[y1:y2, x1:x2][bbox_background_mask] = [0, 0, 0]  # Black background
        
        return masked_frame
    
    def _create_masked_frames(self, frames: List[np.ndarray], segmentation_result: Dict[str, Any]) -> List[np.ndarray]:
        """
        Create masked frames with configurable background handling.
        Can use either background reconstruction or simple black masking.
        """
        use_reconstruction = config.get_bool('stitching', 'use_background_reconstruction', True)
        masked_frames = []
        frames_data = segmentation_result.get('frames_data', [])
        
        for i, frame in enumerate(frames):
            try:
                # Find corresponding frame data
                frame_data = None
                for fd in frames_data:
                    if fd.get('frame_index', -1) == i:
                        frame_data = fd
                        break
                
                if frame_data is None or not frame_data.get('objects'):
                    # No objects found, use original frame
                    masked_frames.append(frame.copy())
                    continue
                
                if use_reconstruction:
                    # Create masked frame with background reconstruction
                    masked_frame = self._reconstruct_background(frame, frame_data, frames, i)
                else:
                    # Use simple black masking (old method)
                    masked_frame = frame.copy()
                    for obj in frame_data.get('objects', []):
                        if 'mask' in obj:
                            obj_mask = obj['mask']
                            masked_frame[obj_mask > 0] = [0, 0, 0]  # Black overlay
                
                masked_frames.append(masked_frame)
                
            except Exception as e:
                logging.warning(f"Failed to create masked frame {i}: {e}")
                # Fallback to original frame
                masked_frames.append(frame.copy())
        
        return masked_frames
    
    def _reconstruct_background(self, current_frame: np.ndarray, frame_data: Dict[str, Any], 
                               all_frames: List[np.ndarray], frame_idx: int) -> np.ndarray:
        """
        Reconstruct background behind objects using content from other frames.
        """
        reconstructed_frame = current_frame.copy()
        
        # Get all object masks for this frame
        combined_mask = np.zeros(current_frame.shape[:2], dtype=np.uint8)
        for obj in frame_data.get('objects', []):
            if 'mask' in obj:
                obj_mask = obj['mask']
                combined_mask = cv2.bitwise_or(combined_mask, obj_mask.astype(np.uint8))
        
        if np.sum(combined_mask) == 0:
            return reconstructed_frame
        
        # Try to fill object areas using content from neighboring frames
        object_areas = combined_mask > 0
        
        # Strategy 1: Use neighboring frames (temporal reconstruction)
        neighbor_indices = []
        for offset in [-2, -1, 1, 2]:  # Check 2 frames before and after
            neighbor_idx = frame_idx + offset
            if 0 <= neighbor_idx < len(all_frames):
                neighbor_indices.append(neighbor_idx)
        
        if neighbor_indices:
            # Create weighted average of neighboring frames for object areas
            background_content = np.zeros_like(current_frame, dtype=np.float64)
            total_weight = 0
            
            for neighbor_idx in neighbor_indices:
                weight = 1.0 / (abs(neighbor_idx - frame_idx))  # Closer frames have higher weight
                background_content += all_frames[neighbor_idx].astype(np.float64) * weight
                total_weight += weight
            
            if total_weight > 0:
                background_content = (background_content / total_weight).astype(np.uint8)
                
                # Use inpainting to blend the background content smoothly
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)
                
                # Replace object areas with reconstructed background
                reconstructed_frame[object_areas] = background_content[object_areas]
                
                # Apply Gaussian blur to the boundary for smooth blending
                blurred_frame = cv2.GaussianBlur(reconstructed_frame, (5, 5), 0)
                
                # Create a soft transition mask
                distance_transform = cv2.distanceTransform(255 - dilated_mask, cv2.DIST_L2, 5)
                transition_mask = np.clip(distance_transform / 10.0, 0, 1)  # Soft falloff over 10 pixels
                
                # Blend original and reconstructed content based on distance from object
                for c in range(3):  # For each color channel
                    reconstructed_frame[:, :, c] = (
                        blurred_frame[:, :, c] * (1 - transition_mask) + 
                        current_frame[:, :, c] * transition_mask
                    ).astype(np.uint8)
        
        else:
            # Fallback: Use simple inpainting if no neighbor frames available
            inpainted = cv2.inpaint(current_frame, combined_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            reconstructed_frame = inpainted
        
        return reconstructed_frame
    
    def _cleanup_panorama_black_areas(self, panorama: np.ndarray) -> np.ndarray:
        """
        Remove black areas in panorama using smart inpainting that excludes border areas.
        """
        if panorama is None:
            return None
            
        try:
            # Create mask for black/very dark areas
            black_threshold = config.get_int('stitching', 'cleanup_black_threshold', 10)
            gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            black_mask = (gray < black_threshold).astype(np.uint8)
            
            # Remove small isolated black pixels (noise)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
            
            # SMART FILTERING: Remove black areas that touch borders
            exclude_borders = config.get_bool('stitching', 'exclude_border_black_areas', True)
            h, w = black_mask.shape
            
            if exclude_borders:
                border_mask = np.zeros_like(black_mask)
                
                # Create border region (configurable width from each edge)
                border_width = config.get_int('stitching', 'border_exclusion_width', 10)
                border_mask[:border_width, :] = 1    # Top border
                border_mask[-border_width:, :] = 1   # Bottom border 
                border_mask[:, :border_width] = 1    # Left border
                border_mask[:, -border_width:] = 1   # Right border
                
                # Find black areas that touch borders using connected components
                num_labels, labels = cv2.connectedComponents(black_mask)
                
                # Create mask for interior black areas only
                interior_black_mask = np.zeros_like(black_mask)
                
                for label in range(1, num_labels):  # Skip background (label 0)
                    component_mask = (labels == label).astype(np.uint8)
                    
                    # Check if this component touches any border
                    if np.any(component_mask & border_mask):
                        # Component touches border - don't inpaint
                        continue
                    else:
                        # Component is interior - add to inpainting mask
                        interior_black_mask |= component_mask
                
                final_mask = interior_black_mask
                excluded_pixels = np.sum(black_mask) - np.sum(interior_black_mask)
            else:
                # Use all black areas
                final_mask = black_mask
                excluded_pixels = 0
            
            # Only inpaint if we have significant interior black areas
            black_area_ratio = np.sum(final_mask) / (h * w)
            
            if black_area_ratio > 0.001:  # More than 0.1% interior black areas
                inpaint_radius = config.get_int('stitching', 'inpaint_radius', 7)
                
                if exclude_borders and excluded_pixels > 0:
                    logging.info(f"Inpainting {black_area_ratio*100:.2f}% interior black areas in panorama")
                    logging.info(f"Excluded {excluded_pixels} border pixels from inpainting")
                else:
                    logging.info(f"Inpainting {black_area_ratio*100:.2f}% black areas in panorama")
                
                # Use Telea inpainting algorithm for better results
                inpainted = cv2.inpaint(panorama, final_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
                return inpainted
            else:
                logging.info("No significant interior black areas found - panorama clean")
                return panorama
                
        except Exception as e:
            logging.warning(f"Failed to cleanup panorama black areas: {e}")
            return panorama
    
    def _process_keypoints(self, segmentation_result: Dict[str, Any], frames: List[np.ndarray]) -> Dict[str, Any]:
        """Process keypoints for all segmented objects and save object images."""
        try:
            # Extract objects from all frames and add cropped images
            all_objects = []
            for frame_data in segmentation_result['frames_data']:
                objects = frame_data.get('objects', [])
                frame_idx = frame_data.get('frame_index', 0)
                
                # Get the actual frame for cropping
                if frame_idx < len(frames):
                    frame = frames[frame_idx]
                    
                    # Create a masked version of the frame where all object backgrounds are black
                    masked_frame = self._create_masked_frame_for_objects(frame, objects)
                    
                    for obj_idx, obj in enumerate(objects):
                        obj['frame_index'] = frame_idx
                        obj['object_id'] = f"frame_{frame_idx}_obj_{obj_idx}"
                        
                        # Crop object image using bounding box from the masked frame
                        bbox = obj.get('bbox', [])
                        if len(bbox) >= 4:
                            x1, y1, x2, y2 = map(int, bbox[:4])
                            # Ensure coordinates are within frame bounds
                            h, w = masked_frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            if x2 > x1 and y2 > y1:
                                cropped_object = masked_frame[y1:y2, x1:x2]
                                obj['cropped_image'] = cropped_object
                                obj['crop_bbox'] = [x1, y1, x2, y2]
                                obj['crop_size'] = [x2-x1, y2-y1]
                            else:
                                obj['cropped_image'] = None
                                obj['crop_bbox'] = bbox
                                obj['crop_size'] = [0, 0]
                        else:
                            obj['cropped_image'] = None
                            obj['crop_bbox'] = []
                            obj['crop_size'] = [0, 0]
                        
                        all_objects.append(obj)
            
            # Process keypoints for all objects
            if all_objects:
                keypoint_result = self.keypointer.extract_keypoints(all_objects)
                
                # Enhance objects with additional metadata
                enhanced_objects = []
                for obj in keypoint_result.get('objects', []):
                    # Add detailed metadata for saving
                    enhanced_obj = {
                        'object_id': obj.get('object_id', 'unknown'),
                        'frame_index': obj.get('frame_index', 0),
                        'class_name': obj.get('class_name', 'unknown'),
                        'confidence': obj.get('confidence', 0.0),
                        'bbox': obj.get('bbox', []),
                        'crop_bbox': obj.get('crop_bbox', []),
                        'crop_size': obj.get('crop_size', [0, 0]),
                        'keypoints': obj.get('keypoints', []),
                        'segmentation_mask': obj.get('segmentation_mask'),
                        'cropped_image': obj.get('cropped_image'),
                        'processed_at': time.time(),
                        'has_keypoints': len(obj.get('keypoints', [])) > 0,
                        'mask_area': obj.get('mask_area', 0)
                    }
                    enhanced_objects.append(enhanced_obj)
                
                return {
                    'objects': enhanced_objects,
                    'total_objects': len(enhanced_objects),
                    'objects_with_keypoints': sum(1 for obj in enhanced_objects if obj['has_keypoints']),
                    'processing_time': keypoint_result.get('processing_time', 0)
                }
            else:
                return {
                    'objects': [],
                    'total_objects': 0,
                    'objects_with_keypoints': 0,
                    'processing_time': 0
                }
                
        except Exception as e:
            logging.error(f"Keypoint processing failed: {e}")
            return {
                'objects': [], 
                'total_objects': 0,
                'objects_with_keypoints': 0,
                'error': str(e)
            }
    
    @track_performance
    def process_scene(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete scene through the pipeline.
        
        Args:
            scene_data: Scene data from video splitter
            
        Returns:
            Processed scene data or Complex scene indicator
        """
        scene_number = scene_data.get('scene_number', 0)
        frames = scene_data.get('frames', [])
        
        if not frames:
            return {
                'scene_type': 'Complex',
                'scene_number': scene_number,
                'error': 'no_frames'
            }
        
        logging.info(f"🎬 Processing scene {scene_number} ({len(frames)} frames)")
        
        try:
            # STEP 1: Frame Segmentation (moved before stitching)
            logging.info(f"🎯 Scene {scene_number}: Step 1/4 - Object segmentation on frames...")
            step_start = time.time()
            frame_segmentation_result = self.segmenter.segment_frames_only(frames)
            step_time = time.time() - step_start
            objects_count = sum(len(frame_data.get('objects', [])) for frame_data in frame_segmentation_result.get('frames_data', []))
            logging.info(f"   ✅ Frame segmentation completed in {step_time:.1f}s - Found {objects_count} objects")
            
            # STEP 2: Create masked frames (overlay masks to hide objects)
            logging.info(f"🎭 Scene {scene_number}: Step 2/4 - Creating masked frames...")
            step_start = time.time()
            masked_frames = self._create_masked_frames(frames, frame_segmentation_result)
            step_time = time.time() - step_start
            logging.info(f"   ✅ Masked frames created in {step_time:.1f}s")
            
            # STEP 3: Stitching with masked frames
            logging.info(f"🧩 Scene {scene_number}: Step 3/4 - Stitching panorama with masked frames...")
            step_start = time.time()
            stitching_result = self.stitcher.stitch_scene(masked_frames)
            step_time = time.time() - step_start
            logging.info(f"   ✅ Stitching completed in {step_time:.1f}s")
            
            if stitching_result['scene_type'] == 'Complex':
                logging.info(f"⚠️  Scene {scene_number} classified as Complex (unsuitable for stitching)")
                return {
                    'scene_type': 'Complex',
                    'scene_number': scene_number,
                    'stitching_result': stitching_result,
                    'frames': frames  # Return original frames for AV1 encoding
                }
            
            panorama = stitching_result['panorama']
            
            # Clean up any remaining black areas in the panorama
            if panorama is not None and config.get_bool('stitching', 'enable_panorama_cleanup', True):
                logging.info(f"🎨 Scene {scene_number}: Cleaning up panorama black areas...")
                panorama = self._cleanup_panorama_black_areas(panorama)
                stitching_result['panorama'] = panorama  # Update the cleaned panorama
            
            homographies = stitching_result['homographies']
            
            # STEP 4: Process keypoints (using original segmentation data)
            logging.info(f"🎯 Scene {scene_number}: Step 4/4 - Keypoint extraction...")
            step_start = time.time()
            keypoint_result = self._process_keypoints(frame_segmentation_result, frames)
            step_time = time.time() - step_start
            keypoints_count = len(keypoint_result.get('objects', []))
            logging.info(f"   ✅ Keypoints completed in {step_time:.1f}s - Processed {keypoints_count} objects")
            
            # Combine all results
            processed_result = {
                'scene_type': stitching_result['scene_type'],
                'scene_number': scene_number,
                'stitching_result': stitching_result,
                'segmentation_result': frame_segmentation_result,
                'keypoint_result': keypoint_result,
                'masked_frames': masked_frames
            }
            
            logging.info(f"🎉 Scene {scene_number} processed successfully")
            return processed_result
            
        except Exception as e:
            logging.error(f"Scene {scene_number} processing failed: {e}")
            return {
                'scene_type': 'Complex',
                'scene_number': scene_number,
                'error': str(e),
                'frames': frames
            }
    
    def _save_scene_objects(self, scene_data: Dict[str, Any], output_dir: Path, scene_number: int):
        """Save individual object images using the dedicated Saver component."""
        if not output_dir:
            return
        
        # Use the Saver component to save objects
        save_result = self.saver.save_scene_objects(scene_data, output_dir, scene_number)
        
        if save_result.get('saved_objects', 0) > 0:
            logging.info(f"Saved {save_result['saved_objects']} objects for scene {scene_number}")
        
        # Save metadata using the Saver component
        metadata_result = self.saver.save_metadata(scene_data, output_dir, scene_number)
        if metadata_result.get('metadata_saved'):
            logging.info(f"Saved metadata for scene {scene_number}")
        
        return save_result
    
    def _log_component_fps_statistics(self):
        """Log average FPS statistics for each component across all processed scenes."""
        try:
            from .decorators import profiler
            
            # Get overall performance summary
            performance_summary = profiler.get_overall_summary()
            
            if not performance_summary:
                logging.info("📊 Component FPS Analysis: No performance data available")
                return
            
            # Define frame processing components and their typical frame processing
            component_mappings = {
                'segment_frames_only_processing': 'Frame Segmentation',
                'stitch_scene_processing': 'Stitching', 
                'extract_keypoints_processing': 'Keypoint Extraction',
                '_detect_scene_cuts_in_batch_processing': 'Scene Detection'
            }
            
            # Calculate total frames processed (estimate from scene data)
            total_frames_processed = 0
            scenes_with_frames = 0
            
            # Estimate frames per scene based on processing times and scene count
            if self.processed_scenes > 0:
                # Rough estimate: 24 fps video, average scene ~2-3 seconds = ~50-70 frames per scene
                estimated_frames_per_scene = 60  # Conservative estimate
                total_frames_processed = self.processed_scenes * estimated_frames_per_scene
                scenes_with_frames = self.processed_scenes
            
            logging.info("📊 Component Performance Analysis (Average FPS):")
            logging.info("=" * 60)
            
            for component_key, component_name in component_mappings.items():
                if component_key in performance_summary:
                    stats = performance_summary[component_key]
                    avg_time = stats['avg_time']
                    call_count = stats['call_count']
                    
                    if avg_time > 0 and call_count > 0:
                        # Calculate FPS based on estimated frames per call
                        if component_key == 'segment_frames_only_processing':
                            # Segmentation processes all frames in a scene
                            estimated_frames_per_call = estimated_frames_per_scene
                        elif component_key == 'stitch_scene_processing':
                            # Stitching processes all frames in a scene
                            estimated_frames_per_call = estimated_frames_per_scene
                        elif component_key == 'extract_keypoints_processing':
                            # Keypoints process individual objects, but we can estimate frames
                            estimated_frames_per_call = estimated_frames_per_scene
                        else:
                            estimated_frames_per_call = estimated_frames_per_scene
                        
                        fps = estimated_frames_per_call / avg_time
                        
                        logging.info(f"🎯 {component_name:20s}: {fps:6.1f} fps (avg {avg_time:5.1f}s per scene)")
                    else:
                        logging.info(f"🎯 {component_name:20s}: No data available")
                else:
                    logging.info(f"🎯 {component_name:20s}: Not measured")
            
            # Overall pipeline FPS
            if self.processing_times and total_frames_processed > 0:
                total_processing_time = sum(self.processing_times)
                overall_fps = total_frames_processed / total_processing_time
                logging.info("=" * 60)
                logging.info(f"🚀 Overall Pipeline    : {overall_fps:6.1f} fps ({total_processing_time:.1f}s total)")
                logging.info(f"📈 Estimated Frames   : {total_frames_processed} ({estimated_frames_per_scene} per scene avg)")
                logging.info(f"🎬 Scenes Processed   : {self.processed_scenes} scenes")
            
        except Exception as e:
            logging.warning(f"Failed to calculate component FPS statistics: {e}")
    
    @track_performance
    def process_video(self, input_video: str, output_dir: str = None, 
                     enable_saving: bool = True) -> Dict[str, Any]:
        """
        Process complete video through the pipeline.
        
        Args:
            input_video: Path to input video
            output_dir: Output directory for results
            enable_saving: Whether to save results to files
            
        Returns:
            Processing summary
        """
        logging.info(f"🎬 Starting PointStream pipeline processing: {input_video}")
        
        # Setup output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            logging.info(f"📁 Output directory: {output_path}")
        
        # Initialize video scene splitter
        logging.info("🎞️  Initializing video scene splitter...")
        splitter = VideoSceneSplitter(
            input_video=input_video,
            output_dir=None,  # We handle saving differently
            enable_encoding=False  # We just want frames
        )
        logging.info("✅ Video splitter ready")
        
        try:
            # Process scenes
            scene_generator = splitter.process_video_realtime_generator()
            
            for scene_data in scene_generator:
                # Handle completion/error status
                if isinstance(scene_data, dict) and scene_data.get('status') in ['complete', 'error']:
                    break
                
                scene_number = scene_data.get('scene_number', 0)
                scene_start_time = scene_data.get('start_time', 0)
                scene_end_time = scene_data.get('end_time', 0)
                scene_duration = scene_end_time - scene_start_time
                processing_start = time.time()
                
                logging.info(f"📺 Processing scene {scene_number} (#{self.processed_scenes + 1}) - Duration: {scene_duration:.2f}s")
                
                # Process scene
                logging.info(f"🚀 Starting sequential processing for scene {scene_number}")
                result = self.process_scene(scene_data)
                
                processing_time = time.time() - processing_start
                self.processing_times.append(processing_time)
                self.processed_scenes += 1
                
                logging.info(f"⏱️  Scene {scene_number} completed in {processing_time:.1f}s - Type: {result.get('scene_type', 'Unknown')}")
                
                # Handle results
                if result.get('scene_type') == 'Complex':
                    # Handle complex scene
                    self.complex_scenes += 1
                    logging.info(f"🎬 Encoding complex scene {scene_number} to video...")
                    encoding_result = self.saver.save_complex_scene_video(
                        frames=scene_data.get('frames', []),
                        output_path=str(output_path / f"scene_{scene_number:04d}_complex.mp4"),
                        fps=scene_data.get('fps', 25.0),
                        scene_number=scene_number
                    )
                    logging.info(f"✅ Scene {scene_number}: Complex -> AV1 encoded")
                    
                    if enable_saving and encoding_result.get('success'):
                        logging.info(f"💾 Complex scene saved: {encoding_result['output_path']}")
                    else:
                        logging.warning(f"⚠️ Failed to encode complex scene {scene_number}: {encoding_result.get('error', 'Unknown error')}")
                
                else:
                    # Handle successfully processed scene
                    logging.info(f"✨ Scene {scene_number}: {result.get('scene_type', 'Unknown')} -> Processed successfully")
                    
                    # Save results if enabled
                    if enable_saving and output_dir:
                        logging.info(f"💾 Saving scene {scene_number} results...")
                        self._save_scene_results(result, output_path, scene_number)
                        logging.info(f"✅ Scene {scene_number} results saved")
                
                # Progress summary (every 5 scenes)
                if self.processed_scenes % 5 == 0:
                    avg_time = sum(self.processing_times) / len(self.processing_times)
                    logging.info(f"📊 Progress: {self.processed_scenes} scenes completed | Avg: {avg_time:.1f}s/scene | Complex: {self.complex_scenes}")
                    logging.info("-" * 80)
            
            # Final cleanup
            splitter.close()
            
            # Calculate component FPS statistics
            self._log_component_fps_statistics()
            
            # Generate summary
            summary = self._generate_processing_summary()
            
            logging.info("PointStream pipeline processing complete")
            return summary
            
        except Exception as e:
            logging.error(f"Pipeline processing failed: {e}")
            raise
    
    def _save_scene_results(self, result: Dict[str, Any], output_path: Path, scene_number: int):
        """Save scene processing results to files."""
        try:
            # Import profiler for timing data
            from .decorators import profiler
            
            # Create subdirectories
            (output_path / "panoramas").mkdir(exist_ok=True)
            (output_path / "results").mkdir(exist_ok=True)
            
            # Save panorama
            stitching_result = result.get('stitching_result', {})
            panorama = stitching_result.get('panorama')
            if panorama is not None:
                panorama_path = output_path / "panoramas" / f"scene_{scene_number:04d}_panorama.jpg"
                cv2.imwrite(str(panorama_path), panorama)
            
            # Save objects (images, masks, keypoints) - use single implementation
            self._save_scene_objects(result, output_path, scene_number)
            
            # Get detailed performance data
            performance_summary = profiler.get_overall_summary()
            
            # Enhanced metadata with detailed object and performance information
            keypoint_result = result.get('keypoint_result', {})
            segmentation_result = result.get('segmentation_result', {})
            
            metadata = {
                'scene_number': scene_number,
                'scene_type': result.get('scene_type'),
                'processing_timestamp': time.time(),
                
                'stitching': {
                    'scene_type': stitching_result.get('scene_type'),
                    'homographies_count': len(stitching_result.get('homographies', [])),
                    'panorama_shape': list(panorama.shape) if panorama is not None else None,
                    'processing_time': stitching_result.get('processing_time', 0)
                },
                
                'segmentation': {
                    'panorama_objects': len(segmentation_result.get('panorama_data', {}).get('objects', [])),
                    'total_frame_objects': sum(len(fd.get('objects', [])) for fd in segmentation_result.get('frames_data', [])),
                    'frames_processed': len(segmentation_result.get('frames_data', [])),
                    'processing_time': segmentation_result.get('processing_time', 0)
                },
                
                'keypoints': {
                    'total_objects': keypoint_result.get('total_objects', 0),
                    'objects_with_keypoints': keypoint_result.get('objects_with_keypoints', 0),
                    'processing_time': keypoint_result.get('processing_time', 0),
                    'objects_saved': len([obj for obj in keypoint_result.get('objects', []) if obj.get('cropped_image') is not None])
                },
                
                'performance': {
                    'detailed_timings': performance_summary,
                    'total_scene_time': sum(timing['total_time'] for timing in performance_summary.values()),
                    'slowest_operation': max(performance_summary.keys(), key=lambda k: performance_summary[k]['avg_time']) if performance_summary else None
                }
            }
            
            metadata_path = output_path / "results" / f"scene_{scene_number:04d}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Log performance summary for this scene
            profiler.log_scene_summary(scene_number)
            
            logging.info(f"💾 Scene {scene_number} results saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Failed to save scene {scene_number} results: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def _generate_processing_summary(self) -> Dict[str, Any]:
        """Generate final processing summary."""
        total_time = sum(self.processing_times)
        
        return {
            'processed_scenes': self.processed_scenes,
            'complex_scenes': self.complex_scenes,
            'simple_scenes': self.processed_scenes - self.complex_scenes,
            'total_processing_time': total_time,
            'average_processing_time': total_time / max(self.processed_scenes, 1),
            'throughput': self.processed_scenes / total_time if total_time > 0 else 0,
            'workflow': 'segmentation_first'
        }


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration with better visibility."""
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Setup console handler with better formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # More visible format for terminal output
    formatter = logging.Formatter(
        '🔥 %(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)
    
    # Ensure our module's logger uses the same setup
    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Suppress verbose third-party logging but keep errors
    logging.getLogger('mmpose').setLevel(logging.ERROR)
    logging.getLogger('mmdet').setLevel(logging.ERROR)
    logging.getLogger('mmengine').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    print(f"🚀 PointStream Pipeline Starting - Log Level: {log_level.upper()}")
    print(f"📊 Terminal output will show progress updates...")
    print("=" * 80)


def main():
    """Main entry point for the PointStream pipeline."""
    parser = argparse.ArgumentParser(
        description="PointStream Pipeline - Simplified Sequential Video Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process video
    python server_simplified.py input.mp4
    
    # Custom output directory
    python server_simplified.py input.mp4 --output-dir ./output
    
    # Process without saving files
    python server_simplified.py input.mp4 --no-saving
    
    # Custom configuration
    python server_simplified.py input.mp4 --config config_custom.ini
        """
    )
    
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--output-dir', default='./pointstream_output',
                       help='Output directory for processed results')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--no-saving', action='store_true',
                       help='Disable saving results to files')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate input
    if not Path(args.input_video).exists():
        print(f"Error: Input video not found: {args.input_video}")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = PointStreamPipeline(config_file=args.config)
        
        # Process video
        summary = pipeline.process_video(
            input_video=args.input_video,
            output_dir=args.output_dir if not args.no_saving else None,
            enable_saving=not args.no_saving
        )
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"POINTSTREAM PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Processed scenes: {summary['processed_scenes']}")
        print(f"Simple scenes: {summary['simple_scenes']}")
        print(f"Complex scenes: {summary['complex_scenes']}")
        print(f"Total processing time: {summary['total_processing_time']:.3f}s")
        print(f"Average time per scene: {summary['average_processing_time']:.3f}s")
        print(f"Throughput: {summary['throughput']:.2f} scenes/second")
        print(f"Workflow: {summary['workflow']}")
        
        if not args.no_saving:
            print(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

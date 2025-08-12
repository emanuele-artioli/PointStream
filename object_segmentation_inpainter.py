#!/usr/bin/env python3
"""
Object Segmentation and Background Inpainter

This script processes frames yielded from the video scene splitter to:
1. Detect and segment objects using YOLO
2. Track the 3 most important objects per scene (by confidence)
3. Extract masked object images
4. Inpaint backgrounds to remove objects
5. Yield processed data for downstream pipeline stages

The script separates processing time from file saving and maintains
the generator pattern for real-time streaming simulation.
"""

import os
import sys
import time
import logging
import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator
import tempfile

try:
    import torch
    from ultralytics import YOLO
    import config
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure you have activated the pointstream environment and installed:")
    print("  pip install ultralytics torch opencv-python")
    sys.exit(1)


class ObjectSegmentationInpainter:
    def __init__(self, output_dir: str = None, config_file: str = None,
                 enable_saving: bool = False, model_name: str = None):
        """
        Initialize the object segmentation and inpainting processor.
        
        Args:
            output_dir: Directory to save object and background images (only used if enable_saving=True)
            config_file: Path to configuration file
            enable_saving: Whether to save processed images to files (for debugging) or just yield data
            model_name: YOLO model name (e.g., 'yolov8n-seg.pt', 'yolov8s-seg.pt', etc.)
        """
        # Load configuration
        if config_file:
            config.load_config(config_file)
        
        self.enable_saving = enable_saving
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir and enable_saving:
            self.output_dir.mkdir(exist_ok=True)
            # Create subdirectories for organization
            (self.output_dir / "objects").mkdir(exist_ok=True)
            (self.output_dir / "backgrounds").mkdir(exist_ok=True)
            (self.output_dir / "masks").mkdir(exist_ok=True)
            (self.output_dir / "panoramas").mkdir(exist_ok=True)
        
        # Get segmentation configuration
        self.model_name = model_name or config.get_str('segmentation', 'yolo_model', 'yolov8n-seg.pt')
        self.confidence_threshold = config.get_float('segmentation', 'confidence_threshold', 0.25)
        self.iou_threshold = config.get_float('segmentation', 'iou_threshold', 0.7)
        self.max_objects = config.get_int('segmentation', 'max_objects_per_frame', 3)
        self.device = config.get_str('segmentation', 'device', 'auto')
        
        # Performance optimization settings
        self.yolo_image_size = config.get_int('segmentation', 'yolo_image_size', 640)
        self.yolo_half_precision = config.get_bool('segmentation', 'yolo_half_precision', True)
        self.yolo_stream = config.get_bool('segmentation', 'yolo_stream', False)
        
        # YOLO class filtering (empty list = all classes)
        classes_config = config.get_list('segmentation', 'classes', [])
        self.classes = classes_config if classes_config else None
        
        # Get inpainting configuration
        self.inpaint_method = config.get_str('inpainting', 'method', 'telea')  # 'telea' or 'navier_stokes'
        self.inpaint_radius = config.get_int('inpainting', 'radius', 3)
        self.dilate_mask = config.get_bool('inpainting', 'dilate_mask', True)
        self.dilate_kernel_size = config.get_int('inpainting', 'dilate_kernel_size', 3)
        
        # Custom selection strategy (applied after YOLO's built-in filtering)
        self.selection_strategy = config.get_str('object_tracking', 'selection_strategy', 'confidence')
        self.min_object_area = config.get_int('object_tracking', 'min_object_area', 500)
        self.frame_skip = config.get_int('object_tracking', 'frame_skip', 1)
        
        # Output format configuration
        self.object_format = config.get_str('object_output', 'object_image_format', 'png')
        self.background_format = config.get_str('object_output', 'background_image_format', 'png')
        
        # Performance tracking
        self.processing_times = []
        self.saving_times = []
        self.processed_scenes = 0
        
        # Initialize YOLO model
        self._initialize_model()
        
        logging.info(f"Object Segmentation Inpainter initialized")
        logging.info(f"YOLO model: {self.model_name}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Max objects per frame (YOLO max_det): {self.max_objects}")
        logging.info(f"Confidence threshold (YOLO): {self.confidence_threshold}")
        logging.info(f"Classes filter (YOLO): {self.classes or 'All classes'}")
        logging.info(f"Custom selection strategy: {self.selection_strategy}")
        logging.info(f"Frame skip: {self.frame_skip} (process every {self.frame_skip} frames)")
        logging.info(f"YOLO image size: {self.yolo_image_size}")
        logging.info(f"YOLO half precision: {self.yolo_half_precision}")
        logging.info(f"YOLO streaming mode: {self.yolo_stream}")
        logging.info(f"Inpainting method: {self.inpaint_method}")
        logging.info(f"Saving enabled: {self.enable_saving}")
        logging.info(f"Purpose: Create clean panoramas from inpainted backgrounds")

    def _initialize_model(self):
        """Initialize the YOLO segmentation model."""
        try:
            logging.info(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(self.model_name)
            
            # Set device
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            logging.info(f"YOLO model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            raise

    def _extract_objects_from_track_results(self, results) -> List[Dict]:
        """
        Extract objects from YOLO track results.
        
        Args:
            results: YOLO track results (already filtered by YOLO's max_det and classes)
                    Can be a list or generator depending on stream parameter
            
        Returns:
            List of object dictionaries with detection and tracking info
        """
        objects = []
        
        # Handle both generator and list results
        if hasattr(results, '__iter__') and not isinstance(results, (list, tuple)):
            # It's a generator, convert to list
            results = list(results)
        
        if len(results) == 0 or results[0].masks is None:
            return objects
        
        result = results[0]  # Assuming single image
        boxes = result.boxes
        masks = result.masks
        
        if boxes is None or masks is None:
            return objects
        
        # Extract object information from tracked results
        for i in range(len(boxes)):
            box = boxes[i]
            mask = masks[i]
            
            # Get class and confidence
            class_id = int(box.cls.cpu().numpy())
            confidence = float(box.conf.cpu().numpy())
            class_name = self.model.names[class_id]
            
            # Get track ID if available (from YOLO tracking)
            track_id = None
            if hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id.cpu().numpy())
            
            # Get mask data (at original image resolution due to retina_masks=True)
            mask_data = mask.data[0].cpu().numpy().astype(np.uint8)
            
            # Check minimum area (only additional filter we apply)
            mask_area = np.sum(mask_data)
            if mask_area < self.min_object_area:
                continue
            
            # Get bounding box
            bbox = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = bbox
            
            # Calculate center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Calculate size (area of bounding box)
            size = (x2 - x1) * (y2 - y1)
            
            objects.append({
                'track_id': track_id,
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': bbox,
                'center': (center_x, center_y),
                'size': size,
                'mask_area': mask_area,
                'mask': mask_data,
                'index': i
            })
        
        return objects

    def _apply_selection_strategy(self, objects: List[Dict], strategy: str = None) -> List[Dict]:
        """
        Apply custom selection strategy to objects that have already been filtered by YOLO.
        YOLO has already applied max_det (keeping top confidence detections) and class filtering.
        
        Args:
            objects: List of objects already filtered by YOLO
            strategy: Selection strategy ('confidence', 'size', 'center', 'area')
            
        Returns:
            List of objects after applying the strategy
        """
        if not objects:
            return objects
        
        strategy = strategy or self.selection_strategy
        
        # Apply custom sorting strategy
        if strategy == 'confidence':
            # Sort by confidence (highest first) - this is redundant since YOLO already does this with max_det
            objects.sort(key=lambda x: x['confidence'], reverse=True)
        elif strategy == 'size':
            # Sort by bounding box size (largest first)
            objects.sort(key=lambda x: x['size'], reverse=True)
        elif strategy == 'center':
            # Sort by distance from image center (closest first)
            if objects:
                img_h, img_w = objects[0]['mask'].shape
                img_center = (img_w / 2, img_h / 2)
                objects.sort(key=lambda x: np.sqrt((x['center'][0] - img_center[0])**2 + 
                                                 (x['center'][1] - img_center[1])**2))
        elif strategy == 'area':
            # Sort by mask area (largest first)
            objects.sort(key=lambda x: x['mask_area'], reverse=True)
        
        # Note: We don't limit the number here since YOLO's max_det already did that
        # But we could add an additional limit if needed:
        # return objects[:self.max_objects]
        
        return objects

    def _create_combined_mask(self, objects: List[Dict], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a combined mask from multiple object masks.
        
        Args:
            objects: List of object dictionaries with masks
            image_shape: Shape of the image (height, width)
            
        Returns:
            Combined binary mask
        """
        combined_mask = np.zeros(image_shape, dtype=np.uint8)
        
        for obj in objects:
            mask = obj['mask']
            combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
        
        # Dilate mask if configured
        if self.dilate_mask and self.dilate_kernel_size > 0:
            kernel = np.ones((self.dilate_kernel_size, self.dilate_kernel_size), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        
        return combined_mask

    def _inpaint_background(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint the background to remove objects.
        
        Args:
            image: Original image
            mask: Binary mask of objects to remove
            
        Returns:
            Inpainted image
        """
        # Convert mask to 8-bit if needed
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Apply inpainting
        if self.inpaint_method == 'telea':
            inpainted = cv2.inpaint(image, mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        elif self.inpaint_method == 'navier_stokes':
            inpainted = cv2.inpaint(image, mask, self.inpaint_radius, cv2.INPAINT_NS)
        else:
            logging.warning(f"Unknown inpainting method: {self.inpaint_method}, using Telea")
            inpainted = cv2.inpaint(image, mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        
        return inpainted

    def _create_scene_panorama(self, inpainted_frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Create a panorama from inpainted background frames using OpenCV Stitcher.
        
        Args:
            inpainted_frames: List of inpainted background images
            
        Returns:
            Panorama image or None if creation fails
        """
        if not inpainted_frames:
            return None
        
        if len(inpainted_frames) == 1:
            return inpainted_frames[0]
        
        try:
            # Get panorama configuration
            stitcher_mode = config.get_str('panorama', 'stitcher_mode', 'panorama')  # 'panorama' or 'scans'
            max_frames_for_stitching = config.get_int('panorama', 'max_frames', 20)
            downsample_factor = config.get_float('panorama', 'downsample_factor', 0.5)
            fallback_to_median = config.get_bool('panorama', 'fallback_to_median', True)
            
            # Limit number of frames for performance (stitching can be slow with many frames)
            frames_to_use = inpainted_frames[:max_frames_for_stitching]
            
            # Downsample frames for faster stitching if they're large
            original_frames = frames_to_use.copy()
            if downsample_factor < 1.0:
                frames_to_use = []
                for frame in original_frames:
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * downsample_factor), int(w * downsample_factor)
                    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    frames_to_use.append(resized_frame)
            
            # Create OpenCV Stitcher
            if stitcher_mode.lower() == 'scans':
                stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
            else:
                stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
            
            # Attempt to stitch the frames
            logging.info(f"Attempting to stitch {len(frames_to_use)} frames into panorama...")
            status, panorama = stitcher.stitch(frames_to_use)
            
            if status == cv2.Stitcher_OK:
                logging.info(f"Panorama stitching successful! Result size: {panorama.shape[:2]}")
                
                # If we downsampled, try to stitch at higher resolution with fewer frames
                if downsample_factor < 1.0 and len(original_frames) > 5:
                    try:
                        # Use fewer frames at original resolution for final panorama
                        step = max(1, len(original_frames) // 10)  # Use every nth frame
                        high_res_frames = original_frames[::step]
                        logging.info(f"Attempting high-res stitching with {len(high_res_frames)} frames...")
                        
                        status_hr, panorama_hr = stitcher.stitch(high_res_frames)
                        if status_hr == cv2.Stitcher_OK:
                            logging.info(f"High-res panorama successful! Result size: {panorama_hr.shape[:2]}")
                            return panorama_hr
                        else:
                            logging.info(f"High-res stitching failed (status: {status_hr}), using low-res result")
                    except Exception as e:
                        logging.warning(f"High-res stitching failed: {e}, using low-res result")
                
                return panorama
                
            else:
                # Stitching failed, log the reason
                status_messages = {
                    cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
                    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
                    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameters adjustment failed"
                }
                error_msg = status_messages.get(status, f"Unknown error (status: {status})")
                logging.warning(f"Panorama stitching failed: {error_msg}")
                
                if fallback_to_median:
                    logging.info("Falling back to median composite approach...")
                    frame_stack = np.stack(inpainted_frames, axis=0)
                    panorama = np.median(frame_stack, axis=0).astype(np.uint8)
                    return panorama
                else:
                    return inpainted_frames[0]  # Return first frame as fallback
                    
        except Exception as e:
            logging.warning(f"Failed to create panorama: {e}")
            if len(inpainted_frames) > 0:
                return inpainted_frames[0]  # Fallback to first frame
            return None

    def _extract_object_images(self, image: np.ndarray, objects: List[Dict]) -> List[np.ndarray]:
        """
        Extract individual object images using their masks.
        
        Args:
            image: Original image
            objects: List of object dictionaries with masks and bounding boxes
            
        Returns:
            List of object images
        """
        object_images = []
        
        for obj in objects:
            mask = obj['mask']
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Crop to bounding box
            cropped_image = image[y1:y2, x1:x2]
            cropped_mask = mask[y1:y2, x1:x2]
            
            # Create RGBA image with transparency
            if len(cropped_image.shape) == 3:
                # Add alpha channel
                object_img = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 4), dtype=np.uint8)
                object_img[:, :, :3] = cropped_image
                object_img[:, :, 3] = cropped_mask * 255  # Alpha channel from mask
            else:
                # Grayscale image
                object_img = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 2), dtype=np.uint8)
                object_img[:, :, 0] = cropped_image
                object_img[:, :, 1] = cropped_mask * 255
            
            object_images.append(object_img)
        
        return object_images

    def _save_scene_data(self, scene_data: Dict[str, Any], scene_number: int) -> Dict[str, Any]:
        """
        Save processed scene data to files (if saving is enabled).
        Now handles frame-by-frame processing data.
        
        Args:
            scene_data: Processed scene data with frames_data
            scene_number: Scene number for file naming
            
        Returns:
            Updated scene data with file paths
        """
        if not self.enable_saving or not self.output_dir:
            return scene_data
        
        saving_start = time.time()
        saved_files = []
        
        try:
            frames_data = scene_data.get('frames_data', [])
            
            # Process each frame's data
            for frame_data in frames_data:
                frame_idx = frame_data['frame_index']
                frame_saved_files = {}
                
                # Save background image for this frame
                if 'inpainted_background' in frame_data:
                    bg_filename = f"scene_{scene_number:04d}_frame_{frame_idx:04d}_background.{self.background_format}"
                    bg_path = self.output_dir / "backgrounds" / bg_filename
                    cv2.imwrite(str(bg_path), frame_data['inpainted_background'])
                    frame_saved_files['background_file'] = str(bg_path)
                
                # Save combined mask for this frame
                if 'combined_mask' in frame_data:
                    mask_filename = f"scene_{scene_number:04d}_frame_{frame_idx:04d}_mask.png"
                    mask_path = self.output_dir / "masks" / mask_filename
                    cv2.imwrite(str(mask_path), frame_data['combined_mask'] * 255)
                    frame_saved_files['mask_file'] = str(mask_path)
                
                # Save individual object images for this frame
                if 'object_images' in frame_data and frame_data['object_images']:
                    object_files = []
                    for i, obj_img in enumerate(frame_data['object_images']):
                        obj_info = frame_data['objects'][i]
                        class_name = obj_info['class_name']
                        confidence = obj_info['confidence']
                        track_id = obj_info.get('track_id', 'unknown')
                        
                        obj_filename = f"scene_{scene_number:04d}_frame_{frame_idx:04d}_track_{track_id}_{class_name}_{confidence:.2f}.{self.object_format}"
                        obj_path = self.output_dir / "objects" / obj_filename
                        cv2.imwrite(str(obj_path), obj_img)
                        object_files.append(str(obj_path))
                    
                    frame_saved_files['object_files'] = object_files
                
                # Add frame metadata
                frame_saved_files['frame_index'] = frame_idx
                frame_saved_files['objects'] = frame_data.get('objects', [])
                frame_saved_files['num_objects'] = frame_data.get('num_objects_detected', 0)
                frame_saved_files['processing_time'] = frame_data.get('processing_time', 0)
                
                saved_files.append(frame_saved_files)
            
            # Save scene panorama if available
            panorama_file = None
            if 'scene_panorama' in scene_data and scene_data['scene_panorama'] is not None:
                panorama_filename = f"scene_{scene_number:04d}_panorama.{self.background_format}"
                panorama_path = self.output_dir / "panoramas" / panorama_filename
                panorama_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(panorama_path), scene_data['scene_panorama'])
                panorama_file = str(panorama_path)
                logging.info(f"Saved scene {scene_number} panorama: {panorama_file}")
            
            # Save scene-level metadata
            metadata = {
                'scene_number': scene_number,
                'total_frames_processed': scene_data.get('total_frames_processed', 0),
                'total_objects_detected': scene_data.get('total_objects_detected', 0),
                'processing_time': scene_data.get('processing_time', 0),
                'panorama_file': panorama_file,
                'frames_data': saved_files
            }
            
            metadata_filename = f"scene_{scene_number:04d}_metadata.json"
            metadata_path = self.output_dir / metadata_filename
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            saving_time = time.time() - saving_start
            self.saving_times.append(saving_time)
            
            # Add saving info to scene data
            scene_data['saved_files'] = saved_files
            scene_data['panorama_file'] = panorama_file
            scene_data['saving_time'] = saving_time
            
            logging.info(f"Scene {scene_number} files saved in {saving_time:.3f}s ({len(frames_data)} frames)")
            
        except Exception as e:
            logging.error(f"Error saving scene {scene_number} files: {e}")
        
        return scene_data

    def process_scene_generator(self, scene_frame_generator: Generator) -> Generator[Dict[str, Any], None, None]:
        """
        Process scenes from the video scene splitter generator.
        
        Args:
            scene_frame_generator: Generator yielding scene data from video_scene_splitter
            
        Yields:
            Processed scene data with objects, masks, and inpainted backgrounds
        """
        logging.info("Starting object segmentation and inpainting processing...")
        
        for scene_data in scene_frame_generator:
            # Handle completion/error status from scene splitter
            if isinstance(scene_data, dict) and scene_data.get('status') in ['complete', 'error']:
                # Add our processing summary to the final status
                if scene_data.get('status') == 'complete':
                    summary = scene_data.get('summary', {})
                    
                    # Aggregate detailed timing from all scenes
                    total_yolo_time = sum(getattr(self, 'total_yolo_time', 0) for _ in range(self.processed_scenes))
                    total_inpaint_time = sum(getattr(self, 'total_inpaint_time', 0) for _ in range(self.processed_scenes))
                    
                    summary['object_processing'] = {
                        'processed_scenes': self.processed_scenes,
                        'total_processing_time': sum(self.processing_times),
                        'total_saving_time': sum(self.saving_times),
                        'average_processing_time': sum(self.processing_times) / max(self.processed_scenes, 1),
                        'processing_fps': self.processed_scenes / sum(self.processing_times) if self.processing_times else 0,
                        'timing_breakdown': {
                            'yolo_time_total': total_yolo_time,
                            'inpaint_time_total': total_inpaint_time,
                            'yolo_percentage': (total_yolo_time / sum(self.processing_times)) * 100 if self.processing_times else 0,
                            'inpaint_percentage': (total_inpaint_time / sum(self.processing_times)) * 100 if self.processing_times else 0
                        }
                    }
                    scene_data['summary'] = summary
                
                yield scene_data
                continue
            
            # Process regular scene data
            if 'frames' not in scene_data or not scene_data['frames']:
                logging.warning(f"Scene {scene_data.get('scene_number', '?')} has no frames, skipping")
                continue
            
            processing_start = time.time()
            
            try:
                # Process frames with optional skipping for performance
                frames = scene_data['frames']
                frame_processing_data = []
                total_objects_in_scene = 0
                
                # Timing for different processing steps
                total_yolo_time = 0
                total_mask_time = 0
                total_inpaint_time = 0
                total_extract_time = 0
                
                # Store inpainted backgrounds for panorama creation
                inpainted_backgrounds = []
                
                # Process every nth frame based on frame_skip setting
                for frame_idx, frame in enumerate(frames):
                    # Skip frames based on frame_skip setting (1=all frames, 2=every other frame, etc.)
                    if frame_idx % self.frame_skip != 0:
                        continue
                        
                    frame_start_time = time.time()
                    
                    # STEP 1: YOLO Detection and Tracking
                    yolo_start = time.time()
                    # Run YOLO tracking with built-in filtering and optimization:
                    # - conf: confidence threshold filter
                    # - iou: non-maximum suppression threshold  
                    # - max_det: maximum detections per frame (keeps top confidence objects)
                    # - classes: class ID filter (None = all classes)
                    # - retina_masks: get masks at original resolution (not scaled resolution)
                    # - imgsz: YOLO's built-in image resizing for optimal inference speed
                    # - half: use half precision (FP16) for faster inference
                    # - stream: memory-efficient processing for better performance
                    results = self.model.track(frame, 
                                             conf=self.confidence_threshold,
                                             iou=self.iou_threshold,
                                             device=self.device,
                                             retina_masks=True,
                                             max_det=self.max_objects,
                                             classes=self.classes,
                                             imgsz=self.yolo_image_size,
                                             half=self.yolo_half_precision,
                                             stream=self.yolo_stream,
                                             verbose=False,
                                             persist=True)  # Persist tracks across frames
                    
                    # Extract objects from YOLO's filtered and tracked results
                    objects = self._extract_objects_from_track_results(results)
                    
                    # Apply our custom selection strategy to YOLO's filtered objects
                    selected_objects = self._apply_selection_strategy(objects)
                    yolo_time = time.time() - yolo_start
                    total_yolo_time += yolo_time
                    
                    total_objects_in_scene += len(selected_objects)
                    
                    # STEP 2: Mask Creation
                    mask_start = time.time()
                    if selected_objects:
                        combined_mask = self._create_combined_mask(selected_objects, frame.shape[:2])
                    else:
                        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    mask_time = time.time() - mask_start
                    total_mask_time += mask_time
                    
                    # STEP 3: Background Inpainting (ALWAYS - needed for panorama)
                    inpaint_start = time.time()
                    if selected_objects:
                        inpainted_background = self._inpaint_background(frame, combined_mask)
                    else:
                        inpainted_background = frame.copy()  # No objects to remove
                    inpaint_time = time.time() - inpaint_start
                    total_inpaint_time += inpaint_time
                    
                    # Store for panorama creation
                    inpainted_backgrounds.append(inpainted_background)
                    
                    # STEP 4: Object Extraction (only if saving)
                    extract_start = time.time()
                    if self.enable_saving and selected_objects:
                        object_images = self._extract_object_images(frame, selected_objects)
                    else:
                        object_images = []  # Skip extraction when not saving
                    extract_time = time.time() - extract_start
                    total_extract_time += extract_time
                    
                    frame_processing_time = time.time() - frame_start_time
                    
                    # Store frame processing data with detailed timing
                    frame_data = {
                        'frame_index': frame_idx,
                        'frame': frame,
                        'objects': [
                            {
                                'track_id': obj['track_id'],
                                'class_id': obj['class_id'],
                                'class_name': obj['class_name'],
                                'confidence': obj['confidence'],
                                'bbox': obj['bbox'].tolist(),
                                'center': obj['center'],
                                'size': obj['size'],
                                'mask_area': obj['mask_area']
                            } for obj in selected_objects
                        ],
                        'object_images': object_images,
                        'combined_mask': combined_mask,
                        'inpainted_background': inpainted_background,
                        'processing_time': frame_processing_time,
                        'yolo_time': yolo_time,
                        'mask_time': mask_time,
                        'inpaint_time': inpaint_time,
                        'extract_time': extract_time,
                        'num_objects_detected': len(selected_objects)
                    }
                    frame_processing_data.append(frame_data)
                
                # STEP 5: Create Scene Panorama from Inpainted Backgrounds
                panorama_start = time.time()
                scene_panorama = self._create_scene_panorama(inpainted_backgrounds)
                panorama_time = time.time() - panorama_start
                
                processing_time = time.time() - processing_start
                self.processing_times.append(processing_time)
                self.processed_scenes += 1
                
                # Create processed scene data with all frames and panorama
                processed_data = {
                    'scene_number': scene_data.get('scene_number'),
                    'original_scene_data': scene_data,  # Keep original data
                    'frames_data': frame_processing_data,  # All processed frames
                    'scene_panorama': scene_panorama,  # Clean background panorama
                    'processing_time': processing_time,
                    'detailed_timing': {
                        'total_yolo_time': total_yolo_time,
                        'total_mask_time': total_mask_time,
                        'total_inpaint_time': total_inpaint_time,
                        'total_extract_time': total_extract_time,
                        'panorama_time': panorama_time,
                        'avg_yolo_per_frame': total_yolo_time / max(len(frame_processing_data), 1),
                        'avg_inpaint_per_frame': total_inpaint_time / max(len(frame_processing_data), 1)
                    },
                    'total_objects_detected': total_objects_in_scene,
                    'total_frames_processed': len(frames),
                    'frames_actually_processed': len(frame_processing_data),
                    'processing_timestamp': time.time()
                }
                
                # Save files if enabled (separately timed)
                processed_data = self._save_scene_data(processed_data, scene_data.get('scene_number', 0))
                
                # Log processing info with detailed timing
                scene_num = scene_data.get('scene_number', '?')
                frames_processed = len(frame_processing_data)
                
                logging.info(f"Scene {scene_num}: processed {frames_processed} frames, "
                           f"detected {total_objects_in_scene} total objects, "
                           f"processed in {processing_time:.3f}s")
                
                # Log detailed timing breakdown
                timing = processed_data['detailed_timing']
                logging.info(f"  Timing breakdown:")
                logging.info(f"    YOLO detection: {total_yolo_time:.3f}s ({timing['avg_yolo_per_frame']:.3f}s/frame)")
                logging.info(f"    Mask creation: {total_mask_time:.3f}s")
                logging.info(f"    Inpainting: {total_inpaint_time:.3f}s ({timing['avg_inpaint_per_frame']:.3f}s/frame)")
                logging.info(f"    Object extraction: {total_extract_time:.3f}s")
                logging.info(f"    Panorama creation: {panorama_time:.3f}s")
                
                # Show percentage breakdown
                if processing_time > 0:
                    yolo_pct = (total_yolo_time / processing_time) * 100
                    inpaint_pct = (total_inpaint_time / processing_time) * 100
                    logging.info(f"  Performance bottlenecks: YOLO {yolo_pct:.1f}%, Inpainting {inpaint_pct:.1f}%")
                
                # Log sample of objects found (from first frame with objects)
                sample_frame_with_objects = next((fd for fd in frame_processing_data if fd['objects']), None)
                if sample_frame_with_objects:
                    obj_names = [obj['class_name'] for obj in sample_frame_with_objects['objects']]
                    obj_confs = [f"{obj['confidence']:.2f}" for obj in sample_frame_with_objects['objects']]
                    track_ids = [str(obj.get('track_id', 'N/A')) for obj in sample_frame_with_objects['objects']]
                    logging.info(f"  Sample objects: {', '.join([f'{name}({conf},id:{tid})' for name, conf, tid in zip(obj_names, obj_confs, track_ids)])}")
                
                yield processed_data
                
            except Exception as e:
                logging.error(f"Error processing scene {scene_data.get('scene_number', '?')}: {e}")
                
                # Yield error information
                yield {
                    'status': 'error',
                    'scene_number': scene_data.get('scene_number'),
                    'error': str(e),
                    'original_scene_data': scene_data
                }

    def create_processing_summary(self) -> Dict[str, Any]:
        """Create a summary of processing statistics."""
        total_processing_time = sum(self.processing_times)
        total_saving_time = sum(self.saving_times)
        
        return {
            'processed_scenes': self.processed_scenes,
            'total_processing_time': total_processing_time,
            'total_saving_time': total_saving_time,
            'average_processing_time': total_processing_time / max(self.processed_scenes, 1),
            'average_saving_time': total_saving_time / max(len(self.saving_times), 1),
            'processing_fps': self.processed_scenes / total_processing_time if total_processing_time > 0 else 0,
            'model_info': {
                'model_name': self.model_name,
                'device': self.device,
                'confidence_threshold': self.confidence_threshold,
                'max_objects_per_frame': self.max_objects,
                'classes_filter': self.classes,
                'selection_strategy': self.selection_strategy
            }
        }


def main():
    """Example usage of the ObjectSegmentationInpainter with the video scene splitter."""
    parser = argparse.ArgumentParser(
        description="Object Segmentation and Background Inpainting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process video with object segmentation (no file saving)
    python object_segmentation_inpainter.py input.mp4
    
    # Enable saving processed images
    python object_segmentation_inpainter.py input.mp4 --output-dir ./output --enable-saving
    
    # Use different YOLO model
    python object_segmentation_inpainter.py input.mp4 --model yolov8s-seg.pt
    
    # Custom configuration
    python object_segmentation_inpainter.py input.mp4 --config config_test.ini
        """
    )
    
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--output-dir', help='Output directory for processed images')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--model', help='YOLO model name (e.g., yolov8n-seg.pt)')
    parser.add_argument('--enable-saving', action='store_true', 
                       help='Enable saving processed images to files')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for scene processing')
    
    args = parser.parse_args()
    
    try:
        # Import video scene splitter
        from video_scene_splitter import VideoSceneSplitter
        
        # Initialize video scene splitter (without encoding to get frames)
        splitter = VideoSceneSplitter(
            input_video=args.input_video,
            batch_size=args.batch_size,
            config_file=args.config,
            enable_encoding=False  # We just want the frames
        )
        
        # Initialize object processor
        processor = ObjectSegmentationInpainter(
            output_dir=args.output_dir,
            config_file=args.config,
            enable_saving=args.enable_saving,
            model_name=args.model
        )
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        print(f"Processing video: {args.input_video}")
        print(f"Saving enabled: {args.enable_saving}")
        print(f"Processing frames for object segmentation...")
        
        # Process scenes through the pipeline
        scene_count = 0
        object_count = 0
        
        # Get scene generator from splitter
        scene_generator = splitter.process_video_realtime_generator()
        
        # Process through object segmentation
        for processed_data in processor.process_scene_generator(scene_generator):
            if processed_data.get('status') == 'complete':
                # Final summary
                summary = processed_data['summary']
                obj_summary = summary.get('object_processing', {})
                
                print(f"\n{'-'*60}")
                print(f"Processing complete!")
                print(f"Processed {scene_count} scenes with {object_count} total objects")
                print(f"Scene processing time: {summary.get('total_processing_time', 0):.3f}s")
                print(f"Object processing time: {obj_summary.get('total_processing_time', 0):.3f}s")
                
                if args.enable_saving:
                    print(f"File saving time: {obj_summary.get('total_saving_time', 0):.3f}s")
                
                print(f"Object processing FPS: {obj_summary.get('processing_fps', 0):.1f} scenes/second")
                break
                
            elif processed_data.get('status') == 'error':
                print(f"❌ Error in scene {processed_data.get('scene_number', '?')}: {processed_data.get('error')}")
                continue
            
            else:
                # Regular processed scene
                scene_count += 1
                num_objects = processed_data.get('num_objects_detected', 0)
                object_count += num_objects
                processing_time = processed_data.get('processing_time', 0)
                
                print(f"Scene {scene_count:2d}: {num_objects} objects detected "
                      f"({processing_time:.3f}s processing)", end="")
                
                if args.enable_saving and 'saved_files' in processed_data:
                    saving_time = processed_data.get('saving_time', 0)
                    print(f" [saved in {saving_time:.3f}s]")
                else:
                    print()
                
                # In a real pipeline, you would use processed_data for the next stage
                # processed_data contains:
                # - 'objects': list of detected object info
                # - 'object_images': list of extracted object images 
                # - 'inpainted_background': background with objects removed
                # - 'combined_mask': mask of all objects
        
        # Clean up
        splitter.close()
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

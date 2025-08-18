#!/usr/bin/env python3
"""
Object Segmentation Component

This component handles object detection and segmentation using YOLO models.
It uses two separate models:
- One with tracking for processing video frames
- One without tracking for processing panorama images

The component returns structured data with bounding boxes, masks, and metadata.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from decorators import log_step, time_step
import config

try:
    import torch
    from ultralytics import YOLO
except ImportError as e:
    logging.error(f"Failed to import required libraries: {e}")
    raise


class Segmenter:
    """Object segmentation component using YOLO models."""
    
    def __init__(self):
        """Initialize the segmenter with YOLO models."""
        # Load configuration
        self.model_path = config.get_str('segmentation', 'yolo_model', 'yolov8n-seg.pt')
        self.confidence_threshold = config.get_float('segmentation', 'confidence_threshold', 0.25)
        self.iou_threshold = config.get_float('segmentation', 'iou_threshold', 0.7)
        self.max_objects = config.get_int('segmentation', 'max_objects_per_frame', 10)
        self.device = config.get_str('segmentation', 'device', 'auto')
        
        # Performance settings
        self.yolo_image_size = config.get_int('segmentation', 'yolo_image_size', 640)
        self.yolo_half_precision = config.get_bool('segmentation', 'yolo_half_precision', True)
        
        # Class filtering
        classes_config = config.get_list('segmentation', 'classes', [])
        self.classes = classes_config if classes_config else None
        
        # Set device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize models
        self._initialize_models()
        
        logging.info("Segmenter initialized")
        logging.info(f"Model: {self.model_path}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Confidence threshold: {self.confidence_threshold}")
        logging.info(f"IoU threshold: {self.iou_threshold}")
        logging.info(f"Max objects per frame: {self.max_objects}")
        logging.info(f"Classes filter: {self.classes or 'All classes'}")
    
    def _initialize_models(self):
        """Initialize YOLO models for tracking and non-tracking."""
        try:
            # Model with tracking for video frames
            logging.info("Loading YOLO model with tracking for frames...")
            self.model_with_tracking = YOLO(self.model_path)
            
            # Model without tracking for panorama processing
            logging.info("Loading YOLO model without tracking for panorama...")
            self.model_without_tracking = YOLO(self.model_path)
            
            logging.info("YOLO models loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize YOLO models: {e}")
            raise
    
    @log_step
    @time_step(track_processing=True)
    def segment_scene(self, frames: List[np.ndarray], panorama: np.ndarray) -> Dict[str, Any]:
        """
        Segment objects in scene frames and panorama.
        
        Args:
            frames: List of scene frames
            panorama: Scene panorama image
            
        Returns:
            Dictionary containing:
            - panorama_data: Segmentation data for panorama
            - frames_data: List of segmentation data for each frame
        """
        if not frames:
            return {
                'panorama_data': {'objects': [], 'masks': [], 'bounding_boxes': []},
                'frames_data': []
            }
        
        logging.info(f"Segmenting {len(frames)} frames and panorama")
        
        # Process panorama (without tracking)
        panorama_data = self._segment_panorama(panorama)
        
        # Process frames (with tracking)
        frames_data = []
        for i, frame in enumerate(frames):
            frame_data = self._segment_frame(frame, frame_index=i)
            frames_data.append(frame_data)
        
        return {
            'panorama_data': panorama_data,
            'frames_data': frames_data
        }
    
    def _segment_panorama(self, panorama: np.ndarray) -> Dict[str, Any]:
        """
        Segment objects in panorama image without tracking.
        
        Args:
            panorama: Panorama image
            
        Returns:
            Dictionary with panorama segmentation data
        """
        if panorama is None:
            return {'objects': [], 'masks': [], 'bounding_boxes': []}
        
        try:
            # Run YOLO detection without tracking
            results = self.model_without_tracking.predict(
                panorama,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                retina_masks=True,
                max_det=self.max_objects,
                classes=self.classes,
                imgsz=self.yolo_image_size,
                half=self.yolo_half_precision,
                verbose=False
            )
            
            return self._extract_detection_data(results, frame_index=None, with_tracking=False)
            
        except Exception as e:
            logging.error(f"Panorama segmentation failed: {e}")
            return {'objects': [], 'masks': [], 'bounding_boxes': []}
    
    def _segment_frame(self, frame: np.ndarray, frame_index: int) -> Dict[str, Any]:
        """
        Segment objects in a single frame with tracking.
        
        Args:
            frame: Input frame
            frame_index: Index of the frame in the sequence
            
        Returns:
            Dictionary with frame segmentation data
        """
        try:
            # Run YOLO tracking
            results = self.model_with_tracking.track(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                retina_masks=True,
                max_det=self.max_objects,
                classes=self.classes,
                imgsz=self.yolo_image_size,
                half=self.yolo_half_precision,
                verbose=False,
                persist=True  # Persist tracks across frames
            )
            
            return self._extract_detection_data(results, frame_index=frame_index, with_tracking=True)
            
        except Exception as e:
            logging.error(f"Frame {frame_index} segmentation failed: {e}")
            return {
                'frame_index': frame_index,
                'objects': [],
                'masks': [],
                'bounding_boxes': []
            }
    
    def _extract_detection_data(self, results, frame_index: Optional[int] = None, 
                               with_tracking: bool = False) -> Dict[str, Any]:
        """
        Extract structured data from YOLO detection results.
        
        Args:
            results: YOLO detection results
            frame_index: Frame index (for frame data)
            with_tracking: Whether tracking data is available
            
        Returns:
            Dictionary with structured detection data
        """
        objects = []
        masks = []
        bounding_boxes = []
        
        # Handle both generator and list results
        if hasattr(results, '__iter__') and not isinstance(results, (list, tuple)):
            results = list(results)
        
        if len(results) == 0 or results[0].masks is None:
            base_data = {
                'objects': objects,
                'masks': masks,
                'bounding_boxes': bounding_boxes
            }
            if frame_index is not None:
                base_data['frame_index'] = frame_index
            return base_data
        
        result = results[0]  # Single image result
        boxes = result.boxes
        detection_masks = result.masks
        
        if boxes is None or detection_masks is None:
            base_data = {
                'objects': objects,
                'masks': masks,
                'bounding_boxes': bounding_boxes
            }
            if frame_index is not None:
                base_data['frame_index'] = frame_index
            return base_data
        
        # Extract object information
        for i in range(len(boxes)):
            box = boxes[i]
            mask = detection_masks[i]
            
            # Basic detection data
            class_id = int(box.cls.cpu().numpy())
            confidence = float(box.conf.cpu().numpy())
            class_name = self.model_with_tracking.names[class_id]
            
            # Bounding box
            bbox = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = bbox
            
            # Mask data
            mask_data = mask.data[0].cpu().numpy().astype(np.uint8)
            
            # Track ID (if available)
            track_id = None
            if with_tracking and hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id.cpu().numpy())
            
            # Calculate additional properties
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            area = width * height
            mask_area = np.sum(mask_data)
            
            # Create object dictionary
            obj_data = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'center': [float(center_x), float(center_y)],
                'width': float(width),
                'height': float(height),
                'area': float(area),
                'mask_area': int(mask_area),
                'mask': mask_data
            }
            
            if track_id is not None:
                obj_data['track_id'] = track_id
            
            objects.append(obj_data)
            masks.append(mask_data)
            bounding_boxes.append([float(x1), float(y1), float(x2), float(y2)])
        
        # Create return data
        result_data = {
            'objects': objects,
            'masks': masks,
            'bounding_boxes': bounding_boxes
        }
        
        if frame_index is not None:
            result_data['frame_index'] = frame_index
        
        return result_data
    
    def extract_object_crops(self, frame: np.ndarray, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract cropped and masked object images from a frame.
        
        Args:
            frame: Source frame
            objects: List of object dictionaries with masks and bounding boxes
            
        Returns:
            List of object dictionaries with added 'cropped_image' and 'masked_image' fields
        """
        enhanced_objects = []
        
        for obj in objects:
            try:
                # Get bounding box
                bbox = obj['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Ensure coordinates are within frame bounds
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                # Extract mask
                mask = obj['mask']
                
                # Crop frame and mask to bounding box
                cropped_frame = frame[y1:y2, x1:x2]
                cropped_mask = mask[y1:y2, x1:x2]
                
                # Create masked image (object with transparent background)
                if len(cropped_frame.shape) == 3:
                    # Color image - add alpha channel
                    masked_image = np.zeros((cropped_frame.shape[0], cropped_frame.shape[1], 4), dtype=np.uint8)
                    masked_image[:, :, :3] = cropped_frame
                    masked_image[:, :, 3] = cropped_mask * 255  # Alpha channel
                else:
                    # Grayscale - add alpha channel
                    masked_image = np.zeros((cropped_frame.shape[0], cropped_frame.shape[1], 2), dtype=np.uint8)
                    masked_image[:, :, 0] = cropped_frame
                    masked_image[:, :, 1] = cropped_mask * 255
                
                # Add to enhanced object data
                enhanced_obj = obj.copy()
                enhanced_obj.update({
                    'cropped_image': cropped_frame,
                    'masked_image': masked_image,
                    'cropped_mask': cropped_mask
                })
                
                enhanced_objects.append(enhanced_obj)
                
            except Exception as e:
                logging.error(f"Failed to extract crop for object {obj.get('class_name', 'unknown')}: {e}")
                # Add object without crops
                enhanced_objects.append(obj.copy())
        
        return enhanced_objects
    
    def create_combined_mask(self, objects: List[Dict[str, Any]], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a combined mask from multiple objects.
        
        Args:
            objects: List of object dictionaries with masks
            image_shape: Shape of the target image (height, width)
            
        Returns:
            Combined binary mask
        """
        if not objects:
            return np.zeros(image_shape[:2], dtype=np.uint8)
        
        combined_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        for obj in objects:
            if 'mask' in obj:
                mask = obj['mask']
                combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
        
        return combined_mask
    
    def filter_objects_by_confidence(self, objects: List[Dict[str, Any]], 
                                   min_confidence: float) -> List[Dict[str, Any]]:
        """
        Filter objects by confidence threshold.
        
        Args:
            objects: List of object dictionaries
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of objects
        """
        return [obj for obj in objects if obj.get('confidence', 0) >= min_confidence]
    
    def filter_objects_by_area(self, objects: List[Dict[str, Any]], 
                              min_area: int) -> List[Dict[str, Any]]:
        """
        Filter objects by minimum area.
        
        Args:
            objects: List of object dictionaries
            min_area: Minimum area in pixels
            
        Returns:
            Filtered list of objects
        """
        return [obj for obj in objects if obj.get('mask_area', 0) >= min_area]
    
    def filter_objects_by_class(self, objects: List[Dict[str, Any]], 
                               allowed_classes: List[str]) -> List[Dict[str, Any]]:
        """
        Filter objects by class names.
        
        Args:
            objects: List of object dictionaries
            allowed_classes: List of allowed class names
            
        Returns:
            Filtered list of objects
        """
        return [obj for obj in objects if obj.get('class_name') in allowed_classes]
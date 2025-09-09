#!/usr/bin/env python3
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from utils.decorators import track_performance
from utils.error_handling import safe_execute
from utils import config

try:
    import torch
    from ultralytics import YOLO, YOLOE
except ImportError:
    raise

class Segmenter:
    def __init__(self):
        self.model_path = config.get_str('segmentation', 'yolo_model', 'yolov8n-seg.pt')
        self.confidence_threshold = config.get_float('segmentation', 'confidence_threshold', 0.25)
        self.iou_threshold = config.get_float('segmentation', 'iou_threshold', 0.7)
        self.max_objects = config.get_int('segmentation', 'max_objects_per_frame', 0)
        self.device = config.get_str('segmentation', 'device', 'auto')
        self.yolo_image_size = config.get_int('segmentation', 'yolo_image_size', 640)
        self.yolo_half_precision = config.get_bool('segmentation', 'yolo_half_precision', True)
        self.agnostic_nms = config.get_bool('segmentation', 'agnostic_nms', True)
        classes_config = config.get_list('segmentation', 'classes', [])
        self.classes = classes_config if classes_config else None
        self.selection_strategy = config.get_str('segmentation', 'selection_strategy', 'confidence')
        self.min_object_area = config.get_int('segmentation', 'min_object_area', 0)
        self.frame_skip = config.get_int('segmentation', 'frame_skip', 1)
        if self.frame_skip < 1:
            self.frame_skip = 1

        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self._initialize_models()
    
    def _initialize_models(self):
        try:
            if 'yoloe' in self.model_path.lower():
                self.model_with_tracking = YOLOE(self.model_path)
            else:
                self.model_with_tracking = YOLO(self.model_path)
        except Exception as e:
            raise
    
    @track_performance
    def segment_frames_only(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        if not frames:
            return {'frames_data': []}
        
        frames_data = []
        for i in range(0, len(frames), self.frame_skip):
            frame = frames[i]
            frame_data = self._segment_frame(frame, frame_index=i)
            frames_data.append(frame_data)
        
        return {'frames_data': frames_data}

    @safe_execute("Frame segmentation", {'frame_index': 0, 'objects': [], 'masks': [], 'bounding_boxes': []})
    def _segment_frame(self, frame: np.ndarray, frame_index: int) -> Dict[str, Any]:
        results = self.model_with_tracking.track(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            retina_masks=True,
            max_det=0,
            classes=self.classes,
            imgsz=self.yolo_image_size,
            half=self.yolo_half_precision,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
            persist=True
        )
        
        return self._extract_detection_data(results, frame_index=frame_index, with_tracking=True)
    
    def _extract_detection_data(self, results, frame_index: Optional[int] = None, 
                               with_tracking: bool = False) -> Dict[str, Any]:
        objects = []
        masks = []
        bounding_boxes = []
        
        if hasattr(results, '__iter__') and not isinstance(results, (list, tuple)):
            results = list(results)
        
        if len(results) == 0 or results[0].masks is None:
            base_data = {'objects': [], 'masks': [], 'bounding_boxes': []}
            if frame_index is not None:
                base_data['frame_index'] = frame_index
            return base_data
        
        result = results[0]
        boxes = result.boxes
        detection_masks = result.masks
        
        if boxes is None or detection_masks is None:
            base_data = {'objects': [], 'masks': [], 'bounding_boxes': []}
            if frame_index is not None:
                base_data['frame_index'] = frame_index
            return base_data
        
        for i in range(len(boxes)):
            box = boxes[i]
            mask = detection_masks[i]
            
            class_id = int(box.cls.cpu().numpy())
            confidence = float(box.conf.cpu().numpy())
            class_name = self.model_with_tracking.names[class_id]
            
            bbox = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = bbox
            
            mask_data = mask.data[0].cpu().numpy().astype(np.uint8)
            
            track_id = None
            if with_tracking and hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id.cpu().numpy())
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            area = width * height
            mask_area = np.sum(mask_data)
            
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

        if self.min_object_area > 0:
            objects = self.filter_objects_by_area(objects, self.min_object_area)

        if self.max_objects > 0 and len(objects) > self.max_objects:
            if self.selection_strategy == 'area':
                objects.sort(key=lambda obj: obj.get('mask_area', 0), reverse=True)
            else:
                objects.sort(key=lambda obj: obj.get('confidence', 0), reverse=True)
            objects = objects[:self.max_objects]

        for obj in objects:
            masks.append(obj['mask'])
            bounding_boxes.append(obj['bbox'])

        result_data = {
            'objects': objects,
            'masks': masks,
            'bounding_boxes': bounding_boxes
        }
        
        if frame_index is not None:
            result_data['frame_index'] = frame_index
        
        return result_data
    
    def extract_object_crops(self, frame: np.ndarray, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enhanced_objects = []
        for obj in objects:
            try:
                bbox = obj['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                mask = obj['mask']
                cropped_frame = frame[y1:y2, x1:x2]
                cropped_mask = mask[y1:y2, x1:x2]
                
                if len(cropped_frame.shape) == 3:
                    masked_image = np.zeros((cropped_frame.shape[0], cropped_frame.shape[1], 4), dtype=np.uint8)
                    masked_image[:, :, :3] = cropped_frame
                    masked_image[:, :, 3] = cropped_mask * 255
                else:
                    masked_image = np.zeros((cropped_frame.shape[0], cropped_frame.shape[1], 2), dtype=np.uint8)
                    masked_image[:, :, 0] = cropped_frame
                    masked_image[:, :, 1] = cropped_mask * 255
                
                enhanced_obj = obj.copy()
                enhanced_obj.update({
                    'cropped_image': cropped_frame,
                    'masked_image': masked_image,
                    'cropped_mask': cropped_mask
                })
                enhanced_objects.append(enhanced_obj)
            except Exception:
                enhanced_objects.append(obj.copy())
        return enhanced_objects
    
    def create_combined_mask(self, objects: List[Dict[str, Any]], image_shape: Tuple[int, int]) -> np.ndarray:
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
        return [obj for obj in objects if obj.get('confidence', 0) >= min_confidence]
    
    def filter_objects_by_area(self, objects: List[Dict[str, Any]], 
                              min_area: int) -> List[Dict[str, Any]]:
        return [obj for obj in objects if obj.get('mask_area', 0) >= min_area]
    
    def filter_objects_by_class(self, objects: List[Dict[str, Any]], 
                               allowed_classes: List[str]) -> List[Dict[str, Any]]:
        return [obj for obj in objects if obj.get('class_name') in allowed_classes]
#!/usr/bin/env python3
"""
Duplicate Detection Filter

This module handles filtering of duplicate detections that might occur
even with agnostic NMS enabled. It uses IoU-based filtering and semantic
similarity to remove redundant detections.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from .decorators import track_performance
from . import config


class DuplicateFilter:
    """Filter for removing duplicate object detections."""
    
    def __init__(self):
        """Initialize the duplicate filter."""
        self.iou_threshold = config.get_float('duplicate_filter', 'iou_threshold', 0.7)
        self.confidence_weight = config.get_float('duplicate_filter', 'confidence_weight', 0.6)
        self.area_weight = config.get_float('duplicate_filter', 'area_weight', 0.4)
        
        logging.info("Duplicate filter initialized")
        logging.info(f"IoU threshold: {self.iou_threshold}")
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate union
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_mask_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate mask overlap ratio between two masks."""
        if mask1 is None or mask2 is None:
            return 0.0
        
        # Ensure masks are the same size
        if mask1.shape != mask2.shape:
            return 0.0
        
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _select_best_detection(self, duplicates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best detection from a group of duplicates."""
        if len(duplicates) == 1:
            return duplicates[0]
        
        best_detection = None
        best_score = -1
        
        for detection in duplicates:
            confidence = detection.get('confidence', 0.0)
            area = detection.get('area', 0.0)
            mask_area = detection.get('mask_area', 0.0)
            
            # Prefer detections with track IDs (more stable)
            track_bonus = 0.1 if detection.get('track_id') is not None else 0.0
            
            # Prefer larger mask areas (better segmentation)
            area_bonus = 0.05 if mask_area > area * 0.8 else 0.0
            
            # Combined score
            score = (confidence * self.confidence_weight + 
                    (area / 10000) * self.area_weight +  # Normalize area
                    track_bonus + area_bonus)
            
            if score > best_score:
                best_score = score
                best_detection = detection
        
        return best_detection
    
    def _find_duplicates(self, objects: List[Dict[str, Any]]) -> List[List[int]]:
        """Find groups of duplicate objects based on IoU and mask overlap."""
        duplicate_groups = []
        processed = set()
        
        for i, obj1 in enumerate(objects):
            if i in processed:
                continue
            
            current_group = [i]
            bbox1 = obj1.get('bbox', [])
            mask1 = obj1.get('mask')
            
            if len(bbox1) < 4:
                continue
            
            for j, obj2 in enumerate(objects[i+1:], i+1):
                if j in processed:
                    continue
                
                bbox2 = obj2.get('bbox', [])
                mask2 = obj2.get('mask')
                
                if len(bbox2) < 4:
                    continue
                
                # Calculate IoU
                iou = self._calculate_iou(bbox1, bbox2)
                
                # Calculate mask overlap if masks are available
                mask_overlap = 0.0
                if mask1 is not None and mask2 is not None:
                    mask_overlap = self._calculate_mask_overlap(mask1, mask2)
                
                # Consider as duplicate if high IoU or high mask overlap
                if iou > self.iou_threshold or mask_overlap > self.iou_threshold:
                    current_group.append(j)
                    processed.add(j)
            
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
                for idx in current_group:
                    processed.add(idx)
        
        return duplicate_groups
    
    @track_performance
    def filter_duplicates(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Filter duplicate detections from a list of objects.
        
        Args:
            objects: List of object dictionaries
            
        Returns:
            Dictionary with filtered objects and statistics
        """
        if not objects:
            return {
                'objects': [],
                'removed_duplicates': 0,
                'original_count': 0,
                'filtered_count': 0
            }
        
        original_count = len(objects)
        
        # Find duplicate groups
        duplicate_groups = self._find_duplicates(objects)
        
        # Create filtered list by keeping only the best from each group
        filtered_objects = []
        removed_indices = set()
        
        for group in duplicate_groups:
            # Get objects in this duplicate group
            group_objects = [objects[i] for i in group]
            
            # Select the best detection
            best_detection = self._select_best_detection(group_objects)
            
            # Find the index of the best detection
            best_idx = None
            for i, obj in enumerate(group_objects):
                if obj is best_detection:
                    best_idx = group[i]
                    break
            
            # Mark others for removal
            for idx in group:
                if idx != best_idx:
                    removed_indices.add(idx)
        
        # Add non-duplicate objects and best objects from duplicate groups
        for i, obj in enumerate(objects):
            if i not in removed_indices:
                filtered_objects.append(obj)
        
        removed_count = original_count - len(filtered_objects)
        
        if removed_count > 0:
            logging.info(f"Removed {removed_count} duplicate detections out of {original_count}")
        
        return {
            'objects': filtered_objects,
            'removed_duplicates': removed_count,
            'original_count': original_count,
            'filtered_count': len(filtered_objects),
            'duplicate_groups_found': len(duplicate_groups)
        }
    
    def filter_by_frame(self, frames_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply duplicate filtering to each frame separately.
        
        Args:
            frames_data: List of frame data dictionaries
            
        Returns:
            List of filtered frame data
        """
        filtered_frames = []
        total_removed = 0
        
        for frame_data in frames_data:
            objects = frame_data.get('objects', [])
            
            if objects:
                filter_result = self.filter_duplicates(objects)
                
                # Update frame data
                filtered_frame = frame_data.copy()
                filtered_frame['objects'] = filter_result['objects']
                filtered_frame['duplicate_filter_stats'] = {
                    'removed_duplicates': filter_result['removed_duplicates'],
                    'original_count': filter_result['original_count'],
                    'filtered_count': filter_result['filtered_count']
                }
                
                total_removed += filter_result['removed_duplicates']
                filtered_frames.append(filtered_frame)
            else:
                filtered_frames.append(frame_data)
        
        if total_removed > 0:
            logging.info(f"Total duplicates removed across all frames: {total_removed}")
        
        return filtered_frames
    
    def update_thresholds(self, iou_threshold: float = None, 
                         confidence_weight: float = None,
                         area_weight: float = None):
        """
        Update filtering thresholds.
        
        Args:
            iou_threshold: New IoU threshold for duplicate detection
            confidence_weight: New weight for confidence in selection
            area_weight: New weight for area in selection
        """
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
            logging.info(f"Updated IoU threshold to {iou_threshold}")
        
        if confidence_weight is not None:
            self.confidence_weight = confidence_weight
            logging.info(f"Updated confidence weight to {confidence_weight}")
        
        if area_weight is not None:
            self.area_weight = area_weight
            logging.info(f"Updated area weight to {area_weight}")

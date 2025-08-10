"""
Advanced tracking improvements for better ID consistency and reduced fragmentation.
"""
import numpy as np
import cv2
from typing import List, Dict, Any, Set
from collections import defaultdict

class AdvancedTrackProcessor:
    """Advanced processing for track ID consistency and deduplication."""
    
    def __init__(self, 
                 iou_threshold: float = 0.3,
                 confidence_boost: float = 0.1,
                 temporal_window: int = 10):
        self.iou_threshold = iou_threshold
        self.confidence_boost = confidence_boost
        self.temporal_window = temporal_window
        
    def apply_advanced_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply advanced non-maximum suppression to reduce overlapping detections."""
        if len(detections) <= 1:
            return detections
        
        print(f"     -> Applying advanced NMS to {len(detections)} detections...")
        
        # Group by frame for frame-wise NMS
        frame_groups = defaultdict(list)
        for det in detections:
            frame_id = det.get('frame_id', 0)
            frame_groups[frame_id].append(det)
        
        processed_detections = []
        
        for frame_id, frame_detections in frame_groups.items():
            if len(frame_detections) <= 1:
                processed_detections.extend(frame_detections)
                continue
                
            # Apply class-aware NMS
            class_groups = defaultdict(list)
            for det in frame_detections:
                class_name = det.get('class_name', 'unknown')
                class_groups[class_name].append(det)
            
            for class_name, class_dets in class_groups.items():
                nms_result = self._apply_class_nms(class_dets)
                processed_detections.extend(nms_result)
        
        print(f"     -> NMS result: {len(processed_detections)} detections")
        return processed_detections
    
    def _apply_class_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply NMS within a single class."""
        if len(detections) <= 1:
            return detections
        
        # Convert to format for NMS
        boxes = []
        confidences = []
        
        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            conf = det.get('confidence', 0.0)
            boxes.append(bbox)
            confidences.append(conf)
        
        # Apply OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, 
            score_threshold=0.1, 
            nms_threshold=self.iou_threshold
        )
        
        if len(indices) > 0:
            return [detections[i] for i in indices.flatten()]
        else:
            # If NMS removes everything, keep the highest confidence detection
            best_idx = np.argmax(confidences)
            return [detections[best_idx]]
    
    def enforce_track_consistency(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enforce consistency in track IDs across temporal sequences."""
        if len(detections) <= 1:
            return detections
            
        print(f"     -> Enforcing track consistency on {len(detections)} detections...")
        
        # Sort by frame and track ID
        detections.sort(key=lambda x: (x.get('frame_id', 0), x.get('track_id', 0)))
        
        # Track ID mapping for consistency
        id_mapping = {}
        next_consistent_id = 1
        
        # Process each detection
        consistent_detections = []
        for det in detections:
            original_id = det.get('track_id', 0)
            
            if original_id not in id_mapping:
                id_mapping[original_id] = next_consistent_id
                next_consistent_id += 1
            
            # Update detection with consistent ID
            consistent_det = det.copy()
            consistent_det['track_id'] = id_mapping[original_id]
            consistent_det['original_track_id'] = original_id
            consistent_detections.append(consistent_det)
        
        print(f"     -> ID mapping: {len(id_mapping)} unique tracks")
        return consistent_detections
    
    def merge_temporal_fragments(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge temporal fragments of the same track."""
        if len(detections) <= 1:
            return detections
            
        print(f"     -> Merging temporal fragments...")
        
        # Group by track ID and class
        track_groups = defaultdict(list)
        for det in detections:
            track_id = det.get('track_id', 0)
            class_name = det.get('class_name', 'unknown')
            key = f"{track_id}_{class_name}"
            track_groups[key].append(det)
        
        merged_detections = []
        
        for group_key, group_detections in track_groups.items():
            if len(group_detections) == 1:
                merged_detections.extend(group_detections)
                continue
            
            # Sort by frame
            group_detections.sort(key=lambda x: x.get('frame_id', 0))
            
            # Check for temporal continuity
            frame_ids = [det.get('frame_id', 0) for det in group_detections]
            gaps = [frame_ids[i+1] - frame_ids[i] for i in range(len(frame_ids)-1)]
            
            if max(gaps) <= self.temporal_window:
                # Merge into single track
                merged_track = self._merge_track_sequence(group_detections)
                merged_detections.append(merged_track)
            else:
                # Keep as separate fragments
                merged_detections.extend(group_detections)
        
        print(f"     -> Merged to {len(merged_detections)} tracks")
        return merged_detections
    
    def _merge_track_sequence(self, track_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a sequence of detections into a single track."""
        # Use the detection with highest confidence as base
        base_detection = max(track_detections, key=lambda x: x.get('confidence', 0))
        
        # Collect all frame information
        frames = []
        bboxes = []
        confidences = []
        
        for det in track_detections:
            frame_id = det.get('frame_id', 0)
            bbox = det.get('bbox', [0, 0, 0, 0])
            conf = det.get('confidence', 0)
            
            frames.append(frame_id)
            bboxes.append(bbox)
            confidences.append(conf)
        
        # Create merged track
        merged = base_detection.copy()
        merged.update({
            'frames': sorted(frames),
            'bboxes': bboxes,
            'confidence': np.mean(confidences),
            'max_confidence': max(confidences),
            'frame_count': len(frames),
            'temporal_span': max(frames) - min(frames) + 1,
            'merged_from': len(track_detections)
        })
        
        return merged
    
    def filter_low_quality_tracks(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out low-quality tracks based on various criteria."""
        if not detections:
            return detections
            
        print(f"     -> Filtering low-quality tracks from {len(detections)} detections...")
        
        filtered = []
        
        for det in detections:
            # Quality criteria
            confidence = det.get('confidence', 0)
            frame_count = len(det.get('frames', [det.get('frame_id', 0)]))
            
            # Check bbox validity
            bbox = det.get('bbox', [0, 0, 0, 0])
            bbox_normalized = det.get('bbox_normalized', bbox)
            
            # Calculate area (handle both normalized and absolute coordinates)
            if max(bbox_normalized) <= 1.0:  # Normalized coordinates
                bbox_area = (bbox_normalized[2] - bbox_normalized[0]) * (bbox_normalized[3] - bbox_normalized[1])
                min_bbox_area = 0.001  # Minimum 0.1% of image area
            else:  # Absolute coordinates
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                min_bbox_area = 100  # Minimum 10x10 pixels
            
            # Quality thresholds
            min_confidence = 0.2
            min_frame_count = 1
            
            if (confidence >= min_confidence and 
                frame_count >= min_frame_count and 
                bbox_area >= min_bbox_area):
                filtered.append(det)
            else:
                print(f"       -> Filtered track {det.get('track_id', '?')}: conf={confidence:.2f}, frames={frame_count}, area={bbox_area}")
        
        print(f"     -> Kept {len(filtered)} high-quality tracks")
        return filtered

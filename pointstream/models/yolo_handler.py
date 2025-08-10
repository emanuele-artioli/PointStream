"""
Model handler for YOLO object detection and tracking with adaptive enhancement.
"""
from typing import List, Dict, Any
import numpy as np
from ultralytics import YOLO
from .. import config
from ..utils.adaptive_metrics import (
    calculate_normalized_tracking_metrics, 
    get_metrics_cache
)
from ..utils.enhanced_processing import TrackConsolidator
from ..utils.advanced_tracking import AdvancedTrackProcessor

class AdaptiveTracker:
    """Enhanced YOLO tracker with adaptive threshold learning and monitoring."""

    def __init__(self, model_path: str):
        """Initialize the adaptive tracker."""
        print(f" -> Initializing adaptive YOLO tracker with model: {model_path}")
        self.model_path = model_path
        self.yolo_model = YOLO(model_path)
        self.metrics_cache = get_metrics_cache()
        self.track_consolidator = TrackConsolidator()
        self.advanced_processor = AdvancedTrackProcessor()
        
        # Enhanced YOLO configuration with ByteTrack
        self.base_yolo_config = {
            'conf': 0.2,        # Lower confidence to catch more detections
            'iou': 0.4,         # Lower IoU to reduce duplicate suppression
            'tracker': 'bytetrack.yaml',  # Use ByteTrack for better tracking
            'persist': True,    # Maintain track IDs across frames
            'verbose': False,
            'save': False,
            'max_det': 30,      # Reasonable max detections
            'agnostic_nms': False,  # Class-aware NMS
            'retina_masks': False,  # Disable masks for speed
            'half': False       # Use full precision for stability
        }

    def track_objects(self, frames: List[np.ndarray], content_type: str = "general") -> List[List[Dict[str, Any]]]:
        """Track objects with adaptive threshold learning."""
        print(f"  -> Tracking {len(frames)} frames with adaptive YOLO...")
        
        # Step 1: Get optimal thresholds for this content type
        optimal_thresholds = self.metrics_cache.get_optimal_thresholds(content_type)
        
        # Step 2: Configure YOLO with learned thresholds
        adaptive_config = self.base_yolo_config.copy()
        if optimal_thresholds:
            adaptive_config.update({
                'conf': optimal_thresholds.get('confidence', 0.5),
                'iou': optimal_thresholds.get('iou', 0.7)
            })
            print(f"     -> Using learned thresholds: conf={adaptive_config['conf']:.2f}, iou={adaptive_config['iou']:.2f}")
        else:
            print(f"     -> Using default thresholds: conf={adaptive_config['conf']:.2f}, iou={adaptive_config['iou']:.2f}")
        
        # Step 3: Track with YOLO
        yolo_results = self._track_with_yolo(frames, adaptive_config)
        
        # Step 3.5: Consolidate fragmented tracks
        flattened_detections = []
        for frame_idx, frame_detections in enumerate(yolo_results):
            for detection in frame_detections:
                detection['frame_id'] = frame_idx
                flattened_detections.append(detection)
        
        print(f"     -> Pre-consolidation: {len(flattened_detections)} detections")
        
        # Apply advanced processing pipeline
        flattened_detections = self.advanced_processor.apply_advanced_nms(flattened_detections)
        flattened_detections = self.advanced_processor.enforce_track_consistency(flattened_detections)
        consolidated_detections = self.track_consolidator.consolidate_tracks(flattened_detections)
        consolidated_detections = self.advanced_processor.merge_temporal_fragments(consolidated_detections)
        consolidated_detections = self.advanced_processor.filter_low_quality_tracks(consolidated_detections)
        
        print(f"     -> Final result: {len(consolidated_detections)} high-quality tracks")
        
        # Convert back to frame-based format
        frame_based_results = [[] for _ in range(len(frames))]
        for detection in consolidated_detections:
            for frame_id in detection.get('frames', [detection.get('frame_id', 0)]):
                if frame_id < len(frame_based_results):
                    frame_based_results[frame_id].append(detection)
        
        # Step 4: Evaluate tracking quality
        tracking_metrics = calculate_normalized_tracking_metrics(frame_based_results)
        print(f"     -> Tracking quality metrics: {tracking_metrics}")
        
        # Step 5: Calculate overall success score
        overall_quality = np.mean(list(tracking_metrics.values()))
        print(f"     -> Overall tracking quality: {overall_quality:.3f}")
        
        # Step 6: Log results for threshold learning
        self.metrics_cache.log_tracking_result(
            content_type, 
            tracking_metrics, 
            "yolo", 
            overall_quality,
            adaptive_config
        )
        
        # Step 7: Optional future enhancement placeholder
        if config.ENABLE_DEEPSORT_FALLBACK and overall_quality < 0.3:
            print(f"     -> Note: Quality very low ({overall_quality:.3f}), consider DeepSORT for future enhancement")
        
        # Clear CUDA cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return yolo_results

    def _track_with_yolo(self, frames: List[np.ndarray], yolo_config: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Track using YOLO with specified configuration."""
        results_generator = self.yolo_model.track(
            source=frames,
            **yolo_config
        )

        all_frames_detections = []
        for frame_results in results_generator:
            current_frame_detections = []
            if frame_results.boxes.id is not None:
                for i in range(len(frame_results.boxes.id)):
                    detection = {
                        "track_id": int(frame_results.boxes.id[i]),
                        "class_id": int(frame_results.boxes.cls[i]),
                        "class_name": self.yolo_model.names[int(frame_results.boxes.cls[i])],
                        "confidence": float(frame_results.boxes.conf[i]),
                        "bbox_normalized": frame_results.boxes.xyxyn[i].cpu().numpy().tolist()
                    }
                    current_frame_detections.append(detection)
            
            all_frames_detections.append(current_frame_detections)
        
        return all_frames_detections

    def get_tracking_statistics(self, content_type: str = "general") -> Dict[str, Any]:
        """Get tracking performance statistics for this content type."""
        return self.metrics_cache.get_content_statistics(content_type)


# Maintain backward compatibility
class YOLOHandler(AdaptiveTracker):
    """Backward compatibility wrapper."""
    
    def track_objects(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Track objects (backward compatible interface)."""
        return super().track_objects(frames, content_type="general")
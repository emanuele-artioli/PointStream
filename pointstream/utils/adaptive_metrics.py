"""
Adaptive tracking and inpainting management with normalized metrics.
"""
import numpy as np
import cv2
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import logging

class MetricsCache:
    """Cache and learn optimal thresholds for different content types."""
    
    def __init__(self, cache_file: str = "pointstream_metrics_cache.json"):
        self.cache_file = Path(cache_file)
        self.metrics_history = defaultdict(list)
        self.optimal_thresholds = defaultdict(lambda: 0.7)  # Default threshold
        self.load_cache()
    
    def load_cache(self):
        """Load cached metrics and thresholds."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.metrics_history = defaultdict(list, data.get('history', {}))
                    self.optimal_thresholds = defaultdict(lambda: 0.7, data.get('thresholds', {}))
        except Exception as e:
            logging.warning(f"Could not load metrics cache: {e}")
    
    def save_cache(self):
        """Save metrics and thresholds to disk."""
        try:
            data = {
                'history': dict(self.metrics_history),
                'thresholds': dict(self.optimal_thresholds),
                'last_updated': time.time()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save metrics cache: {e}")
    
    def log_tracking_result(self, content_type: str, metrics: Dict[str, float], 
                           method_used: str, success_score: float, config_used: Dict[str, Any] = None):
        """Log tracking results for learning optimal thresholds."""
        entry = {
            'metrics': metrics,
            'method': method_used,
            'success_score': success_score,
            'timestamp': time.time(),
            'config': config_used or {}
        }
        
        self.metrics_history[f"{content_type}_tracking"].append(entry)
        
        # Keep only recent entries (last 100 per content type)
        if len(self.metrics_history[f"{content_type}_tracking"]) > 100:
            self.metrics_history[f"{content_type}_tracking"] = \
                self.metrics_history[f"{content_type}_tracking"][-100:]
        
        # Update optimal thresholds based on recent performance
        self._update_optimal_threshold(content_type, "tracking")
        self.save_cache()
    
    def get_optimal_thresholds(self, content_type: str) -> Optional[Dict[str, float]]:
        """Get optimal configuration thresholds for content type."""
        history_key = f"{content_type}_tracking"
        if history_key not in self.metrics_history or len(self.metrics_history[history_key]) < 5:
            return None
        
        # Analyze recent successful runs to find optimal thresholds
        recent_entries = self.metrics_history[history_key][-20:]  # Last 20 runs
        successful_entries = [e for e in recent_entries if e['success_score'] > 0.6]
        
        if not successful_entries:
            return None
        
        # Find average thresholds from successful runs
        conf_values = [e['config'].get('conf', 0.5) for e in successful_entries if 'config' in e]
        iou_values = [e['config'].get('iou', 0.7) for e in successful_entries if 'config' in e]
        
        if not conf_values or not iou_values:
            return None
        
        return {
            'confidence': np.mean(conf_values),
            'iou': np.mean(iou_values)
        }
    
    def get_content_statistics(self, content_type: str) -> Dict[str, Any]:
        """Get performance statistics for a content type."""
        history_key = f"{content_type}_tracking"
        if history_key not in self.metrics_history:
            return {"runs": 0, "avg_quality": 0.0}
        
        entries = self.metrics_history[history_key]
        if not entries:
            return {"runs": 0, "avg_quality": 0.0}
        
        recent_entries = entries[-10:]  # Last 10 runs
        scores = [e['success_score'] for e in recent_entries]
        
        return {
            "runs": len(entries),
            "recent_runs": len(recent_entries),
            "avg_quality": np.mean(scores),
            "best_quality": max(scores),
            "worst_quality": min(scores),
            "optimal_thresholds": self.get_optimal_thresholds(content_type)
        }
    
    def log_inpainting_result(self, content_type: str, mask_complexity: float,
                            method_used: str, success: bool):
        """Log inpainting results for learning."""
        entry = {
            'mask_complexity': mask_complexity,
            'method': method_used,
            'success': success,
            'timestamp': time.time()
        }
        self.metrics_history[f"{content_type}_inpainting"].append(entry)
        self._update_optimal_threshold(f"{content_type}_inpainting")
    
    def _update_optimal_threshold(self, content_type: str, task: str):
        """Update optimal thresholds based on recent performance."""
        history_key = f"{content_type}_{task}"
        if history_key not in self.metrics_history or len(self.metrics_history[history_key]) < 5:
            return
        
        recent_entries = self.metrics_history[history_key][-20:]  # Use recent 20 entries
        
        # For tracking, find the configuration that worked best
        if task == "tracking":
            best_configs = [e for e in recent_entries if e.get('success_score', 0) > 0.6]
            if best_configs:
                # Average the good configurations
                avg_conf = np.mean([e.get('config', {}).get('conf', 0.5) for e in best_configs])
                avg_iou = np.mean([e.get('config', {}).get('iou', 0.7) for e in best_configs])
                
                # Store as threshold info (though we use get_optimal_thresholds for actual values)
                self.optimal_thresholds[history_key] = {
                    'confidence': avg_conf,
                    'iou': avg_iou,
                    'last_updated': time.time()
                }
        
        self.save_cache()
    
    def _calculate_success_rate(self, entries: List[Dict], threshold: float) -> float:
        """Calculate success rate for a given threshold."""
        if not entries:
            return 0.0
        
        # For tracking entries, check if success_score > threshold
        successes = 0
        for entry in entries:
            if 'success_score' in entry and entry['success_score'] > threshold:
                successes += 1
            elif 'success' in entry and entry['success']:
                successes += 1
        
        return successes / len(entries)


def calculate_normalized_tracking_metrics(frame_detections: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    """Calculate normalized tracking quality metrics (0-1 scale, 1.0 = perfect)."""
    
    if not frame_detections or len(frame_detections) < 2:
        return {'continuity': 0.0, 'confidence': 0.0, 'spatial': 0.0, 'length': 0.0}
    
    # Build track history
    tracks = defaultdict(list)
    for frame_idx, detections in enumerate(frame_detections):
        for det in detections:
            track_id = det.get('track_id')
            if track_id is not None:
                tracks[track_id].append((frame_idx, det))
    
    if not tracks:
        return {'continuity': 0.0, 'confidence': 0.0, 'spatial': 0.0, 'length': 0.0}
    
    # 1. Track Continuity (1.0 = no gaps in tracks)
    total_gaps = 0
    for track_data in tracks.values():
        frames = [frame_idx for frame_idx, _ in track_data]
        expected_frames = max(frames) - min(frames) + 1
        actual_frames = len(frames)
        total_gaps += max(0, expected_frames - actual_frames)
    
    max_possible_gaps = len(frame_detections) * len(tracks)
    continuity = 1.0 - (total_gaps / max_possible_gaps) if max_possible_gaps > 0 else 1.0
    
    # 2. Confidence Stability (1.0 = stable confidence)
    all_confidences = []
    for track_data in tracks.values():
        confidences = [det['confidence'] for _, det in track_data]
        if len(confidences) > 1:
            all_confidences.append(np.std(confidences))
    
    avg_confidence_std = np.mean(all_confidences) if all_confidences else 0.0
    confidence_stability = max(0.0, 1.0 - (avg_confidence_std / 0.3))  # Normalize by expected max std
    
    # 3. Spatial Consistency (1.0 = smooth movement)
    spatial_consistency = 0.0
    consistent_tracks = 0
    
    for track_data in tracks.values():
        if len(track_data) < 3:
            continue
        
        # Calculate movement smoothness
        positions = []
        for _, det in sorted(track_data, key=lambda x: x[0]):
            bbox = det['bbox_normalized']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            positions.append([center_x, center_y])
        
        if len(positions) >= 3:
            # Calculate acceleration variance (smooth movement has low acceleration variance)
            velocities = np.diff(positions, axis=0)
            accelerations = np.diff(velocities, axis=0)
            acceleration_magnitude = np.linalg.norm(accelerations, axis=1)
            acceleration_variance = np.var(acceleration_magnitude) if len(acceleration_magnitude) > 0 else 0
            
            # Normalize (assuming max reasonable variance is 0.01)
            track_smoothness = max(0.0, 1.0 - (acceleration_variance / 0.01))
            spatial_consistency += track_smoothness
            consistent_tracks += 1
    
    spatial_consistency = spatial_consistency / consistent_tracks if consistent_tracks > 0 else 1.0
    
    # 4. Track Length Quality (1.0 = optimal length tracks)
    track_lengths = [len(track_data) for track_data in tracks.values()]
    avg_track_length = np.mean(track_lengths) if track_lengths else 0
    ideal_length = min(30, len(frame_detections))  # Ideal is 30 frames or full scene
    length_quality = min(1.0, avg_track_length / ideal_length) if ideal_length > 0 else 0.0
    
    return {
        'continuity': max(0.0, min(1.0, continuity)),
        'confidence': max(0.0, min(1.0, confidence_stability)),
        'spatial': max(0.0, min(1.0, spatial_consistency)),
        'length': max(0.0, min(1.0, length_quality))
    }


def calculate_mask_complexity_score(mask: np.ndarray, size_weight: float = 0.6, 
                                  complexity_weight: float = 0.4) -> float:
    """Calculate normalized mask complexity score (0-1)."""
    if mask is None or mask.size == 0:
        return 0.0
    
    # Size component (0-1): percentage of frame covered
    size_score = np.sum(mask > 0) / mask.size
    
    # Complexity component (0-1): edge density normalized
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask
    
    edges = cv2.Canny(mask_gray.astype(np.uint8), 50, 150)
    complexity_score = np.sum(edges > 0) / mask.size
    
    # Combined score with weights
    combined_score = (size_score * size_weight + complexity_score * complexity_weight)
    return max(0.0, min(1.0, combined_score))


def should_use_advanced_method(metrics: Dict[str, float], threshold: float) -> bool:
    """Decide whether to use advanced method based on balanced approach."""
    if not metrics:
        return False
    
    # Balanced approach: use average of metrics
    avg_metric = np.mean(list(metrics.values()))
    
    # Use advanced method if quality is below threshold
    return avg_metric < threshold


# Global cache instance
_metrics_cache = None

def get_metrics_cache() -> MetricsCache:
    """Get global metrics cache instance."""
    global _metrics_cache
    if _metrics_cache is None:
        _metrics_cache = MetricsCache()
    return _metrics_cache

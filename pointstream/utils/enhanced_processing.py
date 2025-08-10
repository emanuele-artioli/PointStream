"""
Enhanced tracking consolidation and improved background inpainting.
"""
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import logging

class TrackConsolidator:
    """Consolidate fragmented track IDs into coherent tracks."""
    
    def __init__(self, 
                 spatial_threshold: float = 50.0,
                 temporal_threshold: int = 5,
                 appearance_threshold: float = 0.3):
        self.spatial_threshold = spatial_threshold
        self.temporal_threshold = temporal_threshold
        self.appearance_threshold = appearance_threshold
        
    def consolidate_tracks(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate fragmented tracks into coherent sequences."""
        print(f"  -> Consolidating {len(detections)} detections...")
        
        if len(detections) < 2:
            return detections
            
        # Group detections by class
        class_groups = defaultdict(list)
        for det in detections:
            class_name = det.get('class_name', 'unknown')
            class_groups[class_name].append(det)
        
        consolidated = []
        for class_name, class_detections in class_groups.items():
            if len(class_detections) <= 1:
                consolidated.extend(class_detections)
                continue
                
            print(f"     -> Consolidating {len(class_detections)} {class_name} detections")
            
            # Sort by frame
            class_detections.sort(key=lambda x: x.get('frame_id', 0))
            
            # Build spatial-temporal similarity matrix
            similarity_matrix = self._compute_similarity_matrix(class_detections)
            
            # Cluster similar detections
            clusters = self._cluster_detections(similarity_matrix)
            
            # Merge detections within each cluster
            consolidated.extend(self._merge_clusters(class_detections, clusters))
        
        print(f"     -> Consolidated to {len(consolidated)} tracks")
        return consolidated
    
    def _compute_similarity_matrix(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Compute similarity matrix between detections."""
        n = len(detections)
        similarity = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                sim = self._compute_detection_similarity(detections[i], detections[j])
                similarity[i, j] = similarity[j, i] = sim
        
        return similarity
    
    def _compute_detection_similarity(self, det1: Dict[str, Any], det2: Dict[str, Any]) -> float:
        """Compute similarity between two detections."""
        # Temporal distance
        frame1 = det1.get('frame_id', 0)
        frame2 = det2.get('frame_id', 0)
        temporal_dist = abs(frame1 - frame2)
        
        if temporal_dist > self.temporal_threshold:
            return 0.0
        
        # Spatial distance
        bbox1 = det1.get('bbox', [0, 0, 0, 0])
        bbox2 = det2.get('bbox', [0, 0, 0, 0])
        
        center1 = np.array([(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2])
        center2 = np.array([(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2])
        
        spatial_dist = np.linalg.norm(center1 - center2)
        
        if spatial_dist > self.spatial_threshold:
            return 0.0
        
        # Size similarity
        size1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        size2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0
        
        # Confidence similarity
        conf1 = det1.get('confidence', 0)
        conf2 = det2.get('confidence', 0)
        conf_sim = 1.0 - abs(conf1 - conf2)
        
        # Combined similarity
        temporal_sim = max(0, 1.0 - temporal_dist / self.temporal_threshold)
        spatial_sim = max(0, 1.0 - spatial_dist / self.spatial_threshold)
        
        return (temporal_sim * 0.4 + spatial_sim * 0.4 + size_ratio * 0.1 + conf_sim * 0.1)
    
    def _cluster_detections(self, similarity_matrix: np.ndarray, min_similarity: float = 0.6) -> List[int]:
        """Cluster detections using simple thresholding approach."""
        n = len(similarity_matrix)
        clusters = list(range(n))  # Initially each detection is its own cluster
        
        # Simple agglomerative clustering
        for i in range(n):
            for j in range(i+1, n):
                if similarity_matrix[i, j] > min_similarity:
                    # Merge clusters
                    old_cluster = clusters[j]
                    new_cluster = clusters[i]
                    for k in range(n):
                        if clusters[k] == old_cluster:
                            clusters[k] = new_cluster
        
        # Renumber clusters to be consecutive
        unique_clusters = list(set(clusters))
        cluster_map = {old: new for new, old in enumerate(unique_clusters)}
        clusters = [cluster_map[c] for c in clusters]
        
        return clusters
    
    def _merge_clusters(self, detections: List[Dict[str, Any]], clusters: List[int]) -> List[Dict[str, Any]]:
        """Merge detections within each cluster."""
        cluster_groups = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            cluster_groups[cluster_id].append(detections[i])
        
        merged_tracks = []
        for cluster_id, cluster_detections in cluster_groups.items():
            if len(cluster_detections) == 1:
                merged_tracks.append(cluster_detections[0])
            else:
                merged_track = self._merge_detection_group(cluster_detections)
                merged_tracks.append(merged_track)
        
        return merged_tracks
    
    def _merge_detection_group(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a group of detections into a single track."""
        # Use the detection with highest confidence as base
        base_detection = max(detections, key=lambda x: x.get('confidence', 0))
        
        # Merge frame information
        frames = []
        bboxes = []
        confidences = []
        
        for det in detections:
            frames.extend(det.get('frames', [det.get('frame_id', 0)]))
            if 'bboxes' in det:
                bboxes.extend(det['bboxes'])
            else:
                bboxes.append(det.get('bbox', [0, 0, 0, 0]))
            confidences.append(det.get('confidence', 0))
        
        merged = base_detection.copy()
        merged.update({
            'frames': sorted(list(set(frames))),
            'bboxes': bboxes,
            'confidence': np.mean(confidences),
            'consolidated_from': len(detections)
        })
        
        return merged


class EnhancedBackgroundInpainter:
    """Enhanced background inpainting with better object removal."""
    
    def __init__(self):
        self.motion_accumulator = None
        self.frame_buffer = []
        self.max_buffer_size = 20
        
    def create_enhanced_background(self, 
                                 frames: List[np.ndarray], 
                                 detections: List[Dict[str, Any]], 
                                 scene_info: Dict[str, Any]) -> np.ndarray:
        """Create enhanced background with better object removal."""
        print(f"  -> Creating enhanced background from {len(frames)} frames...")
        
        if len(frames) == 0:
            raise ValueError("No frames provided for background creation")
        
        # Method 1: Motion-based background subtraction
        bg_motion = self._motion_based_background(frames)
        
        # Method 2: Median-based background
        bg_median = self._median_background(frames)
        
        # Method 3: Detection-aware background
        bg_detection_aware = self._detection_aware_background(frames, detections)
        
        # Method 4: Temporal consensus background
        bg_consensus = self._temporal_consensus_background(frames)
        
        # Combine methods intelligently
        final_background = self._combine_backgrounds([
            bg_motion, bg_median, bg_detection_aware, bg_consensus
        ], frames, detections)
        
        # Post-process to remove artifacts
        final_background = self._post_process_background(final_background, frames)
        
        print(f"     -> Enhanced background created: {final_background.shape}")
        return final_background
    
    def _motion_based_background(self, frames: List[np.ndarray]) -> np.ndarray:
        """Create background using motion analysis."""
        if len(frames) < 3:
            return frames[0] if frames else np.zeros((360, 640, 3), dtype=np.uint8)
        
        # Convert to grayscale for motion analysis
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        
        # Compute optical flow to identify static regions
        static_mask = np.ones(gray_frames[0].shape, dtype=np.uint8) * 255
        
        for i in range(len(gray_frames) - 1):
            flow = cv2.calcOpticalFlowPyrLK(
                gray_frames[i], gray_frames[i+1], 
                np.array([[x, y] for y in range(0, gray_frames[i].shape[0], 20) 
                         for x in range(0, gray_frames[i].shape[1], 20)], dtype=np.float32),
                None
            )[0]
            
            if flow is not None:
                # Mark regions with significant motion
                for point in flow:
                    if point is not None:
                        x, y = int(point[0]), int(point[1])
                        if 0 <= x < static_mask.shape[1] and 0 <= y < static_mask.shape[0]:
                            cv2.circle(static_mask, (x, y), 15, 0, -1)
        
        # Use static regions to build background
        background = np.zeros_like(frames[0])
        mask_3ch = cv2.cvtColor(static_mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        for frame in frames:
            background = background * (1 - mask_3ch) + frame * mask_3ch
        
        return background.astype(np.uint8)
    
    def _median_background(self, frames: List[np.ndarray]) -> np.ndarray:
        """Create background using median filtering."""
        if not frames:
            return np.zeros((360, 640, 3), dtype=np.uint8)
        
        # Sample frames to avoid memory issues
        sample_frames = frames[::max(1, len(frames)//10)][:10]
        frame_stack = np.stack(sample_frames, axis=0)
        return np.median(frame_stack, axis=0).astype(np.uint8)
    
    def _detection_aware_background(self, 
                                  frames: List[np.ndarray], 
                                  detections: List[Dict[str, Any]]) -> np.ndarray:
        """Create background by explicitly removing detected objects."""
        if not frames:
            return np.zeros((360, 640, 3), dtype=np.uint8)
        
        background = frames[0].copy()
        
        # Create mask for all detections
        detection_mask = np.zeros(background.shape[:2], dtype=np.uint8)
        
        for detection in detections:
            for frame_idx in detection.get('frames', []):
                if frame_idx < len(frames):
                    bbox = detection.get('bbox', [0, 0, 0, 0])
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        x1, y1, x2, y2 = map(int, bbox)
                        detection_mask[y1:y2, x1:x2] = 255
        
        # Dilate mask to ensure complete object removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        detection_mask = cv2.dilate(detection_mask, kernel, iterations=2)
        
        # Inpaint the masked regions
        background = cv2.inpaint(background, detection_mask, 10, cv2.INPAINT_TELEA)
        
        return background
    
    def _temporal_consensus_background(self, frames: List[np.ndarray]) -> np.ndarray:
        """Create background using temporal consensus."""
        if not frames:
            return np.zeros((360, 640, 3), dtype=np.uint8)
        
        # Convert frames to LAB color space for better averaging
        lab_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB) for f in frames]
        
        # Compute pixel-wise temporal statistics
        frame_stack = np.stack(lab_frames, axis=0)
        
        # Use mode (most frequent value) for each pixel
        background_lab = np.zeros_like(lab_frames[0])
        
        for i in range(background_lab.shape[0]):
            for j in range(background_lab.shape[1]):
                for c in range(3):
                    pixel_values = frame_stack[:, i, j, c]
                    # Use histogram to find most frequent value
                    hist, bins = np.histogram(pixel_values, bins=20)
                    mode_idx = np.argmax(hist)
                    background_lab[i, j, c] = bins[mode_idx]
        
        # Convert back to BGR
        background = cv2.cvtColor(background_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return background
    
    def _combine_backgrounds(self, 
                           backgrounds: List[np.ndarray], 
                           frames: List[np.ndarray],
                           detections: List[Dict[str, Any]]) -> np.ndarray:
        """Intelligently combine multiple background estimates."""
        if not backgrounds or not backgrounds[0].size:
            return frames[0] if frames else np.zeros((360, 640, 3), dtype=np.uint8)
        
        # Weight backgrounds based on quality metrics
        weights = []
        
        for bg in backgrounds:
            # Compute quality score based on variance and edge content
            gray_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            
            # Lower variance in static regions is better
            variance_score = 1.0 / (1.0 + np.var(gray_bg))
            
            # Reasonable edge content indicates good background
            edges = cv2.Canny(gray_bg, 50, 150)
            edge_score = np.sum(edges) / (gray_bg.shape[0] * gray_bg.shape[1])
            edge_score = min(edge_score, 0.1) / 0.1  # Normalize
            
            quality = variance_score * 0.7 + edge_score * 0.3
            weights.append(quality)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted combination
        combined = np.zeros_like(backgrounds[0], dtype=np.float32)
        for bg, weight in zip(backgrounds, weights):
            combined += bg.astype(np.float32) * weight
        
        return combined.astype(np.uint8)
    
    def _post_process_background(self, 
                               background: np.ndarray, 
                               frames: List[np.ndarray]) -> np.ndarray:
        """Post-process background to remove artifacts."""
        # Bilateral filter to smooth while preserving edges
        smoothed = cv2.bilateralFilter(background, 9, 75, 75)
        
        # Enhance contrast slightly
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced

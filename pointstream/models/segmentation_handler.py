"""
Segmentation utilities for better background inpainting.
Uses Segment Anything Model (SAM) or fallback methods to create precise masks.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import torch

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("[INFO] Segment Anything Model not available. Using bbox-based masks.")


class SegmentationHandler:
    """Handles object segmentation for precise inpainting masks."""
    
    def __init__(self):
        self.sam_predictor = None
        if SAM_AVAILABLE:
            try:
                # Try to load SAM model - this would need to be downloaded separately
                # For now, we'll use a fallback approach
                self.sam_predictor = None
                print("[INFO] SAM model not loaded. Using improved bbox segmentation.")
            except Exception as e:
                print(f"[INFO] Could not load SAM model: {e}. Using bbox fallback.")
                self.sam_predictor = None
    
    def create_precise_masks(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create precise segmentation masks for detected objects.
        
        Args:
            frame: Input frame
            detections: List of object detections with bboxes
            
        Returns:
            Binary mask with all objects marked for inpainting
        """
        if self.sam_predictor is not None:
            return self._create_sam_masks(frame, detections)
        else:
            return self._create_improved_bbox_masks(frame, detections)
    
    def _create_sam_masks(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Create masks using Segment Anything Model."""
        # This would use SAM to create precise segmentation masks
        # For now, fallback to improved bbox method
        return self._create_improved_bbox_masks(frame, detections)
    
    def _create_improved_bbox_masks(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create improved masks using morphological operations and edge detection.
        This creates more organic shapes instead of rectangles.
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for detection in detections:
            bbox = detection['bbox_normalized']
            confidence = detection.get('confidence', 1.0)
            
            # Convert normalized bbox to pixel coordinates
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            
            # Extract object region
            obj_region = frame[y1:y2, x1:x2]
            if obj_region.size == 0:
                continue
            
            # Create a more organic mask using edge detection and morphology
            region_mask = self._create_organic_mask(obj_region, confidence)
            
            # Place the organic mask back into the full frame mask
            mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], region_mask)
        
        # Apply morphological operations to smooth the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _create_organic_mask(self, region: np.ndarray, confidence: float) -> np.ndarray:
        """
        Create an organic-shaped mask for an object region using edge detection.
        """
        if region.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        
        h, w = region.shape[:2]
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Create initial mask based on intensity and edges
        # Objects typically have different intensity than background
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine edge information with threshold
        mask = cv2.bitwise_or(thresh, edges)
        
        # Fill holes and smooth
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # If confidence is low, make the mask more conservative (smaller)
        if confidence < 0.7:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.erode(mask, kernel, iterations=1)
        
        # Ensure mask covers a reasonable portion of the bbox
        if np.sum(mask) < 0.1 * h * w:
            # Fallback to a simple rectangle if segmentation failed
            mask = np.ones((h, w), dtype=np.uint8) * 255
        
        return mask


# Global instance for reuse
_segmentation_handler = None

def get_segmentation_handler() -> SegmentationHandler:
    """Get or create the global segmentation handler instance."""
    global _segmentation_handler
    if _segmentation_handler is None:
        _segmentation_handler = SegmentationHandler()
    return _segmentation_handler

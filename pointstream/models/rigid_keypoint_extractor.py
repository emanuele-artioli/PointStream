"""
Rigid Object Keypoint Extractor

Extracts meaningful keypoints from rigid objects using computer vision techniques
like edge detection, corner detection, and feature matching.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RigidObjectKeypointExtractor:
    """
    Extract keypoints from rigid objects using multiple CV techniques.
    
    This combines several approaches:
    1. Harris corner detection for stable corner points
    2. Canny edge detection followed by contour analysis
    3. SIFT/ORB feature points for texture-rich objects
    4. Geometric keypoints (centroid, extrema)
    """
    
    def __init__(self, 
                 max_keypoints: int = 20,
                 corner_quality: float = 0.01,
                 corner_min_distance: float = 10,
                 edge_threshold1: int = 50,
                 edge_threshold2: int = 150,
                 use_sift: bool = True):
        """
        Initialize the rigid object keypoint extractor.
        
        Args:
            max_keypoints: Maximum number of keypoints to extract
            corner_quality: Quality level for Harris corner detection
            corner_min_distance: Minimum distance between corners
            edge_threshold1: Lower threshold for Canny edge detection
            edge_threshold2: Upper threshold for Canny edge detection
            use_sift: Whether to use SIFT features (fallback to ORB if False)
        """
        self.max_keypoints = max_keypoints
        self.corner_quality = corner_quality
        self.corner_min_distance = corner_min_distance
        self.edge_threshold1 = edge_threshold1
        self.edge_threshold2 = edge_threshold2
        self.use_sift = use_sift
        
        # Initialize feature detectors
        try:
            if self.use_sift:
                self.feature_detector = cv2.SIFT_create(nfeatures=max_keypoints//2)
            else:
                self.feature_detector = cv2.ORB_create(nfeatures=max_keypoints//2)
        except Exception as e:
            logger.warning(f"Failed to initialize feature detector: {e}")
            self.feature_detector = None
    
    def extract_keypoints(self, 
                         image: np.ndarray, 
                         bbox: Tuple[int, int, int, int],
                         object_mask: Optional[np.ndarray] = None) -> List[Tuple[float, float]]:
        """
        Extract keypoints from a rigid object in the image.
        
        Args:
            image: Input image (BGR format)
            bbox: Bounding box (x, y, width, height)
            object_mask: Optional binary mask for the object
            
        Returns:
            List of (x, y) keypoint coordinates
        """
        try:
            x, y, w, h = bbox
            
            # Extract object region
            object_region = image[y:y+h, x:x+w]
            if object_region.size == 0:
                return self._fallback_keypoints(bbox)
            
            # Convert to grayscale
            if len(object_region.shape) == 3:
                gray = cv2.cvtColor(object_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = object_region
            
            # Create or refine object mask using segmentation if not provided
            if object_mask is None:
                object_mask = self._create_object_mask(object_region, gray)
            else:
                # Use provided mask region
                object_mask = object_mask[y:y+h, x:x+w]
            
            # Apply mask to grayscale image
            if object_mask is not None:
                gray = cv2.bitwise_and(gray, gray, mask=object_mask)
            
            # Collect keypoints from multiple methods
            keypoints = []
            
            # Method 1: Harris corners (only on object pixels)
            corners = self._extract_harris_corners(gray, object_mask)
            keypoints.extend([(x + kp[0], y + kp[1]) for kp in corners])
            
            # Method 2: Enhanced contour-based keypoints
            contour_kps = self._extract_contour_keypoints(gray, object_mask)
            keypoints.extend([(x + kp[0], y + kp[1]) for kp in contour_kps])
            
            # Method 3: Feature detector keypoints
            if self.feature_detector is not None:
                feature_kps = self._extract_feature_keypoints(gray, object_mask)
                keypoints.extend([(x + kp[0], y + kp[1]) for kp in feature_kps])
            
            # Method 4: Geometric keypoints (based on actual object shape)
            geometric_kps = self._extract_geometric_keypoints(gray, object_mask)
            keypoints.extend([(x + kp[0], y + kp[1]) for kp in geometric_kps])
            
            # Remove duplicates and limit number
            keypoints = self._filter_keypoints(keypoints)
            
            if len(keypoints) < 4:  # Ensure minimum keypoints
                fallback_kps = self._fallback_keypoints(bbox)
                keypoints.extend(fallback_kps)
                keypoints = self._filter_keypoints(keypoints)
            
            return keypoints[:self.max_keypoints]
            
        except Exception as e:
            logger.warning(f"Error extracting rigid object keypoints: {e}")
            return self._fallback_keypoints(bbox)
    
    def _create_object_mask(self, object_region: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Create an object mask using multiple segmentation techniques.
        
        Args:
            object_region: RGB object region
            gray: Grayscale object region
            
        Returns:
            Binary mask for the object
        """
        try:
            h, w = gray.shape
            
            # Method 1: GrabCut-based segmentation
            mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Create initial rectangle (slightly inside bbox)
            rect = (5, 5, max(1, w-10), max(1, h-10))
            
            try:
                cv2.grabCut(object_region, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
                object_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            except:
                object_mask = None
            
            # Method 2: Adaptive thresholding + morphology
            if object_mask is None or np.sum(object_mask) < 100:
                # Use adaptive threshold to separate object from background
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
                
                # Morphological operations to clean up
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                object_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
                object_mask = (object_mask > 0).astype('uint8')
            
            # Method 3: Fallback to edge-based mask
            if object_mask is None or np.sum(object_mask) < 50:
                edges = cv2.Canny(gray, 30, 80)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                object_mask = cv2.dilate(edges, kernel, iterations=2)
                object_mask = cv2.fillPoly(object_mask, [np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])], 255)
                object_mask = (object_mask > 0).astype('uint8')
            
            return object_mask if np.sum(object_mask) > 0 else None
            
        except Exception as e:
            logger.warning(f"Error creating object mask: {e}")
            return None

    def _extract_harris_corners(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Tuple[float, float]]:
        """Extract Harris corner points."""
        try:
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_keypoints//4,
                qualityLevel=self.corner_quality,
                minDistance=self.corner_min_distance,
                useHarrisDetector=True,
                mask=mask
            )
            
            if corners is not None:
                return [(float(x[0][0]), float(x[0][1])) for x in corners]
            return []
        except Exception:
            return []
    
    def _extract_contour_keypoints(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Tuple[float, float]]:
        """Extract keypoints from object contours."""
        try:
            # Use mask-aware edge detection
            if mask is not None:
                # Apply mask to focus on object boundaries
                masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
                edges = cv2.Canny(masked_gray, self.edge_threshold1, self.edge_threshold2)
                # Also include mask boundaries as edges
                mask_edges = cv2.Canny(mask * 255, 50, 100)
                edges = cv2.bitwise_or(edges, mask_edges)
            else:
                edges = cv2.Canny(gray, self.edge_threshold1, self.edge_threshold2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            keypoints = []
            for contour in contours:
                if cv2.contourArea(contour) > 20:  # Filter small contours
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Add polygon vertices as keypoints
                    for point in approx:
                        x, y = float(point[0][0]), float(point[0][1])
                        # Verify point is within mask if provided
                        if mask is None or (0 <= int(x) < mask.shape[1] and 
                                          0 <= int(y) < mask.shape[0] and 
                                          mask[int(y), int(x)] > 0):
                            keypoints.append((x, y))
            
            return keypoints
        except Exception:
            return []

    def _extract_feature_keypoints(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Tuple[float, float]]:
        """Extract SIFT/ORB feature keypoints."""
        try:
            if self.feature_detector is None:
                return []
            
            keypoints = self.feature_detector.detect(gray, mask=mask)
            return [(float(kp.pt[0]), float(kp.pt[1])) for kp in keypoints]
        except Exception:
            return []

    def _extract_geometric_keypoints(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Tuple[float, float]]:
        """Extract geometric keypoints (centroid, extrema) based on object mask."""
        try:
            keypoints = []
            
            # Use mask to find object pixels if available, otherwise use non-zero pixels
            if mask is not None:
                y_coords, x_coords = np.where(mask > 0)
            else:
                y_coords, x_coords = np.where(gray > 0)
            
            if len(x_coords) > 0 and len(y_coords) > 0:
                # Centroid of actual object pixels
                cx = float(np.mean(x_coords))
                cy = float(np.mean(y_coords))
                keypoints.append((cx, cy))
                
                # Extrema points on object boundary
                left = (float(np.min(x_coords)), float(y_coords[np.argmin(x_coords)]))
                right = (float(np.max(x_coords)), float(y_coords[np.argmax(x_coords)]))
                top = (float(x_coords[np.argmin(y_coords)]), float(np.min(y_coords)))
                bottom = (float(x_coords[np.argmax(y_coords)]), float(np.max(y_coords)))
                
                keypoints.extend([left, right, top, bottom])
                
                # Add some intermediate points for better coverage
                if len(x_coords) > 10:
                    # Quarter points
                    x_sorted_idx = np.argsort(x_coords)
                    y_sorted_idx = np.argsort(y_coords)
                    
                    q1_x = float(x_coords[x_sorted_idx[len(x_coords)//4]])
                    q1_y = float(y_coords[x_sorted_idx[len(x_coords)//4]])
                    q3_x = float(x_coords[x_sorted_idx[3*len(x_coords)//4]])
                    q3_y = float(y_coords[x_sorted_idx[3*len(x_coords)//4]])
                    
                    keypoints.extend([(q1_x, q1_y), (q3_x, q3_y)])
            
            return keypoints
        except Exception:
            return []
    
    def _filter_keypoints(self, keypoints: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Remove duplicate keypoints and apply spatial filtering."""
        if not keypoints:
            return []
        
        # Remove duplicates with tolerance
        filtered = []
        tolerance = 5.0
        
        for kp in keypoints:
            is_duplicate = False
            for existing in filtered:
                if (abs(kp[0] - existing[0]) < tolerance and 
                    abs(kp[1] - existing[1]) < tolerance):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(kp)
        
        return filtered
    
    def _fallback_keypoints(self, bbox: Tuple[int, int, int, int]) -> List[Tuple[float, float]]:
        """Fallback to bounding box keypoints."""
        x, y, w, h = bbox
        return [
            (float(x), float(y)),           # Top-left
            (float(x + w), float(y)),       # Top-right  
            (float(x + w), float(y + h)),   # Bottom-right
            (float(x), float(y + h)),       # Bottom-left
            (float(x + w//2), float(y + h//2))  # Center
        ]
    
    def visualize_keypoints(self, 
                          image: np.ndarray, 
                          keypoints: List[Tuple[float, float]],
                          color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Visualize keypoints on the image.
        
        Args:
            image: Input image
            keypoints: List of (x, y) keypoint coordinates
            color: Color for keypoint visualization (BGR)
            
        Returns:
            Image with keypoints drawn
        """
        vis_image = image.copy()
        
        for i, (x, y) in enumerate(keypoints):
            cv2.circle(vis_image, (int(x), int(y)), 3, color, -1)
            cv2.putText(vis_image, str(i), (int(x+5), int(y+5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return vis_image


def extract_rigid_object_keypoints(image: np.ndarray, 
                                 bbox: Tuple[int, int, int, int],
                                 object_mask: Optional[np.ndarray] = None,
                                 max_keypoints: int = 20) -> List[Tuple[float, float]]:
    """
    Convenience function to extract keypoints from a rigid object.
    
    Args:
        image: Input image (BGR format)
        bbox: Bounding box (x, y, width, height)
        object_mask: Optional binary mask for the object
        max_keypoints: Maximum number of keypoints to extract
        
    Returns:
        List of (x, y) keypoint coordinates
    """
    extractor = RigidObjectKeypointExtractor(max_keypoints=max_keypoints)
    return extractor.extract_keypoints(image, bbox, object_mask)

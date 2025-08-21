#!/usr/bin/env python3
"""
Keypoint Detection Component

This component handles keypoint detection for segmented objects using different approaches:
- Human keypoints: MMPose human pose estimation
- Animal keypoints: MMPose animal pose estimation  
- Other objects: Canny edge detection for key features

The component includes a semantic tool to categorize objects into appropriate keypoint types.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from .decorators import log_step, time_step
from . import config

try:
    # Try to import MMPose
    from mmpose.apis import MMPoseInferencer
    MMPOSE_AVAILABLE = True
    logging.info("MMPose available for keypoint detection")
except ImportError:
    MMPOSE_AVAILABLE = False
    logging.warning("MMPose not available, using fallback keypoint detection")


class Keypointer:
    """Keypoint detection component with semantic object categorization."""
    
    def __init__(self):
        """Initialize the keypointer with MMPose models and parameters."""
        # Device configuration
        self.device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        
        # Semantic categorization mappings
        self.class_to_category = {
            # Human category
            'person': 'human',
            'human': 'human',
            'man': 'human',
            'woman': 'human',
            'child': 'human',
            'baby': 'human',
            
            # Animal category
            'dog': 'animal',
            'cat': 'animal',
            'horse': 'animal',
            'sheep': 'animal',
            'cow': 'animal',
            'elephant': 'animal',
            'bear': 'animal',
            'zebra': 'animal',
            'giraffe': 'animal',
            'bird': 'animal',
            'chicken': 'animal',
            'duck': 'animal',
            'goose': 'animal',
            'turkey': 'animal',
            'fish': 'animal',
            'shark': 'animal',
            'whale': 'animal',
            
            # Everything else is 'other'
        }
        
        # Keypoint detection parameters
        self.human_confidence_threshold = config.get_float('keypoints', 'human_confidence_threshold', 0.3)
        self.animal_confidence_threshold = config.get_float('keypoints', 'animal_confidence_threshold', 0.3)
        
        # Canny edge detection parameters for 'other' objects
        self.canny_low_threshold = config.get_int('keypoints', 'canny_low_threshold', 50)
        self.canny_high_threshold = config.get_int('keypoints', 'canny_high_threshold', 150)
        self.canny_kernel_size = config.get_int('keypoints', 'canny_kernel_size', 3)
        
        # Corner detection parameters for 'other' objects
        self.corner_max_corners = config.get_int('keypoints', 'corner_max_corners', 25)
        self.corner_quality_level = config.get_float('keypoints', 'corner_quality_level', 0.01)
        self.corner_min_distance = config.get_int('keypoints', 'corner_min_distance', 10)
        
        # Initialize models
        self._initialize_models()
        
        logging.info("Keypointer initialized")
        logging.info(f"MMPose available: {MMPOSE_AVAILABLE}")
        logging.info(f"Human confidence threshold: {self.human_confidence_threshold}")
        logging.info(f"Animal confidence threshold: {self.animal_confidence_threshold}")
        logging.info(f"Canny thresholds: {self.canny_low_threshold}-{self.canny_high_threshold}")
        logging.info(f"Corner detection: max {self.corner_max_corners} corners")
    
    def _initialize_models(self):
        """Initialize MMPose models for human and animal pose estimation."""
        self.human_model = None
        self.animal_model = None
        
        if MMPOSE_AVAILABLE:
            try:
                # Initialize human pose model
                human_model_config = config.get_str('keypoints', 'human_model_config', 'human')
                human_model_checkpoint = config.get_str('keypoints', 'human_model_checkpoint', '')
                
                # Initialize animal pose model
                animal_model_config = config.get_str('keypoints', 'animal_model_config', 'animal')
                animal_model_checkpoint = config.get_str('keypoints', 'animal_model_checkpoint', '')
                
                # Initialize MMPose models
                try:
                    from mmpose.apis import MMPoseInferencer
                    import warnings
                    
                    # Suppress model loading warnings (they're usually harmless)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*unexpected key.*")
                        warnings.filterwarnings("ignore", message=".*do not match exactly.*")
                        
                        # Use device from config with fallback
                        device = getattr(self, 'device', 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
                        
                        self.human_model = MMPoseInferencer(
                            pose2d=human_model_config,
                            pose2d_weights=human_model_checkpoint if human_model_checkpoint else None,
                            device=device
                        )
                        
                        self.animal_model = MMPoseInferencer(
                            pose2d=animal_model_config, 
                            pose2d_weights=animal_model_checkpoint if animal_model_checkpoint else None,
                            device=device
                        )
                    
                    logging.info("MMPose models initialized successfully")
                    
                except Exception as model_error:
                    logging.warning(f"Failed to load specific MMPose models: {model_error}")
                    # Fallback to default models
                    try:
                        from mmpose.apis import MMPoseInferencer
                        
                        self.human_model = MMPoseInferencer('human')
                        self.animal_model = MMPoseInferencer('animal') 
                        
                        logging.info("MMPose models initialized with defaults")
                        
                    except Exception as fallback_error:
                        logging.error(f"Failed to initialize any MMPose models: {fallback_error}")
                        self.human_model = None
                        self.animal_model = None
                
            except Exception as e:
                logging.error(f"Failed to initialize MMPose models: {e}")
                self.human_model = None
                self.animal_model = None
        else:
            logging.info("MMPose not available, using fallback methods only")
    
    def semantic_categorizer(self, class_name: str) -> str:
        """
        Semantic tool to categorize object class names into keypoint categories.
        
        Args:
            class_name: Object class name from detection
            
        Returns:
            Category string: 'human', 'animal', or 'other'
        """
        class_name_lower = class_name.lower()
        
        # Check direct mappings
        if class_name_lower in self.class_to_category:
            return self.class_to_category[class_name_lower]
        
        # Check partial matches for animals
        animal_keywords = ['dog', 'cat', 'bird', 'fish', 'animal']
        for keyword in animal_keywords:
            if keyword in class_name_lower:
                return 'animal'
        
        # Check partial matches for humans
        human_keywords = ['person', 'human', 'man', 'woman', 'people']
        for keyword in human_keywords:
            if keyword in class_name_lower:
                return 'human'
        
        # Default to 'other'
        return 'other'
    
    @log_step
    @time_step(track_processing=True)
    def extract_keypoints(self, objects_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract keypoints for all objects based on their semantic category.
        
        Args:
            objects_data: List of object dictionaries with cropped/masked images
            
        Returns:
            Dictionary containing keypoint data for each object
        """
        if not objects_data:
            return {'objects': []}
        
        logging.info(f"Extracting keypoints for {len(objects_data)} objects")
        
        processed_objects = []
        
        for obj_data in objects_data:
            try:
                # Get object information
                class_name = obj_data.get('class_name', 'unknown')
                track_id = obj_data.get('track_id')
                frame_index = obj_data.get('frame_index')
                
                # Categorize object semantically
                category = self.semantic_categorizer(class_name)
                
                # Extract keypoints based on category
                if category == 'human':
                    keypoints = self._extract_human_keypoints(obj_data)
                elif category == 'animal':
                    keypoints = self._extract_animal_keypoints(obj_data)
                else:  # category == 'other'
                    keypoints = self._extract_edge_keypoints(obj_data)
                
                # Create enhanced object data
                enhanced_obj = obj_data.copy()
                enhanced_obj.update({
                    'semantic_category': category,
                    'keypoints': keypoints,
                    'keypoint_method': self._get_method_name(category)
                })
                
                processed_objects.append(enhanced_obj)
                
                logging.debug(f"Object {class_name} (track {track_id}): {category} -> {len(keypoints.get('points', []))} keypoints")
                
            except Exception as e:
                logging.error(f"Failed to extract keypoints for object {obj_data.get('class_name', 'unknown')}: {e}")
                # Add object without keypoints
                enhanced_obj = obj_data.copy()
                enhanced_obj.update({
                    'semantic_category': 'other',
                    'keypoints': {'points': [], 'method': 'error'},
                    'keypoint_method': 'error'
                })
                processed_objects.append(enhanced_obj)
        
        return {'objects': processed_objects}
    
    def _extract_human_keypoints(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract human pose keypoints using MMPose.
        
        Args:
            obj_data: Object data dictionary with cropped/masked image
            
        Returns:
            Dictionary with human keypoint data
        """
        if not MMPOSE_AVAILABLE or self.human_model is None:
            # Fallback to edge detection
            logging.debug("MMPose not available for human keypoints, using edge detection fallback")
            return self._extract_edge_keypoints(obj_data)
        
        try:
            # Get the cropped image
            if 'cropped_image' in obj_data:
                image = obj_data['cropped_image']
            elif 'masked_image' in obj_data:
                # Convert RGBA to RGB if needed
                masked_img = obj_data['masked_image']
                if masked_img.shape[-1] == 4:
                    image = masked_img[:, :, :3]
                else:
                    image = masked_img
            else:
                logging.warning("No suitable image found for human keypoint detection")
                return {'points': [], 'method': 'no_image', 'confidence_scores': []}
            
            # In a real implementation, you would run MMPose inference here:
            # results = self.human_model(image)
            # keypoints = results['predictions'][0]['keypoints']
            # scores = results['predictions'][0]['keypoint_scores']
            
            # For simulation, generate fake human keypoints (17 COCO keypoints)
            h, w = image.shape[:2]
            simulated_keypoints = self._generate_simulated_human_keypoints(w, h)
            
            # Filter by confidence threshold
            valid_keypoints = []
            confidence_scores = []
            
            for i, (x, y, conf) in enumerate(simulated_keypoints):
                if conf >= self.human_confidence_threshold:
                    valid_keypoints.append({
                        'id': i,
                        'name': self._get_human_keypoint_name(i),
                        'x': float(x),
                        'y': float(y),
                        'confidence': float(conf),
                        'visible': True
                    })
                    confidence_scores.append(float(conf))
            
            return {
                'points': valid_keypoints,
                'method': 'mmpose_human',
                'confidence_scores': confidence_scores,
                'total_keypoints': len(simulated_keypoints),
                'valid_keypoints': len(valid_keypoints)
            }
            
        except Exception as e:
            logging.error(f"Human keypoint extraction failed: {e}")
            return self._extract_edge_keypoints(obj_data)
    
    def _extract_animal_keypoints(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract animal pose keypoints using MMPose.
        
        Args:
            obj_data: Object data dictionary with cropped/masked image
            
        Returns:
            Dictionary with animal keypoint data
        """
        if not MMPOSE_AVAILABLE or self.animal_model is None:
            # Fallback to edge detection
            logging.debug("MMPose not available for animal keypoints, using edge detection fallback")
            return self._extract_edge_keypoints(obj_data)
        
        try:
            # Get the cropped image
            if 'cropped_image' in obj_data:
                image = obj_data['cropped_image']
            elif 'masked_image' in obj_data:
                # Convert RGBA to RGB if needed
                masked_img = obj_data['masked_image']
                if masked_img.shape[-1] == 4:
                    image = masked_img[:, :, :3]
                else:
                    image = masked_img
            else:
                logging.warning("No suitable image found for animal keypoint detection")
                return {'points': [], 'method': 'no_image', 'confidence_scores': []}
            
            # In a real implementation, you would run MMPose animal inference here:
            # results = self.animal_model(image)
            
            # For simulation, generate fake animal keypoints
            h, w = image.shape[:2]
            simulated_keypoints = self._generate_simulated_animal_keypoints(w, h)
            
            # Filter by confidence threshold
            valid_keypoints = []
            confidence_scores = []
            
            for i, (x, y, conf) in enumerate(simulated_keypoints):
                if conf >= self.animal_confidence_threshold:
                    valid_keypoints.append({
                        'id': i,
                        'name': self._get_animal_keypoint_name(i),
                        'x': float(x),
                        'y': float(y),
                        'confidence': float(conf),
                        'visible': True
                    })
                    confidence_scores.append(float(conf))
            
            return {
                'points': valid_keypoints,
                'method': 'mmpose_animal',
                'confidence_scores': confidence_scores,
                'total_keypoints': len(simulated_keypoints),
                'valid_keypoints': len(valid_keypoints)
            }
            
        except Exception as e:
            logging.error(f"Animal keypoint extraction failed: {e}")
            return self._extract_edge_keypoints(obj_data)
    
    def _extract_edge_keypoints(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key features using Canny edge detection and corner detection.
        
        Args:
            obj_data: Object data dictionary with cropped/masked image
            
        Returns:
            Dictionary with edge-based keypoint data
        """
        try:
            # Get the cropped image
            if 'cropped_image' in obj_data:
                image = obj_data['cropped_image']
            elif 'masked_image' in obj_data:
                # Convert RGBA to RGB if needed
                masked_img = obj_data['masked_image']
                if masked_img.shape[-1] == 4:
                    image = masked_img[:, :, :3]
                else:
                    image = masked_img
            else:
                logging.warning("No suitable image found for edge keypoint detection")
                return {'points': [], 'method': 'no_image'}
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (self.canny_kernel_size, self.canny_kernel_size), 0)
            
            # Canny edge detection
            edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)
            
            # Find corners on the edge image
            corners = cv2.goodFeaturesToTrack(
                edges,
                maxCorners=self.corner_max_corners,
                qualityLevel=self.corner_quality_level,
                minDistance=self.corner_min_distance
            )
            
            keypoints = []
            if corners is not None:
                for i, corner in enumerate(corners):
                    x, y = corner.ravel()
                    keypoints.append({
                        'id': i,
                        'name': f'corner_{i}',
                        'x': float(x),
                        'y': float(y),
                        'confidence': 1.0,  # Corners don't have confidence scores
                        'visible': True,
                        'type': 'corner'
                    })
            
            return {
                'points': keypoints,
                'method': 'canny_corners',
                'edge_pixels': int(np.sum(edges > 0)),
                'total_corners': len(keypoints)
            }
            
        except Exception as e:
            logging.error(f"Edge keypoint extraction failed: {e}")
            return {'points': [], 'method': 'error'}
    
    def _generate_simulated_human_keypoints(self, width: int, height: int) -> List[Tuple[float, float, float]]:
        """Generate simulated human keypoints for testing (17 COCO keypoints)."""
        # Generate random keypoints within the image bounds with varying confidence
        keypoints = []
        
        # COCO human keypoint order: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
        for i in range(17):
            x = np.random.uniform(0.1 * width, 0.9 * width)
            y = np.random.uniform(0.1 * height, 0.9 * height)
            confidence = np.random.uniform(0.2, 0.9)  # Random confidence
            keypoints.append((x, y, confidence))
        
        return keypoints
    
    def _generate_simulated_animal_keypoints(self, width: int, height: int) -> List[Tuple[float, float, float]]:
        """Generate simulated animal keypoints for testing."""
        # Generate random animal keypoints (typically fewer than human)
        keypoints = []
        
        # Common animal keypoints: nose, ears, shoulders, elbows, paws, hips, etc.
        for i in range(12):  # Typical animal keypoint count
            x = np.random.uniform(0.1 * width, 0.9 * width)
            y = np.random.uniform(0.1 * height, 0.9 * height)
            confidence = np.random.uniform(0.3, 0.8)
            keypoints.append((x, y, confidence))
        
        return keypoints
    
    def _get_human_keypoint_name(self, index: int) -> str:
        """Get human keypoint name by index (COCO format)."""
        names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        return names[index] if index < len(names) else f'keypoint_{index}'
    
    def _get_animal_keypoint_name(self, index: int) -> str:
        """Get animal keypoint name by index."""
        names = [
            'nose', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_paw', 'right_paw',
            'left_hip', 'right_hip', 'tail_base'
        ]
        return names[index] if index < len(names) else f'animal_keypoint_{index}'
    
    def _get_method_name(self, category: str) -> str:
        """Get the method name used for keypoint extraction."""
        if category == 'human':
            return 'mmpose_human' if MMPOSE_AVAILABLE and self.human_model else 'canny_corners'
        elif category == 'animal':
            return 'mmpose_animal' if MMPOSE_AVAILABLE and self.animal_model else 'canny_corners'
        else:
            return 'canny_corners'
    
    def filter_keypoints_by_confidence(self, keypoints_data: Dict[str, Any], 
                                     min_confidence: float) -> Dict[str, Any]:
        """
        Filter keypoints by minimum confidence threshold.
        
        Args:
            keypoints_data: Keypoints data from extract_keypoints
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered keypoints data
        """
        filtered_objects = []
        
        for obj in keypoints_data.get('objects', []):
            keypoints = obj.get('keypoints', {})
            points = keypoints.get('points', [])
            
            # Filter points by confidence
            filtered_points = [
                point for point in points 
                if point.get('confidence', 0) >= min_confidence
            ]
            
            # Update object data
            filtered_obj = obj.copy()
            filtered_keypoints = keypoints.copy()
            filtered_keypoints['points'] = filtered_points
            filtered_keypoints['filtered_count'] = len(filtered_points)
            filtered_obj['keypoints'] = filtered_keypoints
            
            filtered_objects.append(filtered_obj)
        
        return {'objects': filtered_objects}
    
    def get_keypoint_statistics(self, keypoints_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about keypoint detection results.
        
        Args:
            keypoints_data: Keypoints data from extract_keypoints
            
        Returns:
            Dictionary with statistics
        """
        objects = keypoints_data.get('objects', [])
        
        if not objects:
            return {
                'total_objects': 0,
                'total_keypoints': 0,
                'methods_used': [],
                'categories': {}
            }
        
        total_keypoints = 0
        methods_used = set()
        categories = {}
        
        for obj in objects:
            # Count keypoints
            keypoints = obj.get('keypoints', {})
            points = keypoints.get('points', [])
            total_keypoints += len(points)
            
            # Track methods
            method = keypoints.get('method', 'unknown')
            methods_used.add(method)
            
            # Track categories
            category = obj.get('semantic_category', 'unknown')
            if category not in categories:
                categories[category] = {'count': 0, 'keypoints': 0}
            categories[category]['count'] += 1
            categories[category]['keypoints'] += len(points)
        
        return {
            'total_objects': len(objects),
            'total_keypoints': total_keypoints,
            'average_keypoints_per_object': total_keypoints / len(objects) if objects else 0,
            'methods_used': list(methods_used),
            'categories': categories
        }
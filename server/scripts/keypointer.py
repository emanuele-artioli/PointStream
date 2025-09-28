#!/usr/bin/env python3
"""
Enhanced Fixed Keypoint Detection Component

Improvements:
1. Keypoints are normalized to [0, 1] range relative to crop size
2. Simplified bbox format: [x, y, width, height] 
3. Integer coordinates (no sub-pixel precision)
4. Optional segmentation masks in metadata
5. Proper pose image generation support
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path

from utils.decorators import track_performance
from utils import config

try:
    # Try to import MMPose
    from mmpose.apis import MMPoseInferencer
    MMPOSE_AVAILABLE = True
    logging.info("MMPose available for keypoint detection")
except ImportError:
    MMPOSE_AVAILABLE = False
    logging.warning("MMPose not available, using fallback keypoint detection")

try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import resnet50, ResNet50_Weights
    TORCHVISION_AVAILABLE = True
    logging.info("TorchVision available for appearance vector extraction.")
except ImportError:
    TORCHVISION_AVAILABLE = False
    logging.warning("TorchVision not available, cannot generate appearance vectors.")


class AppearanceFeatureExtractor:
    """Extracts a deep feature vector from an image using a pre-trained ResNet-50 model."""
    def __init__(self, device: str = 'cuda'):
        self.device = device
        if not TORCHVISION_AVAILABLE:
            self.model = None
            self.preprocess = None
            return

        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)

        # Remove the final classification layer (the "head")
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing steps for ResNet-50
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_feature(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extracts a 2048-dimensional feature vector from a single image."""
        if self.model is None or image is None:
            return None

        try:
            # Preprocess the image and add a batch dimension
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Extract features
            features = self.model(input_tensor)

            # Flatten the features and move to CPU
            return features.squeeze().cpu().numpy()
        except Exception as e:
            logging.error(f"Failed to extract appearance feature: {e}")
            return None


class Keypointer:
    """Enhanced keypoint detection component with normalized keypoints and simplified bbox format."""
    
    def __init__(self):
        """Initialize the keypointer with MMPose models and parameters."""
        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Configuration options
        self.include_segmentation_masks = config.get_bool('keypoints', 'include_segmentation_masks', False)
        self.normalize_keypoints = config.get_bool('keypoints', 'normalize_keypoints', True)
        self.integer_coordinates = config.get_bool('keypoints', 'integer_coordinates', True)
        
        # Initialize models
        self._initialize_models()

        # Initialize the appearance feature extractor
        if TORCHVISION_AVAILABLE:
            self.feature_extractor = AppearanceFeatureExtractor(device=self.device)
        else:
            self.feature_extractor = None

        logging.info("Keypointer initialized (ENHANCED FIXED VERSION)")
        logging.info(f"MMPose available: {MMPOSE_AVAILABLE}")
        logging.info(f"TorchVision available: {TORCHVISION_AVAILABLE}")
        logging.info(f"Include segmentation masks: {self.include_segmentation_masks}")
        logging.info(f"Normalize keypoints: {self.normalize_keypoints}")
        logging.info(f"Integer coordinates: {self.integer_coordinates}")
        logging.info(f"Using device: {self.device}")
    
    def _initialize_models(self):
        """Initialize MMPose models for human and animal pose estimation."""
        self.human_model = None
        self.animal_model = None
        
        if MMPOSE_AVAILABLE:
            try:
                # Initialize MMPose models with defaults
                from mmpose.apis import MMPoseInferencer
                import warnings
                
                # Suppress model loading warnings (they're usually harmless)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*unexpected key.*")
                    warnings.filterwarnings("ignore", message=".*do not match exactly.*")
                    
                    self.human_model = MMPoseInferencer('human')
                    self.animal_model = MMPoseInferencer('animal') 
                
                logging.info("MMPose models initialized successfully")
                
            except Exception as e:
                logging.error(f"Failed to initialize MMPose models: {e}")
                self.human_model = None
                self.animal_model = None
        else:
            logging.info("MMPose not available, using fallback methods only")
    
    @track_performance
    def extract_keypoints(self, objects_data: List[Dict[str, Any]], frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extracts appearance and pose vectors for all objects with enhanced metadata format.

        Args:
            objects_data: List of object dictionaries with semantic categories.
            frames: The list of all frames in the scene.

        Returns:
            Dictionary containing the processed objects with enhanced metadata.
        """
        if not objects_data:
            return {'objects': []}

        logging.info(f"Extracting appearance and pose vectors for {len(objects_data)} objects.")
        
        # Group objects by their track ID
        objects_by_track = defaultdict(list)
        for obj in objects_data:
            track_id = obj.get('track_id')
            if track_id is not None:
                objects_by_track[track_id].append(obj)

        # Generate static appearance vector for each track
        appearance_vectors = {}
        if self.feature_extractor:
            for track_id, track_objects in objects_by_track.items():
                # Find the best object instance (highest confidence) to generate the vector from
                best_obj = max(track_objects, key=lambda o: o.get('confidence', 0))

                # Extract feature vector from the cropped image of the best object
                cropped_image = best_obj.get('cropped_image')
                if cropped_image is not None:
                    v_appearance = self.feature_extractor.extract_feature(cropped_image)
                    appearance_vectors[track_id] = v_appearance
                    logging.debug(f"Generated appearance vector for track {track_id}")

        # Process each object to add appearance and pose vectors
        processed_objects = []
        for obj_data in objects_data:
            # First, enhance the bbox format and metadata
            enhanced_obj = self._enhance_object_metadata(obj_data)
            
            track_id = enhanced_obj.get('track_id')
            semantic_category = enhanced_obj.get('semantic_category', 'other')
            
            # Debug logging
            if semantic_category != 'other':
                logging.info(f"Processing object {enhanced_obj.get('object_id', 'unknown')} with category: {semantic_category}")

            # Add the static appearance vector for this track
            if track_id in appearance_vectors:
                enhanced_obj['v_appearance'] = appearance_vectors[track_id]

            # Generate the dynamic pose vector with normalized keypoints
            if semantic_category == 'human':
                pose_vector = self._extract_human_keypoints(enhanced_obj)
                enhanced_obj['p_pose'] = pose_vector
                enhanced_obj['pose_method'] = 'mmpose_human'
                enhanced_obj['keypoints'] = pose_vector  # Set keypoints for client compatibility
            elif semantic_category == 'animal':
                pose_vector = self._extract_animal_keypoints(enhanced_obj)
                enhanced_obj['p_pose'] = pose_vector
                enhanced_obj['pose_method'] = 'mmpose_animal'
                enhanced_obj['keypoints'] = pose_vector  # Set keypoints for client compatibility
            else: # 'other' category
                pose_vector = self._extract_bbox_pose(enhanced_obj, frames)
                enhanced_obj['p_pose'] = pose_vector
                enhanced_obj['pose_method'] = 'bbox_simple'
                enhanced_obj['keypoints'] = pose_vector  # Set keypoints for client compatibility

            # Ensure the client can access the category information
            enhanced_obj['category'] = semantic_category
            logging.debug(f"Set category for object {enhanced_obj.get('object_id', 'unknown')}: {semantic_category}")

            processed_objects.append(enhanced_obj)

        return {'objects': processed_objects}

    def _enhance_object_metadata(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance object metadata with simplified bbox format and optional segmentation masks.
        
        Args:
            obj_data: Original object data
            
        Returns:
            Enhanced object data with simplified format
        """
        enhanced_obj = obj_data.copy()
        
        # Simplify bbox format from [x1, y1, x2, y2] to [x, y, width, height] with integer coordinates
        if 'bbox' in enhanced_obj:
            bbox = enhanced_obj['bbox']
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                
                if self.integer_coordinates:
                    x, y = int(round(x1)), int(round(y1))
                    width, height = int(round(x2 - x1)), int(round(y2 - y1))
                else:
                    x, y = float(x1), float(y1)
                    width, height = float(x2 - x1), float(y2 - y1)
                
                enhanced_obj['bbox'] = [x, y, width, height]
                
                # Also update crop_bbox for consistency
                enhanced_obj['crop_bbox'] = [x, y, x + width, y + height]
                enhanced_obj['crop_size'] = [width, height]
        
        # Optionally include segmentation masks
        if self.include_segmentation_masks and 'mask' in obj_data:
            enhanced_obj['segmentation_mask'] = obj_data['mask'].tolist() if isinstance(obj_data['mask'], np.ndarray) else obj_data['mask']
        
        return enhanced_obj

    def _extract_bbox_pose(self, obj_data: Dict[str, Any], frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extracts bounding box information as normalized pose data.
        Returns bbox as 4 corner points with coordinates normalized to [0, 1] range.
        
        Args:
            obj_data: Object data dictionary.
            frames: List of all frames in the scene (may be empty for fallback cases).

        Returns:
            A dictionary containing normalized bounding box corner points.
        """
        bbox = obj_data.get('bbox')

        if bbox is None or len(bbox) < 4:
            # Return zero bbox
            return {
                'points': [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                'method': 'bbox_no_data'
            }

        try:
            # Bbox is now [x, y, width, height]
            x, y, width, height = bbox[:4]

            if self.normalize_keypoints:
                # Normalize to [0, 1] range relative to crop size
                # For bbox objects, normalize relative to bbox size itself
                bbox_points = [
                    [0.0, 0.0],        # Top-left (relative to bbox)
                    [1.0, 0.0],        # Top-right
                    [1.0, 1.0],        # Bottom-right
                    [0.0, 1.0]         # Bottom-left
                ]
            else:
                # Use absolute coordinates
                x1, y1 = x, y
                x2, y2 = x + width, y + height
                
                bbox_points = [
                    [float(x1), float(y1)],  # Top-left
                    [float(x2), float(y1)],  # Top-right  
                    [float(x2), float(y2)],  # Bottom-right
                    [float(x1), float(y2)]   # Bottom-left
                ]

            return {
                'points': bbox_points,
                'method': 'bbox_normalized' if self.normalize_keypoints else 'bbox_absolute'
            }
        except Exception as e:
            logging.error(f"Bounding box pose extraction failed: {e}")
            return {
                'points': [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                'method': 'bbox_error'
            }

    def _extract_human_keypoints(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract human pose keypoints using MMPose with normalization.
        Returns exactly 17 keypoints for COCO format, normalized to [0, 1] range.
        
        Args:
            obj_data: Object data dictionary with cropped/masked image
            
        Returns:
            Dictionary with normalized human keypoint data (exactly 17 keypoints)
        """
        if not MMPOSE_AVAILABLE or self.human_model is None:
            logging.debug("MMPose not available for human keypoints, using bbox fallback")
            return self._extract_bbox_pose(obj_data, [])
        
        try:
            # Get the cropped image
            image = self._load_image_from_obj_data(obj_data)
            if image is None:
                logging.warning("No suitable image found for human keypoint detection")
                return self._get_zero_human_keypoints()
            
            # Get crop dimensions for normalization
            crop_size = obj_data.get('crop_size', [image.shape[1], image.shape[0]])
            crop_width, crop_height = crop_size
            
            # Run MMPose inference
            results = list(self.human_model(image))
            
            if not results:
                logging.debug("No MMPose results for human keypoints")
                return self._get_zero_human_keypoints()
            
            # Extract keypoints from the first result
            result = results[0]
            if 'predictions' not in result or not result['predictions']:
                logging.debug("No predictions in MMPose result for human")
                return self._get_zero_human_keypoints()
            
            predictions = result['predictions'][0]  # Get first prediction
            if not predictions:
                logging.debug("Empty prediction list for human")
                return self._get_zero_human_keypoints()
                
            prediction = predictions[0]  # Get first detection
            
            keypoints_2d = prediction.get('keypoints', [])
            keypoint_scores = prediction.get('keypoint_scores', [])
            
            if len(keypoints_2d) < 17:
                logging.warning(f"Expected 17 human keypoints, got {len(keypoints_2d)}")
                return self._get_zero_human_keypoints()
            
            # Convert to normalized format: [[x, y], ...]
            formatted_keypoints = []
            for i in range(17):  # COCO human has 17 keypoints
                if i < len(keypoints_2d):
                    x, y = keypoints_2d[i]
                    
                    if self.normalize_keypoints:
                        # Normalize to [0, 1] range relative to crop size
                        norm_x = float(x) / crop_width if crop_width > 0 else 0.0
                        norm_y = float(y) / crop_height if crop_height > 0 else 0.0
                        # Clamp to [0, 1] range
                        norm_x = max(0.0, min(1.0, norm_x))
                        norm_y = max(0.0, min(1.0, norm_y))
                        formatted_keypoints.append([norm_x, norm_y])
                    else:
                        if self.integer_coordinates:
                            formatted_keypoints.append([int(round(x)), int(round(y))])
                        else:
                            formatted_keypoints.append([float(x), float(y)])
                else:
                    # Fill missing keypoints with zeros
                    formatted_keypoints.append([0.0, 0.0])
            
            logging.info(f"Successfully extracted {len(formatted_keypoints)} normalized human keypoints")
            return {
                'points': formatted_keypoints,
                'method': 'mmpose_human_normalized' if self.normalize_keypoints else 'mmpose_human_absolute'
            }
            
        except Exception as e:
            logging.error(f"Human keypoint extraction failed: {e}")
            return self._get_zero_human_keypoints()
    
    def _extract_animal_keypoints(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract animal pose keypoints using MMPose with normalization.
        Returns keypoints normalized to [0, 1] range.
        
        Args:
            obj_data: Object data dictionary with cropped/masked image
            
        Returns:
            Dictionary with normalized animal keypoint data
        """
        if not MMPOSE_AVAILABLE or self.animal_model is None:
            logging.debug("MMPose not available for animal keypoints, using bbox fallback")
            return self._extract_bbox_pose(obj_data, [])
        
        try:
            # Get the cropped image
            image = self._load_image_from_obj_data(obj_data)
            if image is None:
                logging.warning("No suitable image found for animal keypoint detection")
                return self._get_zero_animal_keypoints()
            
            # Get crop dimensions for normalization
            crop_size = obj_data.get('crop_size', [image.shape[1], image.shape[0]])
            crop_width, crop_height = crop_size
            
            # Run MMPose inference
            results = list(self.animal_model(image))
            
            if not results:
                logging.debug("No MMPose results for animal keypoints")
                return self._get_zero_animal_keypoints()
            
            # Extract keypoints from the first result
            result = results[0]
            if 'predictions' not in result or not result['predictions']:
                logging.debug("No predictions in MMPose result for animal")
                return self._get_zero_animal_keypoints()
            
            predictions = result['predictions'][0]  # Get first prediction
            if not predictions:
                logging.debug("Empty prediction list for animal")
                return self._get_zero_animal_keypoints()
                
            prediction = predictions[0]  # Get first detection
            
            keypoints_2d = prediction.get('keypoints', [])
            keypoint_scores = prediction.get('keypoint_scores', [])
            
            if len(keypoints_2d) < 12:  # Most animal models have at least 12 keypoints
                logging.warning(f"Expected at least 12 animal keypoints, got {len(keypoints_2d)}")
                # Still use what we have
            
            # Convert to normalized format: [[x, y], ...]
            formatted_keypoints = []
            for i, (x, y) in enumerate(keypoints_2d):
                if self.normalize_keypoints:
                    # Normalize to [0, 1] range relative to crop size
                    norm_x = float(x) / crop_width if crop_width > 0 else 0.0
                    norm_y = float(y) / crop_height if crop_height > 0 else 0.0
                    # Clamp to [0, 1] range
                    norm_x = max(0.0, min(1.0, norm_x))
                    norm_y = max(0.0, min(1.0, norm_y))
                    formatted_keypoints.append([norm_x, norm_y])
                else:
                    if self.integer_coordinates:
                        formatted_keypoints.append([int(round(x)), int(round(y))])
                    else:
                        formatted_keypoints.append([float(x), float(y)])
            
            # Pad to ensure we have at least 12 keypoints for consistency
            while len(formatted_keypoints) < 12:
                formatted_keypoints.append([0.0, 0.0])
            
            logging.info(f"Successfully extracted {len(formatted_keypoints)} normalized animal keypoints")
            return {
                'points': formatted_keypoints,
                'method': 'mmpose_animal_normalized' if self.normalize_keypoints else 'mmpose_animal_absolute'
            }
            
        except Exception as e:
            logging.error(f"Animal keypoint extraction failed: {e}")
            return self._get_zero_animal_keypoints()
            
    def _load_image_from_obj_data(self, obj_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load image from object data, supporting both file paths and numpy arrays."""
        if 'cropped_image' in obj_data:
            image_data = obj_data['cropped_image']
            
            if isinstance(image_data, str) and Path(image_data).exists():
                # Load from file path
                image = cv2.imread(image_data)
                return image
            elif isinstance(image_data, np.ndarray):
                # Use numpy array directly
                return image_data
            else:
                logging.warning("Invalid cropped image format")
                return None
        else:
            logging.warning("No cropped image found in object data")
            return None
            
    def _get_zero_human_keypoints(self) -> Dict[str, Any]:
        """Return zero keypoints for human (17 keypoints)."""
        return {
            'points': [[0.0, 0.0] for _ in range(17)],
            'method': 'mmpose_human_fallback'
        }
        
    def _get_zero_animal_keypoints(self) -> Dict[str, Any]:
        """Return zero keypoints for animal (12 keypoints).""" 
        return {
            'points': [[0.0, 0.0] for _ in range(12)],
            'method': 'mmpose_animal_fallback'
        }

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
                'categories': {},
                'normalized_keypoints': self.normalize_keypoints,
                'integer_coordinates': self.integer_coordinates
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
            categories[category] = categories.get(category, 0) + 1
        
        return {
            'total_objects': len(objects),
            'total_keypoints': total_keypoints,
            'methods_used': list(methods_used),
            'categories': categories,
            'normalized_keypoints': self.normalize_keypoints,
            'integer_coordinates': self.integer_coordinates
        }
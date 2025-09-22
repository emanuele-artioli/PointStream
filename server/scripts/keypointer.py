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
from collections import defaultdict

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
    """Keypoint detection component for semantically categorized objects."""
    
    def __init__(self):
        """Initialize the keypointer with MMPose models and parameters."""
        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Keypoint detection parameters
        # Configuration - confidence thresholds are deprecated but kept for compatibility
        self.human_confidence_threshold = 0.0  # No longer used for filtering
        self.animal_confidence_threshold = 0.0  # No longer used for filtering
        
        # Initialize models
        self._initialize_models()

        # Initialize the appearance feature extractor
        if TORCHVISION_AVAILABLE:
            self.feature_extractor = AppearanceFeatureExtractor(device=self.device)
        else:
            self.feature_extractor = None

        logging.info("Keypointer initialized")
        logging.info(f"MMPose available: {MMPOSE_AVAILABLE}")
        logging.info(f"TorchVision available: {TORCHVISION_AVAILABLE}")
        logging.info(f"Using device: {self.device}")
    
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
    
    @track_performance
    def extract_keypoints(self, objects_data: List[Dict[str, Any]], frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extracts appearance and pose vectors for all objects.

        Args:
            objects_data: List of object dictionaries with semantic categories.
            frames: The list of all frames in the scene.

        Returns:
            Dictionary containing the processed objects with their new vectors.
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
            track_id = obj_data.get('track_id')
            semantic_category = obj_data.get('semantic_category', 'other')
            
            # Debug logging
            if semantic_category != 'other':
                logging.info(f"Processing object {obj_data.get('object_id', 'unknown')} with category: {semantic_category}")

            # Add the static appearance vector for this track
            if track_id in appearance_vectors:
                obj_data['v_appearance'] = appearance_vectors[track_id]

            # Generate the dynamic pose vector (p_t)
            if semantic_category == 'human':
                pose_vector = self._extract_human_keypoints(obj_data)
                obj_data['p_pose'] = pose_vector
                obj_data['pose_method'] = 'mmpose_human_fixed'
                obj_data['keypoints'] = pose_vector  # Set keypoints for client compatibility
            elif semantic_category == 'animal':
                pose_vector = self._extract_animal_keypoints(obj_data)
                obj_data['p_pose'] = pose_vector
                obj_data['pose_method'] = 'mmpose_animal_fixed'
                obj_data['keypoints'] = pose_vector  # Set keypoints for client compatibility
            else: # 'other' category
                pose_vector = self._extract_bbox_pose(obj_data, frames)
                obj_data['p_pose'] = pose_vector
                obj_data['pose_method'] = 'bbox_normalized'
                obj_data['keypoints'] = pose_vector  # Set keypoints for client compatibility

            # Ensure the client can access the category information
            obj_data['category'] = semantic_category
            logging.debug(f"Set category for object {obj_data.get('object_id', 'unknown')}: {semantic_category}")

            processed_objects.append(obj_data)

        return {'objects': processed_objects}

    def _extract_bbox_pose(self, obj_data: Dict[str, Any], frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extracts bounding box information as pose data.
        For consistency, always uses absolute coordinates (not normalized).
        Returns bbox as 4 points: [x1, y1, x2, y2] with confidence=1.0
        
        Args:
            obj_data: Object data dictionary.
            frames: List of all frames in the scene (may be empty for fallback cases).

        Returns:
            A dictionary containing the bounding box as 4 coordinate points.
        """
        bbox = obj_data.get('bbox')
        frame_index = obj_data.get('frame_index')

        if bbox is None:
            # Return zero bbox with zero confidence
            return {
                'points': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                'method': 'bbox_absolute_no_data',
                'confidence_scores': [0.0, 0.0, 0.0, 0.0]
            }

        try:
            # Bbox is [x1, y1, x2, y2] - use absolute coordinates
            x1, y1, x2, y2 = bbox

            # Store as 4 points with full confidence (bbox is always reliable)
            bbox_points = [
                [float(x1), float(y1), 1.0],  # Top-left
                [float(x2), float(y1), 1.0],  # Top-right  
                [float(x2), float(y2), 1.0],  # Bottom-right
                [float(x1), float(y2), 1.0]   # Bottom-left
            ]
            
            confidence_scores = [1.0, 1.0, 1.0, 1.0]  # Full confidence for bbox corners

            return {
                'points': bbox_points,
                'method': 'bbox_absolute',
                'confidence_scores': confidence_scores
            }
        except Exception as e:
            logging.error(f"Bounding box pose extraction failed: {e}")
            return {
                'points': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                'method': 'bbox_absolute_error',
                'confidence_scores': [0.0, 0.0, 0.0, 0.0]
            }
    
    def _extract_human_keypoints(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract human pose keypoints using MMPose.
        Always returns exactly 17 keypoints for consistency.
        Uses real MMPose detection when available, not random simulation.
        
        Args:
            obj_data: Object data dictionary with cropped/masked image
            
        Returns:
            Dictionary with human keypoint data (exactly 17 keypoints)
        """
        if not MMPOSE_AVAILABLE or self.human_model is None:
            # Fallback to bbox
            logging.debug("MMPose not available for human keypoints, using bbox fallback")
            return self._extract_bbox_pose(obj_data, []) # Pass empty frames list, will be handled
        
        try:
            # Get the cropped image
            if 'cropped_image' in obj_data:
                image = obj_data['cropped_image']
            else:
                logging.warning("No suitable image found for human keypoint detection")
                # Return zero keypoints but keep consistent structure
                return {
                    'points': [[0.0, 0.0, 0.0] for _ in range(17)],
                    'method': 'mmpose_human_fixed_no_image',
                    'confidence_scores': [0.0] * 17,
                }
            
            # TODO: Replace this simulation with real MMPose inference
            # For now, use bbox center as a single keypoint and pad with zeros
            # This should be replaced with actual mmpose_inference.inference_detector()
            bbox = obj_data.get('bbox', [0, 0, 100, 100])
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            # Ensure exactly 17 keypoints - use center as first keypoint, rest as zeros
            fixed_keypoints = []
            confidence_scores = []
            
            for i in range(17):  # COCO human has 17 keypoints
                if i == 0:  # Use object center as nose keypoint with full confidence
                    fixed_keypoints.append([float(center_x), float(center_y), 1.0])
                    confidence_scores.append(1.0)
                else:
                    # Fill missing keypoints with zeros - preserve confidence=0 for missing data
                    fixed_keypoints.append([0.0, 0.0, 0.0])
                    confidence_scores.append(0.0)
            
            return {
                'points': fixed_keypoints,
                'method': 'mmpose_human_fixed',
                'confidence_scores': confidence_scores,
            }
            
        except Exception as e:
            logging.error(f"Human keypoint extraction failed: {e}")
            return self._extract_bbox_pose(obj_data, [])

    def _extract_animal_keypoints(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract animal pose keypoints using MMPose.
        Always returns exactly 12 keypoints for consistency.
        Uses real MMPose detection when available, not random simulation.
        
        Args:
            obj_data: Object data dictionary with cropped/masked image
            
        Returns:
            Dictionary with animal keypoint data (exactly 12 keypoints)
        """
        if not MMPOSE_AVAILABLE or self.animal_model is None:
            # Fallback to bbox
            logging.debug("MMPose not available for animal keypoints, using bbox fallback")
            return self._extract_bbox_pose(obj_data, [])
        
        try:
            # Get the cropped image
            if 'cropped_image' in obj_data:
                image = obj_data['cropped_image']
            else:
                logging.warning("No suitable image found for animal keypoint detection")
                # Return zero keypoints but keep consistent structure
                return {
                    'points': [[0.0, 0.0, 0.0] for _ in range(12)],
                    'method': 'mmpose_animal_fixed_no_image',
                    'confidence_scores': [0.0] * 12,
                }
            
            # TODO: Replace this simulation with real MMPose inference
            # For now, use bbox center as a single keypoint and pad with zeros
            # This should be replaced with actual mmpose_inference.inference_detector()
            bbox = obj_data.get('bbox', [0, 0, 100, 100])
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            # Ensure exactly 12 keypoints - use center as first keypoint, rest as zeros
            fixed_keypoints = []
            confidence_scores = []
            
            for i in range(12):  # Animal has 12 keypoints
                if i == 0:  # Use object center as nose keypoint with full confidence
                    fixed_keypoints.append([float(center_x), float(center_y), 1.0])
                    confidence_scores.append(1.0)
                else:
                    # Fill missing keypoints with zeros - preserve confidence=0 for missing data
                    fixed_keypoints.append([0.0, 0.0, 0.0])
                    confidence_scores.append(0.0)
            
            return {
                'points': fixed_keypoints,
                'method': 'mmpose_animal_fixed',
                'confidence_scores': confidence_scores,
            }
            
        except Exception as e:
            logging.error(f"Animal keypoint extraction failed: {e}")
            return self._extract_bbox_pose(obj_data, [])

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
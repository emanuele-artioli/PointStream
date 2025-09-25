#!/usr/bin/env python3
"""
Object Generator

This module generates objects for each frame using trained generative models.
It uses keypoints and first frame references to generate objects for all frames.
Supports different models for humans, animals, and other objects.
"""

import logging
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

try:
    from utils.decorators import track_performance
    from utils import config
    from client.models.human_cgan import HumanCGAN
    from client.models.animal_cgan import AnimalCGAN
    from client.models.other_cgan import OtherCGAN
    from utils.vector_utils import build_pose_vector, validate_vector_dimensions, calculate_pose_vector_size
except ImportError as e:
    logging.error(f"Failed to import PointStream utilities or models: {e}")
    raise


class ObjectGenerator:
    """
    Generates objects for each frame using trained generative models.
    
    This component takes keypoints and reference images to generate
    objects for each frame using category-specific generative models.
    """
    
    def __init__(self, device: torch.device):
        """Initialize the object generator."""
        self.device = device
        
        # Load configuration
        self.model_type = config.get_str('models', 'model_type', 'cgan')
        self.batch_size = config.get_int('models', 'batch_size', 8)
        
        # Model paths
        self.models_dir = config.get_str('models', 'models_dir', './models')
        self.human_model_path = config.get_str('models', 'human_model_path', './models/human_cgan.pth')
        self.animal_model_path = config.get_str('models', 'animal_model_path', './models/animal_cgan.pth')
        self.other_model_path = config.get_str('models', 'other_model_path', './models/other_cgan.pth')
        
        # Input sizes
        self.human_input_size = config.get_int('models', 'human_input_size', 256)
        self.animal_input_size = config.get_int('models', 'animal_input_size', 256)
        self.other_input_size = config.get_int('models', 'other_input_size', 256)
        
        # Confidence thresholds
        self.human_threshold = config.get_float('models', 'human_pose_confidence_threshold', 0.3)
        self.animal_threshold = config.get_float('models', 'animal_pose_confidence_threshold', 0.3)
        self.other_threshold = config.get_float('models', 'other_confidence_threshold', 0.3)
        
        # Initialize models dictionary
        self.models = {}
        
        # Initialize models
        self._initialize_models()
        
        # Initialize appearance feature extractor
        self._initialize_feature_extractor()
        
        logging.info("🤖 Object Generator initialized")
        logging.info(f"   Model type: {self.model_type}")
        logging.info(f"   Device: {self.device}")
        logging.info(f"   Batch size: {self.batch_size}")
    
    def _initialize_models(self):
        """Initialize all CGAN models."""
        logging.info("🤖 Object Generator initialized")
        logging.info(f"   Device: {self.device}")
        logging.info(f"   Models directory: {self.models_dir}")
        
        # Initialize feature extractor for appearance vectors
        self._initialize_feature_extractor()
        
        # Initialize Human model
        self._initialize_human_model()
        
        # Initialize Animal model
        self._initialize_animal_model()
        
        # Initialize Other model
        self._initialize_other_model()
        
        if self.feature_extractor is not None:
            logging.info(f"✅ Initialized {self.feature_extractor.__class__.__name__} feature extractor for appearance vectors")
    
    def _extract_temporal_context(self, current_obj: Dict[str, Any], all_objects: List[Dict[str, Any]], 
                                 temporal_frames: int) -> List[float]:
        """
        Extract temporal context from previous frames for the same track.
        
        Args:
            current_obj: Current object data
            all_objects: All objects in the scene
            temporal_frames: Number of previous frames to include
            
        Returns:
            Flattened list of keypoints from previous frames
        """
        if temporal_frames == 0:
            return []
        
        try:
            track_id = current_obj.get('track_id')
            current_frame = current_obj.get('frame_index', 0)
            category = current_obj.get('semantic_category', 'other')
            
            if track_id is None:
                # No tracking available, return zeros
                return self._get_empty_temporal_context(category, temporal_frames)
            
            # Find previous frames for the same track
            track_objects = [obj for obj in all_objects 
                           if obj.get('track_id') == track_id and 
                           obj.get('frame_index', 0) < current_frame]
            
            # Sort by frame index (most recent first)
            track_objects.sort(key=lambda x: x.get('frame_index', 0), reverse=True)
            
            temporal_context = []
            frames_collected = 0
            
            for prev_obj in track_objects:
                if frames_collected >= temporal_frames:
                    break
                
                # Extract keypoints from previous frame
                prev_keypoints = self._extract_keypoints_for_temporal_context(prev_obj, category)
                temporal_context.extend(prev_keypoints)
                frames_collected += 1
            
            # Pad with zeros if we don't have enough previous frames
            while frames_collected < temporal_frames:
                empty_keypoints = self._get_empty_keypoints_for_category(category)
                temporal_context.extend(empty_keypoints)
                frames_collected += 1
            
            return temporal_context
            
        except Exception as e:
            logging.warning(f"Error extracting temporal context: {e}")
            return self._get_empty_temporal_context(category, temporal_frames)
    
    def _get_empty_temporal_context(self, category: str, temporal_frames: int) -> List[float]:
        """Get empty temporal context filled with zeros."""
        empty_keypoints = self._get_empty_keypoints_for_category(category)
        return empty_keypoints * temporal_frames
    
    def _get_empty_keypoints_for_category(self, category: str) -> List[float]:
        """Get empty keypoints for a specific category."""
        include_confidence = config.get_bool('keypoints', 'include_confidence_in_vectors', True)
        
        if category == 'human':
            keypoint_count = config.get_int('keypoints', 'human_num_keypoints', 17)
        elif category == 'animal':
            keypoint_count = config.get_int('keypoints', 'animal_num_keypoints', 12)
        else:
            keypoint_count = config.get_int('keypoints', 'other_num_keypoints', 24)
        
        if include_confidence:
            return [0.0] * (keypoint_count * 3)  # x, y, confidence
        else:
            return [0.0] * (keypoint_count * 2)  # x, y only
    
    def _extract_keypoints_for_temporal_context(self, obj: Dict[str, Any], category: str) -> List[float]:
        """Extract keypoints from an object for temporal context."""
        # Use the same keypoint processing as the main pipeline
        if category in ['human', 'animal']:
            return self._normalize_keypoints_to_standard_size(obj, 256, category)
        else:
            # For 'other', extract enhanced keypoints or bbox
            keypoints_data = obj.get('p_pose_data', [])
            if keypoints_data and len(keypoints_data) > 4:
                # Use enhanced keypoints if available
                include_confidence = config.get_bool('keypoints', 'include_confidence_in_vectors', True)
                keypoint_count = config.get_int('keypoints', 'other_num_keypoints', 24)
                
                flattened = []
                for i, kp in enumerate(keypoints_data[:keypoint_count]):
                    if include_confidence and len(kp) >= 3:
                        flattened.extend([kp[0], kp[1], kp[2]])
                    elif len(kp) >= 2:
                        if include_confidence:
                            flattened.extend([kp[0], kp[1], 1.0])  # Default confidence
                        else:
                            flattened.extend([kp[0], kp[1]])
                
                # Pad if needed
                target_size = keypoint_count * (3 if include_confidence else 2)
                while len(flattened) < target_size:
                    flattened.append(0.0)
                
                return flattened[:target_size]
            else:
                # Fallback to bbox
                return self._get_empty_keypoints_for_category(category)
    
    def _check_model_compatibility(self, model_path: str, expected_metadata: dict) -> tuple[bool, dict]:
        """
        Check if a saved model is compatible with current configuration.
        
        Args:
            model_path: Path to the saved model
            expected_metadata: Expected model metadata based on current config
            
        Returns:
            Tuple of (is_compatible, saved_metadata)
        """
        if not Path(model_path).exists():
            return False, {}
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            saved_metadata = checkpoint.get('model_metadata', {})
            
            # Check critical compatibility parameters
            critical_params = ['vector_input_size', 'keypoint_channels', 'include_confidence', 'temporal_frames']
            
            for param in critical_params:
                if param in saved_metadata and param in expected_metadata:
                    if saved_metadata[param] != expected_metadata[param]:
                        logging.warning(f"Model incompatibility in {param}: saved={saved_metadata[param]}, expected={expected_metadata[param]}")
                        return False, saved_metadata
            
            return True, saved_metadata
            
        except Exception as e:
            logging.error(f"Error checking model compatibility: {e}")
            return False, {}
    
    def _initialize_human_model(self):
        """Initialize human model with compatibility checking."""
        try:
            # Get configuration parameters
            keypoint_channels = config.get_int('keypoints', 'human_num_keypoints', 17)
            include_confidence = config.get_bool('keypoints', 'include_confidence_in_vectors', True)
            temporal_frames = config.get_int('keypoints', 'temporal_frames', 0)
            
            # Calculate vector size based on configuration
            if include_confidence:
                pose_size = keypoint_channels * 3  # x, y, confidence
            else:
                pose_size = keypoint_channels * 2  # x, y only
            
            # Include temporal context
            pose_size *= (1 + temporal_frames)
            
            # Total vector size: appearance (2048) + pose
            human_vector_size = 2048 + pose_size
            
            # Expected metadata for compatibility checking
            expected_metadata = {
                'vector_input_size': human_vector_size,
                'keypoint_channels': keypoint_channels,
                'include_confidence': include_confidence,
                'temporal_frames': temporal_frames,
                'model_version': '2.0'
            }
            
            # Check model compatibility
            is_compatible, saved_metadata = self._check_model_compatibility(self.human_model_path, expected_metadata)
            
            if not is_compatible and Path(self.human_model_path).exists():
                logging.warning(f"   ⚠️ Human model incompatible with current config, will need retraining")
                # Move incompatible model to backup
                backup_path = f"{self.human_model_path}.backup"
                Path(self.human_model_path).rename(backup_path)
                logging.info(f"   💾 Moved incompatible model to {backup_path}")
            
            # Use saved vector size if model is compatible, otherwise use calculated size
            model_vector_size = saved_metadata.get('vector_input_size', human_vector_size) if is_compatible else human_vector_size
            
            # Initialize model with appropriate configuration
            self.models['human'] = HumanCGAN(
                input_size=self.human_input_size,
                vector_input_size=model_vector_size,
                temporal_frames=temporal_frames,
                include_confidence=include_confidence
            ).to(self.device)
            
            if Path(self.human_model_path).exists() and is_compatible:
                checkpoint = torch.load(self.human_model_path, map_location=self.device)
                self.models['human'].generator.load_state_dict(checkpoint['generator'])
                logging.info(f"   ✅ Loaded compatible human model: {self.human_model_path}")
            else:
                logging.warning(f"   ⚠️ Human model not found: {self.human_model_path}")
                
        except Exception as e:
            logging.error(f"Failed to initialize human model: {e}")
            self.models['human'] = None
        
        # Initialize Animal model
        self._initialize_animal_model()
        
        # Initialize Other model
        self._initialize_other_model()

    def _initialize_animal_model(self):
        """Initialize animal model with compatibility checking."""
        try:
            # Get configuration parameters
            keypoint_channels = config.get_int('keypoints', 'animal_num_keypoints', 12)
            include_confidence = config.get_bool('keypoints', 'include_confidence_in_vectors', True)
            temporal_frames = config.get_int('keypoints', 'temporal_frames', 0)
            
            # Calculate vector size based on configuration
            if include_confidence:
                pose_size = keypoint_channels * 3  # x, y, confidence
            else:
                pose_size = keypoint_channels * 2  # x, y only
            
            # Include temporal context
            pose_size *= (1 + temporal_frames)
            
            # Total vector size: appearance (2048) + pose
            animal_vector_size = 2048 + pose_size
            
            # Expected metadata for compatibility checking
            expected_metadata = {
                'vector_input_size': animal_vector_size,
                'keypoint_channels': keypoint_channels,
                'include_confidence': include_confidence,
                'temporal_frames': temporal_frames,
                'model_version': '2.0'
            }
            
            # Check model compatibility
            is_compatible, saved_metadata = self._check_model_compatibility(self.animal_model_path, expected_metadata)
            
            if not is_compatible and Path(self.animal_model_path).exists():
                logging.warning(f"   ⚠️ Animal model incompatible with current config, will need retraining")
                # Move incompatible model to backup
                backup_path = f"{self.animal_model_path}.backup"
                Path(self.animal_model_path).rename(backup_path)
                logging.info(f"   💾 Moved incompatible model to {backup_path}")
            
            # Use saved vector size if model is compatible, otherwise use calculated size
            model_vector_size = saved_metadata.get('vector_input_size', animal_vector_size) if is_compatible else animal_vector_size
            
            # Initialize model with appropriate configuration
            self.models['animal'] = AnimalCGAN(
                input_size=self.animal_input_size,
                vector_input_size=model_vector_size,
                temporal_frames=temporal_frames,
                include_confidence=include_confidence
            ).to(self.device)
            
            if Path(self.animal_model_path).exists() and is_compatible:
                checkpoint = torch.load(self.animal_model_path, map_location=self.device)
                self.models['animal'].generator.load_state_dict(checkpoint['generator'])
                logging.info(f"   ✅ Loaded compatible animal model: {self.animal_model_path}")
            else:
                logging.warning(f"   ⚠️ Animal model not found: {self.animal_model_path}")
                
        except Exception as e:
            logging.error(f"Failed to initialize animal model: {e}")
            self.models['animal'] = None
    
    def _initialize_other_model(self):
        """Initialize other model with compatibility checking."""
        try:
            # Get configuration parameters  
            keypoint_channels = config.get_int('keypoints', 'other_num_keypoints', 24)  # Updated to 24
            include_confidence = config.get_bool('keypoints', 'include_confidence_in_vectors', True)
            temporal_frames = config.get_int('keypoints', 'temporal_frames', 0)
            
            # Calculate vector size based on configuration
            if include_confidence:
                pose_size = keypoint_channels * 3  # x, y, confidence
            else:
                pose_size = keypoint_channels * 2  # x, y only
            
            # Include temporal context
            pose_size *= (1 + temporal_frames)
            
            # Total vector size: appearance (2048) + pose
            other_vector_size = 2048 + pose_size
            
            # Expected metadata for compatibility checking
            expected_metadata = {
                'vector_input_size': other_vector_size,
                'keypoint_channels': keypoint_channels,
                'include_confidence': include_confidence,
                'temporal_frames': temporal_frames,
                'model_version': '2.0'
            }
            
            # Check model compatibility
            is_compatible, saved_metadata = self._check_model_compatibility(self.other_model_path, expected_metadata)
            
            if not is_compatible and Path(self.other_model_path).exists():
                logging.warning(f"   ⚠️ Other model incompatible with current config, will need retraining")
                # Move incompatible model to backup
                backup_path = f"{self.other_model_path}.backup"
                Path(self.other_model_path).rename(backup_path)
                logging.info(f"   💾 Moved incompatible model to {backup_path}")
            
            # Use saved vector size if model is compatible, otherwise use calculated size
            model_vector_size = saved_metadata.get('vector_input_size', other_vector_size) if is_compatible else other_vector_size
            
            # Initialize model with appropriate configuration
            self.models['other'] = OtherCGAN(
                input_size=self.other_input_size,
                vector_input_size=model_vector_size,
                temporal_frames=temporal_frames,
                include_confidence=include_confidence
            ).to(self.device)
            
            if Path(self.other_model_path).exists() and is_compatible:
                checkpoint = torch.load(self.other_model_path, map_location=self.device)
                self.models['other'].generator.load_state_dict(checkpoint['generator'])
                logging.info(f"   ✅ Loaded compatible other model: {self.other_model_path}")
            else:
                logging.warning(f"   ⚠️ Other model not found: {self.other_model_path}")
                
        except Exception as e:
            logging.error(f"Failed to initialize other model: {e}")
            self.models['other'] = None
    
    def _initialize_feature_extractor(self):
        """Initialize pre-trained CNN for appearance feature extraction."""
        try:
            # Use ResNet-50 pre-trained on ImageNet for robust feature extraction
            self.feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            
            # Remove the final classification layer to get features
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
            
            # Set to evaluation mode
            self.feature_extractor.eval()
            self.feature_extractor.to(self.device)
            
            # Define image preprocessing pipeline
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),  # ResNet input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
            
            logging.info("   ✅ Initialized ResNet-50 feature extractor for appearance vectors")
            
        except Exception as e:
            logging.error(f"Failed to initialize feature extractor: {e}")
            self.feature_extractor = None
            self.transform = None
    
    @track_performance
    def generate_objects(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate objects for all frames in a scene.
        
        Args:
            scene_data: Scene metadata including object information and keypoints
            
        Returns:
            Dictionary containing generated objects for each frame
        """
        start_time = time.time()
        
        # Group objects by track ID and category
        object_tracks = self._group_objects_by_track(scene_data)
        
        logging.info(f"🎭 Generating objects for {len(object_tracks)} tracks")
        
        # Generate objects for each track
        all_generated_objects = []
        
        for track_id, track_objects in object_tracks.items():
            # Determine track category by consensus (majority vote)
            categories = [obj.get('category', obj.get('semantic_category', 'other')) for obj in track_objects]
            # Count categories and use the most common one
            from collections import Counter
            category_counts = Counter(categories)
            category = category_counts.most_common(1)[0][0]  # Most frequent category
            
            logging.info(f"   🎯 Track {track_id} ({category}): {len(track_objects)} frames")
            if len(category_counts) > 1:
                logging.info(f"      📊 Category distribution: {dict(category_counts)}")
            
            # Generate objects for this track
            track_generated = self._generate_track_objects(track_objects, category, scene_data)
            all_generated_objects.extend(track_generated)
        
        processing_time = time.time() - start_time
        
        result = {
            'generated_objects': all_generated_objects,
            'total_objects': len(all_generated_objects),
            'tracks_processed': len(object_tracks),
            'processing_time': processing_time
        }
        
        logging.info(f"✅ Object generation completed in {processing_time:.2f}s")
        return result
    
    def _group_objects_by_track(self, scene_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Group objects by track ID for consistent generation."""
        tracks = {}
        
        # Load object data from scene metadata - check multiple possible locations
        objects = scene_data.get('objects', [])
        if not objects and 'keypoint_result' in scene_data:
            objects = scene_data['keypoint_result'].get('objects', [])
        
        logging.info(f"🔍 Found {len(objects)} objects in scene metadata")
        
        for obj in objects:
            track_id = obj.get('track_id', f"obj_{obj.get('object_id', 'unknown')}")
            
            if track_id not in tracks:
                tracks[track_id] = []
            
            tracks[track_id].append(obj)
        
        # Sort objects in each track by frame index
        for track_id in tracks:
            tracks[track_id].sort(key=lambda x: x.get('frame_index', 0))
        
        return tracks
    
    def _generate_track_objects(self, track_objects: List[Dict[str, Any]], 
                              category: str, scene_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate objects for a complete track using the new vector-based input.
        
        Args:
            track_objects: List of objects in the track (sorted by frame).
            category: Object category (human, animal, other).
            scene_data: Scene metadata including objects directory path.
            
        Returns:
            List of generated objects for each frame.
        """
        # Store objects directory for appearance vector extraction
        self._current_objects_dir = scene_data.get('objects_dir')
        
        model = self.models.get(category)
        if model is None:
            logging.warning(f"No model available for category: {category}")
            return self._generate_fallback_objects(track_objects)
        
        # Get the static appearance vector from the first object in the track
        # (it should be the same for all objects in the track).
        v_appearance = track_objects[0].get('v_appearance')
        if v_appearance is None:
            # Try to extract appearance vector from the first image of the track
            track_id = track_objects[0].get('track_id')
            category = track_objects[0].get('semantic_category', 'other')
            
            # Get objects directory from scene data
            objects_dir = self._current_objects_dir
            
            if objects_dir and track_id is not None:
                # Construct path to first frame image
                first_frame_image = Path(objects_dir) / f"{category}_track_{track_id}_frame_0000.png"
                logging.info(f"Attempting to extract appearance vector from: {first_frame_image}")
                
                v_appearance = self._extract_appearance_vector_from_image(str(first_frame_image), track_objects)
                
                if v_appearance is not None:
                    logging.info(f"✅ Successfully extracted appearance vector for track {track_id}")
                    # Update all objects in the track with the extracted appearance vector
                    for obj in track_objects:
                        obj['v_appearance'] = v_appearance.tolist()
                else:
                    logging.warning(f"❌ Failed to extract appearance vector for track {track_id}")
            
            if v_appearance is None:
                logging.warning(f"No appearance vector found for track {track_objects[0].get('track_id')} and could not extract from image")
                return self._generate_fallback_objects(track_objects)

        v_appearance_tensor = torch.from_numpy(np.array(v_appearance)).float().to(self.device)

        generated_objects = []
        for obj in track_objects:
            try:
                generated_obj = self._generate_single_object(obj, v_appearance_tensor, category, model, scene_data)
                generated_objects.append(generated_obj)
            except Exception as e:
                logging.warning(f"Failed to generate object for frame {obj.get('frame_index', 0)}: {e}")
                fallback_obj = self._create_fallback_object(obj)
                generated_objects.append(fallback_obj)
        
        return generated_objects

    def _normalize_keypoints_to_standard_size(self, obj: Dict[str, Any], standard_size: int = 256, category: str = None) -> List[float]:
        """
        Normalize keypoints from original crop size to standard model input size.
        Ensures fixed-size output based on object category.
        
        Args:
            obj: Object containing keypoints and crop_size
            standard_size: Target size for model input (e.g., 256)
            category: Override category for normalization (uses track consensus)
            
        Returns:
            Normalized keypoints scaled to standard_size with fixed dimensions
        """
        keypoints_info = obj.get('keypoints', {})
        keypoints_data = keypoints_info.get('points', [])
        keypoints_method = keypoints_info.get('method', 'unknown')
        
        if not keypoints_data:
            raise ValueError(f"No keypoints found for object {obj.get('object_id')}")
        
        # Get original crop size
        crop_size = obj.get('crop_size', [standard_size, standard_size])
        
        # Handle different crop_size formats
        if isinstance(crop_size, (int, float)):
            # Square crop size
            crop_size = [float(crop_size), float(crop_size)]
        elif isinstance(crop_size, (list, tuple)) and len(crop_size) >= 2:
            crop_size = [float(crop_size[0]), float(crop_size[1])]
        else:
            logging.warning(f"Invalid crop_size {crop_size}, using standard size")
            crop_size = [float(standard_size), float(standard_size)]
        
        original_width, original_height = crop_size[0], crop_size[1]
        
        # Calculate scaling factors
        scale_x = standard_size / original_width
        scale_y = standard_size / original_height
        
        # Determine expected keypoint count based on provided category or object category
        if category:
            # Use provided category (track consensus)
            use_category = category
        else:
            # Fall back to object's own category
            use_category = obj.get('category', 'other')
            
        if use_category == 'human':
            expected_keypoints = 17
        elif use_category == 'animal':
            expected_keypoints = 12
        else:
            # For other objects, default to a reasonable number for bbox-based data
            expected_keypoints = 2  # Just use center points for bbox data
        
        # Initialize normalized keypoints with zeros (padding)
        normalized_keypoints = [0.0] * (expected_keypoints * 3)
        
        # Handle different keypoint data formats
        if keypoints_method in ['bbox_normalized', 'bbox_absolute', 'bbox_absolute_no_data', 'bbox_absolute_error']:
            # New consistent bbox format: 4 points with coordinates and confidence
            # Each point is [x, y, confidence]
            if len(keypoints_data) >= 4 and all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in keypoints_data):
                # Extract bbox corners from 4 points format
                points = keypoints_data[:4]  # Take first 4 points
                
                # Convert 4-point format back to bbox format for processing
                x_coords = [pt[0] for pt in points]
                y_coords = [pt[1] for pt in points]
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                
                # Convert to center point and size (for "other" objects)
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1
                
                # Scale to standard size
                norm_center_x = center_x * scale_x
                norm_center_y = center_y * scale_y
                norm_width = width * scale_x
                norm_height = height * scale_y
                
                # First keypoint: center
                normalized_keypoints[0] = norm_center_x
                normalized_keypoints[1] = norm_center_y
                normalized_keypoints[2] = 1.0  # Confidence
                
                # Second keypoint: size representation (if we have space)
                if expected_keypoints >= 2:
                    normalized_keypoints[3] = norm_width
                    normalized_keypoints[4] = norm_height
                    normalized_keypoints[5] = 1.0  # Confidence
            elif len(keypoints_data) >= 4 and all(isinstance(val, (int, float)) for val in keypoints_data):
                # Legacy bbox format: [x1, y1, x2, y2] (flat values)
                x1, y1, x2, y2 = keypoints_data[0], keypoints_data[1], keypoints_data[2], keypoints_data[3]
                # Convert to center point and corner point (for "other" objects)
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                # Scale to standard size
                norm_center_x = center_x * scale_x
                norm_center_y = center_y * scale_y
                norm_corner_x = x2 * scale_x
                norm_corner_y = y2 * scale_y
                
                # First keypoint: center
                normalized_keypoints[0] = norm_center_x
                normalized_keypoints[1] = norm_center_y
                normalized_keypoints[2] = 1.0  # Confidence
                
                # Second keypoint: corner (if we have space)
                if expected_keypoints >= 2:
                    normalized_keypoints[3] = norm_corner_x
                    normalized_keypoints[4] = norm_corner_y
                    normalized_keypoints[5] = 1.0  # Confidence
        else:
            # Traditional keypoint format: [[x, y, confidence], ...] or [[x, y], ...]
            for i, kp in enumerate(keypoints_data):
                if i >= expected_keypoints:
                    break  # Don't exceed expected number
                    
                if isinstance(kp, (list, tuple)) and len(kp) >= 3:  # [x, y, confidence]
                    x, y, conf = kp[0], kp[1], kp[2]
                    # Scale to standard size
                    norm_x = x * scale_x
                    norm_y = y * scale_y
                    base_idx = i * 3
                    normalized_keypoints[base_idx] = norm_x
                    normalized_keypoints[base_idx + 1] = norm_y
                    normalized_keypoints[base_idx + 2] = conf
                elif isinstance(kp, (list, tuple)) and len(kp) >= 2:  # [x, y]
                    x, y = kp[0], kp[1]
                    norm_x = x * scale_x
                    norm_y = y * scale_y
                    base_idx = i * 3
                    normalized_keypoints[base_idx] = norm_x
                    normalized_keypoints[base_idx + 1] = norm_y
                    normalized_keypoints[base_idx + 2] = 1.0  # Default confidence
        
        logging.debug(f"Normalized {len(keypoints_data)} {keypoints_method} keypoints to fixed size {expected_keypoints} for {use_category}")
        return normalized_keypoints

    def _resize_generated_image(self, generated_image: np.ndarray, target_size: List[int]) -> np.ndarray:
        """
        Resize generated image from standard size back to original crop size.
        
        Args:
            generated_image: Generated image at standard size
            target_size: Original crop size [width, height]
            
        Returns:
            Resized image at target size
        """
        if not isinstance(target_size, (list, tuple)) or len(target_size) < 2:
            return generated_image
            
        target_width, target_height = int(target_size[0]), int(target_size[1])
        
        # Resize using high-quality interpolation
        resized_image = cv2.resize(generated_image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        logging.debug(f"Resized generated image from {generated_image.shape[:2]} to {target_height}x{target_width}")
        return resized_image

    def _generate_single_object(self, obj: Dict[str, Any], v_appearance_tensor: torch.Tensor,
                              category: str, model: nn.Module, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single object using the new vector-based model with size standardization."""
        
        # Get standard model input size
        if category == 'human':
            standard_size = self.human_input_size
        elif category == 'animal':
            standard_size = self.animal_input_size
        else:
            standard_size = self.other_input_size
        
        # Extract pose vector for model input with temporal context
        try:
            # Get configuration
            temporal_frames = config.get_int('keypoints', 'temporal_frames', 0)
            include_confidence = config.get_bool('keypoints', 'include_confidence_in_vectors', True)
            
            # Get keypoints data from the object
            if 'p_pose_model_input' in obj:
                # Use pre-processed vector if available
                p_t = obj['p_pose_model_input']
                logging.debug(f"Using pre-processed model input vector of length {len(p_t)} for {category}")
            else:
                # Build vector using centralized utility
                keypoints_data = []
                
                # Try different keypoint data sources
                if 'keypoints' in obj and isinstance(obj['keypoints'], dict) and 'points' in obj['keypoints']:
                    keypoints_data = obj['keypoints']['points']
                elif 'p_pose' in obj and isinstance(obj['p_pose'], dict) and 'points' in obj['p_pose']:
                    keypoints_data = obj['p_pose']['points']
                elif 'p_pose_data' in obj:
                    keypoints_data = obj['p_pose_data']
                else:
                    logging.warning(f"No keypoint data found for object {obj.get('object_id', 'unknown')}")
                    keypoints_data = []
                
                # Extract temporal context if enabled
                temporal_context_data = None
                if temporal_frames > 0:
                    all_objects = scene_data.get('objects', [])
                    temporal_context = self._extract_temporal_context(obj, all_objects, temporal_frames)
                    if temporal_context:
                        # Split temporal context into frames
                        expected_size = calculate_pose_vector_size(category, 0, include_confidence)
                        temporal_context_data = []
                        for i in range(0, len(temporal_context), expected_size):
                            frame_data = temporal_context[i:i + expected_size]
                            if len(frame_data) == expected_size:
                                temporal_context_data.append(frame_data)
                
                # Build pose vector using centralized utility
                p_t = build_pose_vector(
                    keypoints_data=keypoints_data,
                    category=category,
                    temporal_frames=temporal_frames,
                    include_confidence=include_confidence,
                    temporal_context_data=temporal_context_data
                )
                
                logging.debug(f"Built pose vector of length {len(p_t)} for {category}")
            
            # Validate vector dimensions
            is_valid, expected_size, actual_size, error_msg = validate_vector_dimensions(
                p_t, category, temporal_frames, include_confidence
            )
            
            if not is_valid:
                logging.warning(f"Vector validation failed: {error_msg}")
                # Try to fix by padding or truncating
                if actual_size < expected_size:
                    p_t.extend([0.0] * (expected_size - actual_size))
                    logging.debug(f"Padded vector from {actual_size} to {expected_size}")
                elif actual_size > expected_size:
                    p_t = p_t[:expected_size]
                    logging.debug(f"Truncated vector from {actual_size} to {expected_size}")
                    
        except Exception as e:
            logging.warning(f"Vector processing failed for object {obj.get('object_id')}: {e}")
            # Fallback to original method with error handling
            try:
                p_pose_data = obj.get('keypoints', {}).get('points', [])
                if not p_pose_data:
                    raise ValueError(f"No pose vector (p_t) found for object {obj.get('object_id')}")
                
                # Use simple flattening as last resort
                if isinstance(p_pose_data[0], (list, tuple)):
                    p_t = [coord for kp in p_pose_data for coord in kp]
                else:
                    p_t = p_pose_data
                    
                logging.debug(f"Used fallback vector processing, length: {len(p_t)}")
            except Exception as fallback_error:
                logging.error(f"Fallback vector processing also failed: {fallback_error}")
                # Ultimate fallback - return zeros with correct dimensions
                expected_size = calculate_pose_vector_size(category, temporal_frames, include_confidence)
                p_t = [0.0] * expected_size
                logging.warning(f"Using zero vector with size {expected_size} as last resort")

            if category in ['human', 'animal']:
                p_t = [coord for kp in p_pose_data for coord in kp]
            else:
                p_t = p_pose_data

        p_t_tensor = torch.tensor(p_t, dtype=torch.float32).to(self.device)
        
        # Generate object at standard size
        model.eval()
        with torch.no_grad():
            # Add batch dimension
            v_appearance_batch = v_appearance_tensor.unsqueeze(0)
            p_t_batch = p_t_tensor.unsqueeze(0)

            generated_tensor = model.generate(v_appearance_batch, p_t_batch)
            generated_image = generated_tensor.squeeze(0).cpu().numpy()
            
            # Convert from [-1, 1] to [0, 255] and from CHW to HWC
            generated_image = ((generated_image + 1) * 127.5).clip(0, 255).astype(np.uint8)
            if generated_image.shape[0] == 3:
                generated_image = np.transpose(generated_image, (1, 2, 0))
        
        # Resize generated image back to original crop size
        original_crop_size = obj.get('crop_size', [standard_size, standard_size])
        if (original_crop_size[0] != standard_size or original_crop_size[1] != standard_size):
            generated_image = self._resize_generated_image(generated_image, original_crop_size)
            logging.debug(f"Resized object from {standard_size}x{standard_size} to {original_crop_size}")
        
        # Create result object
        return {
            'object_id': obj.get('object_id'),
            'frame_index': obj.get('frame_index'),
            'track_id': obj.get('track_id'),
            'category': category,
            'bbox': obj.get('bbox', []),
            'crop_size': original_crop_size,
            'generated_image': generated_image,
            'generation_method': f'{category}_cgan_vector_standardized',
            'confidence': obj.get('confidence', 0.0)
        }
    
    def _create_fallback_object(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback object when generation fails."""
        category = obj.get('semantic_category', 'other')
        
        # Use original crop size if available, otherwise standard size
        crop_size = obj.get('crop_size', [256, 256])
        if isinstance(crop_size, (list, tuple)) and len(crop_size) >= 2:
            width, height = int(crop_size[0]), int(crop_size[1])
        else:
            if category == 'human':
                width = height = self.human_input_size
            elif category == 'animal':
                width = height = self.animal_input_size
            else:
                width = height = self.other_input_size

        fallback_image = np.zeros((height, width, 3), dtype=np.uint8)

        fallback_obj = {
            'object_id': obj.get('object_id'),
            'frame_index': obj.get('frame_index'),
            'track_id': obj.get('track_id'),
            'category': category,
            'bbox': obj.get('bbox', []),
            'crop_size': crop_size,
            'generated_image': fallback_image,
            'generation_method': 'fallback_error',
            'confidence': 0.0
        }
        return fallback_obj
    
    def _generate_fallback_objects(self, track_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate fallback objects (e.g., black squares) when a model fails or is unavailable."""
        fallback_objects = []
        for obj in track_objects:
            # Determine the input size for the category to generate a correctly sized placeholder
            category = obj.get('semantic_category', 'other')
            if category == 'human':
                size = self.human_input_size
            elif category == 'animal':
                size = self.animal_input_size
            else:
                size = self.other_input_size

            fallback_image = np.zeros((size, size, 3), dtype=np.uint8)

            fallback_obj = {
                'object_id': obj.get('object_id'),
                'frame_index': obj.get('frame_index'),
                'track_id': obj.get('track_id'),
                'category': category,
                'bbox': obj.get('bbox', []),
                'generated_image': fallback_image,
                'generation_method': 'fallback_placeholder',
                'confidence': 0.0
            }
            fallback_objects.append(fallback_obj)
        
        return fallback_objects

    def _extract_appearance_vector_from_image(self, image_path: str, track_objects: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
        """
        Extract robust appearance vector from object image(s) using pre-trained CNN.
        
        Args:
            image_path: Path to the first object image
            track_objects: Optional list of all objects in track for multi-frame aggregation
            
        Returns:
            2048-dimensional appearance vector, or None if extraction fails
        """
        try:
            if self.feature_extractor is None:
                logging.warning("Feature extractor not available, falling back to simple method")
                return self._extract_simple_appearance_vector(image_path)
            
            # Strategy: Use multiple frames (if available) to create robust appearance representation
            image_paths = [image_path]
            
            # If we have track objects, use multiple frames for better appearance representation
            if track_objects and self._current_objects_dir:
                track_id = track_objects[0].get('track_id')
                category = track_objects[0].get('semantic_category', 'other')
                
                # Collect paths for first few frames to get diverse appearance samples
                frame_indices = [0, len(track_objects)//4, len(track_objects)//2, 3*len(track_objects)//4, len(track_objects)-1]
                frame_indices = list(set(frame_indices))  # Remove duplicates
                frame_indices = [idx for idx in frame_indices if idx < len(track_objects)]
                
                image_paths = []
                for frame_idx in frame_indices[:5]:  # Use up to 5 frames
                    frame_image_path = Path(self._current_objects_dir) / f"{category}_track_{track_id}_frame_{frame_idx:04d}.png"
                    if frame_image_path.exists():
                        image_paths.append(str(frame_image_path))
                
                if image_paths:
                    logging.info(f"Using {len(image_paths)} frames for robust appearance extraction")
                else:
                    image_paths = [image_path]  # Fallback to original
            
            # Extract features from all available images
            all_features = []
            
            for img_path in image_paths:
                if not Path(img_path).exists():
                    continue
                    
                # Load and preprocess image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                    
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply preprocessing
                input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
                
                # Extract features using ResNet
                with torch.no_grad():
                    features = self.feature_extractor(input_tensor)
                    # features shape: [1, 2048, 1, 1] -> flatten to [2048]
                    features = features.squeeze().cpu().numpy()
                    
                all_features.append(features)
            
            if not all_features:
                logging.warning(f"No valid images found for appearance extraction")
                return None
            
            # Aggregate features from multiple frames
            if len(all_features) == 1:
                final_features = all_features[0]
            else:
                # Use mean aggregation for multi-frame features
                final_features = np.mean(all_features, axis=0)
                logging.info(f"Aggregated features from {len(all_features)} frames")
            
            # Ensure we have exactly 2048 dimensions
            if final_features.shape[0] != 2048:
                logging.warning(f"Unexpected feature dimension: {final_features.shape[0]}, expected 2048")
                # Pad or truncate to 2048
                if final_features.shape[0] > 2048:
                    final_features = final_features[:2048]
                else:
                    padded = np.zeros(2048)
                    padded[:final_features.shape[0]] = final_features
                    final_features = padded
            
            # Normalize features
            norm = np.linalg.norm(final_features)
            if norm > 1e-8:
                final_features = final_features / norm
            
            logging.debug(f"Extracted CNN appearance vector: shape {final_features.shape}, norm {np.linalg.norm(final_features):.4f}")
            return final_features.astype(np.float32)
            
        except Exception as e:
            logging.warning(f"CNN feature extraction failed for {image_path}: {e}")
            # Fallback to simple method
            return self._extract_simple_appearance_vector(image_path)
    
    def _extract_simple_appearance_vector(self, image_path: str) -> Optional[np.ndarray]:
        """
        Fallback method for appearance vector extraction using simple features.
        
        Args:
            image_path: Path to the object image
            
        Returns:
            2048-dimensional appearance vector, or None if extraction fails
        """
        try:
            if not Path(image_path).exists():
                logging.warning(f"Object image not found: {image_path}")
                return None
                
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Failed to load image: {image_path}")
                return None
                
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size and flatten for appearance features
            image_resized = cv2.resize(image_rgb, (64, 64))  # 64*64*3 = 12288 values
            
            # Normalize to [0, 1]
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Flatten image
            flattened = image_normalized.flatten()  # 12288 values
            
            # Pad or truncate to exactly 2048 dimensions
            if len(flattened) > 2048:
                appearance_vector = flattened[:2048]
            else:
                appearance_vector = np.zeros(2048, dtype=np.float32)
                appearance_vector[:len(flattened)] = flattened
            
            # Normalize
            norm = np.linalg.norm(appearance_vector)
            if norm > 1e-8:
                appearance_vector = appearance_vector / norm
                
            logging.debug(f"Extracted simple appearance vector from {image_path}: shape {appearance_vector.shape}")
            return appearance_vector
            
        except Exception as e:
            logging.warning(f"Failed to extract simple appearance vector from {image_path}: {e}")
            return None

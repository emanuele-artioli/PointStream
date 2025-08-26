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

try:
    from utils.decorators import track_performance
    from utils import config
    from client.models.human_cgan import HumanCGAN
    from client.models.animal_cgan import AnimalCGAN
    from client.models.other_cgan import OtherCGAN
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
        
        # Initialize models
        self._initialize_models()
        
        logging.info("🤖 Object Generator initialized")
        logging.info(f"   Model type: {self.model_type}")
        logging.info(f"   Device: {self.device}")
        logging.info(f"   Batch size: {self.batch_size}")
    
    def _initialize_models(self):
        """Initialize generative models for each object category."""
        self.models = {}
        
        # Initialize Human model
        try:
            self.models['human'] = HumanCGAN(
                input_size=self.human_input_size,
                keypoint_channels=config.get_int('models', 'human_keypoint_channels', 17)
            ).to(self.device)
            
            if Path(self.human_model_path).exists():
                checkpoint = torch.load(self.human_model_path, map_location=self.device)
                self.models['human'].load_state_dict(checkpoint['generator'])
                logging.info(f"   ✅ Loaded human model: {self.human_model_path}")
            else:
                logging.warning(f"   ⚠️ Human model not found: {self.human_model_path}")
        except Exception as e:
            logging.error(f"Failed to initialize human model: {e}")
            self.models['human'] = None
        
        # Initialize Animal model
        try:
            self.models['animal'] = AnimalCGAN(
                input_size=self.animal_input_size,
                keypoint_channels=config.get_int('models', 'animal_keypoint_channels', 20)
            ).to(self.device)
            
            if Path(self.animal_model_path).exists():
                checkpoint = torch.load(self.animal_model_path, map_location=self.device)
                self.models['animal'].load_state_dict(checkpoint['generator'])
                logging.info(f"   ✅ Loaded animal model: {self.animal_model_path}")
            else:
                logging.warning(f"   ⚠️ Animal model not found: {self.animal_model_path}")
        except Exception as e:
            logging.error(f"Failed to initialize animal model: {e}")
            self.models['animal'] = None
        
        # Initialize Other objects model
        try:
            self.models['other'] = OtherCGAN(
                input_size=self.other_input_size,
                feature_channels=config.get_int('models', 'other_feature_channels', 50)
            ).to(self.device)
            
            if Path(self.other_model_path).exists():
                checkpoint = torch.load(self.other_model_path, map_location=self.device)
                self.models['other'].load_state_dict(checkpoint['generator'])
                logging.info(f"   ✅ Loaded other model: {self.other_model_path}")
            else:
                logging.warning(f"   ⚠️ Other model not found: {self.other_model_path}")
        except Exception as e:
            logging.error(f"Failed to initialize other model: {e}")
            self.models['other'] = None
    
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
            category = track_objects[0].get('semantic_category', 'other')
            logging.info(f"   🎯 Track {track_id} ({category}): {len(track_objects)} frames")
            
            # Generate objects for this track
            track_generated = self._generate_track_objects(track_objects, category)
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
        
        # Load object data from scene metadata
        objects = scene_data.get('objects', [])
        
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
                              category: str) -> List[Dict[str, Any]]:
        """
        Generate objects for a complete track.
        
        Args:
            track_objects: List of objects in the track (sorted by frame)
            category: Object category (human, animal, other)
            
        Returns:
            List of generated objects for each frame
        """
        model = self.models.get(category)
        if model is None:
            logging.warning(f"No model available for category: {category}")
            return self._generate_fallback_objects(track_objects)
        
        generated_objects = []
        
        # Get reference image from first frame
        reference_obj = track_objects[0]
        reference_image = self._load_reference_image(reference_obj)
        
        if reference_image is None:
            logging.warning(f"No reference image available for track")
            return self._generate_fallback_objects(track_objects)
        
        # Generate objects for each frame
        for obj in track_objects:
            try:
                generated_obj = self._generate_single_object(
                    obj, reference_image, category, model
                )
                generated_objects.append(generated_obj)
            except Exception as e:
                logging.warning(f"Failed to generate object for frame {obj.get('frame_index', 0)}: {e}")
                # Use fallback (copy reference)
                fallback_obj = self._create_fallback_object(obj, reference_image)
                generated_objects.append(fallback_obj)
        
        return generated_objects
    
    def _generate_single_object(self, obj: Dict[str, Any], reference_image: np.ndarray,
                              category: str, model: nn.Module) -> Dict[str, Any]:
        """Generate a single object using the appropriate model."""
        
        # Prepare input based on category
        if category == 'human':
            model_input = self._prepare_human_input(obj, reference_image)
        elif category == 'animal':
            model_input = self._prepare_animal_input(obj, reference_image)
        else:
            model_input = self._prepare_other_input(obj, reference_image)
        
        if model_input is None:
            raise ValueError("Failed to prepare model input")
        
        # Generate object
        model.eval()
        with torch.no_grad():
            model_input_tensor = torch.from_numpy(model_input).float().to(self.device)
            if len(model_input_tensor.shape) == 3:
                model_input_tensor = model_input_tensor.unsqueeze(0)  # Add batch dimension
            
            generated_tensor = model(model_input_tensor)
            generated_image = generated_tensor.squeeze(0).cpu().numpy()
            
            # Convert from [-1, 1] to [0, 255]
            generated_image = ((generated_image + 1) * 127.5).astype(np.uint8)
            
            # Transpose from CHW to HWC if needed
            if generated_image.shape[0] == 3:
                generated_image = np.transpose(generated_image, (1, 2, 0))
        
        # Create result object
        generated_obj = {
            'object_id': obj.get('object_id'),
            'frame_index': obj.get('frame_index'),
            'track_id': obj.get('track_id'),
            'category': category,
            'bbox': obj.get('bbox', []),
            'generated_image': generated_image,
            'original_size': reference_image.shape[:2],
            'generation_method': f'{category}_cgan',
            'confidence': obj.get('confidence', 0.0)
        }
        
        return generated_obj
    
    def _prepare_human_input(self, obj: Dict[str, Any], 
                           reference_image: np.ndarray) -> Optional[np.ndarray]:
        """Prepare input for human generation model."""
        keypoints = obj.get('keypoints', [])
        
        if not keypoints:
            return None
        
        # Create pose map from keypoints
        pose_map = self._create_pose_map(keypoints, self.human_input_size)
        
        # Resize reference image
        reference_resized = cv2.resize(reference_image, (self.human_input_size, self.human_input_size))
        
        # Combine pose map and reference image
        # Normalize to [-1, 1]
        reference_norm = (reference_resized.astype(np.float32) / 127.5) - 1.0
        pose_norm = (pose_map.astype(np.float32) / 127.5) - 1.0
        
        # Concatenate along channel dimension and transpose to CHW
        combined_input = np.concatenate([reference_norm, pose_norm], axis=2)
        model_input = np.transpose(combined_input, (2, 0, 1))
        
        return model_input
    
    def _prepare_animal_input(self, obj: Dict[str, Any], 
                            reference_image: np.ndarray) -> Optional[np.ndarray]:
        """Prepare input for animal generation model."""
        keypoints = obj.get('keypoints', [])
        
        if not keypoints:
            return None
        
        # Create pose map from keypoints
        pose_map = self._create_pose_map(keypoints, self.animal_input_size)
        
        # Resize reference image
        reference_resized = cv2.resize(reference_image, (self.animal_input_size, self.animal_input_size))
        
        # Combine pose map and reference image
        reference_norm = (reference_resized.astype(np.float32) / 127.5) - 1.0
        pose_norm = (pose_map.astype(np.float32) / 127.5) - 1.0
        
        combined_input = np.concatenate([reference_norm, pose_norm], axis=2)
        model_input = np.transpose(combined_input, (2, 0, 1))
        
        return model_input
    
    def _prepare_other_input(self, obj: Dict[str, Any], 
                           reference_image: np.ndarray) -> Optional[np.ndarray]:
        """Prepare input for other objects generation model."""
        keypoints = obj.get('keypoints', [])
        
        if not keypoints:
            return None
        
        # For other objects, keypoints are edge/corner features
        feature_map = self._create_feature_map(keypoints, self.other_input_size)
        
        # Resize reference image
        reference_resized = cv2.resize(reference_image, (self.other_input_size, self.other_input_size))
        
        # Combine feature map and reference image
        reference_norm = (reference_resized.astype(np.float32) / 127.5) - 1.0
        feature_norm = (feature_map.astype(np.float32) / 127.5) - 1.0
        
        combined_input = np.concatenate([reference_norm, feature_norm], axis=2)
        model_input = np.transpose(combined_input, (2, 0, 1))
        
        return model_input
    
    def _create_pose_map(self, keypoints: List[List[float]], image_size: int) -> np.ndarray:
        """Create pose map from keypoints."""
        pose_map = np.zeros((image_size, image_size, 1), dtype=np.uint8)
        
        for kp in keypoints:
            if len(kp) >= 2:
                x, y = int(kp[0] * image_size), int(kp[1] * image_size)
                if 0 <= x < image_size and 0 <= y < image_size:
                    cv2.circle(pose_map, (x, y), 3, (255,), -1)
        
        return pose_map
    
    def _create_feature_map(self, keypoints: List[List[float]], image_size: int) -> np.ndarray:
        """Create feature map from edge/corner keypoints."""
        feature_map = np.zeros((image_size, image_size, 1), dtype=np.uint8)
        
        for kp in keypoints:
            if len(kp) >= 2:
                x, y = int(kp[0] * image_size), int(kp[1] * image_size)
                if 0 <= x < image_size and 0 <= y < image_size:
                    cv2.circle(feature_map, (x, y), 2, (255,), -1)
        
        return feature_map
    
    def _load_reference_image(self, obj: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load reference image for an object."""
        # Try to load from cropped image first
        if 'cropped_image' in obj and obj['cropped_image'] is not None:
            return obj['cropped_image']
        
        # Try to load from object directory
        objects_dir = obj.get('objects_dir')
        if objects_dir:
            object_id = obj.get('object_id', '')
            image_path = Path(objects_dir) / f"{object_id}_frame_0000.png"
            if image_path.exists():
                return cv2.imread(str(image_path))
        
        return None
    
    def _generate_fallback_objects(self, track_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate fallback objects when model is not available."""
        fallback_objects = []
        
        reference_obj = track_objects[0]
        reference_image = self._load_reference_image(reference_obj)
        
        if reference_image is None:
            # Create black placeholder
            reference_image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        for obj in track_objects:
            fallback_obj = self._create_fallback_object(obj, reference_image)
            fallback_objects.append(fallback_obj)
        
        return fallback_objects
    
    def _create_fallback_object(self, obj: Dict[str, Any], 
                              reference_image: np.ndarray) -> Dict[str, Any]:
        """Create fallback object (copy of reference)."""
        return {
            'object_id': obj.get('object_id'),
            'frame_index': obj.get('frame_index'),
            'track_id': obj.get('track_id'),
            'category': obj.get('semantic_category', 'other'),
            'bbox': obj.get('bbox', []),
            'generated_image': reference_image.copy(),
            'original_size': reference_image.shape[:2],
            'generation_method': 'fallback_copy',
            'confidence': 0.0
        }

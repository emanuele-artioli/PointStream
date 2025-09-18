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
            # For humans: v_appearance (2048) + p_t (17*3=51) = 2099
            human_vector_size = 2048 + config.get_int('models', 'human_keypoint_channels', 17) * 3
            self.models['human'] = HumanCGAN(
                input_size=self.human_input_size,
                vector_input_size=human_vector_size
            ).to(self.device)
            
            if Path(self.human_model_path).exists():
                checkpoint = torch.load(self.human_model_path, map_location=self.device)
                self.models['human'].generator.load_state_dict(checkpoint['generator'])
                logging.info(f"   ✅ Loaded human model: {self.human_model_path}")
            else:
                logging.warning(f"   ⚠️ Human model not found: {self.human_model_path}")
        except Exception as e:
            logging.error(f"Failed to initialize human model: {e}")
            self.models['human'] = None
        
        # Initialize Animal model
        try:
            # For animals: v_appearance (2048) + p_t (e.g., 20*3=60)
            animal_vector_size = 2048 + config.get_int('models', 'animal_keypoint_channels', 20) * 3
            self.models['animal'] = AnimalCGAN(
                input_size=self.animal_input_size,
                vector_input_size=animal_vector_size
            ).to(self.device)
            
            if Path(self.animal_model_path).exists():
                checkpoint = torch.load(self.animal_model_path, map_location=self.device)
                self.models['animal'].generator.load_state_dict(checkpoint['generator'])
                logging.info(f"   ✅ Loaded animal model: {self.animal_model_path}")
            else:
                logging.warning(f"   ⚠️ Animal model not found: {self.animal_model_path}")
        except Exception as e:
            logging.error(f"Failed to initialize animal model: {e}")
            self.models['animal'] = None
        
        # Initialize Other objects model
        try:
            # For other: v_appearance (2048) + p_t (bbox=4) = 2052
            other_vector_size = 2048 + 4
            self.models['other'] = OtherCGAN(
                input_size=self.other_input_size,
                vector_input_size=other_vector_size
            ).to(self.device)
            
            if Path(self.other_model_path).exists():
                checkpoint = torch.load(self.other_model_path, map_location=self.device)
                self.models['other'].generator.load_state_dict(checkpoint['generator'])
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
                              category: str) -> List[Dict[str, Any]]:
        """
        Generate objects for a complete track using the new vector-based input.
        
        Args:
            track_objects: List of objects in the track (sorted by frame).
            category: Object category (human, animal, other).
            
        Returns:
            List of generated objects for each frame.
        """
        model = self.models.get(category)
        if model is None:
            logging.warning(f"No model available for category: {category}")
            return self._generate_fallback_objects(track_objects)
        
        # Get the static appearance vector from the first object in the track
        # (it should be the same for all objects in the track).
        v_appearance = track_objects[0].get('v_appearance')
        if v_appearance is None:
            logging.warning(f"No appearance vector found for track {track_objects[0].get('track_id')}")
            return self._generate_fallback_objects(track_objects)

        v_appearance_tensor = torch.from_numpy(np.array(v_appearance)).float().to(self.device)

        generated_objects = []
        for obj in track_objects:
            try:
                generated_obj = self._generate_single_object(obj, v_appearance_tensor, category, model)
                generated_objects.append(generated_obj)
            except Exception as e:
                logging.warning(f"Failed to generate object for frame {obj.get('frame_index', 0)}: {e}")
                fallback_obj = self._create_fallback_object(obj)
                generated_objects.append(fallback_obj)
        
        return generated_objects

    def _generate_single_object(self, obj: Dict[str, Any], v_appearance_tensor: torch.Tensor,
                              category: str, model: nn.Module) -> Dict[str, Any]:
        """Generate a single object using the new vector-based model."""
        
        # The pose vector is stored in the 'keypoints' field for backward compatibility
        p_pose_data = obj.get('keypoints', {}).get('points', [])
        if not p_pose_data:
             raise ValueError(f"No pose vector (p_t) found for object {obj.get('object_id')}")

        # For humans/animals, p_t is a list of [x, y, conf]. Flatten it.
        if category in ['human', 'animal']:
            p_t = [coord for kp in p_pose_data for coord in kp]
        else: # For 'other', it's already a flat list [x, y, w, h]
            p_t = p_pose_data

        p_t_tensor = torch.tensor(p_t, dtype=torch.float32).to(self.device)
        
        # Combine vectors and generate object
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
        
        # Create result object
        return {
            'object_id': obj.get('object_id'),
            'frame_index': obj.get('frame_index'),
            'track_id': obj.get('track_id'),
            'category': category,
            'bbox': obj.get('bbox', []),
            'generated_image': generated_image,
            'generation_method': f'{category}_cgan_vector',
            'confidence': obj.get('confidence', 0.0)
        }
    
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

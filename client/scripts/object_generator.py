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
        
        # Initialize appearance feature extractor
        self._initialize_feature_extractor()
        
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
            # For animals: v_appearance (2048) + p_t (12*3=36) = 2084
            # Note: Using 12 keypoints to match actual data from keypoint extraction
            animal_vector_size = 2048 + config.get_int('models', 'animal_keypoint_channels', 12) * 3
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
                generated_obj = self._generate_single_object(obj, v_appearance_tensor, category, model)
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
        if keypoints_method == 'bbox_normalized':
            # Keypoints are bbox coordinates: [x1, y1, x2, y2]
            if len(keypoints_data) >= 4:
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
                              category: str, model: nn.Module) -> Dict[str, Any]:
        """Generate a single object using the new vector-based model with size standardization."""
        
        # Get standard model input size
        if category == 'human':
            standard_size = self.human_input_size
        elif category == 'animal':
            standard_size = self.animal_input_size
        else:
            standard_size = self.other_input_size
        
        # Normalize keypoints to standard size
        try:
            if category in ['human', 'animal']:
                normalized_keypoints = self._normalize_keypoints_to_standard_size(obj, standard_size, category)
                p_t = normalized_keypoints
            else:  # For 'other', use bbox normalized to standard size
                bbox = obj.get('bbox', [0, 0, standard_size, standard_size])
                crop_size = obj.get('crop_size', [standard_size, standard_size])
                
                if len(bbox) >= 4 and len(crop_size) >= 2:
                    # Normalize bbox to standard size
                    scale_x = standard_size / crop_size[0]
                    scale_y = standard_size / crop_size[1]
                    norm_bbox = [bbox[0] * scale_x, bbox[1] * scale_y, 
                               bbox[2] * scale_x, bbox[3] * scale_y]
                    p_t = norm_bbox
                else:
                    p_t = [0, 0, standard_size, standard_size]
                    
        except Exception as e:
            logging.warning(f"Keypoint normalization failed for object {obj.get('object_id')}: {e}")
            # Fallback to original method
            p_pose_data = obj.get('keypoints', {}).get('points', [])
            if not p_pose_data:
                raise ValueError(f"No pose vector (p_t) found for object {obj.get('object_id')}")

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

#!/usr/bin/env python3
"""
PointStream Server Pipeline - Sequential Processing

This module implements the main processing pipeline for the PointStream system.
It processes scenes sequentially through all components for clear debugging and logging.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import json
import shutil
import hashlib

# Suppress warnings before importing other modules
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Using a slow image processor')

# Configure PySceneDetect logging to reduce verbosity
pyscene_logger = logging.getLogger('pyscenedetect')
pyscene_logger.setLevel(logging.WARNING)

# Now import other modules
import torch
import cv2
import numpy as np
from PIL import Image

# Import all PointStream components
try:
    from decorators import log_step, time_step
    from stitcher import Stitcher
    from segmenter import Segmenter 
    from keypointer import Keypointer
    from prompter import Prompter
    from inpainter import Inpainter
    from video_scene_splitter import VideoSceneSplitter
    import config
except ImportError as e:
    logging.error(f"Failed to import PointStream components: {e}")
    print("Error: Cannot import required PointStream components")
    print("Make sure all component files are in the same directory")
    sys.exit(1)

# Check for optional dependencies
try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logging.warning("imagehash not available, disabling frame similarity caching")

class PointStreamProcessor:
    """Sequential processor for scene processing."""
    
    def __init__(self):
        """Initialize processor with all components."""
        logging.info("🚀 Initializing PointStream processor...")
        
        # Initialize all components
        self.stitcher = None
        self.segmenter = None
        self.keypointer = None
        self.prompter = None
        self.inpainter = None
        
        self._initialize_components()
        
        logging.info("✅ PointStream processor initialized successfully")
    
    def _initialize_components(self):
        """Initialize all components."""
        try:
            logging.info("🔧 Loading components...")
            
            # Initialize components in order of dependency
            logging.info("   📐 Initializing Stitcher...")
            self.stitcher = Stitcher()
            
            logging.info("   🎯 Initializing Segmenter...")
            self.segmenter = Segmenter()
            
            logging.info("   🫴 Initializing Keypointer...")
            self.keypointer = Keypointer()
            
            logging.info("   💭 Initializing Prompter...")
            self.prompter = Prompter()
            
            logging.info("   🎨 Initializing Inpainter...")
            self.inpainter = Inpainter()
            
            logging.info("✅ All components loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize components: {e}")
            raise
    
    @log_step
    @time_step(track_processing=True)
    def process_scene(self, scene_data: Dict[str, Any], cached_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a complete scene through the pipeline.
        
        Args:
            scene_data: Scene data from video splitter
            cached_prompt: Pre-cached prompt if available
            
        Returns:
            Processed scene data or Complex scene indicator
        """
        scene_number = scene_data.get('scene_number', 0)
        frames = scene_data.get('frames', [])
        
        if not frames:
            return {
                'scene_type': 'Complex',
                'scene_number': scene_number,
                'error': 'no_frames'
            }
        
        logging.info(f"🎬 Processing scene {scene_number} ({len(frames)} frames)")
        
        try:
            # STEP 1: Stitching
            logging.info(f"🧩 Scene {scene_number}: Step 1/5 - Stitching panorama...")
            step_start = time.time()
            stitching_result = self.stitcher.stitch_scene(frames)
            step_time = time.time() - step_start
            logging.info(f"   ✅ Stitching completed in {step_time:.1f}s")
            
            if stitching_result['scene_type'] == 'Complex':
                logging.info(f"⚠️  Scene {scene_number} classified as Complex (unsuitable for stitching)")
                return {
                    'scene_type': 'Complex',
                    'scene_number': scene_number,
                    'stitching_result': stitching_result,
                    'frames': frames  # Return frames for AV1 encoding
                }
            
            panorama = stitching_result['panorama']
            homographies = stitching_result['homographies']
            
            # STEP 2: Segmentation
            logging.info(f"🎯 Scene {scene_number}: Step 2/5 - Object segmentation...")
            step_start = time.time()
            segmentation_result = self.segmenter.segment_scene(frames, panorama)
            step_time = time.time() - step_start
            objects_count = sum(len(frame_data.get('objects', [])) for frame_data in segmentation_result.get('frames_data', []))
            logging.info(f"   ✅ Segmentation completed in {step_time:.1f}s - Found {objects_count} objects")
            
            # STEP 3: Process keypoints
            logging.info(f"🎯 Scene {scene_number}: Step 3/5 - Keypoint extraction...")
            step_start = time.time()
            keypoint_result = self._process_keypoints(segmentation_result, frames)
            step_time = time.time() - step_start
            keypoints_count = len(keypoint_result.get('objects', []))
            logging.info(f"   ✅ Keypoints completed in {step_time:.1f}s - Processed {keypoints_count} objects")
            
            # STEP 4: Prompt generation
            logging.info(f"💭 Scene {scene_number}: Step 4/5 - Generating prompt...")
            step_start = time.time()
            if cached_prompt:
                prompt_result = {'prompt': cached_prompt, 'method': 'cached'}
                step_time = 0.0
                logging.info(f"   ✅ Using cached prompt (0.0s)")
            else:
                prompt_result = self.prompter.generate_prompt(panorama)
                step_time = time.time() - step_start
                logging.info(f"   ✅ Prompt generated in {step_time:.1f}s")
            
            prompt = prompt_result.get('prompt', 'natural scene with ambient lighting')
            
            # STEP 5: Inpainting
            logging.info(f"🎨 Scene {scene_number}: Step 5/5 - Inpainting background...")
            step_start = time.time()
            panorama_masks = segmentation_result['panorama_data']['masks']
            inpainting_result = self.inpainter.inpaint_background(panorama, panorama_masks, prompt)
            step_time = time.time() - step_start
            method = inpainting_result.get('method', 'unknown')
            logging.info(f"   ✅ Inpainting completed in {step_time:.1f}s using {method}")
            
            # Combine all results
            processed_result = {
                'scene_type': stitching_result['scene_type'],
                'scene_number': scene_number,
                'stitching_result': stitching_result,
                'segmentation_result': segmentation_result,
                'keypoint_result': keypoint_result,
                'prompt_result': prompt_result,
                'inpainting_result': inpainting_result
            }
            
            logging.info(f"🎉 Scene {scene_number} processed successfully")
            return processed_result
            
        except Exception as e:
            logging.error(f"Scene {scene_number} processing failed: {e}")
            return {
                'scene_type': 'Complex',
                'scene_number': scene_number,
                'error': str(e),
                'frames': frames
            }
    
    def _process_keypoints(self, segmentation_result: Dict[str, Any], frames: List[np.ndarray]) -> Dict[str, Any]:
        """Process keypoints for all segmented objects and save object images."""
        try:
            # Extract objects from all frames and add cropped images
            all_objects = []
            for frame_data in segmentation_result['frames_data']:
                objects = frame_data.get('objects', [])
                frame_idx = frame_data.get('frame_index', 0)
                
                # Get the actual frame for cropping
                if frame_idx < len(frames):
                    frame = frames[frame_idx]
                    
                    for obj_idx, obj in enumerate(objects):
                        obj['frame_index'] = frame_idx
                        obj['object_id'] = f"frame_{frame_idx}_obj_{obj_idx}"
                        
                        # Crop object image using bounding box
                        bbox = obj.get('bbox', [])
                        if len(bbox) >= 4:
                            x1, y1, x2, y2 = map(int, bbox[:4])
                            # Ensure coordinates are within frame bounds
                            h, w = frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            if x2 > x1 and y2 > y1:
                                cropped_object = frame[y1:y2, x1:x2]
                                obj['cropped_image'] = cropped_object
                                obj['crop_bbox'] = [x1, y1, x2, y2]
                                obj['crop_size'] = [x2-x1, y2-y1]
                            else:
                                obj['cropped_image'] = None
                                obj['crop_bbox'] = bbox
                                obj['crop_size'] = [0, 0]
                        else:
                            obj['cropped_image'] = None
                            obj['crop_bbox'] = []
                            obj['crop_size'] = [0, 0]
                        
                        all_objects.append(obj)
            
            # Process keypoints for all objects
            if all_objects:
                keypoint_result = self.keypointer.extract_keypoints(all_objects)
                
                # Enhance objects with additional metadata
                enhanced_objects = []
                for obj in keypoint_result.get('objects', []):
                    # Add detailed metadata for saving
                    enhanced_obj = {
                        'object_id': obj.get('object_id', 'unknown'),
                        'frame_index': obj.get('frame_index', 0),
                        'class_name': obj.get('class_name', 'unknown'),
                        'confidence': obj.get('confidence', 0.0),
                        'bbox': obj.get('bbox', []),
                        'crop_bbox': obj.get('crop_bbox', []),
                        'crop_size': obj.get('crop_size', [0, 0]),
                        'keypoints': obj.get('keypoints', []),
                        'segmentation_mask': obj.get('segmentation_mask'),
                        'cropped_image': obj.get('cropped_image'),
                        # Add timing and processing info
                        'processed_at': time.time(),
                        'processing_method': obj.get('processing_method', 'unknown')
                    }
                    enhanced_objects.append(enhanced_obj)
                
                keypoint_result['objects'] = enhanced_objects
                return keypoint_result
            else:
                return {'objects': [], 'processing_time': 0.0, 'method': 'no_objects'}
                
        except Exception as e:
            logging.error(f"Keypoint processing failed: {e}")
            return {'objects': [], 'processing_time': 0.0, 'error': str(e)}


class PointStreamPipeline:
    
    def _process_keypoints(self, segmentation_result: Dict[str, Any], frames: List[np.ndarray]) -> Dict[str, Any]:
        """Process keypoints for all segmented objects and save object images."""
        try:
            # Extract objects from all frames and add cropped images
            all_objects = []
            for frame_data in segmentation_result['frames_data']:
                objects = frame_data.get('objects', [])
                frame_idx = frame_data.get('frame_index', 0)
                
                # Get the actual frame for cropping
                if frame_idx < len(frames):
                    frame = frames[frame_idx]
                    
                    for obj_idx, obj in enumerate(objects):
                        obj['frame_index'] = frame_idx
                        obj['object_id'] = f"frame_{frame_idx}_obj_{obj_idx}"
                        
                        # Crop object image using bounding box
                        bbox = obj.get('bbox', [])
                        if len(bbox) >= 4:
                            x1, y1, x2, y2 = map(int, bbox[:4])
                            # Ensure coordinates are within frame bounds
                            h, w = frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            if x2 > x1 and y2 > y1:
                                cropped_object = frame[y1:y2, x1:x2]
                                obj['cropped_image'] = cropped_object
                                obj['crop_bbox'] = [x1, y1, x2, y2]
                                obj['crop_size'] = [x2-x1, y2-y1]
                            else:
                                obj['cropped_image'] = None
                                obj['crop_bbox'] = bbox
                                obj['crop_size'] = [0, 0]
                        else:
                            obj['cropped_image'] = None
                            obj['crop_bbox'] = []
                            obj['crop_size'] = [0, 0]
                        
                        all_objects.append(obj)
            
            # Process keypoints for all objects
            if all_objects:
                keypoint_result = self.keypointer.extract_keypoints(all_objects)
                
                # Enhance objects with additional metadata
                enhanced_objects = []
                for obj in keypoint_result.get('objects', []):
                    # Add detailed metadata for saving
                    enhanced_obj = {
                        'object_id': obj.get('object_id', 'unknown'),
                        'frame_index': obj.get('frame_index', 0),
                        'class_name': obj.get('class_name', 'unknown'),
                        'confidence': obj.get('confidence', 0.0),
                        'bbox': obj.get('bbox', []),
                        'crop_bbox': obj.get('crop_bbox', []),
                        'crop_size': obj.get('crop_size', [0, 0]),
                        'keypoints': obj.get('keypoints', []),
                        'segmentation_mask': obj.get('segmentation_mask'),
                        'cropped_image': obj.get('cropped_image'),
                        # Add timing and processing info
                        'processed_at': time.time(),
                        'has_keypoints': len(obj.get('keypoints', [])) > 0,
                        'mask_area': obj.get('mask_area', 0)
                    }
                    enhanced_objects.append(enhanced_obj)
                
                return {
                    'objects': enhanced_objects,
                    'total_objects': len(enhanced_objects),
                    'objects_with_keypoints': sum(1 for obj in enhanced_objects if obj['has_keypoints']),
                    'processing_time': keypoint_result.get('processing_time', 0)
                }
            else:
                return {
                    'objects': [],
                    'total_objects': 0,
                    'objects_with_keypoints': 0,
                    'processing_time': 0
                }
                
        except Exception as e:
            logging.error(f"Keypoint processing failed: {e}")
            return {
                'objects': [], 
                'total_objects': 0,
                'objects_with_keypoints': 0,
                'error': str(e)
            }

    def _save_scene_objects(self, scene_data: Dict[str, Any], output_dir: Path):
        """Save individual object images and their data."""
        if not output_dir:
            return
            
        scene_number = scene_data.get('scene_number', 0)
        objects = scene_data.get('keypoint_result', {}).get('objects', [])
        
        if not objects:
            return
            
        # Create objects directory
        objects_dir = output_dir / 'objects' / f'scene_{scene_number:04d}'
        objects_dir.mkdir(parents=True, exist_ok=True)
        
        saved_objects = []
        
        for obj in objects:
            try:
                object_id = obj.get('object_id', 'unknown')
                cropped_image = obj.get('cropped_image')
                
                if cropped_image is not None:
                    # Save cropped object image
                    image_filename = f"{object_id}.png"
                    image_path = objects_dir / image_filename
                    cv2.imwrite(str(image_path), cropped_image)
                    
                    # Create object metadata (without the image data)
                    obj_metadata = {k: v for k, v in obj.items() if k != 'cropped_image' and k != 'segmentation_mask'}
                    obj_metadata['saved_image_path'] = str(image_path)
                    obj_metadata['image_filename'] = image_filename
                    
                    # Save segmentation mask if available
                    seg_mask = obj.get('segmentation_mask')
                    if seg_mask is not None:
                        mask_filename = f"{object_id}_mask.png"
                        mask_path = objects_dir / mask_filename
                        cv2.imwrite(str(mask_path), seg_mask * 255)  # Convert to 0-255 range
                        obj_metadata['saved_mask_path'] = str(mask_path)
                        obj_metadata['mask_filename'] = mask_filename
                    
                    saved_objects.append(obj_metadata)
                    
            except Exception as e:
                logging.warning(f"Failed to save object {obj.get('object_id', 'unknown')}: {e}")
        
        # Save objects metadata file
        if saved_objects:
            metadata_path = objects_dir / 'objects_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump({
                    'scene_number': scene_number,
                    'total_objects': len(saved_objects),
                    'objects': saved_objects,
                    'saved_at': time.time()
                }, f, indent=2)
            
            logging.info(f"Saved {len(saved_objects)} objects for scene {scene_number} to {objects_dir}")


class PointStreamPipeline:
    """Main pipeline orchestrator with sequential processing and intelligent caching."""
    
    def __init__(self, config_file: str = None):
        """Initialize the pipeline with configuration."""
        # Load configuration
        if config_file:
            config.load_config(config_file)
        
        # Pipeline configuration
        self.enable_caching = config.get_bool('pipeline', 'enable_caching', True)
        self.cache_dir = Path(config.get_str('pipeline', 'cache_dir', './cache'))
        
        # Create cache directory
        if self.enable_caching:
            self.cache_dir.mkdir(exist_ok=True)
            self.prompt_cache = self._load_prompt_cache()
        else:
            self.prompt_cache = {}
        
        # Initialize the processor
        self.processor = PointStreamProcessor()
        
        # Statistics
        self.processed_scenes = 0
        self.complex_scenes = 0
        self.cache_hits = 0
        self.processing_times = []
        
        logging.info("🚀 PointStream Pipeline initialized")
        logging.info("🔄 Running in SEQUENTIAL mode")
        logging.info(f"💾 Caching enabled: {self.enable_caching}")
        if self.enable_caching:
            logging.info(f"📁 Cache directory: {self.cache_dir}")
            logging.info(f"🗃️  Cached prompts: {len(self.prompt_cache)}")
    
    def _load_prompt_cache(self) -> Dict[str, str]:
        """Load existing prompt cache from disk."""
        cache_file = self.cache_dir / "prompt_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                logging.info(f"Loaded {len(cache)} cached prompts")
                return cache
            except Exception as e:
                logging.warning(f"Failed to load prompt cache: {e}")
        return {}
    
    def _save_prompt_cache(self):
        """Save prompt cache to disk."""
        if not self.enable_caching:
            return
        
        try:
            cache_file = self.cache_dir / "prompt_cache.json"
            with open(cache_file, 'w') as f:
                json.dump(self.prompt_cache, f, indent=2)
            logging.debug(f"Saved {len(self.prompt_cache)} cached prompts")
        except Exception as e:
            logging.warning(f"Failed to save prompt cache: {e}")
    
    def _calculate_perceptual_hash(self, frame: np.ndarray) -> str:
        """
        Calculate perceptual hash of frame for caching.
        
        Args:
            frame: Input frame
            
        Returns:
            Hash string for caching
        """
        try:
            if IMAGEHASH_AVAILABLE:
                # Convert to PIL and calculate perceptual hash
                if len(frame.shape) == 3:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
                
                pil_image = Image.fromarray(frame_rgb)
                phash = str(imagehash.phash(pil_image))
                return phash
            else:
                # Fallback to MD5 hash of resized frame
                small_frame = cv2.resize(frame, (64, 64))
                frame_bytes = small_frame.tobytes()
                md5_hash = hashlib.md5(frame_bytes).hexdigest()
                return md5_hash
                
        except Exception as e:
            logging.warning(f"Hash calculation failed: {e}")
            # Fallback to timestamp-based hash
            return str(int(time.time() * 1000))
    
    def _get_cached_prompt(self, scene_data: Dict[str, Any]) -> Optional[str]:
        """
        Check cache for existing prompt based on scene's first frame.
        
        Args:
            scene_data: Scene data with frames
            
        Returns:
            Cached prompt if found, None otherwise
        """
        if not self.enable_caching or not scene_data.get('frames'):
            return None
        
        try:
            first_frame = scene_data['frames'][0]
            frame_hash = self._calculate_perceptual_hash(first_frame)
            
            # Check for exact match
            if frame_hash in self.prompt_cache:
                self.cache_hits += 1
                logging.debug(f"Cache hit for scene {scene_data.get('scene_number', '?')}")
                return self.prompt_cache[frame_hash]
            
            # Check for similar hashes (only with imagehash)
            if IMAGEHASH_AVAILABLE:
                current_hash = imagehash.hex_to_hash(frame_hash)
                for cached_hash_str, prompt in self.prompt_cache.items():
                    try:
                        cached_hash = imagehash.hex_to_hash(cached_hash_str)
                        if current_hash - cached_hash < 5:  # Hamming distance < 5
                            self.cache_hits += 1
                            logging.debug(f"Similar cache hit for scene {scene_data.get('scene_number', '?')}")
                            return prompt
                    except:
                        continue
            
            return None
            
        except Exception as e:
            logging.warning(f"Cache lookup failed: {e}")
            return None
    
    def _cache_prompt(self, scene_data: Dict[str, Any], prompt: str):
        """
        Cache a prompt for future use.
        
        Args:
            scene_data: Scene data with frames
            prompt: Generated prompt to cache
        """
        if not self.enable_caching or not scene_data.get('frames'):
            return
        
        try:
            first_frame = scene_data['frames'][0]
            frame_hash = self._calculate_perceptual_hash(first_frame)
            self.prompt_cache[frame_hash] = prompt
            
            # Save cache periodically
            if len(self.prompt_cache) % 10 == 0:
                self._save_prompt_cache()
                
        except Exception as e:
            logging.warning(f"Prompt caching failed: {e}")
    
    def _encode_complex_scene(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode complex scene to AV1 video.
        
        Args:
            scene_data: Scene data with frames
            
        Returns:
            Encoding result
        """
        scene_number = scene_data.get('scene_number', 0)
        frames = scene_data.get('frames', [])
        
        if not frames:
            return {'success': False, 'error': 'no_frames'}
        
        logging.info(f"Encoding complex scene {scene_number} with {len(frames)} frames")
        
        try:
            # Create temporary video file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Get video properties (assuming 24 fps)
            height, width = frames[0].shape[:2]
            fps = 24
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            # Write frames
            for frame in frames:
                out.write(frame)
            out.release()
            
            # Re-encode with AV1 using FFmpeg (SVT-AV1 encoder)
            output_path = temp_path.replace('.mp4', '_av1.mp4')
            ffmpeg_cmd = [
                'ffmpeg', '-i', temp_path,
                '-c:v', 'libsvtav1',
                '-crf', '35',
                '-preset', '6',  # Balanced speed/quality
                '-y',  # Overwrite output
                output_path
            ]
            
            encoding_start = time.time()
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            encoding_time = time.time() - encoding_start
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            if result.returncode == 0:
                file_size = os.path.getsize(output_path)
                logging.info(f"Complex scene {scene_number} encoded in {encoding_time:.3f}s, size: {file_size/1024/1024:.2f}MB")
                
                return {
                    'success': True,
                    'output_path': output_path,
                    'encoding_time': encoding_time,
                    'file_size': file_size,
                    'method': 'av1'
                }
            else:
                logging.error(f"FFmpeg encoding failed: {result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr
                }
                
        except Exception as e:
            logging.error(f"Complex scene encoding failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @log_step
    @time_step(track_processing=True)
    def process_video(self, input_video: str, output_dir: str = None, 
                     enable_saving: bool = True) -> Dict[str, Any]:
        """
        Process complete video through the pipeline.
        
        Args:
            input_video: Path to input video
            output_dir: Output directory for results
            enable_saving: Whether to save results to files
            
        Returns:
            Processing summary
        """
        logging.info(f"🎬 Starting PointStream pipeline processing: {input_video}")
        
        # Setup output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            logging.info(f"📁 Output directory: {output_path}")
        
        # Initialize video scene splitter
        logging.info("🎞️  Initializing video scene splitter...")
        splitter = VideoSceneSplitter(
            input_video=input_video,
            output_dir=None,  # We handle saving differently
            enable_encoding=False  # We just want frames
        )
        logging.info("✅ Video splitter ready")
        
        try:
            # Process scenes
            scene_generator = splitter.process_video_realtime_generator()
            
            for scene_data in scene_generator:
                # Handle completion/error status
                if isinstance(scene_data, dict) and scene_data.get('status') in ['complete', 'error']:
                    break
                
                scene_number = scene_data.get('scene_number', 0)
                scene_start_time = scene_data.get('start_time', 0)
                scene_end_time = scene_data.get('end_time', 0)
                scene_duration = scene_end_time - scene_start_time
                processing_start = time.time()
                
                logging.info(f"📺 Processing scene {scene_number} (#{self.processed_scenes + 1}) - Duration: {scene_duration:.2f}s")
                
                # Check for cached prompt
                cached_prompt = self._get_cached_prompt(scene_data)
                if cached_prompt:
                    logging.info(f"💾 Using cached prompt for scene {scene_number}")
                else:
                    logging.info(f"🆕 Will generate new prompt for scene {scene_number}")
                
                # Process scene sequentially
                logging.info(f"🚀 Starting sequential processing for scene {scene_number}")
                result = self.processor.process_scene(scene_data, cached_prompt)
                
                processing_time = time.time() - processing_start
                self.processing_times.append(processing_time)
                self.processed_scenes += 1
                
                logging.info(f"⏱️  Scene {scene_number} completed in {processing_time:.1f}s - Type: {result.get('scene_type', 'Unknown')}")
                
                # Handle results
                if result.get('scene_type') == 'Complex':
                    # Handle complex scene
                    self.complex_scenes += 1
                    logging.info(f"🎬 Encoding complex scene {scene_number} to video...")
                    encoding_result = self._encode_complex_scene(scene_data)
                    logging.info(f"✅ Scene {scene_number}: Complex -> AV1 encoded")
                    
                    if enable_saving and encoding_result.get('success') and output_dir:
                        # Move encoded file to output directory
                        src_path = encoding_result['output_path']
                        dst_path = output_path / f"scene_{scene_number:04d}_complex.mp4"
                        shutil.move(src_path, str(dst_path))
                        logging.info(f"💾 Complex scene saved: {dst_path}")
                
                else:
                    # Handle successfully processed scene
                    logging.info(f"✨ Scene {scene_number}: {result.get('scene_type', 'Unknown')} -> Processed successfully")
                    
                    # Cache the prompt if it was generated
                    prompt_result = result.get('prompt_result', {})
                    if prompt_result.get('method') != 'cached':
                        prompt = prompt_result.get('prompt')
                        if prompt:
                            self._cache_prompt(scene_data, prompt)
                            logging.info(f"💾 Cached new prompt for scene {scene_number}")
                    
                    # Save results if enabled
                    if enable_saving and output_dir:
                        logging.info(f"💾 Saving scene {scene_number} results...")
                        self._save_scene_results(result, output_path, scene_number)
                        logging.info(f"✅ Scene {scene_number} results saved")
                
                # Progress summary
                avg_time = sum(self.processing_times) / len(self.processing_times)
                logging.info(f"📊 Progress: {self.processed_scenes} scenes completed | Avg: {avg_time:.1f}s/scene | Complex: {self.complex_scenes}")
                logging.info("-" * 80)
            
            # Final cleanup
            splitter.close()
            
            # Save final cache
            if self.enable_caching:
                self._save_prompt_cache()
            
            # Generate summary
            summary = self._generate_processing_summary()
            
            logging.info("PointStream pipeline processing complete")
            return summary
            
        except Exception as e:
            logging.error(f"Pipeline processing failed: {e}")
            raise
    
    def _encode_complex_scene(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode complex scene to video file.
        
        Args:
            scene_data: Scene data with frames
            
        Returns:
            Encoding result
        """
        scene_number = scene_data.get('scene_number', 0)
        frames = scene_data.get('frames', [])
        
        if not frames:
            return {'success': False, 'error': 'no_frames'}
        
        logging.info(f"Encoding complex scene {scene_number} with {len(frames)} frames")
        
        try:
            # Create temporary video file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Get video properties
            height, width = frames[0].shape[:2]
            fps = 24
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            # Write frames
            for frame in frames:
                writer.write(frame)
            
            writer.release()
            
            return {
                'success': True,
                'output_path': temp_path,
                'frame_count': len(frames),
                'fps': fps,
                'resolution': (width, height)
            }
            
        except Exception as e:
            logging.error(f"Complex scene encoding failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _save_scene_objects(self, result: Dict[str, Any], output_dir: Path):
        """Save individual object images and their data."""
        if not output_dir:
            return
            
        scene_number = result.get('scene_number', 0)
        objects = result.get('keypoint_result', {}).get('objects', [])
        
        if not objects:
            return
            
        # Create objects directory
        objects_dir = output_dir / 'objects' / f'scene_{scene_number:04d}'
        objects_dir.mkdir(parents=True, exist_ok=True)
        
        saved_objects = []
        
        for obj in objects:
            try:
                object_id = obj.get('object_id', 'unknown')
                cropped_image = obj.get('cropped_image')
                
                if cropped_image is not None:
                    # Save cropped object image
                    image_filename = f"{object_id}.png"
                    image_path = objects_dir / image_filename
                    cv2.imwrite(str(image_path), cropped_image)
                    
                    # Create object metadata (without the image data)
                    obj_metadata = {k: v for k, v in obj.items() if k != 'cropped_image' and k != 'segmentation_mask'}
                    obj_metadata['saved_image_path'] = str(image_path)
                    obj_metadata['image_filename'] = image_filename
                    
                    # Save segmentation mask if available
                    seg_mask = obj.get('segmentation_mask')
                    if seg_mask is not None:
                        mask_filename = f"{object_id}_mask.png"
                        mask_path = objects_dir / mask_filename
                        cv2.imwrite(str(mask_path), seg_mask * 255)  # Convert to 0-255 range
                        obj_metadata['saved_mask_path'] = str(mask_path)
                        obj_metadata['mask_filename'] = mask_filename
                    
                    saved_objects.append(obj_metadata)
                    
            except Exception as e:
                logging.warning(f"Failed to save object {obj.get('object_id', 'unknown')}: {e}")
        
        # Save objects metadata
        if saved_objects:
            metadata_path = objects_dir / 'objects_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump({
                    'scene_number': scene_number,
                    'objects': saved_objects,
                    'total_objects': len(saved_objects)
                }, f, indent=2)
            
            logging.info(f"Saved {len(saved_objects)} objects for scene {scene_number} to {objects_dir}")
    
    def _save_scene_results(self, result: Dict[str, Any], output_path: Path, scene_number: int):
        """Save scene processing results to files."""
        try:
            # Import profiler for timing data
            from decorators import profiler
            
            # Create subdirectories
            (output_path / "panoramas").mkdir(exist_ok=True)
            (output_path / "results").mkdir(exist_ok=True)
            
            # Save panorama
            stitching_result = result.get('stitching_result', {})
            panorama = stitching_result.get('panorama')
            if panorama is not None:
                panorama_path = output_path / "panoramas" / f"scene_{scene_number:04d}_panorama.jpg"
                cv2.imwrite(str(panorama_path), panorama)
            
            # Save inpainted background
            inpainting_result = result.get('inpainting_result', {})
            inpainted_bg = inpainting_result.get('inpainted_image')
            if inpainted_bg is not None:
                bg_path = output_path / "panoramas" / f"scene_{scene_number:04d}_inpainted.jpg"
                cv2.imwrite(str(bg_path), inpainted_bg)
            
            # Save objects (images, masks, keypoints)
            self._save_scene_objects(result, output_path)
            
            # Get detailed performance data
            performance_summary = profiler.get_overall_summary()
            
            # Enhanced metadata with detailed object and performance information
            keypoint_result = result.get('keypoint_result', {})
            segmentation_result = result.get('segmentation_result', {})
            
            metadata = {
                'scene_number': scene_number,
                'scene_type': result.get('scene_type'),
                'worker_id': result.get('worker_id'),
                'gpu_id': result.get('gpu_id'),
                'processing_timestamp': time.time(),
                
                'stitching': {
                    'scene_type': stitching_result.get('scene_type'),
                    'homographies_count': len(stitching_result.get('homographies', [])),
                    'panorama_shape': list(panorama.shape) if panorama is not None else None,
                    'processing_time': stitching_result.get('processing_time', 0)
                },
                
                'segmentation': {
                    'panorama_objects': len(segmentation_result.get('panorama_data', {}).get('objects', [])),
                    'total_frame_objects': sum(len(fd.get('objects', [])) for fd in segmentation_result.get('frames_data', [])),
                    'frames_processed': len(segmentation_result.get('frames_data', [])),
                    'processing_time': segmentation_result.get('processing_time', 0)
                },
                
                'keypoints': {
                    'total_objects': keypoint_result.get('total_objects', 0),
                    'objects_with_keypoints': keypoint_result.get('objects_with_keypoints', 0),
                    'processing_time': keypoint_result.get('processing_time', 0),
                    'objects_saved': len([obj for obj in keypoint_result.get('objects', []) if obj.get('cropped_image') is not None])
                },
                
                'prompt': {
                    'text': result.get('prompt_result', {}).get('prompt', ''),
                    'method': result.get('prompt_result', {}).get('method', 'unknown'),
                    'processing_time': result.get('prompt_result', {}).get('processing_time', 0)
                },
                
                'inpainting': {
                    'method': inpainting_result.get('method'),
                    'success': inpainting_result.get('success', False),
                    'processing_time': inpainting_result.get('processing_time', 0)
                },
                
                'performance': {
                    'detailed_timings': performance_summary,
                    'total_scene_time': sum(timing['total_time'] for timing in performance_summary.values()),
                    'slowest_operation': max(performance_summary.keys(), key=lambda k: performance_summary[k]['avg_time']) if performance_summary else None
                }
            }
            
            metadata_path = output_path / "results" / f"scene_{scene_number:04d}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Log performance summary for this scene
            profiler.log_scene_summary(scene_number)
            
            logging.info(f"💾 Scene {scene_number} results saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Failed to save scene {scene_number} results: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def _generate_processing_summary(self) -> Dict[str, Any]:
        """Generate final processing summary."""
        total_time = sum(self.processing_times)
        
        return {
            'processed_scenes': self.processed_scenes,
            'complex_scenes': self.complex_scenes,
            'simple_scenes': self.processed_scenes - self.complex_scenes,
            'total_processing_time': total_time,
            'average_processing_time': total_time / max(self.processed_scenes, 1),
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(self.processed_scenes, 1),
            'throughput': self.processed_scenes / total_time if total_time > 0 else 0,
            'caching_enabled': self.enable_caching
        }


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration with better visibility."""
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Setup console handler with better formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # More visible format for terminal output
    formatter = logging.Formatter(
        '🔥 %(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)
    
    # Ensure our module's logger uses the same setup
    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Suppress verbose third-party logging but keep errors
    logging.getLogger('mmpose').setLevel(logging.ERROR)
    logging.getLogger('mmdet').setLevel(logging.ERROR)
    logging.getLogger('mmengine').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    print(f"🚀 PointStream Pipeline Starting - Log Level: {log_level.upper()}")
    print(f"📊 Terminal output will show progress updates...")
    print("=" * 80)


def main():
    """Main entry point for the PointStream pipeline."""
    parser = argparse.ArgumentParser(
        description="PointStream Pipeline - Dual-GPU Modular Video Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process video with dual-GPU workers
    python server_pipeline.py input.mp4
    
    # Custom output directory
    python server_pipeline.py input.mp4 --output-dir ./output
    
    # Process without saving files
    python server_pipeline.py input.mp4 --no-saving
    
    # Custom configuration
    python server_pipeline.py input.mp4 --config config_custom.ini
        """
    )
    
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--output-dir', default='./pointstream_output',
                       help='Output directory for processed results')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--no-saving', action='store_true',
                       help='Disable saving results to files')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate input
    if not Path(args.input_video).exists():
        print(f"Error: Input video not found: {args.input_video}")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = PointStreamPipeline(config_file=args.config)
        
        # Process video
        summary = pipeline.process_video(
            input_video=args.input_video,
            output_dir=args.output_dir if not args.no_saving else None,
            enable_saving=not args.no_saving
        )
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"POINTSTREAM PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Processed scenes: {summary['processed_scenes']}")
        print(f"Simple scenes: {summary['simple_scenes']}")
        print(f"Complex scenes: {summary['complex_scenes']}")
        print(f"Total processing time: {summary['total_processing_time']:.3f}s")
        print(f"Average time per scene: {summary['average_processing_time']:.3f}s")
        print(f"Throughput: {summary['throughput']:.2f} scenes/second")
        print(f"Cache hits: {summary['cache_hits']} ({summary['cache_hit_rate']:.1%})")
        print(f"Workers used: {summary['workers_used']}")
        
        if not args.no_saving:
            print(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

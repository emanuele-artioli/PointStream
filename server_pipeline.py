#!/usr/bin/env python3
"""
PointStream Server Pipeline - Dual-GPU Processing

This module implements the main processing pipeline for the PointStream system.
It coordinates between all components using a dual-GPU worker pool architecture
for optimal performance.
"""

import os
import sys
import time
import logging
import argparse
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor

# Multiprocessing setup for CUDA compatibility
# Note: CUDA requires 'spawn' method for proper initialization in workers

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

worker_assignments = {}


def init_worker():
    """Initialize worker process with dedicated GPU."""
    global worker_instance
    try:
        # Get worker assignment based on process ID
        current_process = mp.current_process()
        worker_id = current_process._identity[0] - 1 if current_process._identity else 0
        gpu_id = worker_id % 2  # Assign GPU 0 or 1
        
        # Set GPU environment
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Create and initialize worker
        worker_instance = PointStreamWorker(worker_id, gpu_id)
        logging.info(f"Worker {worker_id} initialized on GPU {gpu_id}")
        
    except Exception as e:
        logging.error(f"Worker initialization failed: {e}")
        raise


def process_scene_worker(scene_data: Dict[str, Any], cached_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Standalone worker function for multiprocessing."""
    global worker_instance
    if worker_instance is None:
        raise RuntimeError("Worker not initialized")
    
    return worker_instance.process_scene(scene_data, cached_prompt)


# Global worker initialization data
worker_assignments = {}


class PointStreamWorker:
    """Individual worker process for scene processing with dedicated GPU."""
    
    def __init__(self, worker_id: int, gpu_id: int):
        """
        Initialize worker with dedicated GPU.
        
        Args:
            worker_id: Worker identifier (0 or 1)
            gpu_id: GPU device ID to use
        """
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        
        # Set GPU affinity FIRST before importing any GPU libraries
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Initialize all components with models loaded once
        self.stitcher = None
        self.segmenter = None
        self.keypointer = None
        self.prompter = None
        self.inpainter = None
        
        self._initialize_models()
        
        logging.info(f"Worker {worker_id} initialized on GPU {gpu_id}")
    
    def _initialize_models(self):
        """Initialize all ML models once per worker."""
        try:
            logging.info(f"Worker {self.worker_id}: Loading models on GPU {self.gpu_id}")
            
            # Initialize components in order of dependency
            self.stitcher = Stitcher()
            self.segmenter = Segmenter()
            self.keypointer = Keypointer()
            self.prompter = Prompter()
            self.inpainter = Inpainter()
            
            logging.info(f"Worker {self.worker_id}: All models loaded successfully")
            
        except Exception as e:
            logging.error(f"Worker {self.worker_id}: Failed to initialize models: {e}")
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
        
        logging.info(f"Worker {self.worker_id}: Processing scene {scene_number} ({len(frames)} frames)")
        
        try:
            # STEP 1: Stitching (GPU-Heavy)
            stitching_result = self.stitcher.stitch_scene(frames)
            
            if stitching_result['scene_type'] == 'Complex':
                logging.info(f"Scene {scene_number} classified as Complex")
                return {
                    'scene_type': 'Complex',
                    'scene_number': scene_number,
                    'stitching_result': stitching_result,
                    'frames': frames  # Return frames for AV1 encoding
                }
            
            panorama = stitching_result['panorama']
            homographies = stitching_result['homographies']
            
            # STEP 2: Segmentation (GPU-Heavy)
            segmentation_result = self.segmenter.segment_scene(frames, panorama)
            
            # STEP 3: Process keypoints (sequential execution)
            keypoint_result = self._process_keypoints(segmentation_result, frames)
            
            # STEP 4: Prompt generation (sequential execution)
            if cached_prompt:
                prompt_result = {'prompt': cached_prompt, 'method': 'cached'}
            else:
                prompt_result = self.prompter.generate_prompt(panorama)
            
            prompt = prompt_result.get('prompt', 'natural scene with ambient lighting')
            
            # STEP 5: Inpainting (GPU-Heavy, uses segmentation results and prompt)
            panorama_masks = segmentation_result['panorama_data']['masks']
            inpainting_result = self.inpainter.inpaint_background(panorama, panorama_masks, prompt)
            
            # Combine all results
            processed_result = {
                'scene_type': stitching_result['scene_type'],
                'scene_number': scene_number,
                'stitching_result': stitching_result,
                'segmentation_result': segmentation_result,
                'keypoint_result': keypoint_result,
                'prompt_result': prompt_result,
                'inpainting_result': inpainting_result,
                'worker_id': self.worker_id,
                'gpu_id': self.gpu_id
            }
            
            logging.info(f"Worker {self.worker_id}: Scene {scene_number} processed successfully")
            return processed_result
            
        except Exception as e:
            logging.error(f"Worker {self.worker_id}: Scene {scene_number} processing failed: {e}")
            return {
                'scene_type': 'Complex',
                'scene_number': scene_number,
                'error': str(e),
                'frames': frames
            }
    
    def _process_keypoints(self, segmentation_result: Dict[str, Any], frames: List[np.ndarray]) -> Dict[str, Any]:
        """Process keypoints for all segmented objects."""
        try:
            # Extract objects from all frames and add cropped images
            all_objects = []
            for frame_data in segmentation_result['frames_data']:
                objects = frame_data.get('objects', [])
                frame_idx = frame_data.get('frame_index', 0)
                
                # Get the actual frame for cropping
                if frame_idx < len(frames):
                    frame = frames[frame_idx]
                    
                    for obj in objects:
                        obj['frame_index'] = frame_idx
                        
                        # Extract cropped image from bounding box
                        bbox = obj.get('bbox')
                        if bbox and len(bbox) == 4:
                            x, y, w, h = bbox
                            # Ensure coordinates are within frame bounds
                            x = max(0, int(x))
                            y = max(0, int(y))
                            w = min(frame.shape[1] - x, int(w))
                            h = min(frame.shape[0] - y, int(h))
                            
                            if w > 0 and h > 0:
                                cropped = frame[y:y+h, x:x+w].copy()
                                obj['cropped_image'] = cropped
                            else:
                                # Fallback for invalid bbox
                                obj['cropped_image'] = np.zeros((64, 64, 3), dtype=np.uint8)
                        else:
                            # No valid bbox, use placeholder
                            obj['cropped_image'] = np.zeros((64, 64, 3), dtype=np.uint8)
                        
                        all_objects.append(obj)
            
            # Process keypoints
            if all_objects:
                return self.keypointer.extract_keypoints(all_objects)
            else:
                return {'objects': []}
                
        except Exception as e:
            logging.error(f"Keypoint processing failed: {e}")
            return {'objects': [], 'error': str(e)}


class PointStreamPipeline:
    """Main pipeline orchestrator with dual-GPU workers and intelligent caching."""
    
    def __init__(self, config_file: str = None):
        """Initialize the pipeline with configuration."""
        # Load configuration
        if config_file:
            config.load_config(config_file)
        
        # Pipeline configuration
        self.num_workers = config.get_int('pipeline', 'num_workers', 2)
        self.enable_caching = config.get_bool('pipeline', 'enable_caching', True)
        self.cache_dir = Path(config.get_str('pipeline', 'cache_dir', './cache'))
        
        # Create cache directory
        if self.enable_caching:
            self.cache_dir.mkdir(exist_ok=True)
            self.prompt_cache = self._load_prompt_cache()
        else:
            self.prompt_cache = {}
        
        # Worker pool
        self.worker_pool = None
        self.worker_processes = []
        
        # Statistics
        self.processed_scenes = 0
        self.complex_scenes = 0
        self.cache_hits = 0
        self.processing_times = []
        
        logging.info("PointStream Pipeline initialized")
        logging.info(f"Workers: {self.num_workers}")
        logging.info(f"Caching enabled: {self.enable_caching}")
        if self.enable_caching:
            logging.info(f"Cache directory: {self.cache_dir}")
            logging.info(f"Cached prompts: {len(self.prompt_cache)}")
    
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
            
            # Re-encode with AV1 using FFmpeg
            output_path = temp_path.replace('.mp4', '_av1.mp4')
            ffmpeg_cmd = [
                'ffmpeg', '-i', temp_path,
                '-c:v', 'libaom-av1',
                '-crf', '30',
                '-b:v', '0',
                '-strict', 'experimental',
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
    
    def _initialize_workers(self):
        """Initialize worker pool with GPU affinity."""
        try:
            logging.info(f"Initializing {self.num_workers} workers")
            
            # Use spawn context for CUDA compatibility
            ctx = mp.get_context('spawn')
            self.worker_pool = ctx.Pool(
                processes=self.num_workers,
                initializer=init_worker
            )
            
            logging.info("Worker pool initialized successfully")
            
        except Exception as e:
            logging.error(f"Worker pool initialization failed: {e}")
            raise
    
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
        logging.info(f"Starting PointStream pipeline processing: {input_video}")
        
        # Setup output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        
        # Initialize video scene splitter
        splitter = VideoSceneSplitter(
            input_video=input_video,
            output_dir=None,  # We handle saving differently
            enable_encoding=False  # We just want frames
        )
        
        # Initialize workers
        self._initialize_workers()
        
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
                
                logging.info(f"Processing scene {scene_number} (scene #{self.processed_scenes + 1}) - Duration: {scene_duration:.2f}s")
                
                # Check for cached prompt
                cached_prompt = self._get_cached_prompt(scene_data)
                if cached_prompt:
                    logging.debug(f"Using cached prompt for scene {scene_number}")
                else:
                    logging.debug(f"Will generate new prompt for scene {scene_number}")
                
                # Process scene with worker
                if self.worker_pool:
                    # Submit to worker pool using standalone function
                    logging.debug(f"Submitting scene {scene_number} to worker pool")
                    future = self.worker_pool.apply_async(
                        process_scene_worker,
                        (scene_data, cached_prompt)
                    )
                    
                    # Get result
                    try:
                        logging.info(f"Waiting for scene {scene_number} processing to complete...")
                        result = future.get()  # No timeout - let it run as fast as possible
                        processing_time = time.time() - processing_start
                        logging.info(f"Scene {scene_number} completed in {processing_time:.1f}s - Type: {result.get('scene_type', 'Unknown')}")
                    except Exception as e:
                        processing_time = time.time() - processing_start
                        logging.error(f"Scene {scene_number} processing failed after {processing_time:.1f}s: {e}")
                        result = {
                            'scene_type': 'Complex',
                            'scene_number': scene_number,
                            'error': str(e)
                        }
                else:
                    # Fallback: process directly (shouldn't happen in normal operation)
                    worker = PointStreamWorker(0, 0)
                    result = worker.process_scene(scene_data, cached_prompt)
                
                processing_time = time.time() - processing_start
                self.processing_times.append(processing_time)
                self.processed_scenes += 1
                
                # Handle results
                if result.get('scene_type') == 'Complex':
                    # Handle complex scene
                    self.complex_scenes += 1
                    encoding_result = self._encode_complex_scene(scene_data)
                    logging.info(f"Scene {scene_number}: Complex -> AV1 encoded")
                    
                    if enable_saving and encoding_result.get('success') and output_dir:
                        # Move encoded file to output directory
                        src_path = encoding_result['output_path']
                        dst_path = output_path / f"scene_{scene_number:04d}_complex.mp4"
                        shutil.move(src_path, str(dst_path))
                        logging.info(f"Complex scene saved: {dst_path}")
                
                else:
                    # Handle successfully processed scene
                    logging.info(f"Scene {scene_number}: {result.get('scene_type', 'Unknown')} -> Processed")
                    
                    # Cache the prompt if it was generated
                    prompt_result = result.get('prompt_result', {})
                    if prompt_result.get('method') != 'cached':
                        prompt = prompt_result.get('prompt')
                        if prompt:
                            self._cache_prompt(scene_data, prompt)
                    
                    # Save results if enabled
                    if enable_saving and output_dir:
                        self._save_scene_results(result, output_path, scene_number)
                
                # Log progress
                logging.info(f"Scene {scene_number}: Processed in {processing_time:.3f}s")
            
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
            
        finally:
            # Clean up workers
            if self.worker_pool:
                self.worker_pool.close()
                self.worker_pool.join()
    
    def _save_scene_results(self, result: Dict[str, Any], output_path: Path, scene_number: int):
        """Save scene processing results to files."""
        try:
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
            
            # Save metadata
            metadata = {
                'scene_number': scene_number,
                'scene_type': result.get('scene_type'),
                'worker_id': result.get('worker_id'),
                'gpu_id': result.get('gpu_id'),
                'stitching': {
                    'scene_type': stitching_result.get('scene_type'),
                    'homographies_count': len(stitching_result.get('homographies', []))
                },
                'segmentation': {
                    'panorama_objects': len(result.get('segmentation_result', {}).get('panorama_data', {}).get('objects', [])),
                    'total_frame_objects': sum(len(fd.get('objects', [])) for fd in result.get('segmentation_result', {}).get('frames_data', []))
                },
                'keypoints': {
                    'total_objects': len(result.get('keypoint_result', {}).get('objects', []))
                },
                'prompt': result.get('prompt_result', {}).get('prompt', ''),
                'inpainting': {
                    'method': inpainting_result.get('method'),
                    'success': inpainting_result.get('success', False)
                }
            }
            
            metadata_path = output_path / "results" / f"scene_{scene_number:04d}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
        except Exception as e:
            logging.error(f"Failed to save scene {scene_number} results: {e}")
    
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
            'workers_used': self.num_workers,
            'caching_enabled': self.enable_caching
        }


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


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

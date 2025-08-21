#!/usr/bin/env python3
"""
Video Scene Splitter - Batch Processing Version

This script processes videos in batches to simulate real-time streaming scenarios,
detecting scene cuts and saving each scene as a separate AV1 video using FFmpeg.
All parameters are configurable through config.ini file.

Features:
- Real-time batch processing simulation
- Detects scene cuts using PySceneDetect
- Enforces maximum scene length
- Outputs scenes in AV1 format for optimal compression
- Preserves original video properties
- Full configuration support
"""

import os
import sys
import subprocess
import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import tempfile
import shutil

# Configure PySceneDetect logging BEFORE importing it to reduce verbosity
pyscene_logger = logging.getLogger('pyscenedetect')
pyscene_logger.setLevel(logging.WARNING)  # Only show warnings and errors

try:
    from scenedetect import detect, ContentDetector, open_video, FrameTimecode
    from scenedetect.detectors import AdaptiveDetector, HistogramDetector, HashDetector, ThresholdDetector
    from scenedetect.scene_manager import SceneManager, Interpolation
    from scenedetect.stats_manager import StatsManager
    from scenedetect.backends import AVAILABLE_BACKENDS
    import cv2
    from . import config
    from .decorators import time_step  # Import decorators for timing
    
    # Configure PySceneDetect logging AFTER import
    import logging
    pyscene_logger = logging.getLogger('pyscenedetect')
    pyscene_logger.setLevel(logging.ERROR)  # Only show errors
    for handler in pyscene_logger.handlers:
        handler.setLevel(logging.ERROR)
    
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure you have activated the pointstream environment and installed all dependencies.")
    sys.exit(1)


class VideoSceneSplitter:
    def __init__(self, input_video: str, output_dir: str = None, 
                 batch_size: int = None, max_scene_frames: int = None, config_file: str = None,
                 enable_encoding: bool = True, detector_type: str = None, video_backend: str = None):
        """
        Initialize the video scene splitter with configuration support.
        
        Args:
            input_video: Path to input video file
            output_dir: Directory to save scene videos (only used if enable_encoding=True)
            batch_size: Number of frames to process in each batch
            max_scene_frames: Maximum frames per scene before forced cut
            config_file: Path to configuration file
            enable_encoding: Whether to encode scenes to files (for debugging) or just yield frame data
            detector_type: Type of scene detector ('content', 'adaptive', 'histogram', 'hash', 'threshold', 'multi')
            video_backend: Video backend to use ('opencv', 'pyav', 'moviepy', 'auto')
        """
        # Load configuration
        if config_file:
            config.load_config(config_file)
        
        self.enable_encoding = enable_encoding
        
        self.input_video = Path(input_video)
        if not self.input_video.exists():
            raise FileNotFoundError(f"Input video not found: {input_video}")
        
        # Set detector type
        self.detector_type = detector_type or config.get_str('scene_detection', 'detector_type', 'content')
        
        # Set video backend
        self.video_backend = video_backend or config.get_str('encoding', 'backend', 'auto')
        
        # Initialize stats manager (OPTIONAL - for analysis only, not production)
        self.use_stats = config.get_bool('scene_detection', 'use_stats_manager', False)
        self.stats_manager = None
        self.stats_file_path = None
        
        # WARNING: Stats caching can give false performance readings for repeated videos
        # In production, always set use_stats_manager = false to get true processing times
        if self.use_stats:
            # Always create a unique stats file to avoid caching false positives
            import time
            timestamp = int(time.time())
            stats_file = self.input_video.parent / f"{self.input_video.stem}_stats_{timestamp}.csv"
            self.stats_manager = StatsManager()
            # DO NOT load existing stats to avoid false performance benefits
            self.stats_file_path = str(stats_file)
            logging.warning("Stats collection enabled - this may slow down processing and should be disabled in production")
        else:
            logging.info("Stats collection disabled - using production mode for accurate timing")
        
        # Set output directory (only used if encoding is enabled)
        if output_dir is None and enable_encoding:
            pattern = config.get_str('scene_detection', 'default_output_pattern', '{input_stem}_scenes')
            self.output_dir = self.input_video.parent / pattern.format(input_stem=self.input_video.stem)
        elif output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = None
        
        if self.output_dir and enable_encoding:
            self.output_dir.mkdir(exist_ok=True)
        
        # Get batch processing configuration
        self.batch_size = batch_size or config.get_int('scene_detection', 'batch_size', 100)
        self.max_scene_frames = max_scene_frames or config.get_int('scene_detection', 'default_max_frames', 1000)
        self.overlap_frames = config.get_int('scene_detection', 'overlap_frames', 10)
        
        # Get video properties
        self.video_fps, self.total_frames = self._get_video_properties()
        
        # Scene detection configuration
        self.scene_threshold = config.get_float('scene_detection', 'threshold', 30.0)
        self.min_scene_len = config.get_int('scene_detection', 'min_scene_len', 15)
        self.luma_only = config.get_bool('scene_detection', 'luma_only', False)
        
        # Enhanced detection parameters
        self.enable_downscaling = config.get_bool('scene_detection', 'enable_downscaling', True)
        self.downscale_factor = config.get_int('scene_detection', 'downscale_factor', 0)  # 0 = auto
        self.adaptive_window = config.get_int('scene_detection', 'adaptive_window_frames', 60)
        self.histogram_bins = config.get_int('scene_detection', 'histogram_bins', 256)
        self.hash_size = config.get_int('scene_detection', 'hash_size', 8)
        
        # Interpolation method for frame scaling/processing
        interpolation_str = config.get_str('scene_detection', 'interpolation_method', 'linear')
        self.interpolation_method = getattr(Interpolation, interpolation_str.upper(), Interpolation.LINEAR)
        
        # Note: event_buffer_length is no longer configurable in PySceneDetect 0.6+
        # It's automatically determined by each detector algorithm
        
        # Multi-detector weights (for when detector_type='multi')
        self.detector_weights = {
            'content': config.get_float('scene_detection', 'content_weight', 1.0),
            'adaptive': config.get_float('scene_detection', 'adaptive_weight', 0.8),
            'histogram': config.get_float('scene_detection', 'histogram_weight', 0.6),
            'hash': config.get_float('scene_detection', 'hash_weight', 0.9)
        }
        
        # Initialize video stream with configurable backend
        self.video_stream = self._initialize_video_stream()
        
        # Processing state for live simulation
        self.processed_scenes = []
        self.scene_counter = 0
        self.pending_scene_start = None
        self.pending_frames_buffer = []  # Frames not yet assigned to a scene
        self.last_batch_end_frame = None  # Track where last batch ended
        self.scene_boundaries = []  # Confirmed scene boundaries from current batch
        
        # Time tracking - SEPARATE core processing from optional operations
        self.core_processing_times = []     # CORE: Only scene detection (production-critical)
        self.stats_collection_times = []    # OPTIONAL: Statistics collection overhead
        self.encoding_times = []            # OPTIONAL: Video encoding (not needed in production)
        self.frame_extraction_times = []    # CORE: Frame extraction time (needed for next stage)
        
        # Open video capture for frame extraction (keep for compatibility)
        self.cap = cv2.VideoCapture(str(self.input_video))
        
        logging.info(f"Input video: {self.input_video}")
        if self.output_dir:
            logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"Video FPS: {self.video_fps}")
        logging.info(f"Total frames: {self.total_frames}")
        logging.info(f"Batch size: {self.batch_size} frames")
        logging.info(f"Max scene length: {self.max_scene_frames} frames")
        logging.info(f"Overlap frames: {self.overlap_frames}")
        logging.info(f"Encoding enabled: {self.enable_encoding}")

    def _get_video_properties(self) -> Tuple[float, int]:
        """Get video FPS and total frame count."""
        cap = cv2.VideoCapture(str(self.input_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return fps, frame_count

    def _extract_scene_frames(self, start_time: float, end_time: float) -> List[any]:
        """
        Extract frames for a scene from the video.
        
        Args:
            start_time: Scene start time in seconds
            end_time: Scene end time in seconds
            
        Returns:
            List of frame arrays (numpy arrays)
        """
        start_frame = int(start_time * self.video_fps)
        end_frame = int(end_time * self.video_fps)
        
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_num in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame.copy())  # Make a copy to avoid reference issues
        
        return frames

    @time_step(track_processing=True)
    def _detect_scene_cuts_in_batch(self, start_frame: int, end_frame: int) -> List[float]:
        """
        Enhanced scene cut detection with multiple detector support and performance optimizations.
        
        Args:
            start_frame: Starting frame number of current batch
            end_frame: Ending frame number of current batch
            
        Returns:
            List of scene cut timestamps within this batch
        """
        # Convert to time for scene detection
        start_time = start_frame / self.video_fps
        end_time = end_frame / self.video_fps
        
        logging.debug(f"Detecting scene cuts in batch: {start_time:.2f}s - {end_time:.2f}s")
        
        try:
            if self.detector_type == 'multi':
                return self._detect_with_multiple_detectors(start_time, end_time)
            else:
                return self._detect_with_single_detector(start_time, end_time)
                
        except Exception as e:
            logging.error(f"Error detecting scene cuts in batch: {e}")
            return []

    def _detect_with_single_detector(self, start_time: float, end_time: float) -> List[float]:
        """Detect scenes using a single detector algorithm."""
        detector = self._create_detector()
        
        # Optional: Stats collection timing
        stats_start_time = time.time() if self.use_stats else None
        
        # Use SceneManager for better control and performance
        scene_manager = SceneManager(stats_manager=self.stats_manager if self.use_stats else None)
        scene_manager.add_detector(detector)
        
        # Optional: Track stats collection overhead
        if self.use_stats and stats_start_time:
            stats_setup_time = time.time() - stats_start_time
            self.stats_collection_times.append(stats_setup_time)
        
        # Enable downscaling for performance if configured
        if self.enable_downscaling:
            if self.downscale_factor == 0:
                # Auto-calculate downscale factor
                from scenedetect.scene_manager import compute_downscale_factor
                scene_manager.auto_downscale = True
            else:
                scene_manager.downscale = self.downscale_factor
                
            # Set interpolation method for downscaling
            scene_manager.interpolation = self.interpolation_method
            logging.debug(f"Using {self.interpolation_method} interpolation for downscaling")
        
        # CORE: Scene detection (this is what we need to be fast in production)
        # PySceneDetect 0.6+ API: Use video seeking and duration instead of start_time/end_time
        if start_time > 0:
            self.video_stream.seek(start_time)
        
        # Calculate duration from start_time to end_time
        duration_seconds = end_time - start_time
        duration = FrameTimecode(duration_seconds, fps=self.video_fps)
        
        scene_manager.detect_scenes(
            video=self.video_stream,
            duration=duration,
            show_progress=False
        )
        
        scene_list = scene_manager.get_scene_list()
        
        # Extract only the cut points (start times of scenes after the first)
        cut_times = []
        for i, scene in enumerate(scene_list):
            scene_start = scene[0].get_seconds()
            # Skip the first scene as it represents continuation from previous batch
            if i > 0 and start_time <= scene_start < end_time:
                cut_times.append(scene_start)
        
        return cut_times

    def _detect_with_multiple_detectors(self, start_time: float, end_time: float) -> List[float]:
        """Detect scenes using multiple detectors and combine results."""
        detector_types = ['content', 'adaptive', 'histogram', 'hash']
        all_cuts = {}
        
        for det_type in detector_types:
            if self.detector_weights.get(det_type, 0) > 0:
                try:
                    detector = self._create_detector(det_type)
                    scene_manager = SceneManager(stats_manager=self.stats_manager)
                    scene_manager.add_detector(detector)
                    
                    if self.enable_downscaling:
                        scene_manager.auto_downscale = True
                    
                    # Use the same updated API as single detector
                    if start_time > 0:
                        self.video_stream.seek(start_time)
                    
                    duration_seconds = end_time - start_time
                    duration = FrameTimecode(duration_seconds, fps=self.video_fps)
                    
                    scene_manager.detect_scenes(
                        video=self.video_stream,
                        duration=duration,
                        show_progress=False
                    )
                    
                    scene_list = scene_manager.get_scene_list()
                    cuts = []
                    for i, scene in enumerate(scene_list):
                        scene_start = scene[0].get_seconds()
                        if i > 0 and start_time <= scene_start < end_time:
                            cuts.append(scene_start)
                    
                    weight = self.detector_weights[det_type]
                    for cut_time in cuts:
                        if cut_time not in all_cuts:
                            all_cuts[cut_time] = 0
                        all_cuts[cut_time] += weight
                    
                    logging.debug(f"{det_type} detector found {len(cuts)} cuts")
                    
                except Exception as e:
                    logging.warning(f"Error with {det_type} detector: {e}")
        
        # Filter cuts based on combined weights
        min_weight = config.get_float('scene_detection', 'multi_detector_threshold', 1.0)
        final_cuts = [cut_time for cut_time, weight in all_cuts.items() if weight >= min_weight]
        final_cuts.sort()
        
        logging.debug(f"Multi-detector approach: {len(final_cuts)} final cuts from {len(all_cuts)} candidates")
        return final_cuts

    def _process_batch_realtime(self, batch_start: int, batch_end: int) -> List[Dict[str, Any]]:
        """
        Process a single batch in true live streaming simulation mode.
        
        In live streaming, we can only work with data available up to the current batch.
        We detect scene cuts in this batch, but can only finalize scenes when we're
        certain about the boundaries (either due to cuts or max duration limits).
        
        Args:
            batch_start: Starting frame number
            batch_end: Ending frame number
            
        Returns:
            List of scene dictionaries with frames, timestamps, and metadata
        """
        logging.info(f"Processing batch: frames {batch_start} - {batch_end}")
        
        # Track CORE processing time (scene detection only)
        core_processing_start = time.time()
        
        batch_start_time = batch_start / self.video_fps
        batch_end_time = batch_end / self.video_fps
        completed_scenes = []
        
        # Detect scene cuts in this batch (CORE processing)
        scene_cuts = self._detect_scene_cuts_in_batch(batch_start, batch_end)
        
        # Track core scene detection time
        core_scene_detection_time = time.time() - core_processing_start
        
        # If this is the first batch, initialize pending scene
        if self.pending_scene_start is None:
            self.pending_scene_start = 0.0
            logging.info("Starting first scene at video beginning")
        
        # Process any scene cuts found in this batch
        for cut_time in scene_cuts:
            # Finalize the pending scene up to this cut
            if self.pending_scene_start is not None:
                scene_data = self._create_scene_data_no_encoding(
                    self.pending_scene_start,
                    cut_time,
                    self.scene_counter + 1
                )
                if scene_data:
                    completed_scenes.append(scene_data)
                    self.scene_counter += 1
                    self.processed_scenes.append((self.pending_scene_start, cut_time))
                    logging.info(f"Finalized scene {self.scene_counter}: {self.pending_scene_start:.2f}s - {cut_time:.2f}s")
                
                # Start new scene after this cut
                self.pending_scene_start = cut_time
        
        # Check if pending scene has reached maximum duration
        if self.pending_scene_start is not None:
            pending_duration = batch_end_time - self.pending_scene_start
            max_duration = self.max_scene_frames / self.video_fps
            
            if pending_duration >= max_duration:
                # Force cut at the end of this batch due to length limit
                scene_data = self._create_scene_data_no_encoding(
                    self.pending_scene_start,
                    batch_end_time,
                    self.scene_counter + 1
                )
                if scene_data:
                    completed_scenes.append(scene_data)
                    self.scene_counter += 1
                    self.processed_scenes.append((self.pending_scene_start, batch_end_time))
                    logging.info(f"Force-cut scene {self.scene_counter} at max duration: {self.pending_scene_start:.2f}s - {batch_end_time:.2f}s")
                
                # Start new scene at the end of this batch
                self.pending_scene_start = batch_end_time
        
        # Store information about this batch for next iteration
        self.last_batch_end_frame = batch_end
        
        # Track CORE processing time (scene detection + scene boundary logic)
        core_processing_time = time.time() - core_processing_start
        self.core_processing_times.append(core_processing_time)
        
        # Now handle optional operations separately
        final_scenes = []
        for scene_data in completed_scenes:
            # Optional: Frame extraction timing (needed for next pipeline stage)
            if scene_data:
                frame_extraction_start = time.time()
                # Frame extraction is already done in _create_scene_data_no_encoding
                frame_extraction_time = time.time() - frame_extraction_start
                self.frame_extraction_times.append(frame_extraction_time)
            
            # Optional: Add encoding if enabled (separately timed)
            if self.enable_encoding and self.output_dir:
                encoding_start = time.time()
                output_file = self._save_scene_with_config(
                    scene_data['start_time'], 
                    scene_data['end_time'], 
                    scene_data['scene_number']
                )
                encoding_time = time.time() - encoding_start
                self.encoding_times.append(encoding_time)
                scene_data['encoded_file'] = output_file
                scene_data['encoding_time'] = encoding_time
            
            final_scenes.append(scene_data)
        
        logging.info(f"CORE processing time: {core_processing_time:.3f}s (production-critical)")
        if scene_cuts:
            logging.info(f"Found {len(scene_cuts)} scene cuts in batch at: {[f'{t:.2f}s' for t in scene_cuts]}")
        else:
            logging.info(f"No scene cuts in batch, pending scene continues")
        
        return final_scenes

    def _create_scene_data(self, start_time: float, end_time: float, scene_num: int) -> Optional[Dict[str, Any]]:
        """
        Create scene data dictionary with frames and metadata.
        
        Args:
            start_time: Scene start time in seconds
            end_time: Scene end time in seconds
            scene_num: Scene number
            
        Returns:
            Dictionary with scene data or None if scene is too short
        """
        duration = end_time - start_time
        
        # Apply quality control filters
        min_scene_duration = config.get_float('scene_detection', 'min_scene_duration', 0.5)
        if duration < min_scene_duration:
            logging.warning(f"Scene {scene_num} too short ({duration:.2f}s), skipping")
            return None
        
        # Extract frames for this scene
        frames = self._extract_scene_frames(start_time, end_time)
        
        scene_data = {
            'scene_number': scene_num,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'frames': frames,
            'frame_count': len(frames),
            'fps': self.video_fps,
            'processing_timestamp': time.time()
        }
        
        # Optional encoding for debugging/visualization
        if self.enable_encoding and self.output_dir:
            encoding_start = time.time()
            output_file = self._save_scene_with_config(start_time, end_time, scene_num)
            encoding_time = time.time() - encoding_start
            self.encoding_times.append(encoding_time)
            scene_data['encoded_file'] = output_file
            scene_data['encoding_time'] = encoding_time
        
        return scene_data

    def _create_scene_data_no_encoding(self, start_time: float, end_time: float, scene_num: int) -> Optional[Dict[str, Any]]:
        """
        Create scene data dictionary with frames and metadata, without encoding.
        This separates pure processing time from encoding time.
        
        Args:
            start_time: Scene start time in seconds
            end_time: Scene end time in seconds
            scene_num: Scene number
            
        Returns:
            Dictionary with scene data or None if scene is too short
        """
        duration = end_time - start_time
        
        # Apply quality control filters
        min_scene_duration = config.get_float('scene_detection', 'min_scene_duration', 0.5)
        if duration < min_scene_duration:
            logging.warning(f"Scene {scene_num} too short ({duration:.2f}s), skipping")
            return None
        
        # Extract frames for this scene
        frames = self._extract_scene_frames(start_time, end_time)
        
        scene_data = {
            'scene_number': scene_num,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'frames': frames,
            'frame_count': len(frames),
            'fps': self.video_fps,
            'processing_timestamp': time.time()
        }
        
        return scene_data

    def _save_scene_with_config(self, start_time: float, end_time: float, scene_num: int) -> Optional[str]:
        """
        Save a scene using configuration settings, with quality control.
        
        Args:
            start_time: Scene start time in seconds
            end_time: Scene end time in seconds
            scene_num: Scene number for filename
            
        Returns:
            Path to saved scene file, or None if failed
        """
        duration = end_time - start_time
        
        # Apply quality control filters
        min_scene_duration = config.get_float('scene_detection', 'min_scene_duration', 0.5)
        max_scene_duration = config.get_float('scene_detection', 'max_scene_duration', 300)
        
        if duration < min_scene_duration:
            logging.warning(f"Scene {scene_num} too short ({duration:.2f}s), skipping")
            return None
        
        if max_scene_duration > 0 and duration > max_scene_duration:
            logging.warning(f"Scene {scene_num} too long ({duration:.2f}s), may need splitting")
        
        # Generate output filename using configuration
        filename_pattern = config.get_str('encoding', 'scene_filename_pattern', 'scene_{number:04d}.{extension}')
        container_format = config.get_str('encoding', 'container_format', 'mp4')
        
        filename = filename_pattern.format(number=scene_num, extension=container_format)
        output_file = self.output_dir / filename
        
        # Build FFmpeg command using configuration
        ffmpeg_binary = config.get_str('encoding', 'ffmpeg_binary', 'ffmpeg')
        cmd = [ffmpeg_binary]
        
        # Add extra input arguments
        extra_input_args = config.get_list('encoding', 'extra_input_args', [])
        cmd.extend(extra_input_args)
        
        # Input and timing
        cmd.extend(['-i', str(self.input_video)])
        cmd.extend(['-ss', str(start_time), '-t', str(duration)])
        
        # Video encoding
        video_codec = config.get_str('encoding', 'video_codec', 'libsvtav1')
        crf = config.get_int('encoding', 'crf', 30)
        preset = config.get_str('encoding', 'preset', '6')
        
        cmd.extend(['-c:v', video_codec])
        cmd.extend(['-crf', str(crf)])
        cmd.extend(['-preset', str(preset)])
        
        # Pixel format and color
        pixel_format = config.get_str('encoding', 'pixel_format', 'yuv420p')
        color_range = config.get_str('encoding', 'color_range', 'tv')
        color_space = config.get_str('encoding', 'color_space', 'bt709')
        
        if pixel_format:
            cmd.extend(['-pix_fmt', pixel_format])
        if color_range:
            cmd.extend(['-color_range', color_range])
        if color_space:
            cmd.extend(['-colorspace', color_space])
        
        # Audio encoding
        audio_codec = config.get_str('encoding', 'audio_codec', 'libopus')
        if audio_codec:
            audio_bitrate = config.get_str('encoding', 'audio_bitrate', '128k')
            audio_sample_rate = config.get_int('encoding', 'audio_sample_rate', 48000)
            audio_channels = config.get_int('encoding', 'audio_channels', 2)
            
            cmd.extend(['-c:a', audio_codec])
            cmd.extend(['-b:a', audio_bitrate])
            cmd.extend(['-ar', str(audio_sample_rate)])
            cmd.extend(['-ac', str(audio_channels)])
        else:
            cmd.extend(['-an'])
        
        # Threading
        ffmpeg_threads = config.get_int('encoding', 'ffmpeg_threads', 0)
        if ffmpeg_threads > 0:
            cmd.extend(['-threads', str(ffmpeg_threads)])
        
        # Container settings
        faststart = config.get_bool('encoding', 'faststart', True)
        if faststart:
            cmd.extend(['-movflags', '+faststart'])
        
        # Misc
        cmd.extend(['-avoid_negative_ts', 'make_zero'])
        extra_output_args = config.get_list('encoding', 'extra_output_args', [])
        cmd.extend(extra_output_args)
        cmd.extend(['-y', str(output_file)])
        
        # Execute with retries
        retry_count = config.get_int('encoding', 'retry_failed', 1)
        timeout = config.get_int('encoding', 'ffmpeg_timeout', 300)
        
        for attempt in range(retry_count + 1):
            try:
                logging.info(f"Encoding scene {scene_num}: {start_time:.2f}s - {end_time:.2f}s (attempt {attempt + 1})")
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=True,
                    timeout=timeout
                )
                
                # Verify output if configured
                verify_output = config.get_bool('scene_detection', 'verify_output_files', True)
                if verify_output:
                    if not self._verify_output_file(str(output_file)):
                        if attempt < retry_count:
                            logging.warning(f"Verification failed, retrying...")
                            continue
                        else:
                            logging.error(f"Scene {scene_num} failed verification after all retries")
                            return None
                
                logging.info(f"Successfully saved: {output_file}")
                return str(output_file)
                
            except subprocess.CalledProcessError as e:
                logging.error(f"FFmpeg error on attempt {attempt + 1}: {e}")
                verbose_ffmpeg = config.get_bool('logging', 'verbose_ffmpeg', False)
                if verbose_ffmpeg:
                    logging.error(f"FFmpeg stderr: {e.stderr}")
                
                if attempt >= retry_count:
                    continue_on_error = config.get_bool('encoding', 'continue_on_error', False)
                    if not continue_on_error:
                        raise
                    else:
                        logging.error(f"Scene {scene_num} failed after all retries, continuing...")
                        return None
            
            except subprocess.TimeoutExpired:
                logging.error(f"FFmpeg timeout on attempt {attempt + 1}")
                if attempt >= retry_count:
                    return None
        
        return None

    def _verify_output_file(self, output_file: str) -> bool:
        """Verify output file meets quality requirements."""
        try:
            file_path = Path(output_file)
            if not file_path.exists():
                return False
            
            # Check file size
            file_size_kb = file_path.stat().st_size / 1024
            min_file_size_kb = config.get_float('encoding', 'min_file_size_kb', 10)
            max_file_size_mb = config.get_float('encoding', 'max_file_size_mb', 1000)
            
            if file_size_kb < min_file_size_kb:
                logging.warning(f"File too small: {file_size_kb:.1f}KB")
                return False
            
            max_size_kb = max_file_size_mb * 1024
            if max_size_kb > 0 and file_size_kb > max_size_kb:
                logging.warning(f"File too large: {file_size_kb:.1f}KB")
                return False
            
            # Check video integrity if configured
            check_integrity = config.get_bool('scene_detection', 'check_video_integrity', True)
            if check_integrity:
                result = subprocess.run(
                    ['ffmpeg', '-v', 'error', '-i', output_file, '-f', 'null', '-'],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode != 0:
                    logging.warning(f"Video integrity check failed: {result.stderr}")
                    return False
            
            return True
        except Exception as e:
            logging.error(f"Error verifying output file: {e}")
            return False

    def _initialize_video_stream(self):
        """Initialize video stream with configurable backend."""
        try:
            if self.video_backend == 'auto':
                # Try backends in order of preference
                for backend in ['pyav', 'opencv', 'moviepy']:
                    if backend in AVAILABLE_BACKENDS:
                        try:
                            video_stream = open_video(str(self.input_video), backend=backend)
                            logging.info(f"Using video backend: {backend}")
                            return video_stream
                        except Exception as e:
                            logging.warning(f"Failed to use {backend} backend: {e}")
                            continue
            else:
                if self.video_backend in AVAILABLE_BACKENDS:
                    video_stream = open_video(str(self.input_video), backend=self.video_backend)
                    logging.info(f"Using video backend: {self.video_backend}")
                    return video_stream
                else:
                    logging.warning(f"Backend {self.video_backend} not available, falling back to OpenCV")
            
            # Fallback to OpenCV
            video_stream = open_video(str(self.input_video), backend='opencv')
            logging.info("Using video backend: opencv (fallback)")
            return video_stream
            
        except Exception as e:
            logging.error(f"Failed to initialize video stream: {e}")
            raise

    def _create_detector(self, detector_type: str = None):
        """Create scene detector based on configuration."""
        detector_type = detector_type or self.detector_type
        
        # Create base detector
        if detector_type == 'content':
            base_detector = ContentDetector(
                threshold=self.scene_threshold,
                min_scene_len=self.min_scene_len,
                luma_only=self.luma_only
            )
        elif detector_type == 'adaptive':
            base_detector = AdaptiveDetector(
                adaptive_threshold=self.scene_threshold,
                min_scene_len=self.min_scene_len,
                window_width=self.adaptive_window,
                luma_only=self.luma_only
            )
        elif detector_type == 'histogram':
            base_detector = HistogramDetector(
                threshold=self.scene_threshold,
                min_scene_len=self.min_scene_len,
                bins=self.histogram_bins
            )
        elif detector_type == 'hash':
            try:
                # Try new API without hash_size parameter
                base_detector = HashDetector(
                    threshold=self.scene_threshold,
                    min_scene_len=self.min_scene_len
                )
            except TypeError:
                # Fallback to older API if new one doesn't work
                try:
                    base_detector = HashDetector(
                        threshold=self.scene_threshold,
                        min_scene_len=self.min_scene_len,
                        hash_size=self.hash_size
                    )
                except Exception as e:
                    logging.warning(f"Error with hash detector: {e}")
                    logging.info("Falling back to ContentDetector")
                    base_detector = ContentDetector(
                        threshold=self.scene_threshold,
                        min_scene_len=self.min_scene_len,
                        luma_only=self.luma_only
                    )
        elif detector_type == 'threshold':
            base_detector = ThresholdDetector(
                threshold=self.scene_threshold,
                min_scene_len=self.min_scene_len
            )
        else:
            logging.warning(f"Unknown detector type: {detector_type}, using ContentDetector")
            base_detector = ContentDetector(
                threshold=self.scene_threshold,
                min_scene_len=self.min_scene_len,
                luma_only=self.luma_only
            )
        
        # Note: event_buffer_length is read-only in PySceneDetect 0.6+
        # It's automatically determined by the detector algorithm
        logging.debug(f"Using default event buffer length ({base_detector.event_buffer_length}) for {detector_type} detector")
        
        return base_detector
        
    def process_video_realtime_generator(self):
        """
        Generator that yields scene data one at a time to simulate real-time streaming.
        
        Yields:
            Dict containing scene data with frames, timestamps, and performance metrics
            OR {'status': 'complete', 'summary': {...}} when processing is finished
        """
        logging.info(f"Starting real-time batch processing generator...")
        overall_start_time = time.time()
        
        # Process video in batches
        for batch_start in range(0, self.total_frames, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.total_frames)
            
            try:
                completed_scenes = self._process_batch_realtime(batch_start, batch_end)
                
                # Yield each completed scene
                for scene_data in completed_scenes:
                    yield scene_data
                
                # Simulate processing delay (optional)
                batch_duration = (batch_end - batch_start) / self.video_fps
                # In real streaming, you might add: time.sleep(batch_duration)
                
            except Exception as e:
                logging.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
                
                # Yield error information
                yield {
                    'status': 'error',
                    'batch_start': batch_start,
                    'batch_end': batch_end,
                    'error': str(e)
                }
                
                if not config.get_bool('encoding', 'continue_on_error', False):
                    break
        
        # Handle any remaining pending scene
        if self.pending_scene_start is not None:
            final_time = self.total_frames / self.video_fps
            scene_data = self._create_scene_data_no_encoding(
                self.pending_scene_start, 
                final_time, 
                self.scene_counter + 1
            )
            if scene_data:
                # Add encoding if enabled (separately timed)
                if self.enable_encoding and self.output_dir:
                    encoding_start = time.time()
                    output_file = self._save_scene_with_config(
                        scene_data['start_time'], 
                        scene_data['end_time'], 
                        scene_data['scene_number']
                    )
                    encoding_time = time.time() - encoding_start
                    self.encoding_times.append(encoding_time)
                    scene_data['encoded_file'] = output_file
                    scene_data['encoding_time'] = encoding_time
                
                self.scene_counter += 1
                self.processed_scenes.append((self.pending_scene_start, final_time))
                yield scene_data
        
        # Clean up
        self.cap.release()
        
        # Save stats if enabled
        if self.stats_manager and self.stats_file_path:
            try:
                self.stats_manager.save_to_csv(self.stats_file_path)
                logging.info(f"Saved scene detection stats to: {self.stats_file_path}")
            except Exception as e:
                logging.error(f"Failed to save stats: {e}")
        
        # Final summary with PRODUCTION-RELEVANT metrics
        core_processing_time = sum(self.core_processing_times)
        frame_extraction_time = sum(self.frame_extraction_times)
        production_processing_time = core_processing_time + frame_extraction_time
        
        # Optional operation times
        stats_collection_time = sum(self.stats_collection_times) if self.stats_collection_times else 0
        total_encoding_time = sum(self.encoding_times) if self.encoding_times else 0
        overall_time = time.time() - overall_start_time
        
        summary = {
            'status': 'complete',
            'summary': {
                'total_scenes': self.scene_counter,
                # PRODUCTION metrics (what matters for real-time performance)
                'core_scene_detection_time': core_processing_time,
                'frame_extraction_time': frame_extraction_time,
                'production_processing_time': production_processing_time,
                'production_frames_per_second': self.total_frames / production_processing_time if production_processing_time > 0 else 0,
                'production_real_time_factor': (self.total_frames / self.video_fps) / production_processing_time if production_processing_time > 0 else 0,
                
                # OPTIONAL metrics (not needed in production)
                'stats_collection_time': stats_collection_time,
                'total_encoding_time': total_encoding_time,
                'overall_time': overall_time,
                
                # Legacy metrics for compatibility
                'total_processing_time': production_processing_time,  # For backward compatibility
                'frames_per_second_processing': self.total_frames / production_processing_time if production_processing_time > 0 else 0,
                'real_time_factor': (self.total_frames / self.video_fps) / production_processing_time if production_processing_time > 0 else 0,
                'average_processing_time_per_scene': production_processing_time / max(self.scene_counter, 1),
                
                'scene_timestamps': self.processed_scenes,
                'input_video': str(self.input_video),
                'video_fps': self.video_fps,
                'total_frames': self.total_frames
            }
        }
        
        logging.info(f"Processing complete! Generated {self.scene_counter} scenes")
        logging.info(f"PRODUCTION processing time: {production_processing_time:.3f}s (core + frame extraction)")
        logging.info(f"  - Core scene detection: {core_processing_time:.3f}s")
        logging.info(f"  - Frame extraction: {frame_extraction_time:.3f}s")
        if stats_collection_time > 0:
            logging.info(f"Optional stats collection: {stats_collection_time:.3f}s")
        if total_encoding_time > 0:
            logging.info(f"Optional encoding: {total_encoding_time:.3f}s")
        logging.info(f"PRODUCTION real-time factor: {summary['summary']['production_real_time_factor']:.2f}x")
        
        yield summary

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


    def create_metadata_file(self, results: Dict[str, Any]):
        """Create comprehensive metadata file for the processing session."""
        export_metadata = config.get_bool('encoding', 'export_metadata', True)
        
        if not export_metadata:
            return
        
        metadata = {
            'input_video': str(self.input_video),
            'processing_mode': 'real_time_batch',
            'batch_size': self.batch_size,
            'max_scene_frames': self.max_scene_frames,
            'overlap_frames': self.overlap_frames,
            'video_properties': {
                'fps': self.video_fps,
                'total_frames': self.total_frames,
                'duration_seconds': self.total_frames / self.video_fps
            },
            'results': results
        }
        
        metadata_filename = config.get_str('encoding', 'metadata_filename', 'metadata.json')
        metadata_file = self.output_dir / metadata_filename
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logging.info(f"Metadata saved to: {metadata_file}")
        except Exception as e:
            logging.error(f"Error saving metadata: {e}")

    def create_scene_list(self, results: Dict[str, Any]):
        """Create a detailed scene list file."""
        scene_list_filename = config.get_str('encoding', 'scene_list_filename', 'scene_list.txt')
        scene_list_file = self.output_dir / scene_list_filename
        
        try:
            with open(scene_list_file, 'w') as f:
                f.write(f"Real-time Scene List for: {self.input_video.name}\n")
                f.write(f"Generated by Video Scene Splitter (Batch Mode)\n")
                f.write(f"Processing Mode: Real-time batch simulation\n")
                f.write(f"Batch Size: {self.batch_size} frames\n")
                f.write("-" * 60 + "\n\n")
                
                for i, (start_time, end_time) in enumerate(results['scene_timestamps'], 1):
                    duration = end_time - start_time
                    start_frame = int(start_time * self.video_fps)
                    end_frame = int(end_time * self.video_fps)
                    
                    f.write(f"Scene {i:4d}: {start_time:8.2f}s - {end_time:8.2f}s ")
                    f.write(f"(duration: {duration:6.2f}s, frames: {start_frame}-{end_frame})\n")
            
            logging.info(f"Scene list saved to: {scene_list_file}")
        except Exception as e:
            logging.error(f"Error creating scene list: {e}")

    def save_processing_statistics(self, results: Dict[str, Any]):
        """Save detailed processing statistics."""
        collect_statistics = config.get_bool('logging', 'collect_statistics', True)
        if not collect_statistics:
            return
        
        statistics_filename = config.get_str('logging', 'statistics_filename', 'processing_stats.json')
        stats_file = self.output_dir / statistics_filename
        
        statistics = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_video': str(self.input_video),
            'processing_mode': 'real_time_batch',
            'video_properties': {
                'fps': self.video_fps,
                'total_frames': self.total_frames,
                'duration_seconds': self.total_frames / self.video_fps
            },
            'batch_configuration': {
                'batch_size': self.batch_size,
                'overlap_frames': self.overlap_frames,
                'max_scene_frames': self.max_scene_frames
            },
            'results': results,
            'performance': {
                'processing_time_seconds': results['total_processing_time'],
                'frames_per_second': self.total_frames / results['total_processing_time'],
                'scenes_per_minute': results['total_scenes'] / (results['total_processing_time'] / 60),
                'average_scene_duration': sum(end - start for start, end in results['scene_timestamps']) / len(results['scene_timestamps']) if results['scene_timestamps'] else 0
            }
        }
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(statistics, f, indent=2)
            logging.info(f"Processing statistics saved to: {stats_file}")
        except Exception as e:
            logging.error(f"Error saving statistics: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Video Scene Splitter - Real-time batch processing for live streaming simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic real-time processing
    python video_scene_splitter.py input.mp4
    
    # Custom batch size and output directory
    python video_scene_splitter.py input.mp4 ./output --batch-size 200
    
    # Use custom configuration
    python video_scene_splitter.py input.mp4 --config config_test.ini
    
    # Show configuration
    python video_scene_splitter.py --show-config
        """
    )
    
    parser.add_argument('input_video', nargs='?', help='Path to input video file')
    parser.add_argument('output_dir', nargs='?', default=None,
                       help='Output directory for scene videos (only used if --enable-encoding)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help=f'Number of frames per batch (default from config: {config.get_int("scene_detection", "batch_size", 100)})')
    parser.add_argument('--max-frames', type=int, default=None,
                       help=f'Maximum frames per scene (default from config: {config.get_int("scene_detection", "default_max_frames", 1000)})')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--show-config', action='store_true', help='Show current configuration and exit')
    parser.add_argument('--enable-encoding', action='store_true', 
                       help='Enable encoding scenes to files (for debugging/visualization)')
    parser.add_argument('--simulate-delay', action='store_true', 
                       help='Add realistic processing delays to simulate live streaming')
    
    # Enhanced detection options
    parser.add_argument('--detector', choices=['content', 'adaptive', 'histogram', 'hash', 'threshold', 'multi'],
                       default=None, help='Scene detection algorithm to use')
    parser.add_argument('--backend', choices=['opencv', 'pyav', 'moviepy', 'auto'],
                       default=None, help='Video backend to use for processing')
    parser.add_argument('--enable-stats', action='store_true',
                       help='Enable statistics collection and caching')
    parser.add_argument('--downscale', type=int, default=None,
                       help='Downscale factor for faster processing (0=auto, 1=no downscale)')
    parser.add_argument('--show-detectors', action='store_true',
                       help='Show available detection algorithms and backends')
    
    args = parser.parse_args()
    
    # Show available detectors and backends if requested
    if args.show_detectors:
        print("=== Available Scene Detection Algorithms ===")
        print("content     - Fast cuts using weighted average of HSV changes (default)")
        print("adaptive    - Fast cuts using rolling average of HSL changes")
        print("histogram   - Fast cuts using HSV histogram changes")
        print("hash        - Fast cuts using perceptual image hashing (fastest)")
        print("threshold   - Fades in/out using pixel intensity changes")
        print("multi       - Combines multiple detectors for better accuracy")
        print(f"\n=== Available Video Backends ===")
        for backend in AVAILABLE_BACKENDS:
            print(f"{backend:<12} - Available")
        print("auto        - Automatically select best available backend")
        return
    
    # Reload config if specified
    if args.config:
        config.load_config(args.config)
    
    # Show configuration if requested
    if args.show_config:
        print("=== Video Scene Splitter Configuration ===")
        for section_name in config.config.sections():
            print(f"\n[{section_name}]")
            for key, value in config.config[section_name].items():
                print(f"  {key} = {value}")
        return
    
    # Require input video if not showing config
    if not args.input_video:
        parser.error("input_video is required unless using --show-config")
    
    try:
        # Initialize splitter
        splitter = VideoSceneSplitter(
            input_video=args.input_video,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_scene_frames=args.max_frames,
            config_file=args.config,
            enable_encoding=args.enable_encoding,
            detector_type=args.detector,
            video_backend=args.backend
        )
        
        # Apply enhanced settings from command line
        if args.enable_stats:
            splitter.use_stats = True
            if not splitter.stats_manager:
                splitter.stats_manager = StatsManager()
        
        if args.downscale is not None:
            splitter.downscale_factor = args.downscale
            splitter.enable_downscaling = args.downscale != 1
        
        # Setup logging to file if configured
        log_to_file = config.get_bool('logging', 'log_to_file', True)
        if log_to_file:
            log_pattern = config.get_str('logging', 'log_filename_pattern', '{input_stem}_processing.log')
            if splitter.output_dir:
                log_file = splitter.output_dir / log_pattern.format(input_stem=splitter.input_video.stem)
            else:
                log_file = Path(log_pattern.format(input_stem=splitter.input_video.stem))
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler)
        
        # Process video using generator approach
        print(f"Starting real-time scene processing...")
        print(f"Detection algorithm: {splitter.detector_type}")
        print(f"Video backend: {splitter.video_backend}")
        print(f"Encoding enabled: {args.enable_encoding}")
        if splitter.enable_downscaling:
            if splitter.downscale_factor == 0:
                print(f"Frame downscaling: Auto")
            else:
                print(f"Frame downscaling: {splitter.downscale_factor}x")
        print(f"Stats collection: {splitter.use_stats}")
        print(f"Processing frames in batches...")
        
        scene_count = 0
        total_processing_time = 0
        
        for scene_data in splitter.process_video_realtime_generator():
            if scene_data.get('status') == 'complete':
                # Final summary
                summary = scene_data['summary']
                print(f"\n{'-'*60}")
                print(f"Real-time processing complete!")
                print(f"Generated {summary['total_scenes']} scenes")
                print(f"Total processing time: {summary['total_processing_time']:.3f} seconds")
                if args.enable_encoding:
                    print(f"Total encoding time: {summary['total_encoding_time']:.3f} seconds")
                print(f"Real-time factor: {summary['real_time_factor']:.2f}x")
                print(f"Processing speed: {summary['frames_per_second_processing']:.1f} frames/second")
                
                if summary['real_time_factor'] >= 1.0:
                    print("✅ Processing is FASTER than real-time!")
                else:
                    print("⚠️  Processing is slower than real-time")
                
                break
                
            elif scene_data.get('status') == 'error':
                print(f"❌ Error in batch {scene_data['batch_start']}-{scene_data['batch_end']}: {scene_data['error']}")
                continue
            
            else:
                # Regular scene data
                scene_count += 1
                scene_number = scene_data['scene_number']
                duration = scene_data['duration']
                frame_count = scene_data['frame_count']
                
                print(f"Scene {scene_number:2d}: {duration:5.2f}s ({frame_count:3d} frames) ", end="")
                
                if args.enable_encoding and 'encoded_file' in scene_data:
                    encoding_time = scene_data.get('encoding_time', 0)
                    print(f"[encoded in {encoding_time:.2f}s]")
                else:
                    print(f"[frames extracted]")
                
                # In a real pipeline, you would pass scene_data['frames'] to the next processing stage
                # For demonstration, we just show we have the frames
                if not args.enable_encoding:
                    # Show we have the actual frame data
                    frames = scene_data['frames']
                    if frames:
                        print(f"   → {len(frames)} frames available for next stage (shape: {frames[0].shape})")
        
        # Clean up
        splitter.close()
        
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        print("\nProcessing interrupted by user")
    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

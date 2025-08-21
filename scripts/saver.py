#!/usr/bin/env python3
"""
Saver Component

This component handles all saving operations for the PointStream pipeline:
- Video encoding with multiple codec support (AVC, HEVC, VP9, AV1)
- Image saving with proper color space handling
- Metadata and JSON file generation
- Transparent object cropping with correct masking

Separates saving logic from core processing for better modularity and debugging.
"""

import cv2
import numpy as np
import json
import logging
import subprocess
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from .decorators import track_performance
from .error_handling import safe_execute, create_error_result, create_success_result
from . import config


class Saver:
    """Handles all saving operations with multiple codec and format support."""
    
    def __init__(self):
        """Initialize the saver with codec and format configurations."""
        # Video encoding settings
        self.video_codec = config.get_str('saver', 'video_codec', 'libx264')
        self.video_quality = config.get_int('saver', 'video_quality', 23)
        self.video_preset = config.get_str('saver', 'video_preset', 'medium')
        self.container_format = config.get_str('saver', 'container_format', 'mp4')
        
        # Image settings
        self.image_format = config.get_str('saver', 'image_format', 'png')
        self.image_quality = config.get_int('saver', 'image_quality', 95)
        
        # FFmpeg settings
        self.ffmpeg_binary = config.get_str('saver', 'ffmpeg_binary', 'ffmpeg')
        self.ffmpeg_threads = config.get_int('saver', 'ffmpeg_threads', 0)
        
        # Output settings
        self.save_individual_objects = config.get_bool('saver', 'save_individual_objects', True)
        self.save_metadata_enabled = config.get_bool('saver', 'save_metadata', True)
        self.save_debug_data = config.get_bool('saver', 'save_debug_data', True)
        
        logging.info(f"Saver initialized")
        logging.info(f"Video codec: {self.video_codec}")
        logging.info(f"Image format: {self.image_format}")
        logging.info(f"Container: {self.container_format}")
        
    @safe_execute("Complex scene video encoding", {'success': False, 'error': 'encoding_failed'})
    @track_performance
    def save_complex_scene_video(self, frames: List[np.ndarray], output_path: str, fps: float, scene_number: int) -> Dict[str, Any]:
        """
        Save complex scene as video with multiple codec support.
        
        Args:
            frames: List of frames in BGR format
            output_path: Output video file path
            fps: Video framerate
            scene_number: Scene number for logging
            
        Returns:
            Dictionary with save result
        """
        if not frames:
            return {'success': False, 'error': 'No frames provided'}
            
        height, width = frames[0].shape[:2]
        
        # Build FFmpeg command based on codec
        ffmpeg_cmd = self._build_video_encode_command(width, height, fps, output_path)
        
        logging.info(f"Encoding scene {scene_number} video with {self.video_codec}")
        logging.info(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {len(frames)}")
        
        encoding_start = time.time()
        
        # Start FFmpeg process with proper buffering
        process = subprocess.Popen(
            ffmpeg_cmd, 
            stdin=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            stdout=subprocess.DEVNULL,  # Don't capture stdout
            bufsize=1024*1024  # 1MB buffer
        )
        
        stderr_output = None
        try:
            # Prepare all frame data in a buffer
            frame_data = bytearray()
            for frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_data.extend(rgb_frame.tobytes())
            
            # Let communicate() handle sending data and closing stdin.
            # This is the main fix.
            try:
                _, stderr_output = process.communicate(input=frame_data, timeout=60)
            except subprocess.TimeoutExpired:
                logging.error("FFmpeg process timed out after 60 seconds")
                process.kill()
                # After killing, communicate again to get any final output
                _, stderr_output = process.communicate()
                
        except Exception as e:
            logging.error(f"Error during FFmpeg communication: {e}")
            # Ensure the process is terminated if something goes wrong
            process.kill()
            process.wait()
            raise e
        
        encoding_time = time.time() - encoding_start
        
        if process.returncode == 0:
            file_size = os.path.getsize(output_path)
            logging.info(f"Scene {scene_number} video encoded in {encoding_time:.3f}s")
            logging.info(f"Output: {output_path} ({file_size/1024/1024:.2f}MB)")
            
            return {
                'success': True,
                'output_path': output_path,
                'encoding_time': encoding_time,
                'file_size_mb': file_size/1024/1024,
                'codec': self.video_codec
            }
        else:
            error_msg = stderr_output.decode('utf-8') if stderr_output else 'Unknown FFmpeg error'
            logging.error(f"FFmpeg failed with return code {process.returncode}")
            logging.error(f"FFmpeg stderr: {error_msg}")
            return {
                'success': False,
                'error': f'FFmpeg failed: {error_msg}',
                    'return_code': process.returncode
                }
    
    def _build_video_encode_command(self, width: int, height: int, fps: float, output_path: str) -> List[str]:
        """Build FFmpeg command for different codecs."""
        cmd = [self.ffmpeg_binary]
        
        # Input settings - RGB format now
        cmd.extend([
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',  # Changed from bgr24 to rgb24
            '-r', str(fps),
            '-i', '-'  # Read from stdin
        ])
        
        # Video codec specific settings
        if self.video_codec == 'libx264':  # H.264/AVC
            cmd.extend([
                '-c:v', 'libx264',
                '-crf', str(self.video_quality),
                '-preset', self.video_preset,
                '-pix_fmt', 'yuv420p'
            ])
        elif self.video_codec == 'libx265':  # H.265/HEVC
            cmd.extend([
                '-c:v', 'libx265',
                '-crf', str(self.video_quality),
                '-preset', self.video_preset,
                '-pix_fmt', 'yuv420p'
            ])
        elif self.video_codec == 'libvpx-vp9':  # VP9
            cmd.extend([
                '-c:v', 'libvpx-vp9',
                '-crf', str(self.video_quality),
                '-b:v', '0',  # Use CRF mode
                '-pix_fmt', 'yuv420p'
            ])
        elif self.video_codec == 'libsvtav1':  # AV1
            cmd.extend([
                '-c:v', 'libsvtav1',
                '-crf', str(self.video_quality),
                '-preset', str(self.video_preset),
                '-pix_fmt', 'yuv420p'
            ])
        else:
            # Fallback to H.264
            cmd.extend([
                '-c:v', 'libx264',
                '-crf', str(self.video_quality),
                '-preset', self.video_preset,
                '-pix_fmt', 'yuv420p'
            ])
        
        # Threading
        if self.ffmpeg_threads > 0:
            cmd.extend(['-threads', str(self.ffmpeg_threads)])
        
        # Output
        cmd.extend(['-y', output_path])
        
        return cmd
    
    @track_performance
    def save_scene_objects(self, scene_data: Dict[str, Any], output_dir: Path, scene_number: int) -> Dict[str, Any]:
        """
        Save individual object images with proper color correction and transparency.
        
        Args:
            scene_data: Scene processing results
            output_dir: Output directory
            scene_number: Scene number
            
        Returns:
            Dictionary with save results
        """
        if not self.save_individual_objects:
            return {'saved_objects': 0}
            
        objects = scene_data.get('keypoint_result', {}).get('objects', [])
        if not objects:
            return {'saved_objects': 0}
        
        # Create objects directory
        objects_dir = output_dir / 'objects' / f'scene_{scene_number:04d}'
        objects_dir.mkdir(parents=True, exist_ok=True)
        
        saved_objects = []
        class_counters = {}
        
        for obj in objects:
            try:
                class_name = obj.get('class_name', 'unknown')
                object_id = obj.get('object_id', 'unknown')
                cropped_image = obj.get('cropped_image')
                
                if cropped_image is not None:
                    # Generate filename
                    if class_name not in class_counters:
                        class_counters[class_name] = 0
                    class_counters[class_name] += 1
                    
                    if class_counters[class_name] == 1:
                        object_filename = class_name
                    else:
                        object_filename = f"{class_name}_{class_counters[class_name]}"
                    
                    # Apply proper background masking with color correction
                    masked_object = self._create_transparent_object(cropped_image, obj)
                    
                    # Save as PNG to support transparency using PIL for correct color handling
                    image_filename = f"{object_filename}.png"
                    image_path = objects_dir / image_filename
                    
                    # Save with PIL to ensure correct RGB color space
                    try:
                        # Convert numpy array to PIL Image
                        if masked_object.shape[2] == 4:  # RGBA
                            pil_image = Image.fromarray(masked_object, 'RGBA')
                        else:  # RGB
                            pil_image = Image.fromarray(masked_object, 'RGB')
                        
                        # Save with PIL (handles RGB correctly)
                        pil_image.save(str(image_path), 'PNG')
                        success = True
                    except Exception as e:
                        logging.warning(f"PIL save failed, falling back to OpenCV: {e}")
                        # Fallback to OpenCV (convert RGB back to BGR for cv2.imwrite)
                        if masked_object.shape[2] == 4:  # RGBA
                            bgra_object = cv2.cvtColor(masked_object, cv2.COLOR_RGBA2BGRA)
                            success = cv2.imwrite(str(image_path), bgra_object)
                        else:  # RGB
                            bgr_object = cv2.cvtColor(masked_object, cv2.COLOR_RGB2BGR)
                            success = cv2.imwrite(str(image_path), bgr_object)
                    
                    if success:
                        obj_metadata = {
                            'object_id': object_id,
                            'class_name': class_name,
                            'filename': object_filename,
                            'saved_image_path': str(image_path),
                            'has_transparency': True,
                            'format': 'PNG'
                        }
                        saved_objects.append(obj_metadata)
                
            except Exception as e:
                logging.warning(f"Failed to save object {class_name}: {e}")
                continue
        
        logging.info(f"Saved {len(saved_objects)} objects for scene {scene_number}")
        return {'saved_objects': len(saved_objects), 'objects_metadata': saved_objects}
    
    def _create_transparent_object(self, cropped_image: np.ndarray, obj: Dict[str, Any]) -> np.ndarray:
        """
        Create transparent object image with proper color handling.
        
        Args:
            cropped_image: Cropped object image in BGR format
            obj: Object data with segmentation mask
            
        Returns:
            RGBA image with transparency
        """
        try:
            # Get segmentation mask and crop bbox
            seg_mask = obj.get('segmentation_mask')
            crop_bbox = obj.get('crop_bbox', [])
            
            if seg_mask is None or len(crop_bbox) < 4:
                # No mask available - convert BGR to RGB then to RGBA for correct colors
                if len(cropped_image.shape) == 3:
                    rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                    rgba_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2RGBA)
                    return rgba_image
                else:
                    rgba_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGBA)
                    return rgba_image
            
            # Extract crop coordinates
            x1, y1, x2, y2 = map(int, crop_bbox[:4])
            
            # Get the mask portion for this crop
            if len(seg_mask.shape) == 2:
                mask_crop = seg_mask[y1:y2, x1:x2]
            else:
                mask_crop = seg_mask[y1:y2, x1:x2]
                if len(mask_crop.shape) > 2:
                    mask_crop = mask_crop[:, :, 0]
            
            # Ensure mask dimensions match cropped image
            crop_h, crop_w = cropped_image.shape[:2]
            if mask_crop.shape != (crop_h, crop_w):
                mask_crop = cv2.resize(mask_crop, (crop_w, crop_h))
            
            # Convert mask to binary
            binary_mask = (mask_crop > 0.5).astype(np.uint8)
            
            # Create RGBA image with proper color space
            if len(cropped_image.shape) == 3:
                # Convert BGR to RGB first, then to RGBA for correct colors
                rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                rgba_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2RGBA)
            else:
                # Grayscale to RGBA
                rgba_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGBA)
            
            # Set alpha channel: 255 for object pixels, 0 for background
            rgba_image[:, :, 3] = binary_mask * 255
            
            return rgba_image
            
        except Exception as e:
            logging.warning(f"Failed to create transparent object: {e}")
            # Fallback to opaque image with correct color conversion
            if len(cropped_image.shape) == 3:
                rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2RGBA)
            else:
                return cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGBA)
    
    @track_performance
    def save_metadata(self, scene_data: Dict[str, Any], output_dir: Path, scene_number: int) -> Dict[str, Any]:
        """Save comprehensive scene metadata as JSON with performance statistics."""
        if not self.save_metadata_enabled:
            return {'metadata_saved': False}
            
        try:
            metadata_file = output_dir / f'scene_{scene_number:04d}_metadata.json'
            
            # Add comprehensive performance and summary information first
            try:
                from .decorators import profiler
                performance_summary = profiler.get_overall_summary()
                
                # Extract results for summary statistics (before cleaning)
                stitching_result = scene_data.get('stitching_result', {})
                segmentation_result = scene_data.get('segmentation_result', {})
                keypoint_result = scene_data.get('keypoint_result', {})
                panorama = stitching_result.get('panorama')
                
                # Create enhanced metadata with summary statistics
                enhanced_metadata = {
                    'scene_number': scene_number,
                    'scene_type': scene_data.get('scene_type'),
                    'processing_timestamp': time.time(),
                    
                    # Summary statistics (calculated before cleaning)
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
                    
                    'performance': {
                        'detailed_timings': performance_summary,
                        'total_scene_time': sum(timing['total_time'] for timing in performance_summary.values()) if performance_summary else 0,
                        'slowest_operation': max(performance_summary.keys(), key=lambda k: performance_summary[k]['avg_time']) if performance_summary else None
                    }
                }
                
                # Now add the cleaned detailed data (excluding masked_frames and segmentation_result)
                clean_metadata = self._clean_metadata_for_json(scene_data)
                
                # Merge the clean detailed data with our summary
                for key, value in clean_metadata.items():
                    if key not in enhanced_metadata:  # Don't overwrite our summary sections
                        enhanced_metadata[key] = value
                
            except Exception as e:
                # Fallback to basic metadata if performance data unavailable
                logging.warning(f"Could not add performance data to metadata: {e}")
                enhanced_metadata = self._clean_metadata_for_json(scene_data)
            
            with open(metadata_file, 'w') as f:
                json.dump(enhanced_metadata, f, indent=2, default=str)
            
            return {'metadata_saved': True, 'metadata_path': str(metadata_file)}
            
        except Exception as e:
            logging.warning(f"Failed to save metadata for scene {scene_number}: {e}")
            return {'metadata_saved': False, 'error': str(e)}
    
    def _clean_metadata_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove large binary data and unwanted sections from metadata before JSON serialization."""
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                # Skip large binary data and unwanted data structures
                if key in ['frames', 'cropped_image', 'segmentation_mask', 'panorama', 'masked_frames', 'segmentation_result']:
                    # Skip these entirely - don't include them in the JSON
                    continue
                else:
                    cleaned[key] = self._clean_metadata_for_json(value)
            return cleaned
        elif isinstance(data, list):
            return [self._clean_metadata_for_json(item) for item in data]
        else:
            return data

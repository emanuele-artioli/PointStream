#!/usr/bin/env python3
"""
Video Assembler

This module assembles composed frames into final video files.
Handles video encoding, frame rate adjustment, and output formatting.
"""

import logging
import time
import numpy as np
import cv2
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
import os

try:
    from ...utils.decorators import track_performance
    from ...utils import config
except ImportError as e:
    logging.error(f"Failed to import PointStream utilities: {e}")
    raise


class VideoAssembler:
    """
    Assembles composed frames into final video files.
    
    This component takes composed frames and creates video files with
    proper encoding settings and metadata.
    """
    
    def __init__(self):
        """Initialize the video assembler."""
        # Video encoding settings
        self.fps = config.get_float('video_reconstruction', 'fps', 25.0)
        self.video_codec = config.get_str('video_reconstruction', 'video_codec', 'libx264')
        self.video_quality = config.get_int('video_reconstruction', 'video_quality', 23)
        self.video_preset = config.get_str('video_reconstruction', 'video_preset', 'medium')
        self.container_format = config.get_str('video_reconstruction', 'container_format', 'mp4')
        
        # Output settings
        self.temp_dir = Path(tempfile.gettempdir()) / "pointstream_assembly"
        self.temp_dir.mkdir(exist_ok=True)
        
        # FFmpeg path
        self.ffmpeg_binary = self._find_ffmpeg()
        
        logging.info("🎬 Video Assembler initialized")
        logging.info(f"   FPS: {self.fps}")
        logging.info(f"   Codec: {self.video_codec}")
        logging.info(f"   Quality: {self.video_quality}")
        logging.info(f"   FFmpeg: {self.ffmpeg_binary}")
    
    def _find_ffmpeg(self) -> str:
        """Find FFmpeg binary."""
        # Check common locations
        ffmpeg_paths = [
            'ffmpeg',  # System PATH
            '/usr/bin/ffmpeg',
            '/usr/local/bin/ffmpeg',
            str(Path.cwd().parent / 'ffmpeg' / 'ffmpeg'),  # Project ffmpeg
        ]
        
        for path in ffmpeg_paths:
            try:
                result = subprocess.run([path, '-version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        # Fallback to 'ffmpeg' and hope it's in PATH
        logging.warning("FFmpeg not found in common locations, using 'ffmpeg' from PATH")
        return 'ffmpeg'
    
    @track_performance
    def assemble_video(self, composed_frames: List[np.ndarray], 
                      fps: float = None, scene_number: int = 0) -> Dict[str, Any]:
        """
        Assemble frames into a video file.
        
        Args:
            composed_frames: List of composed frames
            fps: Frame rate (optional, uses config default)
            scene_number: Scene number for output naming
            
        Returns:
            Assembly result with video path and metadata
        """
        start_time = time.time()
        
        if not composed_frames:
            raise ValueError("No frames provided for assembly")
        
        actual_fps = fps if fps is not None else self.fps
        
        logging.info(f"🎞️  Assembling {len(composed_frames)} frames at {actual_fps} FPS")
        
        # Create temporary directory for this scene
        scene_temp_dir = self.temp_dir / f"scene_{scene_number:04d}"
        scene_temp_dir.mkdir(exist_ok=True)
        
        try:
            # Save frames as temporary images
            frame_paths = self._save_temporary_frames(composed_frames, scene_temp_dir)
            
            # Create video from frames
            output_path = scene_temp_dir / f"scene_{scene_number:04d}_reconstructed.{self.container_format}"
            
            success = self._encode_video(frame_paths, output_path, actual_fps)
            
            if not success:
                raise RuntimeError("Video encoding failed")
            
            # Verify output
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("Output video file is empty or missing")
            
            processing_time = time.time() - start_time
            
            # Get video info
            video_info = self._get_video_info(output_path)
            
            result = {
                'success': True,
                'video_path': str(output_path),
                'frame_count': len(composed_frames),
                'fps': actual_fps,
                'duration': len(composed_frames) / actual_fps,
                'video_info': video_info,
                'processing_time': processing_time,
                'file_size': output_path.stat().st_size
            }
            
            logging.info(f"✅ Video assembly completed in {processing_time:.2f}s")
            logging.info(f"   Output: {output_path}")
            logging.info(f"   Size: {result['file_size'] / 1024 / 1024:.1f} MB")
            
            return result
            
        except Exception as e:
            logging.error(f"Video assembly failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_path': None,
                'processing_time': time.time() - start_time
            }
        finally:
            # Clean up temporary frames (keep video)
            self._cleanup_temporary_frames(scene_temp_dir)
    
    def _save_temporary_frames(self, frames: List[np.ndarray], 
                             temp_dir: Path) -> List[Path]:
        """Save frames as temporary image files."""
        frame_paths = []
        
        for i, frame in enumerate(frames):
            frame_path = temp_dir / f"frame_{i:06d}.png"
            
            # Ensure frame is in correct format
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # BGR to RGB for proper color
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Save frame
            success = cv2.imwrite(str(frame_path), frame_rgb)
            if not success:
                raise RuntimeError(f"Failed to save frame {i}")
            
            frame_paths.append(frame_path)
        
        return frame_paths
    
    def _encode_video(self, frame_paths: List[Path], output_path: Path, fps: float) -> bool:
        """Encode video using FFmpeg."""
        if not frame_paths:
            return False
        
        # Build FFmpeg command
        input_pattern = str(frame_paths[0].parent / "frame_%06d.png")
        
        cmd = [
            self.ffmpeg_binary,
            '-y',  # Overwrite output
            '-framerate', str(fps),
            '-i', input_pattern,
            '-c:v', self.video_codec,
            '-preset', self.video_preset,
            '-crf', str(self.video_quality),
            '-pix_fmt', 'yuv420p',  # Compatibility
            '-movflags', '+faststart',  # Web optimization
            str(output_path)
        ]
        
        try:
            logging.info(f"🔧 Encoding video with command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=frame_paths[0].parent
            )
            
            if result.returncode != 0:
                logging.error(f"FFmpeg encoding failed:")
                logging.error(f"STDOUT: {result.stdout}")
                logging.error(f"STDERR: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logging.error("FFmpeg encoding timed out")
            return False
        except Exception as e:
            logging.error(f"FFmpeg encoding error: {e}")
            return False
    
    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video information using FFprobe."""
        try:
            # Use ffprobe to get video info
            ffprobe_cmd = [
                self.ffmpeg_binary.replace('ffmpeg', 'ffprobe'),
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)
                
                # Extract relevant information
                video_stream = None
                for stream in info.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        video_stream = stream
                        break
                
                if video_stream:
                    return {
                        'width': video_stream.get('width'),
                        'height': video_stream.get('height'),
                        'duration': float(video_stream.get('duration', 0)),
                        'frame_count': int(video_stream.get('nb_frames', 0)),
                        'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                        'codec': video_stream.get('codec_name'),
                        'bitrate': int(video_stream.get('bit_rate', 0))
                    }
            
        except Exception as e:
            logging.warning(f"Failed to get video info: {e}")
        
        return {}
    
    def _cleanup_temporary_frames(self, temp_dir: Path):
        """Clean up temporary frame files."""
        try:
            for frame_file in temp_dir.glob("frame_*.png"):
                frame_file.unlink()
        except Exception as e:
            logging.warning(f"Failed to cleanup temporary frames: {e}")
    
    def create_preview_video(self, composed_frames: List[np.ndarray], 
                           preview_fps: float = 5.0) -> Optional[Path]:
        """
        Create a quick preview video with lower quality/fps.
        
        Args:
            composed_frames: Frames to preview
            preview_fps: Preview frame rate
            
        Returns:
            Path to preview video or None if failed
        """
        if not composed_frames:
            return None
        
        preview_dir = self.temp_dir / "preview"
        preview_dir.mkdir(exist_ok=True)
        
        # Subsample frames for preview
        step = max(1, len(composed_frames) // 50)  # Max 50 frames for preview
        preview_frames = composed_frames[::step]
        
        try:
            result = self.assemble_video(
                preview_frames, 
                fps=preview_fps, 
                scene_number=9999  # Special scene number for preview
            )
            
            if result.get('success'):
                return Path(result['video_path'])
                
        except Exception as e:
            logging.warning(f"Failed to create preview video: {e}")
        
        return None
    
    def batch_assemble_scenes(self, scenes_data: List[Dict[str, Any]], 
                            output_dir: Path) -> Dict[str, Any]:
        """
        Assemble multiple scenes in batch.
        
        Args:
            scenes_data: List of scene data with composed frames
            output_dir: Output directory for videos
            
        Returns:
            Batch assembly results
        """
        output_dir.mkdir(exist_ok=True)
        
        results = []
        total_start_time = time.time()
        
        for i, scene_data in enumerate(scenes_data):
            scene_number = scene_data.get('scene_number', i)
            composed_frames = scene_data.get('composed_frames', [])
            
            if not composed_frames:
                continue
            
            logging.info(f"🎬 Assembling scene {scene_number} ({i+1}/{len(scenes_data)})")
            
            result = self.assemble_video(composed_frames, scene_number=scene_number)
            
            if result.get('success'):
                # Move to final output directory
                temp_path = Path(result['video_path'])
                final_path = output_dir / f"scene_{scene_number:04d}_reconstructed.{self.container_format}"
                
                if temp_path.exists():
                    temp_path.rename(final_path)
                    result['final_path'] = str(final_path)
            
            results.append(result)
        
        total_time = time.time() - total_start_time
        successful_scenes = sum(1 for r in results if r.get('success'))
        
        return {
            'total_scenes': len(scenes_data),
            'successful_scenes': successful_scenes,
            'failed_scenes': len(scenes_data) - successful_scenes,
            'total_time': total_time,
            'results': results
        }

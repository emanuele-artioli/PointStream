#!/usr/bin/env python3
import cv2
import numpy as np
import json
import subprocess
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from utils.decorators import track_performance
from utils.error_handling import safe_execute
from utils import config

class Saver:
    def __init__(self):
        self.video_codec = config.get_str('saver', 'video_codec', 'libx264')
        self.video_quality = config.get_int('saver', 'video_quality', 23)
        self.video_preset = config.get_str('saver', 'video_preset', 'medium')
        self.container_format = config.get_str('saver', 'container_format', 'mp4')
        self.image_format = config.get_str('saver', 'image_format', 'png')
        self.image_quality = config.get_int('saver', 'image_quality', 95)
        self.ffmpeg_binary = config.get_str('saver', 'ffmpeg_binary', 'ffmpeg')
        self.ffmpeg_threads = config.get_int('saver', 'ffmpeg_threads', 0)
        self.save_individual_objects = config.get_bool('saver', 'save_individual_objects', True)
        self.save_metadata_enabled = config.get_bool('saver', 'save_metadata', True)
        self.save_debug_data = config.get_bool('saver', 'save_debug_data', True)
        
    @safe_execute("Complex scene video encoding", {'success': False, 'error': 'encoding_failed'})
    @track_performance
    def save_complex_scene_video(self, frames: List[np.ndarray], output_path: str, fps: float, scene_number: int) -> Dict[str, Any]:
        if not frames:
            return {'success': False, 'error': 'No frames provided'}
            
        height, width = frames[0].shape[:2]
        ffmpeg_cmd = self._build_video_encode_command(width, height, fps, output_path)
        
        encoding_start = time.time()
        
        process = subprocess.Popen(
            ffmpeg_cmd, 
            stdin=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            stdout=subprocess.DEVNULL,
            bufsize=1024*1024
        )
        
        stderr_output = None
        try:
            frame_data = bytearray()
            for frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_data.extend(rgb_frame.tobytes())
            try:
                _, stderr_output = process.communicate(input=frame_data, timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                _, stderr_output = process.communicate()
        except Exception as e:
            process.kill()
            process.wait()
            raise e
        
        encoding_time = time.time() - encoding_start
        
        if process.returncode == 0:
            file_size = os.path.getsize(output_path)
            return {
                'success': True,
                'output_path': output_path,
                'encoding_time': encoding_time,
                'file_size_mb': file_size/1024/1024,
                'codec': self.video_codec
            }
        else:
            error_msg = stderr_output.decode('utf-8') if stderr_output else 'Unknown FFmpeg error'
            return {
                'success': False,
                'error': f'FFmpeg failed: {error_msg}',
                    'return_code': process.returncode
                }
    
    def _build_video_encode_command(self, width: int, height: int, fps: float, output_path: str) -> List[str]:
        cmd = [self.ffmpeg_binary]
        cmd.extend([
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-'
        ])
        
        if self.video_codec == 'libx264':
            cmd.extend(['-c:v', 'libx264', '-crf', str(self.video_quality), '-preset', self.video_preset, '-pix_fmt', 'yuv420p'])
        elif self.video_codec == 'libx265':
            cmd.extend(['-c:v', 'libx265', '-crf', str(self.video_quality), '-preset', self.video_preset, '-pix_fmt', 'yuv420p'])
        elif self.video_codec == 'libvpx-vp9':
            cmd.extend(['-c:v', 'libvpx-vp9', '-crf', str(self.video_quality), '-b:v', '0', '-pix_fmt', 'yuv420p'])
        elif self.video_codec == 'libsvtav1':
            cmd.extend(['-c:v', 'libsvtav1', '-crf', str(self.video_quality), '-preset', str(self.video_preset), '-pix_fmt', 'yuv420p'])
        else:
            cmd.extend(['-c:v', 'libx264', '-crf', str(self.video_quality), '-preset', self.video_preset, '-pix_fmt', 'yuv420p'])
        
        if self.ffmpeg_threads > 0:
            cmd.extend(['-threads', str(self.ffmpeg_threads)])
        
        cmd.extend(['-y', output_path])
        return cmd
    
    @track_performance
    def save_scene_objects(self, scene_data: Dict[str, Any], output_dir: Path, scene_number: int) -> Dict[str, Any]:
        if not self.save_individual_objects:
            return {'saved_objects': 0}
            
        objects = scene_data.get('keypoint_result', {}).get('objects', [])
        if not objects:
            return {'saved_objects': 0}
        
        objects_dir = output_dir / 'objects' / f'scene_{scene_number:04d}'
        objects_dir.mkdir(parents=True, exist_ok=True)
        
        saved_objects = []
        
        for obj in objects:
            try:
                semantic_category = obj.get('semantic_category', 'other')
                original_class_name = obj.get('original_class_name', obj.get('class_name', 'unknown'))
                track_id = obj.get('track_id')
                cropped_image = obj.get('cropped_image')
                frame_index = obj.get('frame_index', 0)
                
                if cropped_image is not None:
                    if track_id is not None:
                        object_filename = f"{semantic_category}_track_{track_id}_frame_{frame_index:04d}"
                    else:
                        object_filename = f"{semantic_category}_{original_class_name}_frame_{frame_index:04d}"
                    
                    masked_object = self._create_transparent_object(cropped_image, obj)
                    image_filename = f"{object_filename}.png"
                    image_path = objects_dir / image_filename
                    
                    try:
                        if masked_object.shape[2] == 4:
                            pil_image = Image.fromarray(masked_object, 'RGBA')
                        else:
                            pil_image = Image.fromarray(masked_object, 'RGB')
                        pil_image.save(str(image_path), 'PNG')
                        success = True
                    except Exception:
                        if masked_object.shape[2] == 4:
                            bgra_object = cv2.cvtColor(masked_object, cv2.COLOR_RGBA2BGRA)
                            success = cv2.imwrite(str(image_path), bgra_object)
                        else:
                            bgr_object = cv2.cvtColor(masked_object, cv2.COLOR_RGB2BGR)
                            success = cv2.imwrite(str(image_path), bgr_object)
                    
                    if success:
                        obj_metadata = {
                            'semantic_category': semantic_category,
                            'original_class_name': original_class_name,
                            'track_id': track_id,
                            'filename': object_filename,
                            'saved_image_path': str(image_path),
                            'has_transparency': True,
                            'format': 'PNG',
                            'semantic_confidence': obj.get('semantic_confidence', 0.0),
                            'classification_method': obj.get('classification_method', 'unknown')
                        }
                        saved_objects.append(obj_metadata)
            except Exception:
                continue
        
        return {'saved_objects': len(saved_objects), 'objects_metadata': saved_objects}
    
    def _create_transparent_object(self, cropped_image: np.ndarray, obj: Dict[str, Any]) -> np.ndarray:
        try:
            seg_mask = obj.get('segmentation_mask')
            crop_bbox = obj.get('crop_bbox', [])
            
            if seg_mask is None or len(crop_bbox) < 4:
                if len(cropped_image.shape) == 3:
                    rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2RGBA)
                else:
                    return cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGBA)
            
            x1, y1, x2, y2 = map(int, crop_bbox[:4])
            
            if len(seg_mask.shape) == 2:
                mask_crop = seg_mask[y1:y2, x1:x2]
            else:
                mask_crop = seg_mask[y1:y2, x1:x2]
                if len(mask_crop.shape) > 2:
                    mask_crop = mask_crop[:, :, 0]
            
            crop_h, crop_w = cropped_image.shape[:2]
            if mask_crop.shape != (crop_h, crop_w):
                mask_crop = cv2.resize(mask_crop, (crop_w, crop_h))
            
            binary_mask = (mask_crop > 0.5).astype(np.uint8)
            
            if len(cropped_image.shape) == 3:
                rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                rgba_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2RGBA)
            else:
                rgba_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGBA)
            
            rgba_image[:, :, 3] = binary_mask * 255
            return rgba_image
        except Exception:
            if len(cropped_image.shape) == 3:
                rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2RGBA)
            else:
                return cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGBA)
    
    @track_performance
    def save_metadata(self, scene_data: Dict[str, Any], output_dir: Path, scene_number: int) -> Dict[str, Any]:
        if not self.save_metadata_enabled:
            return {'metadata_saved': False}
            
        try:
            metadata_file = output_dir / f'scene_{scene_number:04d}_metadata.json'
            try:
                from utils.decorators import profiler
                performance_summary = profiler.get_overall_summary()
                stitching_result = scene_data.get('stitching_result', {})
                segmentation_result = scene_data.get('segmentation_result', {})
                keypoint_result = scene_data.get('keypoint_result', {})
                panorama = stitching_result.get('panorama')
                
                enhanced_metadata = {
                    'scene_number': scene_number,
                    'scene_type': scene_data.get('scene_type'),
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
                    'performance': {
                        'detailed_timings': performance_summary,
                        'total_scene_time': sum(timing['total_time'] for timing in performance_summary.values()) if performance_summary else 0,
                        'slowest_operation': max(performance_summary.keys(), key=lambda k: performance_summary[k]['avg_time']) if performance_summary else None
                    }
                }
                
                clean_metadata = self._clean_metadata_for_json(scene_data)
                for key, value in clean_metadata.items():
                    if key not in enhanced_metadata:
                        enhanced_metadata[key] = value
            except Exception:
                enhanced_metadata = self._clean_metadata_for_json(scene_data)
            
            with open(metadata_file, 'w') as f:
                json.dump(enhanced_metadata, f, indent=2)
            
            return {'metadata_saved': True, 'metadata_path': str(metadata_file), 'metadata_object': enhanced_metadata}
        except Exception as e:
            return {'metadata_saved': False, 'error': str(e), 'metadata_object': None}
    
    def _clean_metadata_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if key in ['frames', 'cropped_image', 'segmentation_mask', 'panorama', 'masked_frames', 'segmentation_result']:
                    continue
                else:
                    cleaned[key] = self._clean_metadata_for_json(value)
            return cleaned
        elif isinstance(data, list):
            return [self._clean_metadata_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

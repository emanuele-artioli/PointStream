#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Using a slow image processor')

import torch
import cv2
import numpy as np
from PIL import Image

from utils.decorators import track_performance
from server.scripts.stitcher import Stitcher
from server.scripts.segmenter import Segmenter
from server.scripts.duplicate_filter import DuplicateFilter
from server.scripts.semantic_classifier import SemanticClassifier
from server.scripts.keypointer import Keypointer
from server.scripts.saver import Saver
from server.scripts.splitter import VideoSceneSplitter
from server.scripts.muxer import MetadataMuxer
from utils import config

class PointStreamPipeline:
    def __init__(self, config_file: str = None):
        if config_file:
            config.load_config(config_file)
        self._initialize_components()
        self.processed_scenes = 0
        self.complex_scenes = 0
    
    def _initialize_components(self):
        try:
            self.stitcher = Stitcher()
            self.segmenter = Segmenter()
            self.semantic_classifier = SemanticClassifier()
            self.keypointer = Keypointer()
            self.saver = Saver()
            self.muxer = MetadataMuxer()
            self.duplicate_filter = DuplicateFilter()
        except Exception as e:
            raise
    
    def _create_masked_frame_for_objects(self, frame: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
        masked_frame = frame.copy()
        for obj in objects:
            if 'mask' in obj:
                obj_mask = obj['mask']
                background_mask = (obj_mask == 0)
                bbox = obj.get('bbox', [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    bbox_background_mask = background_mask[y1:y2, x1:x2]
                    masked_frame[y1:y2, x1:x2][bbox_background_mask] = [0, 0, 0]
        return masked_frame
    
    def _create_masked_frames(self, frames: List[np.ndarray], segmentation_result: Dict[str, Any]) -> List[np.ndarray]:
        use_reconstruction = config.get_bool('stitching', 'use_background_reconstruction', True)
        masked_frames = []
        frames_data = segmentation_result.get('frames_data', [])
        
        for i, frame in enumerate(frames):
            try:
                frame_data = None
                for fd in frames_data:
                    if fd.get('frame_index', -1) == i:
                        frame_data = fd
                        break
                if frame_data is None or not frame_data.get('objects'):
                    masked_frames.append(frame.copy())
                    continue
                if use_reconstruction:
                    masked_frame = self._reconstruct_background(frame, frame_data, frames, i)
                else:
                    masked_frame = frame.copy()
                    for obj in frame_data.get('objects', []):
                        if 'mask' in obj:
                            obj_mask = obj['mask']
                            masked_frame[obj_mask > 0] = [0, 0, 0]
                masked_frames.append(masked_frame)
            except Exception:
                masked_frames.append(frame.copy())
        return masked_frames
    
    def _reconstruct_background(self, current_frame: np.ndarray, frame_data: Dict[str, Any], 
                               all_frames: List[np.ndarray], frame_idx: int) -> np.ndarray:
        reconstructed_frame = current_frame.copy()
        combined_mask = np.zeros(current_frame.shape[:2], dtype=np.uint8)
        for obj in frame_data.get('objects', []):
            if 'mask' in obj:
                obj_mask = obj['mask']
                combined_mask = cv2.bitwise_or(combined_mask, obj_mask.astype(np.uint8))
        if np.sum(combined_mask) == 0:
            return reconstructed_frame
        
        object_areas = combined_mask > 0
        neighbor_indices = []
        for offset in [-2, -1, 1, 2]:
            neighbor_idx = frame_idx + offset
            if 0 <= neighbor_idx < len(all_frames):
                neighbor_indices.append(neighbor_idx)
        
        if neighbor_indices:
            background_content = np.zeros_like(current_frame, dtype=np.float64)
            total_weight = 0
            for neighbor_idx in neighbor_indices:
                weight = 1.0 / (abs(neighbor_idx - frame_idx))
                background_content += all_frames[neighbor_idx].astype(np.float64) * weight
                total_weight += weight
            if total_weight > 0:
                background_content = (background_content / total_weight).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)
                reconstructed_frame[object_areas] = background_content[object_areas]
                blurred_frame = cv2.GaussianBlur(reconstructed_frame, (5, 5), 0)
                distance_transform = cv2.distanceTransform(255 - dilated_mask, cv2.DIST_L2, 5)
                transition_mask = np.clip(distance_transform / 10.0, 0, 1)
                for c in range(3):
                    reconstructed_frame[:, :, c] = (
                        blurred_frame[:, :, c] * (1 - transition_mask) + 
                        current_frame[:, :, c] * transition_mask
                    ).astype(np.uint8)
        else:
            inpainted = cv2.inpaint(current_frame, combined_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            reconstructed_frame = inpainted
        return reconstructed_frame
    
    def _cleanup_panorama_black_areas(self, panorama: np.ndarray) -> np.ndarray:
        if panorama is None:
            return None
        try:
            black_threshold = config.get_int('stitching', 'cleanup_black_threshold', 10)
            gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            black_mask = (gray < black_threshold).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
            exclude_borders = config.get_bool('stitching', 'exclude_border_black_areas', True)
            h, w = black_mask.shape
            if exclude_borders:
                border_mask = np.zeros_like(black_mask)
                border_width = config.get_int('stitching', 'border_exclusion_width', 10)
                border_mask[:border_width, :] = 1
                border_mask[-border_width:, :] = 1
                border_mask[:, :border_width] = 1
                border_mask[:, -border_width:] = 1
                num_labels, labels = cv2.connectedComponents(black_mask)
                interior_black_mask = np.zeros_like(black_mask)
                for label in range(1, num_labels):
                    component_mask = (labels == label).astype(np.uint8)
                    if not np.any(component_mask & border_mask):
                        interior_black_mask |= component_mask
                final_mask = interior_black_mask
            else:
                final_mask = black_mask
            
            black_area_ratio = np.sum(final_mask) / (h * w)
            if black_area_ratio > 0.001:
                inpaint_radius = config.get_int('stitching', 'inpaint_radius', 7)
                inpainted = cv2.inpaint(panorama, final_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
                return inpainted
            else:
                return panorama
        except Exception:
            return panorama
    
    def _process_keypoints(self, segmentation_result: Dict[str, Any], frames: List[np.ndarray]) -> Dict[str, Any]:
        try:
            all_objects = []
            for frame_data in segmentation_result['frames_data']:
                objects = frame_data.get('objects', [])
                frame_idx = frame_data.get('frame_index', 0)
                if frame_idx < len(frames):
                    frame = frames[frame_idx]
                    masked_frame = self._create_masked_frame_for_objects(frame, objects)
                    for obj_idx, obj in enumerate(objects):
                        obj['frame_index'] = frame_idx
                        obj['object_id'] = f"frame_{frame_idx}_obj_{obj_idx}"
                        bbox = obj.get('bbox', [])
                        if len(bbox) >= 4:
                            x1, y1, x2, y2 = map(int, bbox[:4])
                            h, w = masked_frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            if x2 > x1 and y2 > y1:
                                obj['cropped_image'] = masked_frame[y1:y2, x1:x2]
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
            
            if all_objects:
                all_objects = self.semantic_classifier.classify_objects(all_objects)
                keypoint_result = self.keypointer.extract_keypoints(all_objects, frames)
                enhanced_objects = []
                for obj in keypoint_result.get('objects', []):
                    enhanced_obj = {
                        'object_id': obj.get('object_id', 'unknown'),
                        'frame_index': obj.get('frame_index', 0),
                        'class_name': obj.get('class_name', 'other'),
                        'original_class_name': obj.get('original_class_name', 'unknown'),
                        'semantic_category': obj.get('semantic_category', 'other'),
                        'semantic_confidence': obj.get('semantic_confidence', 0.0),
                        'classification_method': obj.get('classification_method', 'unknown'),
                        'track_id': obj.get('track_id'),
                        'confidence': obj.get('confidence', 0.0),
                        'bbox': obj.get('bbox', []),
                        'crop_bbox': obj.get('crop_bbox', []),
                        'crop_size': obj.get('crop_size', [0, 0]),
                        'keypoints': obj.get('keypoints', []),
                        'segmentation_mask': obj.get('segmentation_mask'),
                        'cropped_image': obj.get('cropped_image'),
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
                return {'objects': [], 'total_objects': 0, 'objects_with_keypoints': 0, 'processing_time': 0}
        except Exception as e:
            return {'objects': [], 'total_objects': 0, 'objects_with_keypoints': 0, 'error': str(e)}
    
    @track_performance
    def process_scene(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        scene_number = scene_data.get('scene_number', 0)
        frames = scene_data.get('frames', [])
        if not frames:
            return {'scene_type': 'Complex', 'scene_number': scene_number, 'error': 'no_frames'}
        
        try:
            frame_segmentation_result = self.segmenter.segment_frames_only(frames)
            if frame_segmentation_result.get('frames_data'):
                filtered_frames_data = self.duplicate_filter.filter_by_frame(frame_segmentation_result['frames_data'])
                frame_segmentation_result['frames_data'] = filtered_frames_data
            
            masked_frames = self._create_masked_frames(frames, frame_segmentation_result)
            stitching_result = self.stitcher.stitch_scene(masked_frames)
            
            if stitching_result['scene_type'] == 'Complex':
                return {'scene_type': 'Complex', 'scene_number': scene_number, 'stitching_result': stitching_result, 'frames': frames}
            
            panorama = stitching_result['panorama']
            if panorama is not None and config.get_bool('stitching', 'enable_panorama_cleanup', True):
                panorama = self._cleanup_panorama_black_areas(panorama)
                stitching_result['panorama'] = panorama
            
            keypoint_result = self._process_keypoints(frame_segmentation_result, frames)
            height, width, _ = frames[0].shape

            return {
                'scene_type': stitching_result['scene_type'],
                'scene_number': scene_number,
                'stitching_result': stitching_result,
                'segmentation_result': frame_segmentation_result,
                'keypoint_result': keypoint_result,
                'masked_frames': masked_frames,
                'video_properties': {
                    'resolution': {'width': width, 'height': height},
                    'fps': scene_data.get('fps'),
                    'frame_count': len(frames),
                    'start_time': scene_data.get('start_time'),
                    'end_time': scene_data.get('end_time'),
                    'duration': scene_data.get('duration'),
                }
            }
        except Exception as e:
            return {'scene_type': 'Complex', 'scene_number': scene_number, 'error': str(e), 'frames': frames}
    
    def _save_scene_objects(self, scene_data: Dict[str, Any], output_dir: Path, scene_number: int):
        if not output_dir:
            return
        self.saver.save_scene_objects(scene_data, output_dir, scene_number)
        metadata_result = self.saver.save_metadata(scene_data, output_dir, scene_number)
        metadata_obj = metadata_result.get('metadata_object')
        if metadata_obj:
            pzm_file_path = output_dir / f"scene_{scene_number:04d}_metadata.pzm"
            try:
                self.muxer.compress_metadata_object(metadata_obj, pzm_file_path)
            except Exception:
                pass
        elif metadata_result.get('metadata_saved'):
            json_file_path = Path(metadata_result.get('metadata_path'))
            if json_file_path.exists():
                self.muxer.compress_metadata_file(str(json_file_path))
    
    @track_performance
    def process_video(self, input_video: str, output_dir: str = None, enable_saving: bool = True) -> Dict[str, Any]:
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        
        splitter = VideoSceneSplitter(input_video=input_video, output_dir=None, enable_encoding=False)
        
        try:
            scene_generator = splitter.process_video_realtime_generator()
            for scene_data in scene_generator:
                if isinstance(scene_data, dict) and scene_data.get('status') in ['complete', 'error']:
                    break
                
                scene_number = scene_data.get('scene_number', 0)
                result = self.process_scene(scene_data)
                self.processed_scenes += 1
                
                if result.get('scene_type') == 'Complex':
                    self.complex_scenes += 1
                    if enable_saving and output_dir:
                        self.saver.save_complex_scene_video(
                            frames=scene_data.get('frames', []),
                            output_path=str(output_path / f"scene_{scene_number:04d}_complex.mp4"),
                            fps=scene_data.get('fps', 25.0),
                            scene_number=scene_number
                        )
                else:
                    if enable_saving and output_dir:
                        self._save_scene_results(result, output_path, scene_number)
            
            splitter.close()
            return self._generate_processing_summary()
        except Exception as e:
            raise
    
    def _save_scene_results(self, result: Dict[str, Any], output_path: Path, scene_number: int):
        try:
            (output_path / "panoramas").mkdir(exist_ok=True)
            stitching_result = result.get('stitching_result', {})
            panorama = stitching_result.get('panorama')
            if panorama is not None:
                panorama_path = output_path / "panoramas" / f"scene_{scene_number:04d}_panorama.jpg"
                cv2.imwrite(str(panorama_path), panorama)
            self._save_scene_objects(result, output_path, scene_number)
        except Exception:
            pass
    
    def _generate_processing_summary(self) -> Dict[str, Any]:
        return {
            'processed_scenes': self.processed_scenes,
            'complex_scenes': self.complex_scenes,
            'simple_scenes': self.processed_scenes - self.complex_scenes,
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PointStream Pipeline")
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--output-dir', default='./pointstream_output', help='Output directory')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--no-saving', action='store_true', help='Disable saving results')
    args = parser.parse_args()
    
    if not Path(args.input_video).exists():
        print(f"Error: Input video not found: {args.input_video}")
        sys.exit(1)
    
    try:
        pipeline = PointStreamPipeline(config_file=args.config)
        pipeline.process_video(
            input_video=args.input_video,
            output_dir=args.output_dir if not args.no_saving else None,
            enable_saving=not args.no_saving
        )
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

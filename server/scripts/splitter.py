#!/usr/bin/env python3
import os
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

try:
    from scenedetect import ContentDetector, open_video, FrameTimecode
    from scenedetect.detectors import AdaptiveDetector, HistogramDetector, HashDetector, ThresholdDetector
    from scenedetect.scene_manager import SceneManager, Interpolation
    from scenedetect.backends import AVAILABLE_BACKENDS
    import cv2
    from utils import config
    from utils.decorators import time_step
except ImportError as e:
    raise

class VideoSceneSplitter:
    def __init__(self, input_video: str, output_dir: str = None, 
                 batch_size: int = None, max_scene_frames: int = None, config_file: str = None,
                 enable_encoding: bool = True, detector_type: str = None, video_backend: str = None):
        if config_file:
            config.load_config(config_file)
        
        self.enable_encoding = enable_encoding
        self.input_video = Path(input_video)
        if not self.input_video.exists():
            raise FileNotFoundError(f"Input video not found: {input_video}")
        
        self.detector_type = detector_type or config.get_str('scene_detection', 'detector_type', 'content')
        self.video_backend = video_backend or config.get_str('encoding', 'backend', 'auto')
        
        if output_dir is None and enable_encoding:
            pattern = config.get_str('scene_detection', 'default_output_pattern', '{input_stem}_scenes')
            self.output_dir = self.input_video.parent / pattern.format(input_stem=self.input_video.stem)
        elif output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = None
        
        if self.output_dir and enable_encoding:
            self.output_dir.mkdir(exist_ok=True)
        
        self.batch_size = batch_size or config.get_int('scene_detection', 'batch_size', 100)
        self.max_scene_frames = max_scene_frames or config.get_int('scene_detection', 'default_max_frames', 1000)
        self.overlap_frames = config.get_int('scene_detection', 'overlap_frames', 10)
        self.video_fps, self.total_frames = self._get_video_properties()
        self.scene_threshold = config.get_float('scene_detection', 'threshold', 30.0)
        self.min_scene_len = config.get_int('scene_detection', 'min_scene_len', 15)
        self.luma_only = config.get_bool('scene_detection', 'luma_only', False)
        self.enable_downscaling = config.get_bool('scene_detection', 'enable_downscaling', True)
        self.downscale_factor = config.get_int('scene_detection', 'downscale_factor', 0)
        self.adaptive_window = config.get_int('scene_detection', 'adaptive_window_frames', 60)
        self.histogram_bins = config.get_int('scene_detection', 'histogram_bins', 256)
        self.hash_size = config.get_int('scene_detection', 'hash_size', 8)
        interpolation_str = config.get_str('scene_detection', 'interpolation_method', 'linear')
        self.interpolation_method = getattr(Interpolation, interpolation_str.upper(), Interpolation.LINEAR)
        self.detector_weights = {
            'content': config.get_float('scene_detection', 'content_weight', 1.0),
            'adaptive': config.get_float('scene_detection', 'adaptive_weight', 0.8),
            'histogram': config.get_float('scene_detection', 'histogram_weight', 0.6),
            'hash': config.get_float('scene_detection', 'hash_weight', 0.9)
        }
        
        self.video_stream = self._initialize_video_stream()
        self.processed_scenes = []
        self.scene_counter = 0
        self.pending_scene_start = None
        self.last_batch_end_frame = None
        self.cap = cv2.VideoCapture(str(self.input_video))

    def _get_video_properties(self) -> Tuple[float, int]:
        cap = cv2.VideoCapture(str(self.input_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return fps, frame_count

    def _extract_scene_frames(self, start_time: float, end_time: float) -> List[any]:
        start_frame = int(start_time * self.video_fps)
        end_frame = int(end_time * self.video_fps)
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_num in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame.copy())
        return frames

    @time_step(track_processing=True)
    def _detect_scene_cuts_in_batch(self, start_frame: int, end_frame: int) -> List[float]:
        start_time = start_frame / self.video_fps
        end_time = end_frame / self.video_fps
        try:
            if self.detector_type == 'multi':
                return self._detect_with_multiple_detectors(start_time, end_time)
            else:
                return self._detect_with_single_detector(start_time, end_time)
        except Exception:
            return []

    def _detect_with_single_detector(self, start_time: float, end_time: float) -> List[float]:
        detector = self._create_detector()
        scene_manager = SceneManager(stats_manager=None)
        scene_manager.add_detector(detector)
        if self.enable_downscaling:
            if self.downscale_factor == 0:
                scene_manager.auto_downscale = True
            else:
                scene_manager.downscale = self.downscale_factor
            scene_manager.interpolation = self.interpolation_method
        if start_time > 0:
            self.video_stream.seek(start_time)
        duration_seconds = end_time - start_time
        duration = FrameTimecode(duration_seconds, fps=self.video_fps)
        scene_manager.detect_scenes(video=self.video_stream, duration=duration, show_progress=False)
        scene_list = scene_manager.get_scene_list()
        cut_times = []
        for i, scene in enumerate(scene_list):
            scene_start = scene[0].get_seconds()
            if i > 0 and start_time <= scene_start < end_time:
                cut_times.append(scene_start)
        return cut_times

    def _detect_with_multiple_detectors(self, start_time: float, end_time: float) -> List[float]:
        detector_types = ['content', 'adaptive', 'histogram', 'hash']
        all_cuts = {}
        for det_type in detector_types:
            if self.detector_weights.get(det_type, 0) > 0:
                try:
                    detector = self._create_detector(det_type)
                    scene_manager = SceneManager()
                    scene_manager.add_detector(detector)
                    if self.enable_downscaling:
                        scene_manager.auto_downscale = True
                    if start_time > 0:
                        self.video_stream.seek(start_time)
                    duration_seconds = end_time - start_time
                    duration = FrameTimecode(duration_seconds, fps=self.video_fps)
                    scene_manager.detect_scenes(video=self.video_stream, duration=duration, show_progress=False)
                    scene_list = scene_manager.get_scene_list()
                    cuts = [scene[0].get_seconds() for i, scene in enumerate(scene_list) if i > 0 and start_time <= scene[0].get_seconds() < end_time]
                    weight = self.detector_weights[det_type]
                    for cut_time in cuts:
                        if cut_time not in all_cuts:
                            all_cuts[cut_time] = 0
                        all_cuts[cut_time] += weight
                except Exception:
                    pass
        min_weight = config.get_float('scene_detection', 'multi_detector_threshold', 1.0)
        final_cuts = [cut_time for cut_time, weight in all_cuts.items() if weight >= min_weight]
        final_cuts.sort()
        return final_cuts

    def _process_batch_realtime(self, batch_start: int, batch_end: int) -> List[Dict[str, Any]]:
        batch_end_time = batch_end / self.video_fps
        completed_scenes = []
        scene_cuts = self._detect_scene_cuts_in_batch(batch_start, batch_end)
        
        if self.pending_scene_start is None:
            self.pending_scene_start = 0.0
        
        for cut_time in scene_cuts:
            if self.pending_scene_start is not None:
                scene_data = self._create_scene_data_no_encoding(self.pending_scene_start, cut_time, self.scene_counter + 1)
                if scene_data:
                    completed_scenes.append(scene_data)
                    self.scene_counter += 1
                    self.processed_scenes.append((self.pending_scene_start, cut_time))
                self.pending_scene_start = cut_time
        
        if self.pending_scene_start is not None:
            pending_duration = batch_end_time - self.pending_scene_start
            max_duration = self.max_scene_frames / self.video_fps
            if pending_duration >= max_duration:
                scene_data = self._create_scene_data_no_encoding(self.pending_scene_start, batch_end_time, self.scene_counter + 1)
                if scene_data:
                    completed_scenes.append(scene_data)
                    self.scene_counter += 1
                    self.processed_scenes.append((self.pending_scene_start, batch_end_time))
                self.pending_scene_start = batch_end_time
        
        self.last_batch_end_frame = batch_end
        return completed_scenes

    def _create_scene_data_no_encoding(self, start_time: float, end_time: float, scene_num: int) -> Optional[Dict[str, Any]]:
        duration = end_time - start_time
        min_scene_duration = config.get_float('scene_detection', 'min_scene_duration', 0.5)
        if duration < min_scene_duration:
            return None
        frames = self._extract_scene_frames(start_time, end_time)
        return {
            'scene_number': scene_num,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'frames': frames,
            'frame_count': len(frames),
            'fps': self.video_fps,
            'processing_timestamp': time.time()
        }

    def _initialize_video_stream(self):
        try:
            if self.video_backend == 'auto':
                for backend in ['pyav', 'opencv', 'moviepy']:
                    if backend in AVAILABLE_BACKENDS:
                        try:
                            return open_video(str(self.input_video), backend=backend)
                        except Exception:
                            continue
            else:
                if self.video_backend in AVAILABLE_BACKENDS:
                    return open_video(str(self.input_video), backend=self.video_backend)
            return open_video(str(self.input_video), backend='opencv')
        except Exception as e:
            raise

    def _create_detector(self, detector_type: str = None):
        detector_type = detector_type or self.detector_type
        if detector_type == 'content':
            return ContentDetector(threshold=self.scene_threshold, min_scene_len=self.min_scene_len, luma_only=self.luma_only)
        elif detector_type == 'adaptive':
            return AdaptiveDetector(adaptive_threshold=self.scene_threshold, min_scene_len=self.min_scene_len, window_width=self.adaptive_window, luma_only=self.luma_only)
        elif detector_type == 'histogram':
            return HistogramDetector(threshold=self.scene_threshold, min_scene_len=self.min_scene_len, bins=self.histogram_bins)
        elif detector_type == 'hash':
            try:
                return HashDetector(threshold=self.scene_threshold, min_scene_len=self.min_scene_len)
            except TypeError:
                return HashDetector(threshold=self.scene_threshold, min_scene_len=self.min_scene_len, hash_size=self.hash_size)
        elif detector_type == 'threshold':
            return ThresholdDetector(threshold=self.scene_threshold, min_scene_len=self.min_scene_len)
        else:
            return ContentDetector(threshold=self.scene_threshold, min_scene_len=self.min_scene_len, luma_only=self.luma_only)
        
    def process_video_realtime_generator(self):
        for batch_start in range(0, self.total_frames, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.total_frames)
            try:
                completed_scenes = self._process_batch_realtime(batch_start, batch_end)
                for scene_data in completed_scenes:
                    yield scene_data
            except Exception as e:
                yield {'status': 'error', 'batch_start': batch_start, 'batch_end': batch_end, 'error': str(e)}
                if not config.get_bool('encoding', 'continue_on_error', False):
                    break
        
        if self.pending_scene_start is not None:
            final_time = self.total_frames / self.video_fps
            scene_data = self._create_scene_data_no_encoding(self.pending_scene_start, final_time, self.scene_counter + 1)
            if scene_data:
                self.scene_counter += 1
                self.processed_scenes.append((self.pending_scene_start, final_time))
                yield scene_data
        
        self.cap.release()
        yield {'status': 'complete', 'summary': {'total_scenes': self.scene_counter}}

    def close(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Video Scene Splitter")
    parser.add_argument('input_video', nargs='?', help='Path to input video file')
    parser.add_argument('output_dir', nargs='?', default=None, help='Output directory for scene videos')
    parser.add_argument('--batch-size', type=int, default=None, help='Number of frames per batch')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum frames per scene')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--enable-encoding', action='store_true', help='Enable encoding scenes to files')
    parser.add_argument('--detector', choices=['content', 'adaptive', 'histogram', 'hash', 'threshold', 'multi'], default=None, help='Scene detection algorithm')
    parser.add_argument('--backend', choices=['opencv', 'pyav', 'moviepy', 'auto'], default=None, help='Video backend')
    args = parser.parse_args()

    if not args.input_video:
        parser.error("input_video is required")
    
    try:
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
        
        for scene_data in splitter.process_video_realtime_generator():
            if scene_data.get('status') == 'complete':
                break
            elif scene_data.get('status') == 'error':
                print(f"Error processing batch: {scene_data.get('error')}")
                continue
            else:
                print(f"Processed scene {scene_data['scene_number']}")

        splitter.close()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
Stage 1: Scene Analysis and Motion Classification.
"""
from typing import List, Dict, Any, Generator, Tuple
import cv2
import numpy as np
from scenedetect.detectors import ContentDetector
from .. import config
from ..utils.video_utils import get_video_properties, read_frames_with_indices

Scene = Dict[str, Any] # Type alias for clarity

def _detect_scenes_stream(video_path: str, threshold: float) -> Generator[Tuple[List[np.ndarray], int, int], None, None]:
    """Processes a video frame-by-frame to detect scenes in a streaming fashion."""
    detector = ContentDetector(threshold=threshold)
    scene_buffer = []
    last_cut_frame = 0
    
    for frame_num, frame in read_frames_with_indices(video_path):
        cut_detected = detector.process_frame(frame_num, frame)

        if cut_detected and scene_buffer:
            yield scene_buffer, last_cut_frame, frame_num - 1
            scene_buffer = [frame]
            last_cut_frame = frame_num
        else:
            scene_buffer.append(frame)

    if scene_buffer:
        end_frame_num = last_cut_frame + len(scene_buffer) - 1
        yield scene_buffer, last_cut_frame, end_frame_num

def _classify_camera_motion(scene_frames: List[np.ndarray]) -> str:
    """Analyzes a list of frames to classify the global camera motion."""
    if len(scene_frames) < 5:
        return "STATIC"

    scale = config.MOTION_DOWNSAMPLE_FACTOR
    prev_gray = cv2.cvtColor(cv2.resize(scene_frames[0], (0,0), fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    
    magnitudes = []
    for i in range(1, len(scene_frames)):
        curr_gray = cv2.cvtColor(cv2.resize(scene_frames[i], (0,0), fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(np.mean(magnitude))
        prev_gray = curr_gray

    avg_motion = np.mean(magnitudes)
    
    if avg_motion < config.MOTION_CLASSIFIER_THRESHOLD:
        return "STATIC"
    elif avg_motion < config.MOTION_CLASSIFIER_THRESHOLD * 5:
        return "SIMPLE"
    else:
        return "COMPLEX"

def run_analysis_pipeline(video_path: str) -> Generator[Scene, None, None]:
    """
    The main orchestrator for the scene analysis stage. This is a generator
    that yields processed scenes one by one.

    Args:
        video_path: The path to the input video.

    Yields:
        Scene dictionaries, each containing metadata AND the frames for that scene.
    """
    if not get_video_properties(video_path):
        return

    print("--- Starting Stage 1: Scene Analysis (Streaming) ---")
    scene_generator = _detect_scenes_stream(video_path, config.SCENE_DETECTOR_THRESHOLD)
    
    for i, (scene_frames, start_frame, end_frame) in enumerate(scene_generator):
        if not scene_frames:
            continue
            
        motion_type = _classify_camera_motion(scene_frames)
        
        scene_info: Scene = {
            "scene_index": i,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "motion_type": motion_type,
            "frames": scene_frames
        }
        print(f"  -> Stage 1 yielding Scene {i} (Frames {start_frame}-{end_frame}): {motion_type}")
        yield scene_info
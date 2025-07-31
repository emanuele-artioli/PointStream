"""
Stage 1: Scene Analysis and Motion Classification.

This module takes a video, splits it into scenes, and classifies the camera
motion in each scene as STATIC, SIMPLE (pan/zoom), or COMPLEX.

This implementation uses a single-pass, streaming approach.
"""

from typing import List, Dict, Any, Generator, Tuple
import cv2
import numpy as np
from scenedetect.detectors import ContentDetector
from .. import config
from ..utils.video_utils import get_video_properties, read_frames_with_indices

# A type alias for clarity
Scene = Dict[str, Any]

def detect_scenes_stream(video_path: str, threshold: float) -> Generator[Tuple[List[np.ndarray], int, int], None, None]:
    """
    Processes a video frame-by-frame to detect scenes in a streaming fashion.
    This version uses the shared `read_frames_with_indices` utility.

    Args:
        video_path: Path to the video file.
        threshold: The threshold for detecting a scene change.

    Yields:
        A tuple containing (list_of_frames, start_frame_number, end_frame_number).
    """
    detector = ContentDetector(threshold=threshold)
    scene_buffer = []
    last_cut_frame = 0
    
    # Use the refactored utility function for cleaner code
    for frame_num, frame in read_frames_with_indices(video_path):
        cut_detected = detector.process_frame(frame_num, frame)

        if cut_detected and scene_buffer:
            yield scene_buffer, last_cut_frame, frame_num - 1
            scene_buffer = [frame]
            last_cut_frame = frame_num
        else:
            scene_buffer.append(frame)

    # Yield the final scene after the loop finishes
    if scene_buffer:
        # The end frame is the last frame number read
        end_frame_num = last_cut_frame + len(scene_buffer) - 1
        yield scene_buffer, last_cut_frame, end_frame_num


def classify_camera_motion(scene_frames: List[np.ndarray]) -> str:
    """
    Analyzes a list of frames to classify the global camera motion.

    Args:
        scene_frames: A list of frames belonging to a single scene.

    Returns:
        A string classification: 'STATIC', 'SIMPLE' (pan/zoom/tilt), or 'COMPLEX'.
    """
    if len(scene_frames) < 5:  # Not enough frames to analyze
        return "STATIC"

    # For performance, analyze motion on downsampled, grayscale frames
    scale = config.MOTION_DOWNSAMPLE_FACTOR
    prev_frame_gray = cv2.cvtColor(cv2.resize(scene_frames[0], (0, 0), fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    
    motion_vectors = []

    for i in range(1, len(scene_frames)):
        frame_gray = cv2.cvtColor(cv2.resize(scene_frames[i], (0, 0), fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        
        # --- Ablation 1.1 (Motion Metric Robustness) can be tested here ---
        # Method A: Dense Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.mean(magnitude)
        motion_vectors.append(avg_magnitude)
        
        # Method B: Affine Transformation (can be added for the ablation study)
        # M, _ = cv2.estimateAffine2D(prev_frame_gray, frame_gray)
        
        prev_frame_gray = frame_gray

    avg_motion = np.mean(motion_vectors)
    
    # Simple thresholding for now. More complex logic can be added.
    if avg_motion < config.MOTION_CLASSIFIER_THRESHOLD:
        return "STATIC"
    # A more advanced check would see if the affine transform was consistently found.
    # For now, we'll just use a higher threshold for complex motion.
    elif avg_motion < config.MOTION_CLASSIFIER_THRESHOLD * 5:
        return "SIMPLE"
    else:
        return "COMPLEX"


def run_analysis_pipeline(video_path: str) -> Generator[Scene, None, None]:
    """
    The main orchestrator for the scene analysis stage.
    This is now a GENERATOR that yields processed scenes one by one.

    Args:
        video_path: The path to the input video.

    Yields:
        Scene dictionaries, each containing metadata AND the frames for that scene.
    """
    if not get_video_properties(video_path):
        return

    print("--- Starting Stage 1: Scene Analysis (Streaming) ---")
    scene_generator = detect_scenes_stream(video_path, config.SCENE_DETECTOR_THRESHOLD)
    
    for i, (scene_frames, start_frame, end_frame) in enumerate(scene_generator):
        if not scene_frames:
            continue
            
        motion_type = classify_camera_motion(scene_frames)
        
        scene_info: Scene = {
            "scene_index": i,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "motion_type": motion_type,
            "frames": scene_frames  # <-- CRITICAL: Pass the frames along
        }
        print(f"  -> Stage 1 yielding Scene {i} (Frames {start_frame}-{end_frame}): {motion_type}")
        yield scene_info
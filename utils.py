import subprocess
import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from skimage.metrics import structural_similarity as ssim
import cv2

def classify_scene_motion(frames: List[np.ndarray]) -> str:
    """
    Classifies a list of frames as 'static' or 'dynamic' based on optical flow.
    This version samples which CONSECUTIVE pairs to analyze for better accuracy.
    """
    if len(frames) < 2:
        return "static"

    MOTION_THRESHOLD = 0.5 
    MAX_FRAMES_TO_SAMPLE = 45
    frame_interval = max(1, len(frames) // MAX_FRAMES_TO_SAMPLE)
    
    # 1. Pre-process all frames once for efficiency (grayscale + downscale)
    processed_frames = [
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25) 
        for f in frames
    ]

    total_flow = 0
    flow_calculations = 0
    
    # 2. Loop through consecutive frames, but only calculate flow periodically
    for i in range(1, len(processed_frames)):
        # This check determines if we run the expensive calculation for this pair
        if i % frame_interval == 0:
            prev_frame = processed_frames[i-1]
            current_frame = processed_frames[i]
        
            flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
            total_flow += np.mean(magnitude)
            flow_calculations += 1

    if flow_calculations == 0:
        return "static"

    avg_motion = total_flow / flow_calculations
    scene_type = "dynamic" if avg_motion > MOTION_THRESHOLD else "static"
    print(f"  -> Analyzed motion for segment. Avg: {avg_motion:.2f}. Type: {scene_type.upper()}")
    
    return scene_type

def save_video_segment(video_path: str, start_frame: int, end_frame: int, fps: float, output_path: str):
    """Saves a video segment using FFmpeg by re-encoding for accuracy."""
    start_time = start_frame / fps
    end_time = end_frame / fps
    
    print(f"  -> Saving and re-encoding segment to '{output_path}'...")
    command = [
        'ffmpeg', '-y', 
        '-i', video_path,
        '-ss', str(start_time), 
        '-to', str(end_time),
        '-avoid_negative_ts', '1', 
        output_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_video_properties(video_path: str) -> Tuple[int, float]:
    """Gets total frame count and FPS of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps

def extract_frames(video_path: str, frame_range: Tuple[int, int]) -> List[np.ndarray]:
    """Extracts a range of frames from a video file."""
    start_frame, end_frame = frame_range
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame_pos = start_frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effective_end_frame = min(end_frame, total_frames - 1)

    while current_frame_pos <= effective_end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        current_frame_pos += 1
    
    cap.release()
    return frames

def detect_scene_changes(frames: List[np.ndarray], threshold: float, analysis_window: int) -> Dict[int, float]:
    """Detects significant scene changes within a list of frames."""
    if len(frames) < 2:
        return {}

    processed_frames = [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.25, fy=0.25)
        for frame in frames
    ]
    scene_changes = {}

    def _find_best_cut_in_range(start: int, end: int) -> Optional[Tuple[int, float]]:
        """Finds the single most significant cut in a small window."""
        min_score, cut_location = 1.0, None
        if start >= end: return None
        for i in range(start + 1, min(end + 1, len(processed_frames))):
            score, _ = ssim(processed_frames[i-1], processed_frames[i], full=True)
            if score < min_score:
                min_score, cut_location = score, i
        if cut_location is not None and min_score < threshold:
            return cut_location, min_score
        return None

    def _find_changes_recursive(start: int, end: int):
        """Recursively searches for scene changes."""
        if (end - start) <= analysis_window:
            best_cut = _find_best_cut_in_range(start, end)
            if best_cut:
                cut_idx, cut_score = best_cut
                if not any(abs(cut_idx - exist_cut) < analysis_window // 2 for exist_cut in scene_changes):
                    scene_changes[cut_idx] = cut_score
            return
        
        if end >= len(processed_frames): end = len(processed_frames) -1
        boundary_score, _ = ssim(processed_frames[start], processed_frames[end], full=True)
        if boundary_score < threshold:
            mid = (start + end) // 2
            _find_changes_recursive(start, mid)
            _find_changes_recursive(mid + 1, end)

    _find_changes_recursive(0, len(frames) - 1)
    return scene_changes
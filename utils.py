import subprocess
import numpy as np
from typing import List, Tuple, Dict, Optional
from skimage.metrics import structural_similarity as ssim
import cv2

def extract_frames(
    video_path: str, 
    frame_range: Tuple[int, int]
) -> List[np.ndarray]:
    """
    Extract frames from video using OpenCV and return as numpy arrays.
    
    Args:
        video_path (str): Path to the input video file
        frame_range (Tuple[int, int]): Tuple of (start_frame, end_frame) inclusive
        
    Returns:
        List[np.ndarray]: List of frames as numpy arrays (BGR format)
    """
    start_frame, end_frame = frame_range
    frame_count = end_frame - start_frame + 1
    
    # Open video with OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frames = []
    
    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract exactly frame_count frames sequentially
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def detect_scene_changes(
    frames: List[np.ndarray], 
    threshold: float = 0.2, 
    analysis_window: int = 25
) -> Dict[int, float]:
    """
    Detects single, significant scene changes using a corrected hybrid approach.

    This method uses recursion to find small, unstable windows. It then finds
    the single best cut within that window and stops, preventing cut clusters.

    Args:
        frames (List[np.ndarray]): A list of frames (BGR NumPy arrays).
        threshold (float): The SSIM score below which a change is considered significant.
        analysis_window (int): The window size for performing the final "best cut" analysis.

    Returns:
        Dict[int, float]: A dictionary of significant scene change frame indices.
    """
    if len(frames) < 2:
        return {}

    # --- Step 1: Pre-process all frames once for efficiency ---
    processed_frames = [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.25, fy=0.25)
        for frame in frames
    ]

    scene_changes = {}

    def _find_best_cut_in_range(start: int, end: int) -> Optional[Tuple[int, float]]:
        """Sequentially scans a range to find the single lowest SSIM score."""
        min_score = 1.0
        cut_location = None

        if start >= end:
            return None

        for i in range(start + 1, end + 1):
            score, _ = ssim(processed_frames[i-1], processed_frames[i], full=True)
            if score < min_score:
                min_score = score
                cut_location = i
        
        if cut_location is not None and min_score < threshold:
            return cut_location, min_score
        return None

    def _find_changes_recursive(start: int, end: int):
        # Base Case: The segment is small enough for final analysis.
        if (end - start) <= analysis_window:
            best_cut = _find_best_cut_in_range(start, end)
            if best_cut:
                cut_frame_index, cut_score = best_cut
                # Check if a cut already exists nearby to prevent duplicates
                # This is an extra safeguard.
                is_too_close = any(abs(cut_frame_index - existing_cut) < analysis_window // 2 for existing_cut in scene_changes)
                if not is_too_close:
                    scene_changes[cut_frame_index] = cut_score
            # CRUCIAL: Stop recursing this branch after analysis.
            return

        # Recursive Step: For segments larger than the analysis window.
        boundary_score, _ = ssim(processed_frames[start], processed_frames[end], full=True)

        if boundary_score < threshold:
            mid = (start + end) // 2
            _find_changes_recursive(start, mid)
            _find_changes_recursive(mid + 1, end)
        # If boundary_score is high, we prune the segment and do nothing.

    # Start the recursive search
    _find_changes_recursive(0, len(frames) - 1)
    return scene_changes
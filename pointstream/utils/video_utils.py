"""
Generic video processing utilities using OpenCV.
"""
from typing import Generator, Tuple, Optional
import cv2
import numpy as np

def get_video_properties(video_path: str) -> Optional[Tuple[int, float, int, int]]:
    """
    Retrieves essential properties from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        A tuple containing (frame_count, fps, width, height), or None if
        the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    return frame_count, fps, width, height


def read_frames_with_indices(video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    A generator function that reads frames from a video file one by one,
    yielding both the frame index and the frame itself.

    This is memory-efficient and highly reusable.

    Args:
        video_path: Path to the video file.

    Yields:
        A tuple of (frame_index, frame_as_numpy_array).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_index, frame
        frame_index += 1
    
    cap.release()
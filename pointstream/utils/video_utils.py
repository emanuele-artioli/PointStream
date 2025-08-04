"""
Generic video processing utilities using OpenCV and FFmpeg.
"""
from typing import Generator, Tuple, Optional, List # <-- FIX: Added 'List' import
import cv2
import numpy as np
import subprocess

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
    A generator that reads frames from a video file one by one, yielding
    both the frame index and the frame itself. This is highly memory-efficient.

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

def save_frames_as_video(output_path: str, frames: List[np.ndarray], fps: float):
    """
    Saves a list of NumPy frames to a video file using a direct FFmpeg pipe
    for maximum compatibility and performance.

    Args:
        output_path: The path to save the output video file.
        frames: A list of frames (as NumPy arrays in BGR format).
        fps: The desired frames per second for the output video.
    """
    if not frames:
        print("Warning: No frames provided to save_frames_as_video.")
        return

    height, width, _ = frames[0].shape
    
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',  # Frame size
        '-pix_fmt', 'bgr24',       # Input pixel format from OpenCV
        '-r', str(fps),            # Frames per second
        '-i', '-',                 # Input comes from stdin
        '-an',                     # No audio
        '-vcodec', 'libx264',      # Use H.264 codec
        '-pix_fmt', 'yuv420p',     # Standard pixel format for compatibility
        output_path
    ]
    
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for frame in frames:
            process.stdin.write(frame.tobytes())
            
        process.stdin.close()
        process.wait()
        if process.returncode != 0:
            print(f"Error during FFmpeg execution: {process.stderr.read().decode()}")
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Please ensure FFmpeg is installed and in your system's PATH.")
    except Exception as e:
        print(f"An unexpected error occurred while saving the video: {e}")
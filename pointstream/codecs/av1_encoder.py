"""
AV1 encoder for handling complex scenes in PointStream.
Complex scenes that cannot be effectively analyzed are encoded using traditional AV1 codec.
"""
import subprocess
from pathlib import Path
from typing import List
import tempfile
import cv2
import numpy as np
from .. import config


def encode_complex_scene_av1(frames: List[np.ndarray], output_path: str, fps: float = 30.0) -> str:
    """
    Encode a complex scene using AV1 codec.
    
    Args:
        frames: List of frame arrays
        output_path: Path to save the encoded video
        fps: Frame rate for the video
        
    Returns:
        Path to the encoded AV1 file
    """
    print(f"  -> Encoding complex scene with AV1 ({len(frames)} frames)")
    
    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save frames as temporary images
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = temp_path / f"frame_{i:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
        
        # Use ffmpeg to encode with AV1
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-framerate', str(fps),
            '-i', str(temp_path / 'frame_%06d.png'),
            '-c:v', 'libaom-av1',  # AV1 codec
            '-crf', '30',  # Quality setting (lower = higher quality)
            '-b:v', '0',   # Variable bitrate
            '-threads', '4',  # Use multiple threads
            str(output_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"     -> AV1 encoding successful: {output_file}")
                return str(output_file)
            else:
                print(f"     -> AV1 encoding failed: {result.stderr}")
                # Fallback to H.264 if AV1 fails
                return _encode_fallback_h264(frames, output_path, fps)
        
        except subprocess.TimeoutExpired:
            print("     -> AV1 encoding timed out, using H.264 fallback")
            return _encode_fallback_h264(frames, output_path, fps)
        except FileNotFoundError:
            print("     -> ffmpeg not found, using OpenCV H.264 fallback")
            return _encode_fallback_h264(frames, output_path, fps)


def _encode_fallback_h264(frames: List[np.ndarray], output_path: str, fps: float = 30.0) -> str:
    """Fallback H.264 encoding using OpenCV when AV1/ffmpeg is not available."""
    print("     -> Using H.264 fallback encoding")
    
    output_file = Path(output_path).with_suffix('.mp4')
    h, w = frames[0].shape[:2]
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_file), fourcc, fps, (w, h))
    
    try:
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"     -> H.264 encoding successful: {output_file}")
        return str(output_file)
    except Exception as e:
        print(f"     -> H.264 encoding failed: {e}")
        out.release()
        return None


def get_av1_file_size(file_path: str) -> int:
    """Get the file size of an AV1 encoded file in bytes."""
    try:
        return Path(file_path).stat().st_size
    except FileNotFoundError:
        return 0

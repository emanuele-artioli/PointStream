# Take a video, find scene cuts, and save each scene as a separate video file.

import os
import subprocess
import json
from typing import List, Tuple


def _available_video_encoders() -> set:
    command = ["ffmpeg", "-hide_banner", "-encoders"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    encoders = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def _run_encode_command(command: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True)


def _build_ffmpeg_base(video_path: str, start_time: float, end_time: float) -> List[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-y",
        "-ss", f"{start_time:.6f}",
        "-to", f"{end_time:.6f}",
        "-i", video_path,
    ]


def _encode_scene_with_fallback(video_path: str, start_time: float, end_time: float, output_path: str) -> None:
    base = _build_ffmpeg_base(video_path, start_time, end_time)
    available = _available_video_encoders()

    hw_candidates = [
        ("av1_nvenc", ["-c:v", "av1_nvenc", "-cq", "22", "-preset", "p4", "-pix_fmt", "yuv420p"]),
        ("av1_qsv", ["-c:v", "av1_qsv", "-global_quality", "22", "-preset", "medium", "-pix_fmt", "nv12"]),
        ("av1_amf", ["-c:v", "av1_amf", "-quality", "quality", "-pix_fmt", "nv12"]),
        ("av1_vaapi", ["-vaapi_device", "/dev/dri/renderD128", "-vf", "format=nv12,hwupload", "-c:v", "av1_vaapi", "-qp", "22"]),
    ]

    for encoder_name, encoder_args in hw_candidates:
        if encoder_name not in available:
            continue
        hw_command = base + encoder_args + ["-an", output_path]
        hw_result = _run_encode_command(hw_command)
        if hw_result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return

    sw_command = base + [
        "-hwaccel", "none",
        "-c:v", "libsvtav1",
        "-crf", "15",
        "-preset", "3",
        "-pix_fmt", "yuv420p10le",
        "-an",
        output_path,
    ]
    sw_result = _run_encode_command(sw_command)
    if sw_result.returncode != 0:
        raise RuntimeError(
            "Software AV1 encoding failed with libsvtav1.\n"
            f"Command: {' '.join(sw_command)}\n"
            f"stderr:\n{sw_result.stderr}"
        )


def _ffprobe_video_stream(video_path: str) -> dict:
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in: {video_path}")
    stream = streams[0]
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video resolution for: {video_path}")
    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    return {"width": width, "height": height, "duration": duration}


def _detect_scene_timestamps(video_path: str, threshold: float) -> List[float]:
    escaped_path = video_path.replace("'", "'\\''")
    filter_graph = f"movie='{escaped_path}',select=gt(scene\\,{threshold})"
    command = [
        "ffprobe",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", filter_graph,
        "-show_entries", "frame=pts_time",
        "-of", "csv=p=0",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    timestamps = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped:
            timestamps.append(float(stripped))
    return timestamps

def find_scene_cuts(video_path: str, threshold: float = 0.35) -> List[Tuple[float, float]]:
    """
    Find scene cuts in a video using FFmpeg scene-change detection.

    Args:
        video_path (str): Path to the input video file.
        threshold (float): Threshold for histogram difference to detect scene cuts.

    Returns:
        List[Tuple[float, float]]: List of tuples containing start and end timestamps (seconds) of each scene.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    stream_info = _ffprobe_video_stream(video_path)
    duration = stream_info["duration"]
    if duration <= 0:
        raise RuntimeError(f"Could not read video duration for: {video_path}")

    cut_points = _detect_scene_timestamps(video_path, threshold)
    boundaries = [0.0] + cut_points + [duration]
    scene_cuts: List[Tuple[float, float]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start > 1e-6:
            scene_cuts.append((start, end))
    return scene_cuts

def save_scenes(video_path: str, scene_cuts: List[Tuple[float, float]], output_dir: str):
    """
    Save each scene as a separate video file using ffmpeg.

    Args:
        video_path (str): Path to the input video file.
        scene_cuts (List[Tuple[float, float]]): List of tuples containing start and end timestamps (seconds) of each scene.
        output_dir (str): Directory to save the output scene videos.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not scene_cuts:
        raise RuntimeError("No scene ranges to save.")

    for i, (start_time, end_time) in enumerate(scene_cuts):
        if end_time <= start_time:
            continue
        output_path = os.path.join(output_dir, f"scene_{i:03d}.mp4")
        _encode_scene_with_fallback(video_path, start_time, end_time, output_path)
        
if __name__ == "__main__":    
    video_path = "/home/itec/emanuele/Datasets/input/federer_djokovic.mp4"  # Path to your input video
    output_dir = "/home/itec/emanuele/Datasets/federer_djokovic/libsvtav1_crf15_pre3"  # Directory to save the output scene videos
    scene_cuts = find_scene_cuts(video_path)
    print(f"Detected {len(scene_cuts)} scene(s)")
    save_scenes(video_path, scene_cuts, output_dir)
    print(f"Saved scenes to: {output_dir}")
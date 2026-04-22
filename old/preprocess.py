# Take a video, find scene cuts, and save each scene as a separate video file.

import argparse
import os
import subprocess
import json
from typing import List, Tuple, Optional


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


def _codec_params_to_ffmpeg_args(codec_params: List[str]) -> List[str]:
    args: List[str] = []
    for param in codec_params:
        token = param.strip()
        if not token:
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if not key.startswith("-"):
                key = f"-{key}"
            args.extend([key, value])
        else:
            args.append(token)
    return args


def _encode_clip(
    input_path: str,
    output_path: str,
    codec: str,
    codec_args: List[str],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-y",
    ]

    if start_time is not None and end_time is not None:
        command.extend(["-ss", f"{start_time:.6f}", "-to", f"{end_time:.6f}"])

    command.extend([
        "-i", input_path,
        "-map", "0:v:0",
        "-c:v", codec,
    ])
    command.extend(codec_args)
    command.extend(["-an", output_path])

    result = _run_encode_command(command)
    if result.returncode != 0:
        raise RuntimeError(
            "Encoding failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stderr:\n{result.stderr}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Encoding failed to produce valid output file: {output_path}")


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


def _scene_cuts_cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, "scene_cuts.json")


def _video_fingerprint(video_path: str) -> dict:
    stat = os.stat(video_path)
    return {
        "video_path": os.path.abspath(video_path),
        "video_size": stat.st_size,
        "video_mtime": stat.st_mtime,
    }


def _save_scene_cuts_cache(
    cache_path: str,
    video_path: str,
    threshold: float,
    scene_cuts: List[Tuple[float, float]],
) -> None:
    payload = {
        "version": 1,
        "threshold": threshold,
        "fingerprint": _video_fingerprint(video_path),
        "scene_cuts": [[start, end] for start, end in scene_cuts],
    }
    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _load_scene_cuts_cache(
    cache_path: str,
    video_path: str,
    threshold: float,
) -> Optional[List[Tuple[float, float]]]:
    if not os.path.exists(cache_path):
        return None

    with open(cache_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if payload.get("version") != 1:
        return None

    cached_threshold = float(payload.get("threshold", -1.0))
    if abs(cached_threshold - threshold) > 1e-9:
        return None

    cached_fingerprint = payload.get("fingerprint", {})
    current_fingerprint = _video_fingerprint(video_path)
    if cached_fingerprint != current_fingerprint:
        return None

    scene_cuts_raw = payload.get("scene_cuts", [])
    scene_cuts: List[Tuple[float, float]] = []
    for pair in scene_cuts_raw:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        start_time = float(pair[0])
        end_time = float(pair[1])
        if end_time > start_time:
            scene_cuts.append((start_time, end_time))

    return scene_cuts if scene_cuts else None


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


def get_or_create_scene_cuts(
    video_path: str,
    output_dir: str,
    threshold: float = 0.35,
) -> List[Tuple[float, float]]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cache_path = _scene_cuts_cache_path(output_dir)
    cached_scene_cuts = _load_scene_cuts_cache(cache_path, video_path, threshold)
    if cached_scene_cuts is not None:
        print(f"Loaded {len(cached_scene_cuts)} scene(s) from cache: {cache_path}")
        return cached_scene_cuts

    scene_cuts = find_scene_cuts(video_path, threshold=threshold)
    _save_scene_cuts_cache(cache_path, video_path, threshold, scene_cuts)
    print(f"Detected {len(scene_cuts)} scene(s) and saved cache: {cache_path}")
    return scene_cuts

def save_scenes(
    video_path: str,
    scene_cuts: List[Tuple[float, float]],
    output_dir: str,
    codec: str,
    codec_params: List[str],
):
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

    codec_args = _codec_params_to_ffmpeg_args(codec_params)

    for i, (start_time, end_time) in enumerate(scene_cuts):
        if end_time <= start_time:
            continue
        output_path = os.path.join(output_dir, f"scene_{i:03d}.mp4")
        _encode_clip(
            input_path=video_path,
            output_path=output_path,
            codec=codec,
            codec_args=codec_args,
            start_time=start_time,
            end_time=end_time,
        )


def transcode_scene_folder(input_dir: str, output_dir: str, codec: str, codec_params: List[str]) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scene_files = []
    for name in sorted(os.listdir(input_dir)):
        lower_name = name.lower()
        if lower_name.startswith("scene_") and lower_name.endswith((".mp4", ".mov", ".mkv", ".webm")):
            scene_files.append(name)

    if not scene_files:
        raise RuntimeError(f"No scene files found in folder: {input_dir}")

    codec_args = _codec_params_to_ffmpeg_args(codec_params)
    skipped_count = 0
    encoded_count = 0
    for name in scene_files:
        input_path = os.path.join(input_dir, name)
        output_path = os.path.join(output_dir, name)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            skipped_count += 1
            print(f"Skipping existing scene: {output_path}")
            continue
        _encode_clip(
            input_path=input_path,
            output_path=output_path,
            codec=codec,
            codec_args=codec_args,
        )
        encoded_count += 1

    print(
        f"Folder transcoding complete. Encoded: {encoded_count}, "
        f"skipped existing: {skipped_count}."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a video into scenes and encode, or transcode a scene folder.")
    parser.add_argument("--video-path", required=True, help="Input video path or a folder containing scene_*.mp4 clips.")
    parser.add_argument("--output-path", required=True, help="Output folder path.")
    parser.add_argument("--codec", required=True, help="FFmpeg video codec name (e.g., libsvtav1, av1_nvenc, libx264).")
    parser.add_argument(
        "--codec-params",
        nargs="*",
        default=[],
        help="Codec params as key=value tokens (example: crf=25 preset=4 pix_fmt=yuv420p).",
    )
    parser.add_argument("--threshold", type=float, default=0.35, help="Scene detection threshold (file mode only).")
    return parser.parse_args()
        

if __name__ == "__main__":
    args = _parse_args()
    video_path = args.video_path
    output_path = args.output_path
    codec = args.codec
    codec_params = args.codec_params

    if os.path.isdir(video_path):
        transcode_scene_folder(video_path, output_path, codec, codec_params)
        print(f"Transcoded scene folder to: {output_path}")
    else:
        scene_cuts = get_or_create_scene_cuts(video_path, output_path, threshold=args.threshold)
        save_scenes(video_path, scene_cuts, output_path, codec=codec, codec_params=codec_params)
        print(f"Saved scenes to: {output_path}")
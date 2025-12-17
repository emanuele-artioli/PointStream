from dataclasses import dataclass
import os
import subprocess
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

import tempfile
import json

import cv2
import numpy as np


@dataclass
class PointStreamConfig:
    reference_video: str = "/home/itec/emanuele/Datasets/DAVIS/avc_encoded/bear.mp4"
    experiment_folder: str = "/home/itec/emanuele/pointstream/experiments"
    width: int = 640  # experiment resolution
    height: int = 360  # experiment resolution
    frame_stride: int = 1  # Frame sampling stride (1=all frames, 2=every other frame, etc.)
    max_frames: int = 20  # Limit frames for testing (None = all frames)
    framerate: float = None  # If None, use source video framerate
    codecs = ["libx264", "libx265", "libsvtav1"]
    # Use CRF values (lower = higher quality).
    crf_values = {
        "libx264": [25, 30, 35, 40, 45, 50],
        "libx265": [25, 30, 35, 40, 45, 50],
        "libsvtav1": [38, 43, 48, 53, 58, 63],
    }

# encode with ffmpeg, returning success status
def encode_video(input_path: str, output_path: str, framerate: float, codec: str, crf: int = None) -> bool:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-i", input_path,
        "-c:v", codec,
        "-r", str(framerate),
        "-pix_fmt", "yuv420p",
    ]

    # Use CRF where supported (libx264, libx265, libsvtav1 accept -crf)
    if crf is not None:
        command += ["-crf", str(crf)]

    # Suppress encoder-specific info messages
    if codec == "libx265":
        # x265 accepts x265-params; set log-level to warning to silence info messages
        command += ["-x265-params", "log-level=warning"]
    elif codec == "libsvtav1":
        # SVT-AV1 accepts svtav1-params; set log-level to warning to silence info messages
        command += ["-svtav1-params", "progress=0"]

    command.append(output_path)

    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Encoding failed for codec={codec}, crf={crf}: {e}")
        return False

# compute vmaf score between two videos, returning the score
def compute_vmaf(reference_path: str, distorted_path: str, framerate: float) -> float:
    # Use ffmpeg's libvmaf filter and write JSON log, then parse the JSON to get the aggregate VMAF score.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        json_path = tmp.name

        # Escape the log path for use in the filter string
        escaped_log_path = json_path.replace(":", "\\:")
        
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-i", distorted_path,
            "-i", reference_path,
            "-filter_complex",
            f"[0:v]scale=1920x1080:flags=bicubic[main];[1:v]scale=1920x1080:flags=bicubic,format=pix_fmts=yuv420p,fps=fps={framerate}[ref];[main][ref]libvmaf=log_fmt=json:log_path={escaped_log_path}",
            "-f", "null", "-"
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)

            # Read and parse JSON log
            with open(json_path, 'r') as f:
                vmaf_data = json.load(f)
                aggregate_vmaf = vmaf_data['pooled_metrics']['vmaf']['mean']
                return aggregate_vmaf
        except subprocess.CalledProcessError as e:
            print(f"VMAF computation failed: {e}")
            print(f"stderr: {e.stderr}")
            return None
        except Exception as e:
            print(f"Error reading VMAF results: {e}")
            return None
        finally:
            if os.path.exists(json_path):
                os.remove(json_path)
 
# Encode video with ffmpeg given a list of codecs and crf values, returning the vmaf score and size for each.
def compute_bdrates(input_path: str, output_folder: str, framerate: float, codecs: List[str], crf_values: Dict[str, List[int]]) -> Dict[Tuple[str, int], Tuple[float, int]]:
    results = {}
    for codec in codecs:
        for crf in crf_values.get(codec, []):
            start_time = datetime.now()
            encoded_path = os.path.join(output_folder, f"{codec}_crf{crf}.mp4")
            success = encode_video(input_path, encoded_path, framerate, codec, crf)
            encoding_time = datetime.now() - start_time
            if not success:
                continue
            vmaf_score = compute_vmaf(input_path, encoded_path, framerate)
            try:
                file_size = os.path.getsize(encoded_path)
            except Exception:
                file_size = 0
            results[(codec, crf)] = (vmaf_score, file_size)
            # Rename encoded file to include VMAF score and elapsed time
            if vmaf_score is not None:
                new_encoded_path = os.path.join(output_folder, f"{codec}_crf{crf}_vmaf{int(np.around(vmaf_score, 0))}_time{int(np.around(encoding_time / timedelta(milliseconds=1), 0))}ms.mp4")
                os.rename(encoded_path, new_encoded_path)
    return results

# Load video frames into numpy array with OpenCV
def load_frames(video_path: str, frame_stride: int, max_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_stride == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        frame_count += 1
    cap.release()
    return np.array(frames)


if __name__ == "__main__":
    config = PointStreamConfig()

    # Name experiment with current timestamp
    experiment_path = os.path.join(config.experiment_folder, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(experiment_path, exist_ok=True)

    config.framerate = config.framerate or float(cv2.VideoCapture(config.reference_video).get(cv2.CAP_PROP_FPS)) / config.frame_stride

    # Compute BDRates
    bdrate_results = compute_bdrates(
        input_path=config.reference_video,
        output_folder=experiment_path,
        framerate=config.framerate,
        codecs=config.codecs,
        crf_values=config.crf_values,
    )

    
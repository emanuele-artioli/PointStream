"""
Utility functions for PointStream.
"""


import csv
import json
import os
import numpy as np
import torch
import cv2
import subprocess
import shutil
import sys
from datetime import datetime
from pathlib import Path
from torch import nn
# YOLO is required only by the server, so we import it lazily in the functions that need it to avoid forcing a hard dependency on the client environment.
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
    8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
    47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

DEFAULT_ANIMATE_ANYONE_DIR = Path("/home/itec/emanuele/Moore-AnimateAnyone")


def load_video_frames(video_path: str) -> list[np.ndarray]:
    """
    Load video frames from a video file.
    Args:
        video_path: Path to the video file
    Returns:
        A list of video frames as numpy arrays
    """
    
    ffprobe_command = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {video_path}"
    ffprobe_process = subprocess.Popen(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    ffprobe_output, ffprobe_error = ffprobe_process.communicate()
    width, height = ffprobe_output.decode().strip().split("x")
    ffmpeg_command = f"ffmpeg -i {video_path} -vf 'scale={width}:{height}' -f image2pipe -vcodec rawvideo -pix_fmt bgr24 -"
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    
    frames = []
    while True:
        raw_frame = process.stdout.read(int(width) * int(height) * 3)  # width * height * channels
        if not raw_frame:
            break
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((int(height), int(width), 3))
        frames.append(frame)
    
    return frames


def resize_and_pad_image(
    image: np.ndarray,
    image_size: int | tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
    """
    Resize an image while preserving aspect ratio, then pad with black bands.

    Args:
        image: Input image as a 2D mask or 3D image array.
        image_size: Target output size as int (square) or (height, width).
        interpolation: OpenCV interpolation mode used for resizing.

    Returns:
        Resized and padded image with the requested output size.
    """

    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array.")
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W) or (H, W, C).")

    if isinstance(image_size, int):
        target_h = image_size
        target_w = image_size
    elif isinstance(image_size, tuple) and len(image_size) == 2:
        target_h = int(image_size[0])
        target_w = int(image_size[1])
    else:
        raise ValueError("image_size must be an int or a (height, width) tuple.")

    if target_h <= 0 or target_w <= 0:
        raise ValueError("image_size values must be positive.")

    src_h, src_w = image.shape[:2]
    if src_h == 0 or src_w == 0:
        raise ValueError("image must have non-zero spatial dimensions.")

    if src_w >= src_h:
        scale = target_w / float(src_w)
    else:
        scale = target_h / float(src_h)

    # Fallback for mismatched source/target aspect ratios where one dimension could overflow.
    scale = min(scale, target_w / float(src_w), target_h / float(src_h))

    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)

    if image.ndim == 2:
        padded = np.zeros((target_h, target_w), dtype=image.dtype)
    else:
        padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)

    pad_top = (target_h - resized_h) // 2
    pad_left = (target_w - resized_w) // 2
    padded[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = resized

    return padded


def export_reid_model_to_tensorrt(
    model_path: str,
    half: bool = True,
    dynamic: bool = True,
    batch: int = 32,
    imgsz: int | tuple[int, int] = 640,
    overwrite: bool = False,
    ) -> str:
    """
    Export a YOLO classification model to TensorRT for BoT-SORT ReID.

    The exported `.engine` file is placed in the same folder as `model_path`.

    Args:
        model_path: Path to the source YOLO classification model (`.pt` or `.engine`).
        half: Export in FP16 mode.
        dynamic: Enable dynamic TensorRT shapes.
        batch: Max batch size for TensorRT optimization.
        imgsz: Export input size. Use 640 (or larger) for BoT-SORT ReID compatibility.
        overwrite: Replace an existing `.engine` file if present.

    Returns:
        Absolute path to the TensorRT engine model.
    """

    source_path = Path(model_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Model not found: {source_path}")

    if source_path.suffix == ".engine":
        return str(source_path)

    target_path = source_path.with_suffix(".engine")
    if target_path.exists() and not overwrite:
        return str(target_path)
    if target_path.exists() and overwrite:
        target_path.unlink()

    _require_ultralytics()
    model = YOLO(str(source_path))

    head = model.model.model[-1]
    pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1))
    pool.f, pool.i = head.f, head.i
    model.model.model[-1] = pool

    exported = Path(model.export(format="engine", half=half, dynamic=dynamic, batch=batch, imgsz=imgsz))
    exported_path = exported if exported.is_absolute() else Path.cwd() / exported
    exported_path = exported_path.resolve()

    if exported_path.suffix != ".engine":
        candidates = sorted(exported_path.parent.glob("*.engine"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError("TensorRT export completed but no .engine file was found.")
        exported_path = candidates[0].resolve()

    if exported_path != target_path:
        shutil.move(str(exported_path), str(target_path))

    return str(target_path)


def run_yolo(
    frames: list[np.ndarray], 
    exp_folder: str,
    classes: list = ["person"],
    imgsz: tuple = (360, 640),
    max_det: int = 50,
    model: str | object = "/home/itec/emanuele/Models/YOLO/yolo26x.pt",
    tracker: str | None = "/home/itec/emanuele/Models/YOLO/trackers/botsort.yaml",
    task: str = "detect",
    ) -> dict:

    """
    Run YOLO detection or segmentation with optional tracking.

    Args:
        frames: List of video frames as numpy arrays
        exp_folder: Path to the experiment folder where results will be saved
        classes: List of COCO class names to include
        imgsz: Image size for model inference
        max_det: Maximum detections per frame
        model: Path to YOLO model weights or preloaded model
        tracker: Tracker configuration path. If None, uses predict mode.
        task: Either "detect" (default) or "segment".

    Yields:
        A dictionary with frame metadata, classes, bboxes, track ids, and optionally masks when task="segment".
    """
    
    
    def _patch_botsort_reid_singleton_embeddings() -> None:
        """
        Patch Ultralytics BoT-SORT ReID to normalize singleton embedding shapes.

        With TensorRT ReID backends, Ultralytics can occasionally return per-detection
        features with shape ``(1, D)`` (instead of ``(D,)``) when only one detection
        is present. This later becomes a 3D array in embedding-distance matching and
        crashes with ``ValueError: XB must be a 2-dimensional array``.
        """

        try:
            from ultralytics.trackers.bot_sort import ReID
        except Exception:
            return

        if getattr(ReID, "_pointstream_singleton_patch", False):
            return

        original_call = ReID.__call__

        def _patched_call(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
            raw_features = original_call(self, img, dets)
            normalized_features: list[np.ndarray] = []
            for feature in raw_features:
                feature_array = np.asarray(feature, dtype=np.float32)
                if feature_array.ndim > 1:
                    feature_array = feature_array.reshape(-1)
                normalized_features.append(feature_array)
            return normalized_features

        ReID.__call__ = _patched_call
        ReID._pointstream_singleton_patch = True


    if task not in {"detect", "segment"}:
        raise ValueError(f"Invalid task '{task}'. Expected 'detect' or 'segment'.")
    
    yolo_model = YOLO(model) if isinstance(model, str) else model
    
    common_kwargs = dict(
        source=frames,
        conf=0.25,
        iou=0.45,
        imgsz=imgsz,
        half=True,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        batch=16,
        max_det=max_det,
        classes=[k for k, v in COCO_CLASSES.items() if v in classes],
        retina_masks=True,
        project=exp_folder,
        stream=True,
        verbose=False,
    )

    if tracker is None:
        results = yolo_model.predict(**common_kwargs)
    else:
        if "botsort" in str(tracker).lower():
            _patch_botsort_reid_singleton_embeddings()
        results = yolo_model.track(
            tracker=tracker,  # TODO: ablation test of running with and without tracking, tracker with different configurations, and tracker converted to TensorRT or not, and comparing speed and accuracy.
            persist=True,
            **common_kwargs,
            # compile=True,
        )

    for r in results:
        output = {
            "frame": r.orig_img,
            "resolution": r.orig_shape,
            "track_ids": r.boxes.id,
            "class_names": [COCO_CLASSES.get(int(cls)) for cls in r.boxes.cls],
            "bboxes": r.boxes.xyxy,
            "speed": r.speed,
        }
        if task == "segment":
            output["masks"] = r.masks.data if r.masks is not None else None
        yield output
        

def stitch_panorama(
    frames: list[np.ndarray],
    ) -> tuple[np.ndarray, dict]:
    """
    Stitch video frames into a panorama and return metadata needed for reversal.

    Returns:
        A tuple of (panorama_image, panorama_data).
        `panorama_data` contains frame size, canvas size, translation matrix,
        and homographies needed by `animate_panorama`.
    """

    if not frames:
        raise ValueError("frames must contain at least one frame")

    base_h, base_w = frames[0].shape[:2]
    if any(frame.shape[:2] != (base_h, base_w) for frame in frames):
        raise ValueError("all frames must have the same spatial resolution")

    if len(frames) == 1:
        identity = np.eye(3, dtype=np.float64)
        panorama_data = {
            "num_frames": 1,
            "frame_width": int(base_w),
            "frame_height": int(base_h),
            "canvas_width": int(base_w),
            "canvas_height": int(base_h),
            "translation": identity.copy(),
            "global_homographies": [identity.copy()],
            "composite_homographies": [identity.copy()],
        }
        return frames[0].copy(), panorama_data

    orb = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def estimate_pairwise_homography(current_frame: np.ndarray, previous_frame: np.ndarray) -> np.ndarray:
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        current_kp, current_des = orb.detectAndCompute(current_gray, None)
        previous_kp, previous_des = orb.detectAndCompute(previous_gray, None)

        if current_des is None or previous_des is None:
            return np.eye(3, dtype=np.float64)

        matches = matcher.match(current_des, previous_des)
        if len(matches) < 4:
            return np.eye(3, dtype=np.float64)

        matches = sorted(matches, key=lambda m: m.distance)
        best_matches = matches[: min(200, len(matches))]

        src_pts = np.float32([current_kp[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([previous_kp[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography is None or not np.all(np.isfinite(homography)):
            return np.eye(3, dtype=np.float64)

        return homography

    global_h = [np.eye(3, dtype=np.float64)]
    cumulative_h = np.eye(3, dtype=np.float64)

    for frame_idx in range(1, len(frames)):
        h_rel = estimate_pairwise_homography(frames[frame_idx], frames[frame_idx - 1])
        cumulative_h = cumulative_h @ h_rel
        global_h.append(cumulative_h.copy())

    corners = np.float32([[0, 0], [0, base_h], [base_w, base_h], [base_w, 0]]).reshape(-1, 1, 2)
    warped_corners = [cv2.perspectiveTransform(corners, homography) for homography in global_h]
    all_corners = np.concatenate(warped_corners, axis=0)

    xmin, ymin = all_corners.min(axis=0).ravel()
    xmax, ymax = all_corners.max(axis=0).ravel()

    tx = -float(xmin)
    ty = -float(ymin)
    translation = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)

    canvas_w = max(1, int(np.ceil(float(xmax - xmin))))
    canvas_h = max(1, int(np.ceil(float(ymax - ymin))))
    panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    composite_homographies = []

    for frame, homography in zip(frames, global_h):
        homography_to_panorama = translation @ homography
        composite_homographies.append(homography_to_panorama.copy())
        warped = cv2.warpPerspective(frame, homography_to_panorama, (canvas_w, canvas_h))
        mask = np.any(warped != 0, axis=2)
        panorama[mask] = warped[mask]

    panorama_data = {
        "num_frames": int(len(frames)),
        "frame_width": int(base_w),
        "frame_height": int(base_h),
        "canvas_width": int(canvas_w),
        "canvas_height": int(canvas_h),
        "translation": translation.copy(),
        "global_homographies": [homography.copy() for homography in global_h],
        "composite_homographies": composite_homographies,
    }

    return panorama, panorama_data


def animate_panorama(
    panorama: np.ndarray,
    panorama_data: dict,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: tuple[int, int, int] = (0, 0, 0),
    ) -> list[np.ndarray]:
    """
    Reconstruct background frames from a panorama using stitch metadata.

    Args:
        panorama: Stitched panorama image.
        panorama_data: Metadata produced by `stitch_panorama`.
        interpolation: OpenCV interpolation flag for warping.
        border_mode: OpenCV border mode for out-of-bounds pixels.
        border_value: Constant border value when border_mode is BORDER_CONSTANT.

    Returns:
        Reconstructed background frames in original frame coordinates.
    """

    if panorama is None or not isinstance(panorama, np.ndarray):
        raise ValueError("panorama must be a numpy array.")
    if panorama.ndim != 3 or panorama.shape[2] != 3:
        raise ValueError("panorama must be a BGR image with shape (H, W, 3).")
    if not isinstance(panorama_data, dict):
        raise ValueError("panorama_data must be a dictionary returned by stitch_panorama().")

    frame_width = int(panorama_data.get("frame_width", 0))
    frame_height = int(panorama_data.get("frame_height", 0))
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("panorama_data must contain positive frame_width and frame_height values.")

    homographies_to_panorama = panorama_data.get("composite_homographies")
    if homographies_to_panorama is None:
        translation = panorama_data.get("translation")
        global_homographies = panorama_data.get("global_homographies")
        if translation is None or global_homographies is None:
            raise ValueError(
                "panorama_data must contain either composite_homographies or both translation and global_homographies."
            )

        translation_matrix = np.asarray(translation, dtype=np.float64)
        if translation_matrix.shape != (3, 3):
            raise ValueError("translation must have shape (3, 3).")

        homographies_to_panorama = [
            translation_matrix @ np.asarray(homography, dtype=np.float64)
            for homography in global_homographies
        ]

    if len(homographies_to_panorama) == 0:
        return []

    reconstructed_frames = []
    for homography_to_panorama in homographies_to_panorama:
        matrix = np.asarray(homography_to_panorama, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("Each homography must have shape (3, 3).")

        try:
            inverse_homography = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            inverse_homography = np.eye(3, dtype=np.float64)

        frame = cv2.warpPerspective(
            panorama,
            inverse_homography,
            (frame_width, frame_height),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        reconstructed_frames.append(frame)

    return reconstructed_frames


def encode_video_libsvtav1(
    output_path: str,
    fps: float,
    crf: int = 25,
    frame_folder: str | None = None,
    frames: list[np.ndarray] | None = None,
    frame_pattern: str = "%06d.png",
    ) -> str:
    """
    Encode frames to AV1 video with ffmpeg + libsvtav1.

    Provide exactly one input source:
      - frame_folder: directory containing numbered PNG files
      - frames: list of BGR uint8 numpy arrays

    Args:
        output_path: Destination video path.
        fps: Target frame rate.
        crf: AV1 constant rate factor (lower is higher quality).
        frame_folder: Folder with frame images for ffmpeg file input mode.
        frames: In-memory frames for ffmpeg pipe input mode.
        frame_pattern: Input filename pattern when using frame_folder.

    Returns:
        Absolute path of the encoded video.
    """

    using_folder = frame_folder is not None
    using_frames = frames is not None
    if using_folder == using_frames:
        raise ValueError("Provide exactly one of frame_folder or frames.")

    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg executable was not found in PATH.")

    safe_fps = float(fps) if fps and float(fps) > 0 else 25.0
    target_path = Path(output_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if using_folder:
        source_dir = Path(frame_folder).expanduser().resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Frame folder not found: {source_dir}")

        pattern_path = source_dir / frame_pattern
        command = [
            ffmpeg_exe,
            "-y",
            "-framerate",
            str(safe_fps),
            "-i",
            str(pattern_path),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libsvtav1",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(int(crf)),
            "-b:v",
            "0",
            str(target_path),
        ]

        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode != 0:
            stderr = process.stderr.strip()
            raise RuntimeError(f"ffmpeg folder encoding failed: {stderr}")
        return str(target_path)

    if not frames:
        raise ValueError("frames must contain at least one frame.")

    first = frames[0]
    if first is None or not isinstance(first, np.ndarray):
        raise ValueError("frames must be a list of numpy arrays.")
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("frames must be BGR images with shape (H, W, 3).")

    height, width = first.shape[:2]
    target_width = width + (width % 2)
    target_height = height + (height % 2)
    command = [
        ffmpeg_exe,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{target_width}x{target_height}",
        "-r",
        str(safe_fps),
        "-i",
        "-",
        "-c:v",
        "libsvtav1",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(int(crf)),
        "-b:v",
        "0",
        str(target_path),
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            if frame is None or not isinstance(frame, np.ndarray):
                continue
            if frame.ndim != 3 or frame.shape[2] != 3:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if target_width != width or target_height != height:
                frame = cv2.copyMakeBorder(
                    frame,
                    0,
                    target_height - height,
                    0,
                    target_width - width,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        return_code = process.wait()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
        if return_code != 0:
            raise RuntimeError(f"ffmpeg array encoding failed: {stderr.strip()}")
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()

    return str(target_path)


def extract_dwpose_keypoints(
    frames_folder: str,
    det_model: str | None = None,
    pose_model: str | None = None,
    score_threshold: float = 0.3,
    ) -> list[np.ndarray]:
    """
    Run DWPose on a frame folder and return per-frame best-person keypoints.

    Args:
        frames_folder: Folder containing source frame PNGs.
        det_model: Optional path to DWPose detection ONNX model.
        pose_model: Optional path to DWPose pose ONNX model.
        score_threshold: Visibility threshold used to hide low-confidence keypoints.

    Returns:
        List of arrays with shape (N, 2), where columns are [x, y].
        Keypoints with score below ``score_threshold`` are set to -1.
        Empty arrays indicate no detected person for that frame.
    """
    from dwpose import Wholebody, extract_best_person

    source_dir = Path(frames_folder).expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Frames folder not found: {source_dir}")

    frame_paths = sorted(source_dir.glob("*.png"))
    if not frame_paths:
        return []

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    wholebody = Wholebody(device=device, det_model_path=det_model, pose_model_path=pose_model)

    extracted_keypoints: list[np.ndarray] = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            extracted_keypoints.append(np.zeros((0, 2), dtype=np.float32))
            continue

        keypoints, scores = wholebody(frame)
        best_keypoints, best_scores = extract_best_person(keypoints, scores)

        if best_keypoints is None or best_scores is None:
            extracted_keypoints.append(np.zeros((0, 2), dtype=np.float32))
            continue

        keypoint_array = np.asarray(best_keypoints, dtype=np.float32).copy()
        score_array = np.asarray(best_scores, dtype=np.float32)
        keypoint_array[score_array < float(score_threshold)] = -1.0
        extracted_keypoints.append(keypoint_array)

    return extracted_keypoints


def save_dwpose_keypoints_csv(
    keypoint_frames: list[np.ndarray],
    output_csv_path: str,
    frame_indices: list[int] | None = None,
    bboxes: list[np.ndarray | list[float] | tuple[float, float, float, float] | None] | None = None,
    ) -> str:
    """
    Save DWPose keypoints to CSV.

    Args:
        keypoint_frames: Per-frame arrays with columns [x, y].
        output_csv_path: Destination CSV file path.
        frame_indices: Optional source frame indices aligned with ``keypoint_frames``.
        bboxes: Optional per-frame ``[x1, y1, x2, y2]`` bounding boxes aligned with
            ``keypoint_frames``.

    Returns:
        Absolute path to the saved CSV file.
    """

    target_path = Path(output_csv_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    num_frames = len(keypoint_frames)
    resolved_frame_indices = list(range(num_frames)) if frame_indices is None else [int(idx) for idx in frame_indices]
    if len(resolved_frame_indices) != num_frames:
        raise ValueError("frame_indices must have the same length as keypoint_frames.")

    resolved_bboxes = [None] * num_frames if bboxes is None else list(bboxes)
    if len(resolved_bboxes) != num_frames:
        raise ValueError("bboxes must have the same length as keypoint_frames when provided.")

    with target_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "keypoints", "bbox"])
        for frame_index, frame_keypoints, frame_bbox in zip(
            resolved_frame_indices,
            keypoint_frames,
            resolved_bboxes,
        ):
            serialized_keypoints = json.dumps(
                np.asarray(frame_keypoints, dtype=np.float32).tolist(),
                separators=(",", ":"),
            )

            if frame_bbox is None:
                bbox_values: list[float] = []
            else:
                bbox_array = np.asarray(frame_bbox, dtype=np.float32).reshape(-1)
                bbox_values = [float(value) for value in bbox_array[:4]] if bbox_array.size >= 4 else []

            serialized_bbox = json.dumps(bbox_values, separators=(",", ":"))
            writer.writerow([int(frame_index), serialized_keypoints, serialized_bbox])

    return str(target_path)


def load_dwpose_keypoints_csv(
    csv_path: str,
    return_metadata: bool = False,
    ) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """
    Load DWPose keypoints from CSV saved by ``save_dwpose_keypoints_csv``.

    Args:
        csv_path: Path to CSV file containing serialized keypoints.
        return_metadata: If True, also return per-frame bounding boxes and source
            frame indices.

    Returns:
        List of per-frame arrays with columns [x, y].
        When ``return_metadata=True``, returns a tuple:
        ``(keypoints, bboxes, frame_indices)``.
    """

    source_path = Path(csv_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"CSV file not found: {source_path}")

    loaded_keypoints: list[np.ndarray] = []
    loaded_bboxes: list[np.ndarray] = []
    loaded_frame_indices: list[int] = []
    with source_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or "keypoints" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'keypoints' column.")

        has_bbox_column = "bbox" in reader.fieldnames
        has_frame_index_column = "frame_index" in reader.fieldnames

        for row_number, row in enumerate(reader):
            serialized = (row.get("keypoints") or "[]").strip()
            values = json.loads(serialized)
            frame_keypoints = np.asarray(values, dtype=np.float32)

            if frame_keypoints.size == 0:
                loaded_keypoints.append(np.zeros((0, 2), dtype=np.float32))
            else:
                if frame_keypoints.ndim != 2 or frame_keypoints.shape[1] < 2:
                    raise ValueError("Each keypoint row must have at least 2 values [x, y].")
                loaded_keypoints.append(frame_keypoints[:, :2].copy())

            if has_frame_index_column:
                raw_frame_index = (row.get("frame_index") or "").strip()
                if raw_frame_index:
                    try:
                        frame_index_value = int(float(raw_frame_index))
                    except ValueError as exc:
                        raise ValueError(f"Invalid frame_index value at row {row_number}: '{raw_frame_index}'") from exc
                else:
                    frame_index_value = int(row_number)
            else:
                frame_index_value = int(row_number)
            loaded_frame_indices.append(frame_index_value)

            bbox_values: list[float] = []
            if has_bbox_column:
                serialized_bbox = (row.get("bbox") or "[]").strip()
                if serialized_bbox:
                    bbox_parsed = json.loads(serialized_bbox)
                    bbox_array = np.asarray(bbox_parsed, dtype=np.float32).reshape(-1)
                    if bbox_array.size >= 4:
                        bbox_values = [float(value) for value in bbox_array[:4]]

            if bbox_values:
                loaded_bboxes.append(np.asarray(bbox_values, dtype=np.float32))
            else:
                loaded_bboxes.append(np.asarray([-1.0, -1.0, -1.0, -1.0], dtype=np.float32))

    if return_metadata:
        return loaded_keypoints, loaded_bboxes, loaded_frame_indices

    return loaded_keypoints


def scale_frame_to_bbox(
    animate_anyone_frames: list[str | Path | np.ndarray | None],
    bbox_coordinates: list[np.ndarray | list[float] | tuple[float, float, float, float] | None],
    output_frame_size: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
    ) -> list[np.ndarray]:
    """
    Map AnimateAnyone frames back to bbox space by inverting resize+pad geometry.

    Frames are generated from crops previously processed by ``resize_and_pad_image``,
    which first rescales and then adds letterbox padding. This function estimates and
    removes that padding from each generated frame using bbox aspect ratio, then
    rescales the de-letterboxed content to the target bbox.

    Args:
        animate_anyone_frames: Per-frame AnimateAnyone inputs as PNG paths or arrays.
        bbox_coordinates: Per-frame bboxes as ``[x1, y1, x2, y2]``.
        output_frame_size: Canvas size as ``(height, width)``.
        interpolation: OpenCV interpolation mode used during resizing.

    Returns:
        List of RGBA overlays with shape ``(H, W, 4)``.
    """

    if len(animate_anyone_frames) != len(bbox_coordinates):
        raise ValueError("animate_anyone_frames and bbox_coordinates must have the same length.")

    if not isinstance(output_frame_size, tuple) or len(output_frame_size) != 2:
        raise ValueError("output_frame_size must be a (height, width) tuple.")

    output_h = int(output_frame_size[0])
    output_w = int(output_frame_size[1])
    if output_h <= 0 or output_w <= 0:
        raise ValueError("output_frame_size values must be positive.")

    def _invert_resize_and_pad(frame_data: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Undo resize_and_pad_image geometry using target bbox aspect ratio."""

        frame_h, frame_w = frame_data.shape[:2]
        if target_w <= 0 or target_h <= 0:
            return frame_data

        # resize_and_pad_image scales the longer bbox side to the full target side
        # and pads the shorter side centrally. Here we crop that central content.
        if target_w >= target_h:
            content_w = frame_w
            content_h = max(1, int(round(frame_h * (float(target_h) / float(target_w)))))
            content_h = min(content_h, frame_h)
            pad_top = max(0, (frame_h - content_h) // 2)
            pad_bottom = max(0, frame_h - content_h - pad_top)
            cropped = frame_data[pad_top:frame_h - pad_bottom, :]
        else:
            content_h = frame_h
            content_w = max(1, int(round(frame_w * (float(target_w) / float(target_h)))))
            content_w = min(content_w, frame_w)
            pad_left = max(0, (frame_w - content_w) // 2)
            pad_right = max(0, frame_w - content_w - pad_left)
            cropped = frame_data[:, pad_left:frame_w - pad_right]

        if cropped.size == 0:
            return frame_data

        return cv2.resize(cropped, (target_w, target_h), interpolation=interpolation)

    scaled_frames: list[np.ndarray] = []

    for frame_input, frame_bbox in zip(animate_anyone_frames, bbox_coordinates):
        canvas = np.zeros((output_h, output_w, 4), dtype=np.uint8)

        if frame_input is None or frame_bbox is None:
            scaled_frames.append(canvas)
            continue

        if isinstance(frame_input, np.ndarray):
            frame_data = frame_input.copy()
        else:
            frame_path = Path(frame_input).expanduser()
            frame_data = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)

        if frame_data is None:
            scaled_frames.append(canvas)
            continue

        if frame_data.ndim == 2:
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGRA)
        elif frame_data.ndim == 3 and frame_data.shape[2] == 3:
            inferred_alpha = np.where(np.any(frame_data > 0, axis=2), 255, 0).astype(np.uint8)
            frame_data = np.dstack([frame_data, inferred_alpha])
        elif frame_data.ndim != 3 or frame_data.shape[2] < 4:
            scaled_frames.append(canvas)
            continue
        elif frame_data.shape[2] > 4:
            frame_data = frame_data[:, :, :4]

        bbox_array = np.asarray(frame_bbox, dtype=np.float32).reshape(-1)
        if bbox_array.size < 4:
            scaled_frames.append(canvas)
            continue

        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox_array[:4]]
        target_w = x2 - x1
        target_h = y2 - y1
        if target_w <= 0 or target_h <= 0:
            scaled_frames.append(canvas)
            continue

        resized = _invert_resize_and_pad(frame_data, target_w=target_w, target_h=target_h)

        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(output_w, x2)
        dst_y2 = min(output_h, y2)
        if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
            scaled_frames.append(canvas)
            continue

        src_x1 = dst_x1 - x1
        src_y1 = dst_y1 - y1
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]
        scaled_frames.append(canvas)

    return scaled_frames


def overlay_object_on_background_video(
    background_frames: list[np.ndarray],
    object_frames: list[np.ndarray | None],
    ) -> list[np.ndarray]:
    """
    Alpha-composite per-frame object overlays over background video frames.

    Args:
        background_frames: Base BGR video frames.
        object_frames: Per-frame overlays. Supports BGRA with alpha or BGR masks.

    Returns:
        Blended BGR frames.
    """

    if len(background_frames) != len(object_frames):
        raise ValueError("background_frames and object_frames must have the same length.")

    composited_frames: list[np.ndarray] = []

    for background_frame, object_frame in zip(background_frames, object_frames):
        if background_frame is None or not isinstance(background_frame, np.ndarray):
            raise ValueError("Each background frame must be a valid numpy array.")
        if background_frame.ndim != 3 or background_frame.shape[2] != 3:
            raise ValueError("background_frames must contain BGR images with shape (H, W, 3).")

        blended_frame = background_frame.copy()

        if object_frame is None:
            composited_frames.append(blended_frame)
            continue
        if not isinstance(object_frame, np.ndarray):
            composited_frames.append(blended_frame)
            continue

        overlay_frame = object_frame
        if overlay_frame.shape[:2] != blended_frame.shape[:2]:
            overlay_frame = cv2.resize(
                overlay_frame,
                (blended_frame.shape[1], blended_frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        if overlay_frame.ndim == 2:
            overlay_bgr = cv2.cvtColor(overlay_frame, cv2.COLOR_GRAY2BGR)
            alpha = (overlay_frame > 0).astype(np.float32)[..., None]
        elif overlay_frame.ndim == 3 and overlay_frame.shape[2] == 4:
            overlay_bgr = overlay_frame[:, :, :3]
            alpha = (overlay_frame[:, :, 3:4].astype(np.float32) / 255.0)
        elif overlay_frame.ndim == 3 and overlay_frame.shape[2] >= 3:
            overlay_bgr = overlay_frame[:, :, :3]
            alpha = np.any(overlay_bgr > 0, axis=2).astype(np.float32)[..., None]
        else:
            composited_frames.append(blended_frame)
            continue

        if overlay_bgr.dtype != np.uint8:
            overlay_bgr = np.clip(overlay_bgr, 0, 255).astype(np.uint8)

        blended = (
            overlay_bgr.astype(np.float32) * alpha
            + blended_frame.astype(np.float32) * (1.0 - alpha)
        )
        composited_frames.append(np.clip(blended, 0, 255).astype(np.uint8))

    return composited_frames


def convert_dwpose_keypoints_to_skeleton_frames(
    keypoint_frames: list[np.ndarray],
    output_folder: str,
    frame_size: tuple[int, int],
    score_threshold: float = 0.3,
    ) -> None:
    """
    Convert extracted DWPose keypoints into skeleton PNG frames.

    Args:
        keypoint_frames: Output of ``extract_dwpose_keypoints`` with [x, y] values.
        output_folder: Destination folder for per-frame DWPose PNGs.
        frame_size: Output frame size as (height, width).
        score_threshold: Visibility threshold for pose conversion.
    """
    from dwpose import draw_pose, keypoints_to_pose_dict

    if not keypoint_frames:
        return

    if not isinstance(frame_size, tuple) or len(frame_size) != 2:
        raise ValueError("frame_size must be a (height, width) tuple.")
    height = int(frame_size[0])
    width = int(frame_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("frame_size values must be positive.")

    destination_dir = Path(output_folder).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    for existing_frame in destination_dir.glob("*.png"):
        existing_frame.unlink()

    output_size = None
    written_frames = 0

    for frame_keypoints in keypoint_frames:
        keypoint_array = np.asarray(frame_keypoints, dtype=np.float32)

        if keypoint_array.size == 0:
            dwpose_frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            if keypoint_array.ndim != 2 or keypoint_array.shape[1] < 2:
                raise ValueError("Each keypoint frame must have shape (N, >=2) with [x, y].")

            visibility_scores = np.where(
                (keypoint_array[:, 0] >= 0.0) & (keypoint_array[:, 1] >= 0.0),
                1.0,
                0.0,
            ).astype(np.float32)

            pose = keypoints_to_pose_dict(
                keypoint_array[:, :2].copy(),
                visibility_scores,
                width,
                height,
                score_threshold=score_threshold,
            )
            dwpose_frame = draw_pose(pose, height, width)

        if output_size is None:
            output_size = (dwpose_frame.shape[1], dwpose_frame.shape[0])

        if (dwpose_frame.shape[1], dwpose_frame.shape[0]) != output_size:
            dwpose_frame = cv2.resize(dwpose_frame, output_size, interpolation=cv2.INTER_NEAREST)

        frame_output_path = destination_dir / f"{written_frames:06d}.png"
        cv2.imwrite(str(frame_output_path), dwpose_frame)
        written_frames += 1


def save_dwpose(
    frames_folder: str,
    output_folder: str,
    det_model: str | None = None,
    pose_model: str | None = None,
    frame_size: tuple[int, int] | None = None,
    score_threshold: float = 0.3,
    ) -> None:
    """
    Convenience wrapper that extracts DWPose keypoints and writes skeleton frames.

    Args:
        frames_folder: Folder containing source frame PNGs.
        output_folder: Destination folder for per-frame DWPose PNGs.
        det_model: Optional path to DWPose detection ONNX model.
        pose_model: Optional path to DWPose pose ONNX model.
        frame_size: Optional output frame size as (height, width).
        score_threshold: Visibility threshold for pose conversion.
    """
    source_dir = Path(frames_folder).expanduser().resolve()
    destination_dir = Path(output_folder).expanduser().resolve()
    if destination_dir == source_dir:
        raise ValueError("output_folder must be different from frames_folder.")

    keypoint_frames = extract_dwpose_keypoints(
        frames_folder=frames_folder,
        det_model=det_model,
        pose_model=pose_model,
        score_threshold=score_threshold,
    )
    if not keypoint_frames:
        return

    resolved_frame_size = frame_size
    if resolved_frame_size is None:
        source_frame_paths = sorted(source_dir.glob("*.png"))
        source_frame = None
        for frame_path in source_frame_paths:
            source_frame = cv2.imread(str(frame_path))
            if source_frame is not None:
                break
        if source_frame is None:
            return
        source_height, source_width = source_frame.shape[:2]
        resolved_frame_size = (int(source_height), int(source_width))

    convert_dwpose_keypoints_to_skeleton_frames(
        keypoint_frames=keypoint_frames,
        output_folder=output_folder,
        frame_size=resolved_frame_size,
        score_threshold=score_threshold,
    )


def _setup_animate_anyone_import_path(animate_anyone_dir: str | Path) -> Path:
    """Ensure the AnimateAnyone repository is importable by Python."""
    repo_path = Path(animate_anyone_dir).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"AnimateAnyone repository not found: {repo_path}")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    return repo_path


def run_animate_anyone(
    ref_image_path: str,
    skeleton_frames_dir: str,
    config_path: str | None = None,
    animate_anyone_dir: str | Path = DEFAULT_ANIMATE_ANYONE_DIR,
    width: int = 512,
    height: int = 512,
    length: int | None = None,
    steps: int = 30,
    cfg: float = 3.5,
    seed: int = 42,
    save_video: bool = False,
    fps: float | None = None,
    save_dir: str | None = None,
    output_filename: str | None = None,
    include_input_grid: bool = True,
    filename_prefix: str = "frame",
    transparent_threshold: int = 8,
    ) -> str:
    """
    Run AnimateAnyone Pose2Video inference and save either frames (default) or video.

    Default behavior writes RGBA PNG frames with transparent alpha where generated
    background pixels are black. Set ``save_video=True`` to write an MP4 instead.

    Args:
        ref_image_path: Path to the reference RGB image.
        skeleton_frames_dir: Folder containing ordered skeleton PNG frames.
        config_path: AnimateAnyone YAML config path. If relative, it is resolved from
            the AnimateAnyone repository root. Defaults to
            ``configs/prompts/run_finetuned.yaml``.
        animate_anyone_dir: Path to the Moore-AnimateAnyone repository root.
        width: Output frame width.
        height: Output frame height.
        length: Optional cap for sequence length. Uses all available skeleton frames
            when ``None``.
        steps: Diffusion denoising steps.
        cfg: Classifier-free guidance scale.
        seed: Random seed.
        save_video: If True, save MP4 output. If False (default), save PNG frames.
        fps: Output frame rate for video mode. Defaults to 12 when omitted.
        save_dir: Output directory for generated assets.
        output_filename: Optional output video filename in video mode.
        include_input_grid: In video mode, if True save 3-row grid with reference,
            pose input, and generated result.
        filename_prefix: In frame mode, prefix for each output frame name.
        transparent_threshold: In frame mode, pixels with all RGB channels <= this
            value are treated as background (alpha=0).

    Returns:
        Absolute path to generated MP4 (video mode) or output frame directory
        (frame mode).
    """
    # Keep imports local so the rest of PointStream still runs in environments
    # that do not include AnimateAnyone dependencies.
    from PIL import Image
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler
    from einops import repeat
    from omegaconf import OmegaConf
    from torchvision import transforms
    from transformers import CLIPVisionModelWithProjection

    if not torch.cuda.is_available():
        raise RuntimeError("AnimateAnyone Pose2Video requires CUDA, but no CUDA device was detected.")

    threshold = int(transparent_threshold)
    if threshold < 0 or threshold > 255:
        raise ValueError("transparent_threshold must be in [0, 255].")

    aa_dir = _setup_animate_anyone_import_path(animate_anyone_dir)
    default_config = aa_dir / "configs" / "prompts" / "run_finetuned.yaml"
    chosen_config = default_config if config_path is None else Path(config_path).expanduser()
    if not chosen_config.is_absolute():
        chosen_config = aa_dir / chosen_config
    chosen_config = chosen_config.resolve()
    if not chosen_config.exists():
        raise FileNotFoundError(f"AnimateAnyone config not found: {chosen_config}")

    ref_path = Path(ref_image_path).expanduser().resolve()
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")

    skeleton_dir = Path(skeleton_frames_dir).expanduser().resolve()
    if not skeleton_dir.exists():
        raise FileNotFoundError(f"Skeleton folder not found: {skeleton_dir}")

    skeleton_paths = sorted(skeleton_dir.glob("*.png"))
    if not skeleton_paths:
        raise ValueError(f"No skeleton PNG frames found in: {skeleton_dir}")

    previous_cwd = Path.cwd()
    try:
        os.chdir(aa_dir)

        from src.models.pose_guider import PoseGuider
        from src.models.unet_2d_condition import UNet2DConditionModel
        from src.models.unet_3d import UNet3DConditionModel
        from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline

        config = OmegaConf.load(str(chosen_config))
        weight_dtype = torch.float16 if config.weight_dtype == "fp16" else torch.float32

        vae = AutoencoderKL.from_pretrained(
            config.pretrained_vae_path,
        ).to("cuda", dtype=weight_dtype)

        reference_unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=weight_dtype, device="cuda")

        inference_config_path = Path(config.inference_config)
        if not inference_config_path.is_absolute():
            inference_config_path = (aa_dir / inference_config_path).resolve()
        infer_config = OmegaConf.load(str(inference_config_path))
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device="cuda")

        pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
            dtype=weight_dtype,
            device="cuda",
        )

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            config.image_encoder_path,
        ).to(dtype=weight_dtype, device="cuda")

        scheduler_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs, resolve=True)
        scheduler = DDIMScheduler(**scheduler_kwargs)

        denoising_unet.load_state_dict(
            torch.load(config.denoising_unet_path, map_location="cpu"),
            strict=False,
        )
        reference_unet.load_state_dict(
            torch.load(config.reference_unet_path, map_location="cpu"),
            strict=False,
        )
        pose_guider.load_state_dict(
            torch.load(config.pose_guider_path, map_location="cpu"),
            strict=False,
        )

        pipe = Pose2VideoPipeline(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        ).to("cuda", dtype=weight_dtype)

        generator = torch.manual_seed(int(seed))

        with Image.open(ref_path) as ref_image_file:
            ref_image_pil = ref_image_file.convert("RGB")

        max_length = len(skeleton_paths) if length is None else min(int(length), len(skeleton_paths))
        if max_length <= 0:
            raise ValueError("length must be positive when provided.")

        pose_list = []
        for skeleton_path in skeleton_paths[:max_length]:
            with Image.open(skeleton_path) as skeleton_file:
                pose_list.append(skeleton_file.convert("RGB"))

        generated_video = pipe(
            ref_image_pil,
            pose_list,
            int(width),
            int(height),
            int(max_length),
            int(steps),
            float(cfg),
            generator=generator,
        ).videos

        if save_video:
            from src.utils.util import save_videos_grid

            pose_transform = transforms.Compose([
                transforms.Resize((int(height), int(width))),
                transforms.ToTensor(),
            ])

            ref_image_tensor = pose_transform(ref_image_pil).unsqueeze(1).unsqueeze(0)
            ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=max_length)
            pose_tensor = torch.stack([pose_transform(pose_image) for pose_image in pose_list], dim=0)
            pose_tensor = pose_tensor.transpose(0, 1).unsqueeze(0)

            if save_dir is None:
                date_fragment = datetime.now().strftime("%Y%m%d")
                time_fragment = datetime.now().strftime("%H%M")
                save_path = aa_dir / "output" / date_fragment / f"{time_fragment}--pointstream"
            else:
                save_path = Path(save_dir).expanduser()
                if not save_path.is_absolute():
                    save_path = (Path(previous_cwd) / save_path).resolve()
            save_path.mkdir(parents=True, exist_ok=True)

            video_basename = output_filename or f"pose2video_{int(height)}x{int(width)}_cfg{float(cfg):.1f}.mp4"
            output_path = (save_path / video_basename).resolve()
            output_fps = float(fps) if fps and float(fps) > 0 else 12.0

            if include_input_grid:
                video_to_save = torch.cat([ref_image_tensor, pose_tensor, generated_video], dim=0)
                n_rows = 3
            else:
                video_to_save = generated_video
                n_rows = 1

            save_videos_grid(video_to_save, str(output_path), n_rows=n_rows, fps=output_fps)
            return str(output_path)

        if save_dir is None:
            date_fragment = datetime.now().strftime("%Y%m%d")
            time_fragment = datetime.now().strftime("%H%M")
            save_path = aa_dir / "output" / date_fragment / f"{time_fragment}--pointstream-frames"
        else:
            save_path = Path(save_dir).expanduser()
            if not save_path.is_absolute():
                save_path = (Path(previous_cwd) / save_path).resolve()
        save_path.mkdir(parents=True, exist_ok=True)

        for existing_png in save_path.glob("*.png"):
            existing_png.unlink()

        if generated_video.ndim != 5 or generated_video.shape[0] < 1:
            raise RuntimeError("Unexpected output shape from Pose2Video pipeline.")

        video_tensor = generated_video[0].detach().float().cpu()  # (C, F, H, W)
        if video_tensor.shape[0] != 3:
            raise RuntimeError("Expected generated video tensor with 3 color channels.")

        frame_tensor = video_tensor.permute(1, 2, 3, 0)  # (F, H, W, C)
        if float(frame_tensor.min()) < 0.0:
            frame_tensor = (frame_tensor + 1.0) / 2.0
        frame_tensor = frame_tensor.clamp(0.0, 1.0)
        frame_array = (frame_tensor.numpy() * 255.0).round().astype(np.uint8)

        for frame_index, rgb_frame in enumerate(frame_array):
            alpha = np.where(np.all(rgb_frame <= threshold, axis=2), 0, 255).astype(np.uint8)
            rgba_frame = np.dstack([rgb_frame, alpha])
            frame_path = save_path / f"{filename_prefix}_{frame_index:06d}.png"
            Image.fromarray(rgba_frame, mode="RGBA").save(frame_path)

        return str(save_path.resolve())
    finally:
        os.chdir(previous_cwd)



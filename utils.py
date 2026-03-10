"""
Utility functions for PointStream.
"""


import csv
import json
import numpy as np
import torch
import cv2
from ultralytics import YOLO
import subprocess
import shutil
from pathlib import Path
from torch import nn


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
    model: str | YOLO = "/home/itec/emanuele/Models/YOLO/yolo26x.pt",
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
        model: Path to YOLO model weights or preloaded model
        tracker: Tracker configuration path. If None, uses predict mode.
        task: Either "detect" (default) or "segment".

    Yields:
        A dictionary with frame metadata, classes, bboxes, track ids, and optionally masks when task="segment".
    """

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
        batch=4,
        max_det=10,
        classes=[k for k, v in COCO_CLASSES.items() if v in classes],
        retina_masks=True,
        project=exp_folder,
        stream=True,
        verbose=False,
    )

    if tracker is None:
        results = yolo_model.predict(**common_kwargs)
    else:
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
    ) -> str:
    """
    Save DWPose keypoints to CSV.

    Args:
        keypoint_frames: Per-frame arrays with columns [x, y].
        output_csv_path: Destination CSV file path.

    Returns:
        Absolute path to the saved CSV file.
    """

    target_path = Path(output_csv_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with target_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "keypoints"])
        for frame_index, frame_keypoints in enumerate(keypoint_frames):
            serialized = json.dumps(np.asarray(frame_keypoints, dtype=np.float32).tolist(), separators=(",", ":"))
            writer.writerow([int(frame_index), serialized])

    return str(target_path)


def load_dwpose_keypoints_csv(
    csv_path: str,
    ) -> list[np.ndarray]:
    """
    Load DWPose keypoints from CSV saved by ``save_dwpose_keypoints_csv``.

    Args:
        csv_path: Path to CSV file containing serialized keypoints.

    Returns:
        List of per-frame arrays with columns [x, y].
    """

    source_path = Path(csv_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"CSV file not found: {source_path}")

    loaded_keypoints: list[np.ndarray] = []
    with source_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or "keypoints" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'keypoints' column.")

        for row in reader:
            serialized = (row.get("keypoints") or "[]").strip()
            values = json.loads(serialized)
            frame_keypoints = np.asarray(values, dtype=np.float32)

            if frame_keypoints.size == 0:
                loaded_keypoints.append(np.zeros((0, 2), dtype=np.float32))
                continue
            if frame_keypoints.ndim != 2 or frame_keypoints.shape[1] < 2:
                raise ValueError("Each keypoint row must have at least 2 values [x, y].")

            loaded_keypoints.append(frame_keypoints[:, :2].copy())

    return loaded_keypoints


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



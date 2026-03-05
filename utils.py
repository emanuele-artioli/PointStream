"""
Utility functions for PointStream.
"""


import numpy as np
import torch
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
        
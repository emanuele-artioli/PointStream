"""Hand matte helpers: SAM color masks, letterboxed alpha, DWB2 pose roundtrip.

The generator is trained to paint the hand on black and to emit the hand's
alpha. The alpha target is the hand class in the segmentation video, cropped
with the same letterbox as the RGB crop. Pose samples are the landmarks that
survive a DWB2 encode/decode, which is the packet the demo actually sends.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from demo.pipeline.foreground_segmenter import letterbox_crop
from demo.pipeline.hand_keypoints import FrameHandPose
from demo.pipeline.keypoint_compressor import KeypointCompressor
from demo.pipeline.maps.pose_delta import decode_pose_stream, encode_pose_stream

_EMPTY_FACE = b"PF\x00"
_EMPTY_BODY = b"PB\x00"


def hand_alpha_bgr(frame: np.ndarray) -> np.ndarray:
    """Hand class in a colored mask frame. SAM and YOLOE paint the hand red."""
    red = frame[:, :, 2].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 0].astype(np.int16)
    hand = (red > 70) & (red > green + 25) & (red > blue + 25)
    return np.where(hand, np.uint8(255), np.uint8(0))


def letterbox_alpha(mask: np.ndarray, bbox: list[int], target_size: int = 256) -> np.ndarray:
    """Nearest-neighbor letterbox so the alpha lines up with `letterbox_crop`."""
    gray = mask if mask.ndim == 2 else mask[:, :, 0]
    packed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas, _ = letterbox_crop(packed, bbox, target_size=target_size, interpolation=cv2.INTER_NEAREST)
    return canvas[:, :, 0]


def matte_bgr(crop: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Keep crop pixels where alpha is set. Everywhere else is black."""
    out = np.zeros_like(crop)
    out[alpha > 127] = crop[alpha > 127]
    return out


def read_hand_alphas(path: Path, width: int, height: int, max_frames: int) -> list[np.ndarray]:
    """Decode a colored mask video into full-frame hand alphas."""
    frames = _read_bgr(path, width, height, max_frames)
    return [hand_alpha_bgr(frame) for frame in frames]


def union_hand_alphas(primary: list[np.ndarray], fallback: list[np.ndarray]) -> list[np.ndarray]:
    """A pixel is hand if either segmentation says so. Covers frames one model missed."""
    count = max(len(primary), len(fallback))
    out: list[np.ndarray] = []
    for index in range(count):
        left = primary[index] if index < len(primary) else None
        right = fallback[index] if index < len(fallback) else None
        if left is None or right is None:
            chosen = left if left is not None else right
            if chosen is None:
                continue
            out.append(chosen)
            continue
        if right.shape[:2] != left.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
        out.append(np.maximum(left, right))
    return out


def _read_bgr(path: Path, width: int, height: int, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
        frames.append(frame)
    cap.release()
    if frames:
        return frames
    ffmpeg = "ffmpeg"
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-frames:v",
        str(max_frames),
        "-vf",
        f"scale={width}:{height}:flags=neighbor",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0 or not res.stdout:
        tail = res.stderr.decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"could not read mask video {path}: {tail}")
    frame_bytes = width * height * 3
    count = min(max_frames, len(res.stdout) // frame_bytes)
    return [
        np.frombuffer(res.stdout[i * frame_bytes : (i + 1) * frame_bytes], dtype=np.uint8).reshape(height, width, 3)
        for i in range(count)
    ]


def dwb2_roundtrip(
    poses: list[FrameHandPose],
    frame_w: int,
    frame_h: int,
) -> tuple[list[FrameHandPose], int]:
    """Quantize, delta-code, and decode. Returned hands are what the generator may see."""
    packed = [
        (KeypointCompressor.compress_frame(pose, frame_w, frame_h), _EMPTY_FACE, _EMPTY_BODY)
        for pose in poses
    ]
    blob = encode_pose_stream(packed)
    decoded = decode_pose_stream(blob)
    out: list[FrameHandPose] = []
    for pose, (packet, _face, _body) in zip(poses, decoded):
        hands = KeypointCompressor.decompress_frame(packet, frame_w, frame_h)
        out.append(FrameHandPose(frame_idx=pose.frame_idx, hands=hands))
    return out, len(blob)

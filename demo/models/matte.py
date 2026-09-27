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
from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand
from demo.pipeline.keypoint_compressor import KeypointCompressor
from demo.pipeline.maps.pose_delta import decode_pose_stream, encode_pose_stream

_EMPTY_FACE = b"PF\x00"
_EMPTY_BODY = b"PB\x00"

# Mild composite curve. Speckles under the floor go transparent; the rim stays visible.
ALPHA_COMPOSITE_GAMMA = 1.5
ALPHA_SPECKLE_FLOOR = 0.15
# Soft edge width, centered on the SAM contour (half inside, half outside).
ALPHA_FALLOFF_PX = 1.5


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


def soft_edge_alpha(binary_u8: np.ndarray, falloff_px: float = ALPHA_FALLOFF_PX) -> np.ndarray:
    """Binary hand -> [0,1] with a 1–2 px falloff straddling the SAM contour.

    Interior past half the band is opaque. The rim is about 0.5 on the contour,
    fading to 0 half a band outside the mask.
    """
    mask = (binary_u8 > 127).astype(np.uint8)
    if int(mask.max()) == 0:
        return np.zeros(mask.shape, dtype=np.float32)
    half = max(float(falloff_px), 1e-3) / 2.0
    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    alpha = np.zeros(mask.shape, dtype=np.float32)
    alpha[dist_in >= half] = 1.0
    inner = (mask > 0) & (dist_in < half)
    alpha[inner] = 0.5 + 0.5 * (dist_in[inner] / half)
    outer = (mask == 0) & (dist_out > 0) & (dist_out < half)
    alpha[outer] = 0.5 * (1.0 - dist_out[outer] / half)
    return alpha


def steep_alpha(
    alpha: np.ndarray,
    gamma: float = ALPHA_COMPOSITE_GAMMA,
    floor: float = ALPHA_SPECKLE_FLOOR,
) -> np.ndarray:
    """Drop speckles under the floor, then a mild gamma. Not a hard 0.5 cut."""
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    a = np.where(a < float(floor), np.float32(0.0), a)
    return np.power(a, float(gamma))


def interpolate_hand_alphas(
    alphas: list[np.ndarray],
    *,
    min_hand_px: int = 200,
    max_gap: int = 8,
) -> list[np.ndarray]:
    """Fill holes of a few frames by linear blend of the nearest valid masks."""
    if not alphas:
        return alphas
    valid = [i for i, m in enumerate(alphas) if int(np.count_nonzero(m)) >= min_hand_px]
    if not valid:
        return alphas
    out = [m.copy() for m in alphas]
    for i in range(len(out)):
        if int(np.count_nonzero(out[i])) >= min_hand_px:
            continue
        prev = max((v for v in valid if v < i), default=None)
        nxt = min((v for v in valid if v > i), default=None)
        if prev is None and nxt is None:
            continue
        if prev is None:
            if nxt - i <= max_gap:
                out[i] = alphas[nxt].copy()
            continue
        if nxt is None:
            if i - prev <= max_gap:
                out[i] = alphas[prev].copy()
            continue
        gap = nxt - prev
        if gap > max_gap * 2:
            # Prefer nearer neighbor if the hole is long.
            out[i] = alphas[prev].copy() if (i - prev) <= (nxt - i) else alphas[nxt].copy()
            continue
        t = (i - prev) / float(gap)
        left = alphas[prev].astype(np.float32)
        right = alphas[nxt].astype(np.float32)
        if right.shape != left.shape:
            right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
        blended = (1.0 - t) * left + t * right
        out[i] = np.where(blended >= 127.5, np.uint8(255), np.uint8(0))
    return out


def matte_bgr(crop: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Keep crop pixels where alpha is set. Everywhere else is black."""
    out = np.zeros_like(crop)
    a = alpha
    if a.dtype != np.float32 and a.dtype != np.float64:
        a = (a.astype(np.float32) / 255.0) if a.max() > 1 else a.astype(np.float32)
    else:
        a = a.astype(np.float32)
    if a.ndim == 2:
        a3 = a[:, :, None]
    else:
        a3 = a
    out = (crop.astype(np.float32) * a3).astype(np.uint8)
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


def _clamp_box(x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int) -> list[int]:
    x1 = max(0, min(frame_w - 2, x1))
    y1 = max(0, min(frame_h - 2, y1))
    x2 = max(x1 + 1, min(frame_w, x2))
    y2 = max(y1 + 1, min(frame_h, y2))
    return [x1, y1, x2, y2]


def stabilize_hand_boxes(
    poses: list[FrameHandPose],
    frame_w: int,
    frame_h: int,
    *,
    smooth: float = 0.35,
    scale_tol: float = 0.10,
) -> list[FrameHandPose]:
    """Track a smoothed crop center and hold width and height until the span changes.

    A one-pixel fingertip wobble used to resize the box and rescale every in-box
    landmark. Size now stays put until width or height moves by more than
    ``scale_tol``. The center still follows the hand, so a translation shifts
    the codes together instead of zooming them. Landmarks stay in absolute pixels.
    """
    center: dict[str, list[float]] = {}
    size: dict[str, list[float]] = {}
    out: list[FrameHandPose] = []
    for pose in poses:
        hands: list[SingleHand] = []
        for hand in pose.hands:
            key = hand.handedness.lower()
            x1, y1, x2, y2 = (float(v) for v in hand.bbox)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
            if key not in center:
                center[key] = [cx, cy]
                size[key] = [w, h]
            else:
                pcx, pcy = center[key]
                center[key] = [
                    smooth * cx + (1.0 - smooth) * pcx,
                    smooth * cy + (1.0 - smooth) * pcy,
                ]
                hw, hh = size[key]
                if abs(w - hw) / hw > scale_tol or abs(h - hh) / hh > scale_tol:
                    size[key] = [w, h]
            cx, cy = center[key]
            w, h = size[key]
            box = _clamp_box(
                int(round(cx - w / 2.0)),
                int(round(cy - h / 2.0)),
                int(round(cx + w / 2.0)),
                int(round(cy + h / 2.0)),
                frame_w,
                frame_h,
            )
            hands.append(
                SingleHand(
                    handedness=hand.handedness,
                    confidence=hand.confidence,
                    bbox=box,
                    landmarks_norm=hand.landmarks_norm,
                    landmarks_pixel=hand.landmarks_pixel,
                )
            )
        out.append(FrameHandPose(frame_idx=pose.frame_idx, hands=hands))
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
    *,
    stabilize: bool = True,
) -> tuple[list[FrameHandPose], int]:
    """Quantize, delta-code, and decode. Returned hands are what the generator may see."""
    if stabilize:
        poses = stabilize_hand_boxes(poses, frame_w, frame_h)
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
